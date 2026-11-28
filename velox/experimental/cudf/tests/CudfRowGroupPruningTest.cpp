/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/cudf/connectors/hive/CudfRowGroupPruning.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/type/tests/SubfieldFiltersBuilder.h"

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>

#include <cstring>
#include <limits>
#include <numeric>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

namespace pq = cudf::io::parquet;
using Indices = std::vector<cudf::size_type>;
using Footers = std::vector<pq::FileMetaData>;
using common::test::SubfieldFiltersBuilder;
using ::facebook::velox::common::testutil::TempFilePath;

class CudfRowGroupPruningTest : public exec::test::CudfHiveConnectorTestBase {
 protected:
  static Footers readFooters(const std::string& path) {
    std::vector<std::unique_ptr<cudf::io::datasource>> sources;
    sources.push_back(cudf::io::datasource::create(path));
    return cudf::io::read_parquet_footers(sources);
  }

  static std::vector<Indices> allRowGroups(const Footers& files) {
    std::vector<Indices> indices(files.size());
    for (size_t src = 0; src < files.size(); ++src) {
      indices[src].resize(files[src].row_groups.size());
      std::iota(indices[src].begin(), indices[src].end(), 0);
    }
    return indices;
  }

  // Writes one row group per vector and returns the file's path. The file
  // lives until the test ends.
  std::string write(const std::vector<RowVectorPtr>& vectors) {
    tempFiles_.push_back(TempFilePath::create());
    writeToFile(tempFiles_.back()->getPath(), vectors);
    return tempFiles_.back()->getPath();
  }

  // Host decision over every row group of 'files'.
  static HostRowGroupPruning prune(
      const Footers& files,
      const common::SubfieldFilters& filters,
      const RowTypePtr& tableType,
      bool caseInsensitiveColumnNames = false) {
    return pruneRowGroupsWithStatistics(
        files,
        allRowGroups(files),
        prepareRowGroupPruningFilters(filters, *tableType),
        caseInsensitiveColumnNames);
  }

  // Row groups cuDF's statistics pass keeps for the same filters. 'rowType'
  // lists the file's columns in file order; 'decimalTypes' gives the storage
  // type of decimals the file does not store at their Velox width.
  static cudf::size_type cudfKeptRowGroups(
      const std::string& path,
      const common::SubfieldFilters& filters,
      const RowTypePtr& rowType,
      const SubfieldFilterDecimalTypes* decimalTypes = nullptr) {
    cudf::ast::tree tree;
    std::vector<std::unique_ptr<cudf::scalar>> scalars;
    const auto& expr = createAstFromSubfieldFilters(
        filters, tree, scalars, rowType, decimalTypes);
    auto options =
        cudf::io::parquet_reader_options::builder(cudf::io::source_info(path))
            .build();
    options.set_filter(expr);
    const auto result = cudf::io::read_parquet(options);
    return result.metadata.num_row_groups_after_stats_filter.value_or(
        result.metadata.num_input_row_groups);
  }

  // Checks that the host keeps exactly 'expected' of the file's row groups
  // and decides every filter, and that cuDF's statistics pass keeps as many.
  void expectPruned(
      const std::string& path,
      const common::SubfieldFilters& filters,
      const RowTypePtr& rowType,
      const Indices& expected,
      const SubfieldFilterDecimalTypes* decimalTypes = nullptr) {
    const auto result = prune(readFooters(path), filters, rowType);
    ASSERT_EQ(result.rowGroupIndices.size(), 1);
    EXPECT_EQ(result.rowGroupIndices.front(), expected);
    EXPECT_TRUE(result.complete);
    EXPECT_EQ(
        cudfKeptRowGroups(path, filters, rowType, decimalTypes),
        static_cast<cudf::size_type>(expected.size()));
  }

  // Checks that the host keeps exactly 'expected' of the file's row groups
  // and decides every filter, for footers the test patched, which no file
  // holds for cuDF to read.
  static void expectDecided(
      const Footers& files,
      const common::SubfieldFilters& filters,
      const RowTypePtr& tableType,
      const Indices& expected) {
    const auto result = prune(files, filters, tableType);
    ASSERT_EQ(result.rowGroupIndices.size(), 1);
    EXPECT_EQ(result.rowGroupIndices.front(), expected);
    EXPECT_TRUE(result.complete);
  }

  // Checks that the host keeps every row group and leaves the decision to
  // cuDF's statistics pass.
  static void expectLeftToCudf(
      const Footers& files,
      const common::SubfieldFilters& filters,
      const RowTypePtr& tableType) {
    const auto result = prune(files, filters, tableType);
    EXPECT_EQ(result.rowGroupIndices, allRowGroups(files));
    EXPECT_FALSE(result.complete);
  }

  // Checks that the host keeps every row group with nothing left to decide.
  static void expectKept(
      const Footers& files,
      const common::SubfieldFilters& filters,
      const RowTypePtr& tableType,
      bool caseInsensitiveColumnNames = false) {
    const auto result =
        prune(files, filters, tableType, caseInsensitiveColumnNames);
    EXPECT_EQ(result.rowGroupIndices, allRowGroups(files));
    EXPECT_TRUE(result.complete);
  }

  static common::SubfieldFilters filters(
      const std::string& column,
      std::unique_ptr<common::Filter> filter) {
    return SubfieldFiltersBuilder().add(column, std::move(filter)).build();
  }

  static std::unique_ptr<common::Filter> bigintRange(int64_t min, int64_t max) {
    return std::make_unique<common::BigintRange>(min, max, false);
  }

  static std::unique_ptr<common::Filter> bytesRange(
      const std::string& min,
      const std::string& max) {
    return std::make_unique<common::BytesRange>(
        min, false, false, max, false, false, false);
  }

  // The DOUBLE or REAL ranges > or >= 'lower' and < or <= 'upper'.
  template <typename T>
  static std::unique_ptr<common::Filter> above(T lower, bool exclusive) {
    return std::make_unique<common::FloatingPointRange<T>>(
        lower, false, exclusive, T{}, true, false, false);
  }

  template <typename T>
  static std::unique_ptr<common::Filter> below(T upper, bool exclusive) {
    return std::make_unique<common::FloatingPointRange<T>>(
        T{}, true, false, upper, false, exclusive, false);
  }

  static std::unique_ptr<common::Filter> anyOf(
      std::unique_ptr<common::Filter> first,
      std::unique_ptr<common::Filter> second,
      bool nullAllowed = false) {
    std::vector<std::unique_ptr<common::Filter>> filters;
    filters.push_back(std::move(first));
    filters.push_back(std::move(second));
    return std::make_unique<common::MultiRange>(
        std::move(filters), nullAllowed);
  }

  // Three row groups of three ascending values, 1..9, in the column of 'type'
  // built by 'makeColumn(firstValue)'.
  template <typename MakeColumn>
  std::string writeAscending(MakeColumn makeColumn) {
    std::vector<RowVectorPtr> vectors;
    for (int group = 0; group < 3; ++group) {
      vectors.push_back(makeRowVector({"c0"}, {makeColumn(3 * group + 1)}));
    }
    return write(vectors);
  }

  // The file of writeAscending() for integer-backed Velox types.
  template <typename T>
  std::string writeAscendingIntegers(
      const TypePtr& type = CppToType<T>::create()) {
    return writeAscending([&](int64_t first) {
      return makeFlatVector<T>(
          {static_cast<T>(first),
           static_cast<T>(first + 1),
           static_cast<T>(first + 2)},
          type);
    });
  }

  // The file of writeAscending() for DOUBLE and REAL.
  template <typename T>
  std::string writeAscendingFloatingPoints() {
    return writeAscending([&](int64_t first) {
      return makeFlatVector<T>(
          {static_cast<T>(first),
           static_cast<T>(first + 1),
           static_cast<T>(first + 2)});
    });
  }

  // The file of writeAscending() for VARCHAR and VARBINARY: "a".."i".
  std::string writeAscendingStrings(const TypePtr& type = VARCHAR()) {
    return writeAscending([&](int64_t first) {
      return makeFlatVector<std::string>(
          {std::string(1, 'a' + first - 1),
           std::string(1, 'a' + first),
           std::string(1, 'a' + first + 1)},
          type);
    });
  }

  // Applies 'patch' to the statistics of every chunk of the single column.
  template <typename Patch>
  static void patchStatistics(Footers& files, Patch patch) {
    for (auto& rowGroup : files.front().row_groups) {
      patch(rowGroup.columns.front().meta_data.statistics);
    }
  }

  std::vector<std::shared_ptr<TempFilePath>> tempFiles_;
};

TEST_F(CudfRowGroupPruningTest, integerRanges) {
  const auto rowType = ROW({"c0"}, {BIGINT()});
  const auto path = writeAscendingIntegers<int64_t>();
  // Every row group lies below the range, overlaps it, or only the second.
  expectPruned(path, filters("c0", bigintRange(1'000, 1'000)), rowType, {});
  expectPruned(
      path,
      filters("c0", bigintRange(0, std::numeric_limits<int64_t>::max())),
      rowType,
      {0, 1, 2});
  expectPruned(path, filters("c0", bigintRange(4, 6)), rowType, {1});
  expectPruned(
      path,
      filters("c0", std::make_unique<common::NegatedBigintRange>(1, 3, false)),
      rowType,
      {1, 2});
  // Narrower integer types read INT32 chunks.
  expectPruned(
      writeAscendingIntegers<int32_t>(),
      filters("c0", bigintRange(4, 6)),
      ROW({"c0"}, {INTEGER()}),
      {1});
  expectPruned(
      writeAscendingIntegers<int16_t>(),
      filters("c0", bigintRange(4, 6)),
      ROW({"c0"}, {SMALLINT()}),
      {1});
  expectPruned(
      writeAscendingIntegers<int32_t>(DATE()),
      filters("c0", bigintRange(4, 6)),
      ROW({"c0"}, {DATE()}),
      {1});
}

// IN lists are tested value by value: a bitmask for a narrow span of values,
// a hash table for a wide one.
TEST_F(CudfRowGroupPruningTest, integerInLists) {
  const auto rowType = ROW({"c0"}, {BIGINT()});
  const auto path = writeAscendingIntegers<int64_t>();
  expectPruned(
      path,
      filters("c0", common::createBigintValues({2, 8}, false)),
      rowType,
      {0, 2});
  expectPruned(
      path,
      filters(
          "c0", common::createBigintValues({2, 8, 1'000'000'000'000}, false)),
      rowType,
      {0, 2});
  expectPruned(
      path,
      filters("c0", common::createBigintValues({10, 11}, false)),
      rowType,
      {});
}

TEST_F(CudfRowGroupPruningTest, stringFilters) {
  const auto rowType = ROW({"c0"}, {VARCHAR()});
  const auto path = writeAscendingStrings();
  expectPruned(path, filters("c0", bytesRange("d", "f")), rowType, {1});
  expectPruned(path, filters("c0", bytesRange("x", "z")), rowType, {});
  expectPruned(
      path,
      filters(
          "c0",
          std::make_unique<common::BytesValues>(
              std::vector<std::string>{"b", "h"}, false)),
      rowType,
      {0, 2});
  // VARBINARY compares bytes like VARCHAR.
  expectPruned(
      writeAscendingStrings(VARBINARY()),
      filters("c0", bytesRange("d", "f")),
      ROW({"c0"}, {VARBINARY()}),
      {1});
}

// Exclusive bounds prune the row group whose bound equals the literal, as
// cuDF's pass does, and an OR of ranges keeps the union.
TEST_F(CudfRowGroupPruningTest, floatingPointRanges) {
  const auto doubleType = ROW({"c0"}, {DOUBLE()});
  const auto doublePath = writeAscendingFloatingPoints<double>();
  expectPruned(
      doublePath,
      filters(
          "c0",
          std::make_unique<common::DoubleRange>(
              4.0, false, false, 6.0, false, false, false)),
      doubleType,
      {1});
  expectPruned(doublePath, filters("c0", above(3.0, true)), doubleType, {1, 2});
  expectPruned(
      doublePath, filters("c0", above(3.0, false)), doubleType, {0, 1, 2});
  expectPruned(doublePath, filters("c0", below(7.0, true)), doubleType, {0, 1});
  expectPruned(
      doublePath, filters("c0", below(7.0, false)), doubleType, {0, 1, 2});
  expectPruned(
      doublePath,
      filters(
          "c0",
          std::make_unique<common::DoubleRange>(
              6.0, false, true, 7.0, false, true, false)),
      doubleType,
      {});
  expectPruned(
      doublePath,
      filters("c0", anyOf(below(2.0, true), above(8.0, true))),
      doubleType,
      {0, 2});
  expectPruned(
      doublePath,
      filters("c0", anyOf(below(1.0, true), above(9.0, true))),
      doubleType,
      {});
  // c0 <> 5.0 prunes only a row group of 5.0 alone.
  const auto constantPath = write(
      {makeRowVector({"c0"}, {makeFlatVector<double>({1.0, 1.0, 1.0})}),
       makeRowVector({"c0"}, {makeFlatVector<double>({5.0, 5.0, 5.0})}),
       makeRowVector({"c0"}, {makeFlatVector<double>({9.0, 9.0, 9.0})})});
  expectPruned(
      constantPath,
      filters("c0", anyOf(below(5.0, true), above(5.0, true))),
      doubleType,
      {0, 2});

  const auto floatType = ROW({"c0"}, {REAL()});
  const auto floatPath = writeAscendingFloatingPoints<float>();
  expectPruned(
      floatPath,
      filters(
          "c0",
          std::make_unique<common::FloatRange>(
              4.0f, false, false, 6.0f, false, false, false)),
      floatType,
      {1});
  expectPruned(floatPath, filters("c0", above(3.0f, true)), floatType, {1, 2});
  expectPruned(floatPath, filters("c0", below(7.0f, true)), floatType, {0, 1});
}

// Nulls decide a floating-point filter as they do the others.
TEST_F(CudfRowGroupPruningTest, floatingPointNulls) {
  const auto rowType = ROW({"c0"}, {DOUBLE()});
  const auto path = write(
      {makeRowVector(
           {"c0"},
           {makeNullableFlatVector<double>(
               {std::nullopt, std::nullopt, std::nullopt})}),
       makeRowVector(
           {"c0"}, {makeNullableFlatVector<double>({4.0, std::nullopt, 6.0})}),
       makeRowVector({"c0"}, {makeFlatVector<double>({7.0, 8.0, 9.0})})});
  expectPruned(path, filters("c0", above(3.0, true)), rowType, {1, 2});
  // A filter that accepts nulls: c0 > 100.0 OR c0 IS NULL as a range, and
  // c0 < 2.0 OR c0 > 100.0 OR c0 IS NULL as an OR of ranges, with the IS NULL
  // as the OR's flag or as one of its filters.
  expectPruned(
      path,
      filters(
          "c0",
          std::make_unique<common::DoubleRange>(
              100.0, false, true, 0.0, true, false, true)),
      rowType,
      {0, 1});
  expectPruned(
      path,
      filters("c0", anyOf(below(2.0, true), above(100.0, true), true)),
      rowType,
      {0, 1});
  expectPruned(
      path,
      filters(
          "c0",
          anyOf(std::make_unique<common::IsNull>(), above(100.0, true), true)),
      rowType,
      {0, 1});
}

// A chunk with only one of its bounds is left to cuDF's pass, whatever the
// type. The writer here records both, so the footers are patched.
TEST_F(CudfRowGroupPruningTest, oneSidedBounds) {
  const auto dropMax = [](pq::Statistics& stats) {
    stats.max_value.reset();
    stats.max.reset();
  };
  const auto dropMin = [](pq::Statistics& stats) {
    stats.min_value.reset();
    stats.min.reset();
  };
  constexpr double kInf = std::numeric_limits<double>::infinity();

  // Values from 1 up: none is below 0, and infinity may be among them.
  const auto doubleType = ROW({"c0"}, {DOUBLE()});
  const auto doublePath = writeAscendingFloatingPoints<double>();
  auto files = readFooters(doublePath);
  patchStatistics(files, dropMax);
  expectLeftToCudf(files, filters("c0", below(0.0, true)), doubleType);
  expectLeftToCudf(
      files,
      filters(
          "c0",
          std::make_unique<common::DoubleRange>(
              kInf, false, false, kInf, false, false, false)),
      doubleType);
  files = readFooters(doublePath);
  patchStatistics(files, dropMin);
  expectLeftToCudf(files, filters("c0", above(100.0, true)), doubleType);

  files = readFooters(writeAscendingFloatingPoints<float>());
  patchStatistics(files, dropMax);
  expectLeftToCudf(
      files, filters("c0", below(0.0f, true)), ROW({"c0"}, {REAL()}));

  files = readFooters(writeAscendingIntegers<int64_t>());
  patchStatistics(files, dropMax);
  expectLeftToCudf(
      files, filters("c0", bigintRange(-10, 0)), ROW({"c0"}, {BIGINT()}));

  files = readFooters(writeAscendingStrings());
  patchStatistics(files, dropMax);
  expectLeftToCudf(
      files, filters("c0", bytesRange("x", "z")), ROW({"c0"}, {VARCHAR()}));
}

// Some writers put a NaN into a floating-point bound; such a chunk is left to
// cuDF's pass too.
TEST_F(CudfRowGroupPruningTest, nanBound) {
  const auto doubleType = ROW({"c0"}, {DOUBLE()});
  const auto path = writeAscendingFloatingPoints<double>();
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  std::vector<uint8_t> nanBytes(sizeof(double));
  std::memcpy(nanBytes.data(), &kNaN, sizeof(double));

  auto files = readFooters(path);
  patchStatistics(
      files, [&](pq::Statistics& stats) { stats.max_value = nanBytes; });
  expectLeftToCudf(files, filters("c0", below(0.0, true)), doubleType);
  files = readFooters(path);
  patchStatistics(
      files, [&](pq::Statistics& stats) { stats.min_value = nanBytes; });
  expectLeftToCudf(files, filters("c0", above(100.0, true)), doubleType);
}

// A row group of only false values cannot hold true, and the other way round.
TEST_F(CudfRowGroupPruningTest, booleanValues) {
  const auto rowType = ROW({"c0"}, {BOOLEAN()});
  const auto path = write(
      {makeRowVector({"c0"}, {makeFlatVector<bool>({false, false, false})}),
       makeRowVector({"c0"}, {makeFlatVector<bool>({false, true, false})}),
       makeRowVector({"c0"}, {makeFlatVector<bool>({true, true, true})})});
  expectPruned(
      path,
      filters("c0", std::make_unique<common::BoolValue>(true, false)),
      rowType,
      {1, 2});
  expectPruned(
      path,
      filters("c0", std::make_unique<common::BoolValue>(false, false)),
      rowType,
      {0, 1});

  // The bounds decide without a null count too.
  auto files = readFooters(path);
  patchStatistics(
      files, [](pq::Statistics& stats) { stats.null_count.reset(); });
  expectDecided(
      files,
      filters("c0", std::make_unique<common::BoolValue>(true, false)),
      rowType,
      {1, 2});
  expectDecided(
      files,
      filters("c0", std::make_unique<common::BoolValue>(false, false)),
      rowType,
      {0, 1});
}

// cuDF's AST builder has no timestamp range, so only the host is checked.
TEST_F(CudfRowGroupPruningTest, timestampRange) {
  const auto files = readFooters(writeAscending([&](int64_t first) {
    return makeFlatVector<Timestamp>(
        {Timestamp(first, 0),
         Timestamp(first + 1, 0),
         Timestamp(first + 2, 0)});
  }));
  const auto result = prune(
      files,
      filters(
          "c0",
          std::make_unique<common::TimestampRange>(
              Timestamp(4, 0), Timestamp(6, 0), false)),
      ROW({"c0"}, {TIMESTAMP()}));
  EXPECT_EQ(result.rowGroupIndices.front(), Indices{1});
  EXPECT_TRUE(result.complete);
}

// Decimals stored as INT64 (short) and FIXED_LEN_BYTE_ARRAY (long) by cuDF's
// writer, including an IN list over long decimals.
TEST_F(CudfRowGroupPruningTest, decimalFilters) {
  const auto rowType =
      ROW({"c_short", "c_long"}, {DECIMAL(12, 2), DECIMAL(20, 2)});
  std::vector<RowVectorPtr> vectors;
  for (int group = 0; group < 3; ++group) {
    const int64_t base = 3 * group + 1;
    vectors.push_back(makeRowVector(
        rowType->names(),
        {makeFlatVector<int64_t>(
             {100 * base, 100 * (base + 1), 100 * (base + 2)}, DECIMAL(12, 2)),
         makeFlatVector<int128_t>(
             {100 * base, 100 * (base + 1), 100 * (base + 2)},
             DECIMAL(20, 2))}));
  }
  const auto path = write(vectors);
  expectPruned(path, filters("c_short", bigintRange(400, 600)), rowType, {1});
  expectPruned(
      path,
      filters("c_short", common::createBigintValues({200, 800}, false)),
      rowType,
      {0, 2});
  expectPruned(
      path,
      filters(
          "c_long",
          std::make_unique<common::HugeintRange>(
              int128_t{400}, int128_t{600}, false)),
      rowType,
      {1});
  expectPruned(
      path,
      filters(
          "c_long",
          std::make_unique<common::HugeintRange>(
              int128_t{-100}, int128_t{50}, false)),
      rowType,
      {});
  expectPruned(
      path,
      filters(
          "c_long",
          common::createHugeintValues({int128_t{200}, int128_t{800}}, false)),
      rowType,
      {0, 2});
}

// Short decimals stored as INT32 by Velox's Parquet writer, including
// negative values.
TEST_F(CudfRowGroupPruningTest, int32Decimals) {
  const auto rowType = ROW({"c0"}, {DECIMAL(5, 2)});
  auto file = TempFilePath::create();
  writeCompactDecimalParquet(
      file,
      {makeRowVector(
           {"c0"},
           {makeFlatVector<int64_t>({-300, -200, -100}, DECIMAL(5, 2))}),
       makeRowVector(
           {"c0"}, {makeFlatVector<int64_t>({-50, 0, 50}, DECIMAL(5, 2))}),
       makeRowVector(
           {"c0"}, {makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(5, 2))})});
  const auto path = file->getPath();
  const auto files = readFooters(path);
  ASSERT_EQ(files.front().row_groups.size(), 3);
  ASSERT_EQ(
      files.front().row_groups.front().columns.front().meta_data.type,
      pq::Type::INT32);
  // cuDF reads the column as DECIMAL32, so its filter literal must be one.
  const SubfieldFilterDecimalTypes decimalTypes{
      {"c0", SubfieldFilterDecimalType{cudf::type_id::DECIMAL32, 2}}};
  expectPruned(
      path,
      filters("c0", bigintRange(-250, -150)),
      rowType,
      {0},
      &decimalTypes);
  expectPruned(
      path, filters("c0", bigintRange(-60, 60)), rowType, {1}, &decimalTypes);
  expectPruned(
      path,
      filters("c0", bigintRange(-100, 100)),
      rowType,
      {0, 1, 2},
      &decimalTypes);
  expectPruned(
      path, filters("c0", bigintRange(301, 400)), rowType, {}, &decimalTypes);
}

// Nulls: an all-null row group, a row group with some nulls and one without.
TEST_F(CudfRowGroupPruningTest, nulls) {
  const auto rowType = ROW({"c0"}, {BIGINT()});
  const auto path = write(
      {makeRowVector(
           {"c0"},
           {makeNullableFlatVector<int64_t>(
               {std::nullopt, std::nullopt, std::nullopt})}),
       makeRowVector(
           {"c0"}, {makeNullableFlatVector<int64_t>({4, std::nullopt, 6})}),
       makeRowVector({"c0"}, {makeFlatVector<int64_t>({7, 8, 9})})});
  expectPruned(
      path, filters("c0", std::make_unique<common::IsNull>()), rowType, {0, 1});
  expectPruned(
      path,
      filters("c0", std::make_unique<common::IsNotNull>()),
      rowType,
      {1, 2});
  // An all-null row group holds no value in any range, unless the filter
  // accepts nulls.
  expectPruned(path, filters("c0", bigintRange(4, 6)), rowType, {1});
  expectPruned(
      path,
      filters("c0", std::make_unique<common::BigintRange>(4, 6, true)),
      rowType,
      {0, 1});
  expectPruned(path, filters("c0", bigintRange(100, 200)), rowType, {});
  // The ranges of an OR may allow nulls the OR itself rejects. cuDF's AST
  // ignores their flags, so the row group holding a null is not kept for them,
  // on an integer or a short decimal column.
  const auto outside = [](int64_t below, int64_t above) {
    std::vector<std::unique_ptr<common::BigintRange>> ranges;
    ranges.push_back(std::make_unique<common::BigintRange>(0, below, true));
    ranges.push_back(
        std::make_unique<common::BigintRange>(above, 10 * above, true));
    return std::make_unique<common::BigintMultiRange>(std::move(ranges), false);
  };
  expectPruned(path, filters("c0", outside(3, 10)), rowType, {});
  expectPruned(
      write({makeRowVector(
          {"c0"},
          {makeNullableFlatVector<int64_t>(
              {400, std::nullopt, 600}, DECIMAL(12, 2))})}),
      filters("c0", outside(300, 1'000)),
      ROW({"c0"}, {DECIMAL(12, 2)}),
      {});
  // A column without nulls: IS NULL rules out every row group, IS NOT NULL
  // none.
  const auto noNullsPath = writeAscendingIntegers<int64_t>();
  expectPruned(
      noNullsPath,
      filters("c0", std::make_unique<common::IsNull>()),
      rowType,
      {});
  expectPruned(
      noNullsPath,
      filters("c0", std::make_unique<common::IsNotNull>()),
      rowType,
      {0, 1, 2});

  // An empty row group holds no row, null or not.
  auto files = readFooters(noNullsPath);
  files.front().row_groups[1].num_rows = 0;
  expectDecided(
      files, filters("c0", std::make_unique<common::IsNull>()), rowType, {});
}

// Filters on two columns prune the union of what each rules out.
TEST_F(CudfRowGroupPruningTest, severalFilters) {
  const auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  std::vector<RowVectorPtr> vectors;
  for (int group = 0; group < 3; ++group) {
    const int64_t base = 3 * group + 1;
    vectors.push_back(makeRowVector(
        rowType->names(),
        {makeFlatVector<int64_t>({base, base + 1, base + 2}),
         makeFlatVector<std::string>(
             {std::string(1, 'a' + 3 * group),
              std::string(1, 'b' + 3 * group),
              std::string(1, 'c' + 3 * group)})}));
  }
  const auto path = write(vectors);
  expectPruned(
      path,
      SubfieldFiltersBuilder()
          .add("c0", bigintRange(4, 9))
          .add("c1", bytesRange("a", "c"))
          .build(),
      rowType,
      {});
  expectPruned(
      path,
      SubfieldFiltersBuilder()
          .add("c0", bigintRange(4, 9))
          .add("c1", bytesRange("a", "f"))
          .build(),
      rowType,
      {1});
}

// Columns the statistics say nothing about keep their row groups: a column the
// file lacks, a nested column, a one-level list, a chunk without statistics.
TEST_F(CudfRowGroupPruningTest, keptWithoutStatistics) {
  const auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto files = readFooters(writeAscendingIntegers<int64_t>());
  expectKept(files, filters("c1", bigintRange(1'000, 1'000)), rowType);
  expectKept(
      files,
      filters("c0", std::make_unique<common::IsNotNull>()),
      ROW({"c0"}, {ARRAY(BIGINT())}));

  // A one-level list is a repeated leaf whose null count counts its empty
  // lists; it is not tested either.
  auto& element = files.front().schema[1];
  ASSERT_EQ(element.name, "c0");
  element.repetition_type = pq::FieldRepetitionType::REPEATED;
  for (auto& rowGroup : files.front().row_groups) {
    rowGroup.columns.front().meta_data.statistics.null_count =
        rowGroup.num_rows;
  }
  expectKept(
      files, filters("c0", std::make_unique<common::IsNotNull>()), rowType);

  // No statistics at all.
  files = readFooters(writeAscendingIntegers<int64_t>());
  for (auto& rowGroup : files.front().row_groups) {
    rowGroup.columns.front().meta_data.statistics = pq::Statistics{};
  }
  expectKept(files, filters("c0", bigintRange(1'000, 1'000)), rowType);
}

// Storage the host test does not decode leaves the decision to cuDF's pass.
TEST_F(CudfRowGroupPruningTest, leftToCudf) {
  const auto rowType = ROW({"c0"}, {BIGINT()});
  const auto path = writeAscendingIntegers<int64_t>();
  const auto unmatched = filters("c0", bigintRange(1'000, 1'000));

  // An unsigned integer, by logical and by converted type.
  auto files = readFooters(path);
  files.front().schema[1].logical_type =
      pq::LogicalType(pq::LogicalType::INTEGER);
  files.front().schema[1].logical_type->int_type = pq::IntType{64, false};
  expectLeftToCudf(files, unmatched, rowType);
  files = readFooters(path);
  files.front().schema[1].converted_type = pq::ConvertedType::UINT_64;
  expectLeftToCudf(files, unmatched, rowType);

  // A column stored as another physical type than the table reads.
  files = readFooters(path);
  expectLeftToCudf(files, unmatched, ROW({"c0"}, {DOUBLE()}));
  expectLeftToCudf(files, unmatched, ROW({"c0"}, {VARCHAR()}));

  // An OR of ranges on an integer column, which has no range test.
  expectLeftToCudf(
      readFooters(path),
      filters("c0", anyOf(bigintRange(100, 200), bigintRange(300, 400))),
      rowType);

  // An IN list inside an OR, whose range test answers from the list's span.
  expectLeftToCudf(
      readFooters(writeAscendingStrings()),
      filters(
          "c0",
          anyOf(
              std::make_unique<common::BytesValues>(
                  std::vector<std::string>{"b", "h"}, false),
              bytesRange("x", "z"))),
      ROW({"c0"}, {VARCHAR()}));

  // An INT64 column without a timestamp unit, and an INT96 timestamp.
  files = readFooters(path);
  const auto timestampFilter = filters(
      "c0",
      std::make_unique<common::TimestampRange>(
          Timestamp(1'000, 0), Timestamp(1'000, 0), false));
  expectLeftToCudf(files, timestampFilter, ROW({"c0"}, {TIMESTAMP()}));
  for (auto& rowGroup : files.front().row_groups) {
    rowGroup.columns.front().meta_data.type = pq::Type::INT96;
  }
  expectLeftToCudf(files, timestampFilter, ROW({"c0"}, {TIMESTAMP()}));

  // Decimals: a long decimal the file stores at another scale, and one with
  // only the deprecated bounds.
  const auto decimalType = ROW({"c0", "c1"}, {DECIMAL(20, 2), BIGINT()});
  const auto decimalPath = write(
      {makeRowVector(
           decimalType->names(),
           {makeFlatVector<int128_t>({100, 200, 300}, DECIMAL(20, 2)),
            makeFlatVector<int64_t>({1, 2, 3})}),
       makeRowVector(
           decimalType->names(),
           {makeFlatVector<int128_t>({400, 500, 600}, DECIMAL(20, 2)),
            makeFlatVector<int64_t>({4, 5, 6})})});
  const auto decimalFilter = filters(
      "c0",
      std::make_unique<common::HugeintRange>(
          int128_t{1'000}, int128_t{2'000}, false));
  files = readFooters(decimalPath);
  expectLeftToCudf(
      files, decimalFilter, ROW({"c0", "c1"}, {DECIMAL(20, 3), BIGINT()}));
  for (auto& rowGroup : files.front().row_groups) {
    auto& stats = rowGroup.columns.front().meta_data.statistics;
    stats.min = std::move(stats.min_value);
    stats.max = std::move(stats.max_value);
    stats.min_value.reset();
    stats.max_value.reset();
  }
  expectLeftToCudf(files, decimalFilter, decimalType);
  // ... or with the maximum alone in the deprecated field, which cuDF's pass
  // reads and the host does not.
  files = readFooters(decimalPath);
  for (auto& rowGroup : files.front().row_groups) {
    auto& stats = rowGroup.columns.front().meta_data.statistics;
    stats.max = std::move(stats.max_value);
    stats.max_value.reset();
  }
  expectLeftToCudf(files, decimalFilter, decimalType);

  // A decided filter still prunes beside one left to cuDF.
  files = readFooters(decimalPath);
  files.front().schema[1].decimal_scale = 3;
  const auto result = pruneRowGroupsWithStatistics(
      files,
      allRowGroups(files),
      prepareRowGroupPruningFilters(
          SubfieldFiltersBuilder()
              .add("c0", decimalFilter.begin()->second->clone())
              .add("c1", bigintRange(4, 6))
              .build(),
          *decimalType),
      false);
  EXPECT_EQ(result.rowGroupIndices.front(), Indices{1});
  EXPECT_FALSE(result.complete);
}

// Mixed-case file names match lower-case table names only when the reader
// lower-cases file names.
TEST_F(CudfRowGroupPruningTest, columnNames) {
  const auto files = readFooters(write(
      {makeRowVector({"C0"}, {makeFlatVector<int64_t>({1, 2, 3})}),
       makeRowVector({"C0"}, {makeFlatVector<int64_t>({7, 8, 9})})}));
  const auto rowType = ROW({"c0"}, {BIGINT()});
  const auto unmatched = filters("c0", bigintRange(1'000, 1'000));
  expectKept(files, unmatched, rowType, false);
  const auto result = prune(files, unmatched, rowType, true);
  EXPECT_EQ(result.rowGroupIndices.front(), Indices{});
  EXPECT_TRUE(result.complete);
}

// Several files, each with a subset of its row groups.
TEST_F(CudfRowGroupPruningTest, severalSources) {
  Footers files;
  for (int i = 0; i < 2; ++i) {
    auto footers = readFooters(writeAscendingIntegers<int64_t>());
    files.push_back(std::move(footers.front()));
  }
  const auto result = pruneRowGroupsWithStatistics(
      files,
      {Indices{0, 2}, Indices{1, 2}},
      prepareRowGroupPruningFilters(
          filters("c0", bigintRange(5, 8)), *ROW({"c0"}, {BIGINT()})),
      false);
  EXPECT_EQ(
      result.rowGroupIndices,
      (std::vector<Indices>{Indices{2}, Indices{1, 2}}));
  EXPECT_TRUE(result.complete);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
