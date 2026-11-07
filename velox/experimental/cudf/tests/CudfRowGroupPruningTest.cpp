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
        filters,
        *tableType,
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
  const auto path = writeAscending([&](int64_t first) {
    return makeFlatVector<std::string>(
        {std::string(1, 'a' + first - 1),
         std::string(1, 'a' + first),
         std::string(1, 'a' + first + 1)});
  });
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
  const auto binaryType = ROW({"c0"}, {VARBINARY()});
  const auto binaryPath = writeAscending([&](int64_t first) {
    return makeFlatVector<std::string>(
        {std::string(1, 'a' + first - 1),
         std::string(1, 'a' + first),
         std::string(1, 'a' + first + 1)},
        VARBINARY());
  });
  expectPruned(
      binaryPath, filters("c0", bytesRange("d", "f")), binaryType, {1});
}

TEST_F(CudfRowGroupPruningTest, floatingPointRanges) {
  expectPruned(
      writeAscending([&](int64_t first) {
        return makeFlatVector<double>(
            {static_cast<double>(first),
             static_cast<double>(first + 1),
             static_cast<double>(first + 2)});
      }),
      filters(
          "c0",
          std::make_unique<common::DoubleRange>(
              4.0, false, false, 6.0, false, false, false)),
      ROW({"c0"}, {DOUBLE()}),
      {1});
  expectPruned(
      writeAscending([&](int64_t first) {
        return makeFlatVector<float>(
            {static_cast<float>(first),
             static_cast<float>(first + 1),
             static_cast<float>(first + 2)});
      }),
      filters(
          "c0",
          std::make_unique<common::FloatRange>(
              4.0f, false, false, 6.0f, false, false, false)),
      ROW({"c0"}, {REAL()}),
      {1});
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

  // A decided filter still prunes beside one left to cuDF.
  files = readFooters(decimalPath);
  files.front().schema[1].decimal_scale = 3;
  const auto result = pruneRowGroupsWithStatistics(
      files,
      allRowGroups(files),
      SubfieldFiltersBuilder()
          .add("c0", decimalFilter.begin()->second->clone())
          .add("c1", bigintRange(4, 6))
          .build(),
      *decimalType,
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
      filters("c0", bigintRange(5, 8)),
      *ROW({"c0"}, {BIGINT()}),
      false);
  EXPECT_EQ(
      result.rowGroupIndices,
      (std::vector<Indices>{Indices{2}, Indices{1, 2}}));
  EXPECT_TRUE(result.complete);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
