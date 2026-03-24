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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/Mutation.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

class ParquetReaderWideningTest : public ParquetTestBase {
 public:
  std::unique_ptr<dwio::common::RowReader> createWideningRowReader(
      const RowVectorPtr& writeData,
      const RowTypePtr& readSchema,
      bool allowInt32Narrowing = false) {
    auto* sink = write(writeData);
    dwio::common::ReaderOptions readerOptions{leafPool_.get()};
    readerOptions.setFileSchema(readSchema);
    readerOptions.setAllowInt32Narrowing(allowInt32Narrowing);
    auto reader = createReaderInMemory(*sink, readerOptions);
    auto rowReaderOpts = getReaderOpts(readSchema);
    rowReaderOpts.setScanSpec(makeScanSpec(readSchema));
    return reader->createRowReader(rowReaderOpts);
  }

  /// Writes Parquet data with one schema and reads it back with a wider schema,
  /// then verifies the result matches the expected output.
  void assertWideningReads(
      const RowVectorPtr& writeData,
      const RowTypePtr& readSchema,
      const RowVectorPtr& expected) {
    auto rowReader = createWideningRowReader(writeData, readSchema);
    assertReadWithReaderAndExpected(
        readSchema, *rowReader, expected, *leafPool_);
  }

  /// Writes Parquet data and reads it back with a narrower schema
  /// (allowInt32Narrowing enabled), then verifies the result.
  void assertNarrowingReads(
      const RowVectorPtr& writeData,
      const RowTypePtr& readSchema,
      const RowVectorPtr& expected) {
    auto rowReader = createWideningRowReader(
        writeData, readSchema, /*allowInt32Narrowing=*/true);
    assertReadWithReaderAndExpected(
        readSchema, *rowReader, expected, *leafPool_);
  }

  /// Writes Parquet data, reads with widening schema + filter, verifies result.
  void assertWideningWithFilter(
      const RowVectorPtr& writeData,
      const RowTypePtr& readSchema,
      std::unique_ptr<common::Filter> filter,
      const RowVectorPtr& expected,
      bool allowInt32Narrowing = false) {
    auto* sink = write(writeData);
    dwio::common::ReaderOptions readerOptions{leafPool_.get()};
    readerOptions.setFileSchema(readSchema);
    readerOptions.setAllowInt32Narrowing(allowInt32Narrowing);
    auto reader = createReaderInMemory(*sink, readerOptions);
    auto rowReaderOpts = getReaderOpts(readSchema);
    auto scanSpec = makeScanSpec(readSchema);
    auto* child = scanSpec->getOrCreateChild(common::Subfield("col"));
    child->setFilter(std::move(filter));
    rowReaderOpts.setScanSpec(scanSpec);
    auto rowReader = reader->createRowReader(rowReaderOpts);
    assertReadWithReaderAndExpected(
        readSchema, *rowReader, expected, *leafPool_);
  }

  /// Verifies that reading in-memory Parquet data with a mismatched schema
  /// throws an exception whose message contains both the source type name and
  /// "is not allowed for requested type".
  void assertWideningThrows(
      const RowVectorPtr& writeData,
      const RowTypePtr& readSchema,
      const std::string& sourceTypeName) {
    auto* sink = write(writeData);
    dwio::common::ReaderOptions readerOptions{leafPool_.get()};
    readerOptions.setFileSchema(readSchema);
    VELOX_ASSERT_THROW(
        createReaderInMemory(*sink, readerOptions),
        "Converted type " + sourceTypeName +
            " is not allowed for requested type");
  }
};

// Comprehensive test matrix covering all combinations:
// - Nulls: No nulls, With nulls
// - Dictionary: Enabled, Disabled
// - Filter: None, IsNull, IsNotNull, Value filter
// - Density: Dense (no deletions), Non-dense (with deletions/mutations)

enum class FloatToDoubleFilter {
  kNone,
  kIsNull,
  kIsNotNull,
  kGreaterThanOrEqual, // Value filter: greater than or equal to a threshold
  kMultiRange, // MultiRange filter: a < X OR a > Y
};

struct FloatToDoubleSpec {
  std::vector<std::optional<float>> values;
  std::vector<int64_t> ids;
  bool enableDictionary{true};
  FloatToDoubleFilter filter{FloatToDoubleFilter::kNone};
  std::optional<double> filterValue; // Value for value-based filters
  std::optional<double> filterLowerValue; // Lower bound for MultiRange filter
  std::optional<double> filterUpperValue; // Upper bound for MultiRange filter
  std::vector<vector_size_t> deletedRows;
};

struct FloatToDoubleTestParam {
  bool hasNulls;
  bool enableDictionary;
  FloatToDoubleFilter filter;
  bool isDense;

  std::string toString() const {
    return fmt::format(
        "Nulls_{}_Dict_{}_Filter_{}_Dense_{}",
        hasNulls ? "Yes" : "No",
        enableDictionary ? "Yes" : "No",
        filterName(filter),
        isDense ? "Yes" : "No");
  }

  static std::string filterName(FloatToDoubleFilter filter) {
    switch (filter) {
      case FloatToDoubleFilter::kNone:
        return "None";
      case FloatToDoubleFilter::kIsNull:
        return "IsNull";
      case FloatToDoubleFilter::kIsNotNull:
        return "IsNotNull";
      case FloatToDoubleFilter::kGreaterThanOrEqual:
        return "GreaterThanOrEqual";
      case FloatToDoubleFilter::kMultiRange:
        return "MultiRange";
      default:
        return "Unknown";
    }
  }
};

class FloatToDoubleEvolutionTest
    : public ParquetReaderWideningTest,
      public testing::WithParamInterface<FloatToDoubleTestParam> {
 public:
  static std::vector<FloatToDoubleTestParam> getTestParams() {
    std::vector<FloatToDoubleTestParam> params;
    for (bool hasNulls : {false, true}) {
      for (bool enableDictionary : {false, true}) {
        // When hasNulls is false, only test kNone, kGreaterThanOrEqual, and
        // kMultiRange filter (kIsNull would match nothing, kIsNotNull is
        // equivalent to kNone)
        std::vector<FloatToDoubleFilter> filters;
        if (hasNulls) {
          filters = {
              FloatToDoubleFilter::kNone,
              FloatToDoubleFilter::kIsNull,
              FloatToDoubleFilter::kIsNotNull,
              FloatToDoubleFilter::kGreaterThanOrEqual,
              FloatToDoubleFilter::kMultiRange};
        } else {
          filters = {
              FloatToDoubleFilter::kNone,
              FloatToDoubleFilter::kGreaterThanOrEqual,
              FloatToDoubleFilter::kMultiRange};
        }

        for (auto filter : filters) {
          for (bool isDense : {true, false}) {
            params.push_back({hasNulls, enableDictionary, filter, isDense});
          }
        }
      }
    }
    return params;
  }

  void runFloatToDoubleScenario(const FloatToDoubleSpec& spec);
};

void FloatToDoubleEvolutionTest::runFloatToDoubleScenario(
    const FloatToDoubleSpec& spec) {
  ASSERT_EQ(spec.values.size(), spec.ids.size());
  const vector_size_t numRows = spec.ids.size();

  auto floatVector = makeNullableFlatVector<float>(spec.values);
  auto idVector =
      makeFlatVector<int64_t>(numRows, [&](auto row) { return spec.ids[row]; });

  RowVectorPtr writeData = makeRowVector({floatVector, idVector});
  RowTypePtr writeSchema = ROW({"float_col", "id"}, {REAL(), BIGINT()});

  auto sink = std::make_unique<MemorySink>(
      1024 * 1024, dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto sinkPtr = sink.get();

  parquet::WriterOptions writerOptions;
  writerOptions.memoryPool = leafPool_.get();
  writerOptions.enableDictionary = spec.enableDictionary;

  auto writer = std::make_unique<facebook::velox::parquet::Writer>(
      std::move(sink), writerOptions, rootPool_, writeSchema);
  writer->write(writeData);
  writer->close();

  RowTypePtr readSchema = ROW({"float_col", "id"}, {DOUBLE(), BIGINT()});

  dwio::common::ReaderOptions readerOptions{leafPool_.get()};
  readerOptions.setFileSchema(readSchema);

  std::string dataBuf(sinkPtr->data(), sinkPtr->size());
  auto file = std::make_shared<InMemoryReadFile>(std::move(dataBuf));
  auto buffer = std::make_unique<dwio::common::BufferedInput>(
      file, readerOptions.memoryPool());
  auto reader =
      std::make_unique<ParquetReader>(std::move(buffer), readerOptions);

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.select(
      std::make_shared<facebook::velox::dwio::common::ColumnSelector>(
          readSchema, readSchema->names()));
  auto scanSpec = makeScanSpec(readSchema);

  // Apply IsNull or IsNotNull filter if specified
  switch (spec.filter) {
    case FloatToDoubleFilter::kNone:
      break;
    case FloatToDoubleFilter::kIsNull: {
      auto* floatChild =
          scanSpec->getOrCreateChild(common::Subfield("float_col"));
      floatChild->setFilter(exec::isNull());
      break;
    }
    case FloatToDoubleFilter::kIsNotNull: {
      auto* floatChild =
          scanSpec->getOrCreateChild(common::Subfield("float_col"));
      floatChild->setFilter(exec::isNotNull());
      break;
    }
    case FloatToDoubleFilter::kGreaterThanOrEqual: {
      ASSERT_TRUE(spec.filterValue.has_value());
      auto* floatChild =
          scanSpec->getOrCreateChild(common::Subfield("float_col"));
      floatChild->setFilter(
          exec::greaterThanOrEqualDouble(spec.filterValue.value()));
      break;
    }
    case FloatToDoubleFilter::kMultiRange: {
      ASSERT_TRUE(spec.filterLowerValue.has_value());
      ASSERT_TRUE(spec.filterUpperValue.has_value());
      auto* floatChild =
          scanSpec->getOrCreateChild(common::Subfield("float_col"));
      // Create a MultiRange filter: a < lower OR a > upper
      floatChild->setFilter(
          exec::orFilter(
              exec::lessThanDouble(spec.filterLowerValue.value()),
              exec::greaterThanDouble(spec.filterUpperValue.value())));
      break;
    }
  }

  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);

  std::vector<bool> deletedFlags(numRows, false);
  for (auto index : spec.deletedRows) {
    ASSERT_LT(index, numRows);
    deletedFlags[index] = true;
  }

  std::vector<vector_size_t> expectedIndices;
  expectedIndices.reserve(numRows);
  for (vector_size_t row = 0; row < numRows; ++row) {
    if (deletedFlags[row]) {
      continue;
    }

    bool passes = false;
    switch (spec.filter) {
      case FloatToDoubleFilter::kNone:
        passes = true;
        break;
      case FloatToDoubleFilter::kIsNull:
        passes = !spec.values[row].has_value();
        break;
      case FloatToDoubleFilter::kIsNotNull:
        passes = spec.values[row].has_value();
        break;
      case FloatToDoubleFilter::kGreaterThanOrEqual:
        passes = spec.values[row].has_value() &&
            static_cast<double>(*spec.values[row]) >= spec.filterValue.value();
        break;
      case FloatToDoubleFilter::kMultiRange:
        passes = spec.values[row].has_value() &&
            (static_cast<double>(*spec.values[row]) <
                 spec.filterLowerValue.value() ||
             static_cast<double>(*spec.values[row]) >
                 spec.filterUpperValue.value());
        break;
    }

    if (passes) {
      expectedIndices.push_back(row);
    }
  }

  std::vector<std::optional<double>> expectedDoubles(expectedIndices.size());
  for (size_t i = 0; i < expectedIndices.size(); ++i) {
    const auto originalIndex = expectedIndices[i];
    if (!spec.values[originalIndex].has_value()) {
      expectedDoubles[i] = std::nullopt;
    } else {
      expectedDoubles[i] = static_cast<double>(*spec.values[originalIndex]);
    }
  }

  auto expectedFloat = makeNullableFlatVector<double>(expectedDoubles);
  auto expectedId = makeFlatVector<int64_t>(
      expectedIndices.size(),
      [&](auto row) { return spec.ids[expectedIndices[row]]; });
  RowVectorPtr expected = makeRowVector({expectedFloat, expectedId});

  if (spec.deletedRows.empty() && spec.filter != FloatToDoubleFilter::kIsNull &&
      spec.filter != FloatToDoubleFilter::kIsNotNull &&
      spec.filter != FloatToDoubleFilter::kGreaterThanOrEqual &&
      spec.filter != FloatToDoubleFilter::kMultiRange) {
    assertReadWithReaderAndExpected(
        readSchema, *rowReader, expected, *leafPool_);
    return;
  }

  VectorPtr result = BaseVector::create(readSchema, 0, leafPool_.get());
  vector_size_t scanned = 0;
  std::vector<uint64_t> deleted(bits::nwords(numRows), 0);
  if (spec.deletedRows.empty()) {
    scanned = rowReader->next(numRows, result);
  } else {
    for (auto index : spec.deletedRows) {
      bits::setBit(deleted.data(), index);
    }
    dwio::common::Mutation mutation;
    mutation.deletedRows = deleted.data();
    scanned = rowReader->next(numRows, result, &mutation);
  }

  EXPECT_GT(scanned, 0);
  EXPECT_GE(scanned, expected->size());
  ASSERT_TRUE(result != nullptr);
  auto rowVector = result->as<RowVector>();
  ASSERT_TRUE(rowVector != nullptr);
  ASSERT_EQ(rowVector->size(), expected->size());
  assertEqualVectorPart(expected, result, 0);
}

TEST_P(FloatToDoubleEvolutionTest, readFloatToDouble) {
  const auto& param = GetParam();
  FloatToDoubleSpec spec;
  constexpr vector_size_t kSize = 200;
  spec.enableDictionary = param.enableDictionary;
  spec.values.resize(kSize);
  spec.ids.resize(kSize);

  for (vector_size_t row = 0; row < kSize; ++row) {
    if (param.hasNulls && row % 5 == 0) {
      spec.values[row] = std::nullopt;
    } else {
      // Use a value pattern that works for both dictionary and direct encoding
      float val =
          static_cast<float>(row % 10) * 1.1f + static_cast<float>(row) * 0.01f;
      spec.values[row] = val;
    }
    spec.ids[row] = row;
  }

  spec.filter = param.filter;

  // Set filter value for value-based filters
  if (param.filter == FloatToDoubleFilter::kGreaterThanOrEqual) {
    // Filter values greater than or equal to 5.0 (this should match
    // approximately half the rows)
    spec.filterValue = 5.0;
  } else if (param.filter == FloatToDoubleFilter::kMultiRange) {
    // Filter values < 3.0 OR > 7.0
    spec.filterLowerValue = 3.0;
    spec.filterUpperValue = 7.0;
  }

  if (!param.isDense) {
    // Add some deleted rows scattered throughout
    spec.deletedRows = {5, 20, 55, 99, 150, 199};
  }

  runFloatToDoubleScenario(spec);
}

INSTANTIATE_TEST_SUITE_P(
    FloatToDoubleEvolution,
    FloatToDoubleEvolutionTest,
    testing::ValuesIn(FloatToDoubleEvolutionTest::getTestParams()),
    [](const testing::TestParamInfo<FloatToDoubleTestParam>& info) {
      return info.param.toString();
    });

// Type widening tests: verify reading Parquet columns with a wider target
// type than the physical type stored in the file.

TEST_F(ParquetReaderWideningTest, intToShortDecimalWidening) {
  auto writeData = makeRowVector({makeFlatVector<int32_t>(
      {0, 1, -1, 100, -100, 2'147'483'647, -2'147'483'648})});
  auto expected = makeRowVector({makeFlatVector<int64_t>(
      {0, 100, -100, 10'000, -10'000, 214'748'364'700LL, -214'748'364'800LL},
      DECIMAL(12, 2))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(12, 2)}), expected);
}

TEST_F(ParquetReaderWideningTest, smallintToShortDecimalWidening) {
  auto writeData = makeRowVector(
      {makeFlatVector<int16_t>({0, 1, -1, 100, 32'767, -32'768})});
  auto expected = makeRowVector({makeFlatVector<int64_t>(
      {0, 100, -100, 10'000, 3'276'700, -3'276'800}, DECIMAL(12, 2))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(12, 2)}), expected);
}

TEST_F(ParquetReaderWideningTest, tinyintToShortDecimalWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int8_t>({0, 1, -1, 100, 127, -128})});
  auto expected = makeRowVector({makeFlatVector<int64_t>(
      {0, 100, -100, 10'000, 12'700, -12'800}, DECIMAL(12, 2))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(12, 2)}), expected);
}

// Parquet stores TINYINT as INT32, so the minimum precision for decimal
// widening is precision-scale >= 10 (same as INT32). Test exact boundary.
TEST_F(ParquetReaderWideningTest, tinyintToDecimalMinPrecisionWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int8_t>({0, 1, -1, 127, -128})});
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({0, 1, -1, 127, -128}, DECIMAL(10, 0))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(10, 0)}), expected);
}

// Parquet stores SMALLINT as INT32, so decimal widening requires
// precision-scale >= 10 (same as INT32). Test exact boundary.
TEST_F(ParquetReaderWideningTest, smallintToDecimalMinPrecisionWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int16_t>({0, 1, -1, 32'767, -32'768})});
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({0, 1, -1, 32'767, -32'768}, DECIMAL(10, 0))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(10, 0)}), expected);
}

// Byte -> Long Decimal. Parquet stores TINYINT as INT32.
TEST_F(ParquetReaderWideningTest, tinyintToLongDecimalWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int8_t>({0, 1, -1, 127, -128})});
  auto expected = makeRowVector(
      {makeFlatVector<int128_t>({0, 1, -1, 127, -128}, DECIMAL(20, 0))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(20, 0)}), expected);
}

// Short -> Long Decimal. Parquet stores SMALLINT as INT32.
TEST_F(ParquetReaderWideningTest, smallintToLongDecimalWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int16_t>({0, 1, -1, 32'767, -32'768})});
  auto expected = makeRowVector(
      {makeFlatVector<int128_t>({0, 1, -1, 32'767, -32'768}, DECIMAL(20, 0))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(20, 0)}), expected);
}

TEST_F(ParquetReaderWideningTest, bigintToLongDecimalWidening) {
  auto writeData = makeRowVector({makeFlatVector<int64_t>(
      {0, 1, -1, 1'000'000'000'000LL, -1'000'000'000'000LL})});
  auto expected = makeRowVector({makeFlatVector<int128_t>(
      {0,
       100'000,
       -100'000,
       static_cast<int128_t>(1'000'000'000'000LL) * 100'000,
       static_cast<int128_t>(-1'000'000'000'000LL) * 100'000},
      DECIMAL(25, 5))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(25, 5)}), expected);
}

TEST_F(ParquetReaderWideningTest, decimalToDecimalWidening) {
  auto writeData = makeRowVector(
      {makeFlatVector<int64_t>({1111, 2222, 3333, -4444, 0}, DECIMAL(7, 2))});
  // Each value v becomes v * 10^(4-2) = v * 100.
  auto expected = makeRowVector({makeFlatVector<int64_t>(
      {111'100, 222'200, 333'300, -444'400, 0}, DECIMAL(10, 4))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(10, 4)}), expected);
}

TEST_F(ParquetReaderWideningTest, decimalToDecimalPrecisionOnlyWidening) {
  auto writeData = makeRowVector(
      {makeFlatVector<int64_t>({1111, 2222, 3333, -4444, 0}, DECIMAL(7, 2))});
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({1111, 2222, 3333, -4444, 0}, DECIMAL(10, 2))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(10, 2)}), expected);
}

// Decimal long -> long, precision-only widening (scale stays the same).
TEST_F(ParquetReaderWideningTest, decimalLongToLongPrecisionOnlyWidening) {
  auto writeData = makeRowVector(
      {makeFlatVector<int128_t>({1111, -2222, 0}, DECIMAL(20, 2))});
  auto expected = makeRowVector(
      {makeFlatVector<int128_t>({1111, -2222, 0}, DECIMAL(22, 2))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(22, 2)}), expected);
}

TEST_F(ParquetReaderWideningTest, decimalToDecimalWideningWithNulls) {
  auto writeData = makeRowVector({makeNullableFlatVector<int64_t>(
      {std::nullopt, 1111, std::nullopt, -4444, 0, std::nullopt},
      DECIMAL(7, 2))});
  // Each value v becomes v * 10^(4-2) = v * 100.
  auto expected = makeRowVector({makeNullableFlatVector<int64_t>(
      {std::nullopt, 111'100, std::nullopt, -444'400, 0, std::nullopt},
      DECIMAL(10, 4))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(10, 4)}), expected);
}

TEST_F(ParquetReaderWideningTest, intToDecimalWideningWithNulls) {
  auto writeData = makeRowVector({makeNullableFlatVector<int32_t>(
      {std::nullopt, 42, std::nullopt, -7, 0, std::nullopt})});
  auto expected = makeRowVector({makeNullableFlatVector<int64_t>(
      {std::nullopt, 4200, std::nullopt, -700, 0, std::nullopt},
      DECIMAL(12, 2))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(12, 2)}), expected);
}

// All-null column: getDecimalValues must handle ConstantVector (all nulls)
// without crashing when scaleAdjust > 0.
TEST_F(ParquetReaderWideningTest, intToDecimalWideningAllNull) {
  auto writeData = makeRowVector({makeNullableFlatVector<int32_t>(
      {std::nullopt, std::nullopt, std::nullopt})});
  auto expected = makeRowVector({makeNullableFlatVector<int64_t>(
      {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(12, 2)}), expected);
}

TEST_F(ParquetReaderWideningTest, decimalToDecimalWideningAllNull) {
  auto writeData = makeRowVector({makeNullableFlatVector<int64_t>(
      {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(7, 2))});
  auto expected = makeRowVector({makeNullableFlatVector<int64_t>(
      {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(10, 4))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(10, 4)}), expected);
}

// INT32 -> SMALLINT narrowing. Parquet stores both as INT32; reading with
// a narrower type truncates to 16 bits via static_cast.
TEST_F(ParquetReaderWideningTest, intToSmallintNarrowing) {
  auto writeData =
      makeRowVector({makeFlatVector<int32_t>({0, 1, -1, 100, -100})});
  auto expected =
      makeRowVector({makeFlatVector<int16_t>({0, 1, -1, 100, -100})});
  assertNarrowingReads(writeData, ROW({"col"}, {SMALLINT()}), expected);
}

TEST_F(ParquetReaderWideningTest, smallintToTinyintNarrowing) {
  auto writeData =
      makeRowVector({makeFlatVector<int16_t>({0, 1, -1, 42, -42})});
  auto expected = makeRowVector({makeFlatVector<int8_t>({0, 1, -1, 42, -42})});
  assertNarrowingReads(writeData, ROW({"col"}, {TINYINT()}), expected);
}

TEST_F(ParquetReaderWideningTest, shortToIntegerWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int16_t>({0, 1, -1, 32'767, -32'768})});
  auto expected =
      makeRowVector({makeFlatVector<int32_t>({0, 1, -1, 32'767, -32'768})});
  assertWideningReads(writeData, ROW({"col"}, {INTEGER()}), expected);
}

TEST_F(ParquetReaderWideningTest, intToBigintWidening) {
  auto writeData = makeRowVector(
      {makeFlatVector<int32_t>({0, 1, -1, 2'147'483'647, -2'147'483'648})});
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({0, 1, -1, 2'147'483'647, -2'147'483'648})});
  assertWideningReads(writeData, ROW({"col"}, {BIGINT()}), expected);
}

// INT32 -> SMALLINT overflow: 32768 truncates to -32768.
TEST_F(ParquetReaderWideningTest, intToSmallintOverflow) {
  auto writeData = makeRowVector({makeFlatVector<int32_t>({32'768})});
  auto expected = makeRowVector({makeFlatVector<int16_t>({-32'768})});
  assertNarrowingReads(writeData, ROW({"col"}, {SMALLINT()}), expected);
}

// INT16 -> TINYINT overflow: 128 truncates to -128.
TEST_F(ParquetReaderWideningTest, smallintToTinyintOverflow) {
  auto writeData = makeRowVector({makeFlatVector<int16_t>({128})});
  auto expected = makeRowVector({makeFlatVector<int8_t>({-128})});
  assertNarrowingReads(writeData, ROW({"col"}, {TINYINT()}), expected);
}

// INT32 -> DOUBLE works because sizeof(int32_t)=4 != sizeof(double)=8,
// so getFlatValues takes the upcastScalarValues path.
TEST_F(ParquetReaderWideningTest, intToDoubleWidening) {
  auto writeData = makeRowVector({makeFlatVector<int32_t>(
      {0, 1, -1, 100, -100, 2'147'483'647, -2'147'483'648})});
  auto expected = makeRowVector({makeFlatVector<double>(
      {0.0, 1.0, -1.0, 100.0, -100.0, 2'147'483'647.0, -2'147'483'648.0})});
  assertWideningReads(writeData, ROW({"col"}, {DOUBLE()}), expected);
}

TEST_F(ParquetReaderWideningTest, intToDoubleWideningWithNulls) {
  auto writeData = makeRowVector({makeNullableFlatVector<int32_t>(
      {std::nullopt, 42, std::nullopt, -7, 0, std::nullopt})});
  auto expected = makeRowVector({makeNullableFlatVector<double>(
      {std::nullopt, 42.0, std::nullopt, -7.0, 0.0, std::nullopt})});
  assertWideningReads(writeData, ROW({"col"}, {DOUBLE()}), expected);
}

// Byte/Short -> Double widening. Parquet stores both as INT32.
TEST_F(ParquetReaderWideningTest, tinyintToDoubleWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int8_t>({0, 1, -1, 127, -128})});
  auto expected =
      makeRowVector({makeFlatVector<double>({0.0, 1.0, -1.0, 127.0, -128.0})});
  assertWideningReads(writeData, ROW({"col"}, {DOUBLE()}), expected);
}

TEST_F(ParquetReaderWideningTest, smallintToDoubleWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int16_t>({0, 1, -1, 32'767, -32'768})});
  auto expected = makeRowVector(
      {makeFlatVector<double>({0.0, 1.0, -1.0, 32'767.0, -32'768.0})});
  assertWideningReads(writeData, ROW({"col"}, {DOUBLE()}), expected);
}

// INT -> Decimal with scale=0 (exact boundary: p-s=10 for INT32).
TEST_F(ParquetReaderWideningTest, intToDecimalScale0Widening) {
  auto writeData =
      makeRowVector({makeFlatVector<int32_t>({0, 42, -42, 2'147'483'647})});
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({0, 42, -42, 2'147'483'647}, DECIMAL(10, 0))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(10, 0)}), expected);
}

// INT -> Decimal with scale=1 (p-s=10, minimum boundary with nonzero scale).
TEST_F(ParquetReaderWideningTest, intToDecimalScale1Widening) {
  auto writeData = makeRowVector({makeFlatVector<int32_t>({0, 5, -5, 100})});
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({0, 50, -50, 1000}, DECIMAL(11, 1))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(11, 1)}), expected);
}

// Byte -> Decimal with nonzero scale. Values multiplied by 10^scale.
TEST_F(ParquetReaderWideningTest, tinyintToDecimalWithScaleWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int8_t>({0, 1, -1, 127, -128})});
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({0, 10, -10, 1'270, -1'280}, DECIMAL(11, 1))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(11, 1)}), expected);
}

// Short -> Decimal with nonzero scale. Values multiplied by 10^scale.
TEST_F(ParquetReaderWideningTest, smallintToDecimalWithScaleWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int16_t>({0, 1, -1, 32'767, -32'768})});
  auto expected = makeRowVector({makeFlatVector<int64_t>(
      {0, 10, -10, 327'670, -327'680}, DECIMAL(11, 1))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(11, 1)}), expected);
}

// INT32 -> Long Decimal (crossing short/long boundary).
TEST_F(ParquetReaderWideningTest, intToLongDecimalWidening) {
  auto writeData = makeRowVector(
      {makeFlatVector<int32_t>({0, 1, -1, 2'147'483'647, -2'147'483'648})});
  auto expected = makeRowVector({makeFlatVector<int128_t>(
      {0, 1, -1, 2'147'483'647, -2'147'483'648}, DECIMAL(20, 0))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(20, 0)}), expected);
}

// BIGINT -> Decimal(38,0), maximum precision long decimal.
TEST_F(ParquetReaderWideningTest, bigintToMaxPrecisionDecimalWidening) {
  auto writeData = makeRowVector(
      {makeFlatVector<int64_t>({0, 1, -1, 9'223'372'036'854'775'807LL})});
  auto expected = makeRowVector({makeFlatVector<int128_t>(
      {0, 1, -1, static_cast<int128_t>(9'223'372'036'854'775'807LL)},
      DECIMAL(38, 0))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(38, 0)}), expected);
}

// BIGINT -> Decimal(21,1), INT64 with nonzero scale.
TEST_F(ParquetReaderWideningTest, bigintToDecimalWithScaleWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int64_t>({0, 1, -1, 999'999})});
  auto expected = makeRowVector(
      {makeFlatVector<int128_t>({0, 10, -10, 9'999'990}, DECIMAL(21, 1))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(21, 1)}), expected);
}

// Decimal short -> long decimal crossing: file stores int64 (short decimal),
// requested type is int128 (long decimal). getIntValues upcasts int64->int128.
TEST_F(ParquetReaderWideningTest, decimalShortToLongWidening) {
  auto writeData = makeRowVector(
      {makeFlatVector<int64_t>({1111, -2222, 0, 99999}, DECIMAL(5, 2))});
  auto expected = makeRowVector(
      {makeFlatVector<int128_t>({1111, -2222, 0, 99999}, DECIMAL(20, 2))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(20, 2)}), expected);
}

TEST_F(ParquetReaderWideningTest, decimalShortToLongWithScaleWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int64_t>({1111, -2222, 0}, DECIMAL(5, 2))});
  // v * 10^(4-2) = v * 100
  auto expected = makeRowVector(
      {makeFlatVector<int128_t>({111'100, -222'200, 0}, DECIMAL(20, 4))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(20, 4)}), expected);
}

// INT -> Decimal(38,0) max precision.
TEST_F(ParquetReaderWideningTest, smallintToMaxPrecisionDecimalWidening) {
  auto writeData = makeRowVector({makeFlatVector<int16_t>({0, 1, -1, 32'767})});
  auto expected = makeRowVector(
      {makeFlatVector<int128_t>({0, 1, -1, 32'767}, DECIMAL(38, 0))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(38, 0)}), expected);
}

TEST_F(ParquetReaderWideningTest, intToMaxPrecisionDecimalWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int32_t>({0, 1, -1, 2'147'483'647})});
  auto expected = makeRowVector(
      {makeFlatVector<int128_t>({0, 1, -1, 2'147'483'647}, DECIMAL(38, 0))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(38, 0)}), expected);
}

// Decimal(5,2) -> (10,7) -- short to short with large scale increase.
TEST_F(ParquetReaderWideningTest, decimalWideningLargeScaleIncrease) {
  auto writeData =
      makeRowVector({makeFlatVector<int64_t>({1111, -2222, 0}, DECIMAL(5, 2))});
  // v * 10^(7-2) = v * 100000
  auto expected = makeRowVector({makeFlatVector<int64_t>(
      {111'100'000, -222'200'000, 0}, DECIMAL(10, 7))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(10, 7)}), expected);
}

// Decimal(5,2) -> (20,17) -- short to long with large scale increase.
TEST_F(ParquetReaderWideningTest, decimalShortToLongLargeScaleWidening) {
  auto writeData =
      makeRowVector({makeFlatVector<int64_t>({1111, -2222, 0}, DECIMAL(5, 2))});
  // v * 10^(17-2) = v * 10^15
  auto expected = makeRowVector({makeFlatVector<int128_t>(
      {static_cast<int128_t>(1111) * 1'000'000'000'000'000LL,
       static_cast<int128_t>(-2222) * 1'000'000'000'000'000LL,
       0},
      DECIMAL(20, 17))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(20, 17)}), expected);
}

// Decimal(10,2) -> (12,4) -- short to short with precision and scale increase.
TEST_F(ParquetReaderWideningTest, decimalShortWideningPrecisionAndScale) {
  auto writeData = makeRowVector(
      {makeFlatVector<int64_t>({12345, -67890, 0}, DECIMAL(10, 2))});
  // v * 10^(4-2) = v * 100
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({1'234'500, -6'789'000, 0}, DECIMAL(12, 4))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(12, 4)}), expected);
}

// Decimal(10,2) -> (20,12) -- short to long with large scale increase.
TEST_F(
    ParquetReaderWideningTest,
    decimalShortToLongLargeScaleWideningHighPrecision) {
  auto writeData = makeRowVector(
      {makeFlatVector<int64_t>({12345, -67890, 0}, DECIMAL(10, 2))});
  // v * 10^(12-2) = v * 10^10
  auto expected = makeRowVector({makeFlatVector<int128_t>(
      {static_cast<int128_t>(12345) * 10'000'000'000LL,
       static_cast<int128_t>(-67890) * 10'000'000'000LL,
       0},
      DECIMAL(20, 12))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(20, 12)}), expected);
}

// Decimal(20,2) -> (22,4) -- long to long with scale increase.
TEST_F(ParquetReaderWideningTest, decimalLongToLongWithScaleWidening) {
  auto writeData = makeRowVector(
      {makeFlatVector<int128_t>({12345, -67890, 0}, DECIMAL(20, 2))});
  // v * 10^(4-2) = v * 100
  auto expected = makeRowVector(
      {makeFlatVector<int128_t>({1'234'500, -6'789'000, 0}, DECIMAL(22, 4))});
  assertWideningReads(writeData, ROW({"col"}, {DECIMAL(22, 4)}), expected);
}

TEST_F(ParquetReaderWideningTest, typeWideningRejectionIncompatibleTypes) {
  // INT32 -> FLOAT is not supported. FLOAT has only ~7 significant digits
  // vs INT32's 10, which would cause silent precision loss.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int32_t>({1, 2, 3})}),
      ROW({"col"}, {REAL()}),
      "INTEGER");

  // BIGINT -> DOUBLE is not supported.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})}),
      ROW({"col"}, {DOUBLE()}),
      "BIGINT");

  // BIGINT -> INTEGER/SMALLINT/TINYINT narrowing is not supported.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})}),
      ROW({"col"}, {INTEGER()}),
      "BIGINT");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})}),
      ROW({"col"}, {SMALLINT()}),
      "BIGINT");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})}),
      ROW({"col"}, {TINYINT()}),
      "BIGINT");

  // DOUBLE -> FLOAT/INTEGER is not supported.
  assertWideningThrows(
      makeRowVector({makeFlatVector<double>({1.0, 2.0, 3.0})}),
      ROW({"col"}, {REAL()}),
      "DOUBLE");
  assertWideningThrows(
      makeRowVector({makeFlatVector<double>({1.0, 2.0, 3.0})}),
      ROW({"col"}, {INTEGER()}),
      "DOUBLE");

  // FLOAT -> INTEGER/BIGINT is not supported.
  assertWideningThrows(
      makeRowVector({makeFlatVector<float>({1.0f, 2.0f, 3.0f})}),
      ROW({"col"}, {INTEGER()}),
      "REAL");
  assertWideningThrows(
      makeRowVector({makeFlatVector<float>({1.0f, 2.0f})}),
      ROW({"col"}, {BIGINT()}),
      "REAL");

  // BIGINT -> FLOAT is not supported (precision loss).
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1, 2})}),
      ROW({"col"}, {REAL()}),
      "BIGINT");
}

TEST_F(ParquetReaderWideningTest, typeWideningRejectionIntDecimalPrecision) {
  // INT32 -> DECIMAL(8,0). p-s=8 < 10, insufficient for INT32.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int32_t>({1, 2, 3})}),
      ROW({"col"}, {DECIMAL(8, 0)}),
      "INTEGER");

  // TINYINT -> DECIMAL(9,0). Parquet stores TINYINT as INT32, so the
  // minimum precision is p-s >= 10. p-s=9 < 10.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int8_t>({1, 2, 3})}),
      ROW({"col"}, {DECIMAL(9, 0)}),
      "TINYINT");

  // SMALLINT -> DECIMAL(9,0). Same as TINYINT: stored as INT32, p-s=9 < 10.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int16_t>({1, 2, 3})}),
      ROW({"col"}, {DECIMAL(9, 0)}),
      "SMALLINT");

  // BIGINT -> DECIMAL(18,0). p-s=18 < 20, insufficient for INT64.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})}),
      ROW({"col"}, {DECIMAL(18, 0)}),
      "BIGINT");

  // Exact boundary: INT32 -> DECIMAL(9,0), p-s=9 < 10.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int32_t>({1, 2, 3})}),
      ROW({"col"}, {DECIMAL(9, 0)}),
      "INTEGER");

  // Exact boundary: BIGINT -> DECIMAL(19,0), p-s=19 < 20.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})}),
      ROW({"col"}, {DECIMAL(19, 0)}),
      "BIGINT");

  // Rejection with nonzero scale: p-s must still meet the threshold.
  // TINYINT -> DECIMAL(3,1). p-s=2 < 10.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int8_t>({1, 2})}),
      ROW({"col"}, {DECIMAL(3, 1)}),
      "TINYINT");

  // INT32 -> DECIMAL(10,1). p-s=9 < 10.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int32_t>({1, 2})}),
      ROW({"col"}, {DECIMAL(10, 1)}),
      "INTEGER");

  // BIGINT -> DECIMAL(20,1). p-s=19 < 20.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1, 2})}),
      ROW({"col"}, {DECIMAL(20, 1)}),
      "BIGINT");

  // INT->Decimal with insufficient precision (various small precisions).
  auto tinyintData = makeRowVector({makeFlatVector<int8_t>({1})});
  auto smallintData = makeRowVector({makeFlatVector<int16_t>({1})});
  auto intData = makeRowVector({makeFlatVector<int32_t>({1})});
  auto bigintData = makeRowVector({makeFlatVector<int64_t>({1})});
  assertWideningThrows(tinyintData, ROW({"col"}, {DECIMAL(1, 0)}), "TINYINT");
  assertWideningThrows(tinyintData, ROW({"col"}, {DECIMAL(2, 0)}), "TINYINT");
  assertWideningThrows(tinyintData, ROW({"col"}, {DECIMAL(3, 0)}), "TINYINT");
  assertWideningThrows(smallintData, ROW({"col"}, {DECIMAL(3, 0)}), "SMALLINT");
  assertWideningThrows(smallintData, ROW({"col"}, {DECIMAL(4, 0)}), "SMALLINT");
  assertWideningThrows(smallintData, ROW({"col"}, {DECIMAL(5, 0)}), "SMALLINT");
  assertWideningThrows(intData, ROW({"col"}, {DECIMAL(5, 0)}), "INTEGER");
  assertWideningThrows(bigintData, ROW({"col"}, {DECIMAL(10, 0)}), "BIGINT");

  // INT->Decimal with nonzero scale, insufficient precision.
  assertWideningThrows(tinyintData, ROW({"col"}, {DECIMAL(4, 1)}), "TINYINT");
  assertWideningThrows(smallintData, ROW({"col"}, {DECIMAL(6, 1)}), "SMALLINT");
  assertWideningThrows(smallintData, ROW({"col"}, {DECIMAL(5, 1)}), "SMALLINT");
}

TEST_F(
    ParquetReaderWideningTest,
    typeWideningRejectionDecimalPrecisionDecrease) {
  // Decimal precision decrease.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1111}, DECIMAL(10, 2))}),
      ROW({"col"}, {DECIMAL(5, 2)}),
      "DECIMAL(10, 2)");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int128_t>({1111}, DECIMAL(20, 2))}),
      ROW({"col"}, {DECIMAL(5, 2)}),
      "DECIMAL(20, 2)");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1111}, DECIMAL(12, 2))}),
      ROW({"col"}, {DECIMAL(10, 2)}),
      "DECIMAL(12, 2)");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int128_t>({1111}, DECIMAL(20, 2))}),
      ROW({"col"}, {DECIMAL(10, 2)}),
      "DECIMAL(20, 2)");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int128_t>({1111}, DECIMAL(22, 2))}),
      ROW({"col"}, {DECIMAL(20, 2)}),
      "DECIMAL(22, 2)");

  // Decimal precision+scale decrease (precInc < 0).
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1111}, DECIMAL(10, 7))}),
      ROW({"col"}, {DECIMAL(5, 2)}),
      "DECIMAL(10, 7)");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int128_t>({1111}, DECIMAL(20, 17))}),
      ROW({"col"}, {DECIMAL(5, 2)}),
      "DECIMAL(20, 17)");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1111}, DECIMAL(12, 4))}),
      ROW({"col"}, {DECIMAL(10, 2)}),
      "DECIMAL(12, 4)");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int128_t>({1111}, DECIMAL(20, 17))}),
      ROW({"col"}, {DECIMAL(10, 2)}),
      "DECIMAL(20, 17)");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int128_t>({1111}, DECIMAL(22, 4))}),
      ROW({"col"}, {DECIMAL(20, 2)}),
      "DECIMAL(22, 4)");
}

TEST_F(ParquetReaderWideningTest, typeWideningRejectionDecimalScaleViolation) {
  // DECIMAL(7,2) -> DECIMAL(8,5). scaleInc=3 > precInc=1.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1111, 2222}, DECIMAL(7, 2))}),
      ROW({"col"}, {DECIMAL(8, 5)}),
      "DECIMAL(7, 2)");

  // DECIMAL(7,2) -> DECIMAL(6,2). Precision decrease is not allowed.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1111, 2222}, DECIMAL(7, 2))}),
      ROW({"col"}, {DECIMAL(6, 2)}),
      "DECIMAL(7, 2)");

  // DECIMAL(7,4) -> DECIMAL(8,2). Scale narrowing (scaleInc < 0) is rejected.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1111, 2222}, DECIMAL(7, 4))}),
      ROW({"col"}, {DECIMAL(8, 2)}),
      "DECIMAL(7, 4)");

  // Scale decrease with precision increase (scaleInc < 0).
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1111}, DECIMAL(10, 6))}),
      ROW({"col"}, {DECIMAL(12, 4)}),
      "DECIMAL(10, 6)");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int128_t>({1111}, DECIMAL(20, 7))}),
      ROW({"col"}, {DECIMAL(22, 5)}),
      "DECIMAL(20, 7)");

  // Precision decrease with scale increase.
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1111}, DECIMAL(12, 4))}),
      ROW({"col"}, {DECIMAL(10, 6)}),
      "DECIMAL(12, 4)");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int128_t>({1111}, DECIMAL(22, 5))}),
      ROW({"col"}, {DECIMAL(20, 7)}),
      "DECIMAL(22, 5)");

  // scaleInc > precInc (not enough precision for the scale increase).
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1111}, DECIMAL(5, 2))}),
      ROW({"col"}, {DECIMAL(6, 4)}),
      "DECIMAL(5, 2)");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int64_t>({1111}, DECIMAL(10, 4))}),
      ROW({"col"}, {DECIMAL(12, 7)}),
      "DECIMAL(10, 4)");
  assertWideningThrows(
      makeRowVector({makeFlatVector<int128_t>({1111}, DECIMAL(20, 5))}),
      ROW({"col"}, {DECIMAL(22, 8)}),
      "DECIMAL(20, 5)");
}

// Verify allowInt32Narrowing flag on ReaderOptions.
TEST_F(ParquetReaderWideningTest, allowInt32Narrowing) {
  // Write INT32 data with values that exercise truncation edge cases.
  auto data = makeRowVector(
      {"c1"},
      {makeFlatVector<int32_t>(
          {0,
           127,
           128,
           255,
           256,
           32767,
           32768,
           65535,
           -1,
           -128,
           -129,
           std::numeric_limits<int32_t>::min(),
           std::numeric_limits<int32_t>::max()})});
  auto* sink = write(data);

  // Default: flag is false.
  dwio::common::ReaderOptions readerOptions{leafPool_.get()};
  ASSERT_FALSE(readerOptions.allowInt32Narrowing());

  // INT32->TINYINT narrowing rejected by default.
  readerOptions.setFileSchema(ROW({"c1"}, {TINYINT()}));
  VELOX_ASSERT_THROW(
      createReaderInMemory(*sink, readerOptions),
      "is not allowed for requested type");

  // INT32->SMALLINT narrowing rejected by default.
  readerOptions.setFileSchema(ROW({"c1"}, {SMALLINT()}));
  VELOX_ASSERT_THROW(
      createReaderInMemory(*sink, readerOptions),
      "is not allowed for requested type");

  // Annotated type-matching always works without the flag.
  // INT_8 -> TINYINT: write as TINYINT (produces INT_8 annotation), read back.
  {
    auto readSchema = ROW({"c1"}, {TINYINT()});
    auto tinyData =
        makeRowVector({"c1"}, {makeFlatVector<int8_t>({-128, -1, 0, 1, 127})});
    auto* tinySink = write(tinyData);
    readerOptions.setFileSchema(readSchema);
    auto reader = createReaderInMemory(*tinySink, readerOptions);
    auto rowReaderOpts = getReaderOpts(readSchema);
    rowReaderOpts.setScanSpec(makeScanSpec(readSchema));
    auto rowReader = reader->createRowReader(rowReaderOpts);
    assertReadWithReaderAndExpected(
        readSchema, *rowReader, tinyData, *leafPool_);
  }

  // INT_16 -> SMALLINT: write as SMALLINT (produces INT_16 annotation), read
  // back.
  {
    auto readSchema = ROW({"c1"}, {SMALLINT()});
    auto smallData = makeRowVector(
        {"c1"}, {makeFlatVector<int16_t>({-32768, -1, 0, 1, 32767})});
    auto* smallSink = write(smallData);
    readerOptions.setFileSchema(readSchema);
    auto reader = createReaderInMemory(*smallSink, readerOptions);
    auto rowReaderOpts = getReaderOpts(readSchema);
    rowReaderOpts.setScanSpec(makeScanSpec(readSchema));
    auto rowReader = reader->createRowReader(rowReaderOpts);
    assertReadWithReaderAndExpected(
        readSchema, *rowReader, smallData, *leafPool_);
  }

  // With flag enabled, narrowing is allowed with silent truncation.
  readerOptions.setAllowInt32Narrowing(true);

  // INT32->TINYINT: values are truncated via static_cast<int8_t>.
  {
    auto readSchema = ROW({"c1"}, {TINYINT()});
    readerOptions.setFileSchema(readSchema);
    auto reader = createReaderInMemory(*sink, readerOptions);

    auto rowReaderOpts = getReaderOpts(readSchema);
    rowReaderOpts.setScanSpec(makeScanSpec(readSchema));
    auto rowReader = reader->createRowReader(rowReaderOpts);

    auto expected = makeRowVector(
        {"c1"},
        {makeFlatVector<int8_t>(
            {static_cast<int8_t>(0),
             static_cast<int8_t>(127),
             static_cast<int8_t>(128), // -128
             static_cast<int8_t>(255), // -1
             static_cast<int8_t>(256), // 0
             static_cast<int8_t>(32767), // -1
             static_cast<int8_t>(32768), // 0
             static_cast<int8_t>(65535), // -1
             static_cast<int8_t>(-1), // -1
             static_cast<int8_t>(-128), // -128
             static_cast<int8_t>(-129), // 127
             static_cast<int8_t>(std::numeric_limits<int32_t>::min()),
             static_cast<int8_t>(std::numeric_limits<int32_t>::max())})});
    assertReadWithReaderAndExpected(
        readSchema, *rowReader, expected, *leafPool_);
  }

  // INT32->SMALLINT: values are truncated via static_cast<int16_t>.
  {
    auto readSchema = ROW({"c1"}, {SMALLINT()});
    readerOptions.setFileSchema(readSchema);
    auto reader = createReaderInMemory(*sink, readerOptions);

    auto rowReaderOpts = getReaderOpts(readSchema);
    rowReaderOpts.setScanSpec(makeScanSpec(readSchema));
    auto rowReader = reader->createRowReader(rowReaderOpts);

    auto expected = makeRowVector(
        {"c1"},
        {makeFlatVector<int16_t>(
            {static_cast<int16_t>(0),
             static_cast<int16_t>(127),
             static_cast<int16_t>(128),
             static_cast<int16_t>(255),
             static_cast<int16_t>(256),
             static_cast<int16_t>(32767),
             static_cast<int16_t>(32768), // -32768
             static_cast<int16_t>(65535), // -1
             static_cast<int16_t>(-1), // -1
             static_cast<int16_t>(-128), // -128
             static_cast<int16_t>(-129), // -129
             static_cast<int16_t>(std::numeric_limits<int32_t>::min()),
             static_cast<int16_t>(std::numeric_limits<int32_t>::max())})});
    assertReadWithReaderAndExpected(
        readSchema, *rowReader, expected, *leafPool_);
  }
}

// INT -> Integer widening + filter.
TEST_F(ParquetReaderWideningTest, tinyintToSmallintWideningWithFilter) {
  auto writeData =
      makeRowVector({"col"}, {makeFlatVector<int8_t>({-10, 0, 10, 50, 127})});
  auto expected =
      makeRowVector({"col"}, {makeFlatVector<int16_t>({0, 10, 50, 127})});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {SMALLINT()}),
      exec::greaterThanOrEqual(0),
      expected);
}

TEST_F(ParquetReaderWideningTest, tinyintToIntegerWideningWithFilter) {
  auto writeData =
      makeRowVector({"col"}, {makeFlatVector<int8_t>({-10, 0, 10, 50, 127})});
  auto expected =
      makeRowVector({"col"}, {makeFlatVector<int32_t>({0, 10, 50, 127})});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {INTEGER()}),
      exec::greaterThanOrEqual(0),
      expected);
}

TEST_F(ParquetReaderWideningTest, shortToIntegerWideningWithFilter) {
  auto writeData = makeRowVector(
      {"col"}, {makeFlatVector<int16_t>({-100, 0, 50, 100, 32'767})});
  auto expected =
      makeRowVector({"col"}, {makeFlatVector<int32_t>({50, 100, 32'767})});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {INTEGER()}),
      exec::greaterThanOrEqual(50),
      expected);
}

TEST_F(ParquetReaderWideningTest, intToBigintWideningWithFilter) {
  auto writeData = makeRowVector(
      {"col"}, {makeFlatVector<int32_t>({-100, 0, 50, 100, 2'000'000})});
  auto expected =
      makeRowVector({"col"}, {makeFlatVector<int64_t>({50, 100, 2'000'000})});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {BIGINT()}),
      exec::greaterThanOrEqual(50),
      expected);
}

// INT -> DOUBLE widening + filter.
TEST_F(ParquetReaderWideningTest, intToDoubleWideningWithFilter) {
  auto writeData =
      makeRowVector({"col"}, {makeFlatVector<int32_t>({-100, 0, 50, 100})});
  auto expected =
      makeRowVector({"col"}, {makeFlatVector<double>({50.0, 100.0})});
  // BigintRange filter.
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DOUBLE()}),
      exec::greaterThanOrEqual(50),
      expected);
}

TEST_F(ParquetReaderWideningTest, tinyintToDoubleWideningWithFilter) {
  auto writeData =
      makeRowVector({"col"}, {makeFlatVector<int8_t>({-10, 0, 10, 50})});
  auto expected =
      makeRowVector({"col"}, {makeFlatVector<double>({0.0, 10.0, 50.0})});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DOUBLE()}),
      exec::greaterThanOrEqual(0),
      expected);
}

// DoubleRange filter not yet supported for widened columns. See #16895.
TEST_F(
    ParquetReaderWideningTest,
    DISABLED_intToDoubleWideningWithDoubleRangeFilter) {
  auto writeData =
      makeRowVector({"col"}, {makeFlatVector<int32_t>({-100, 0, 50, 100})});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DOUBLE()}),
      exec::greaterThanOrEqualDouble(50.0),
      makeRowVector({"col"}, {makeFlatVector<double>({50.0, 100.0})}));
}

// INT -> Decimal widening + filter.
TEST_F(ParquetReaderWideningTest, intToDecimalWideningWithFilter) {
  auto writeData =
      makeRowVector({"col"}, {makeFlatVector<int32_t>({-100, 0, 50, 100})});
  // Scale 0.
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(10, 0)}),
      exec::greaterThanOrEqual(50),
      makeRowVector(
          {"col"}, {makeFlatVector<int64_t>({50, 100}, DECIMAL(10, 0))}));
}

// Scale > 0 not yet supported for filter pushdown with widening. See #16895.
TEST_F(
    ParquetReaderWideningTest,
    DISABLED_intToDecimalWithScaleWideningWithFilter) {
  auto writeData =
      makeRowVector({"col"}, {makeFlatVector<int32_t>({-100, 0, 50, 100})});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(12, 2)}),
      exec::greaterThanOrEqual(5'000),
      makeRowVector(
          {"col"}, {makeFlatVector<int64_t>({5'000, 10'000}, DECIMAL(12, 2))}));
}

// HugeintRange filter not yet supported for widened columns. See #16895.
TEST_F(ParquetReaderWideningTest, DISABLED_bigintToDecimalWideningWithFilter) {
  auto writeData =
      makeRowVector({"col"}, {makeFlatVector<int64_t>({-100, 0, 50, 100})});
  // BIGINT -> Decimal(25, 5).
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(25, 5)}),
      exec::greaterThanOrEqualHugeint(int128_t(50) * 100'000),
      makeRowVector(
          {"col"},
          {makeFlatVector<int128_t>(
              {int128_t(50) * 100'000, int128_t(100) * 100'000},
              DECIMAL(25, 5))}));
  // BIGINT -> Decimal(38, 0).
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(38, 0)}),
      exec::greaterThanOrEqualHugeint(int128_t(50)),
      makeRowVector(
          {"col"}, {makeFlatVector<int128_t>({50, 100}, DECIMAL(38, 0))}));
  // BIGINT -> Decimal(21, 1).
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(21, 1)}),
      exec::greaterThanOrEqualHugeint(int128_t(50) * 10),
      makeRowVector(
          {"col"},
          {makeFlatVector<int128_t>(
              {int128_t(50) * 10, int128_t(100) * 10}, DECIMAL(21, 1))}));
}

// Decimal -> Decimal (short->short) + filter.
TEST_F(ParquetReaderWideningTest, decimalShortToShortWideningWithFilter) {
  auto writeData = makeRowVector(
      {"col"},
      {makeFlatVector<int64_t>({-1'000, 0, 5'000, 10'000}, DECIMAL(7, 2))});
  // Scale unchanged: DECIMAL(7,2) -> DECIMAL(10,2).
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(10, 2)}),
      exec::greaterThanOrEqual(5'000),
      makeRowVector(
          {"col"}, {makeFlatVector<int64_t>({5'000, 10'000}, DECIMAL(10, 2))}));
}

// Scale changed: DECIMAL(7,2) -> DECIMAL(10,4). See #16895.
TEST_F(
    ParquetReaderWideningTest,
    DISABLED_decimalScaleChangeWideningWithFilter) {
  auto writeData = makeRowVector(
      {"col"},
      {makeFlatVector<int64_t>({-1'000, 0, 5'000, 10'000}, DECIMAL(7, 2))});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(10, 4)}),
      exec::greaterThanOrEqual(500'000),
      makeRowVector(
          {"col"},
          {makeFlatVector<int64_t>({500'000, 1'000'000}, DECIMAL(10, 4))}));
}

// Cases have different failure modes: same-scale fails on HugeintRange crash,
// scale-change fails on unscaled value mismatch. See #16895.
TEST_F(
    ParquetReaderWideningTest,
    DISABLED_decimalShortToLongWideningWithFilter) {
  auto writeData = makeRowVector(
      {"col"},
      {makeFlatVector<int64_t>({-1'000, 0, 5'000, 10'000}, DECIMAL(5, 2))});
  // Scale unchanged: DECIMAL(5,2) -> DECIMAL(20,2).
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(20, 2)}),
      exec::greaterThanOrEqualHugeint(int128_t(5'000)),
      makeRowVector(
          {"col"},
          {makeFlatVector<int128_t>({5'000, 10'000}, DECIMAL(20, 2))}));
  // Scale changed: DECIMAL(5,2) -> DECIMAL(20,4).
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(20, 4)}),
      exec::greaterThanOrEqualHugeint(int128_t(500'000)),
      makeRowVector(
          {"col"},
          {makeFlatVector<int128_t>({500'000, 1'000'000}, DECIMAL(20, 4))}));
  // Large scale increase: DECIMAL(5,2) -> DECIMAL(10,7).
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(10, 7)}),
      exec::greaterThanOrEqual(500'000'000),
      makeRowVector(
          {"col"},
          {makeFlatVector<int64_t>(
              {500'000'000, 1'000'000'000}, DECIMAL(10, 7))}));
  // Precision and scale increase: DECIMAL(5,2) -> DECIMAL(12,4).
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(12, 4)}),
      exec::greaterThanOrEqual(500'000),
      makeRowVector(
          {"col"},
          {makeFlatVector<int64_t>({500'000, 1'000'000}, DECIMAL(12, 4))}));
}

// HugeintRange filter with scale change not yet supported. See #16895.
TEST_F(
    ParquetReaderWideningTest,
    DISABLED_decimalLongToLongWideningWithFilter) {
  auto writeData = makeRowVector(
      {"col"},
      {makeFlatVector<int128_t>({-1'000, 0, 5'000, 10'000}, DECIMAL(20, 2))});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(22, 4)}),
      exec::greaterThanOrEqualHugeint(int128_t(500'000)),
      makeRowVector(
          {"col"},
          {makeFlatVector<int128_t>({500'000, 1'000'000}, DECIMAL(22, 4))}));
}

// INT32 narrowing + filter.
TEST_F(ParquetReaderWideningTest, intNarrowingFilterBehavior) {
  // INT32 -> TINYINT with filter x in [0, 127] (TINYINT range).
  // INT32 value 200 fails filter (200 > 127), so it is filtered out.
  auto writeData =
      makeRowVector({"col"}, {makeFlatVector<int32_t>({-10, 0, 50, 127, 200})});
  auto expected =
      makeRowVector({"col"}, {makeFlatVector<int8_t>({0, 50, 127})});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {TINYINT()}),
      std::make_unique<common::BigintRange>(0, 127, /*nullAllowed=*/false),
      expected,
      /*allowInt32Narrowing=*/true);
}

// Null filter tests.
TEST_F(ParquetReaderWideningTest, intToBigintWideningNullFilter) {
  auto writeData = makeRowVector(
      {"col"},
      {makeNullableFlatVector<int32_t>({0, std::nullopt, 50, std::nullopt})});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {BIGINT()}),
      exec::isNull(),
      makeRowVector(
          {"col"},
          {makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt})}));
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {BIGINT()}),
      exec::isNotNull(),
      makeRowVector({"col"}, {makeFlatVector<int64_t>({0, 50})}));
}

TEST_F(ParquetReaderWideningTest, intToDoubleWideningNullFilter) {
  auto writeData = makeRowVector(
      {"col"},
      {makeNullableFlatVector<int32_t>({0, std::nullopt, 50, std::nullopt})});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DOUBLE()}),
      exec::isNull(),
      makeRowVector(
          {"col"},
          {makeNullableFlatVector<double>({std::nullopt, std::nullopt})}));
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DOUBLE()}),
      exec::isNotNull(),
      makeRowVector({"col"}, {makeFlatVector<double>({0.0, 50.0})}));
}

TEST_F(ParquetReaderWideningTest, intToDecimalScale0NullFilter) {
  auto writeData = makeRowVector(
      {"col"},
      {makeNullableFlatVector<int32_t>({0, std::nullopt, 50, std::nullopt})});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(10, 0)}),
      exec::isNull(),
      makeRowVector(
          {"col"},
          {makeNullableFlatVector<int64_t>(
              {std::nullopt, std::nullopt}, DECIMAL(10, 0))}));
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(10, 0)}),
      exec::isNotNull(),
      makeRowVector(
          {"col"}, {makeFlatVector<int64_t>({0, 50}, DECIMAL(10, 0))}));
}

TEST_F(ParquetReaderWideningTest, decimalPrecisionOnlyNullFilter) {
  auto writeData = makeRowVector(
      {"col"},
      {makeNullableFlatVector<int64_t>(
          {1'000, std::nullopt, 5'000, std::nullopt}, DECIMAL(7, 2))});
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(10, 2)}),
      exec::isNull(),
      makeRowVector(
          {"col"},
          {makeNullableFlatVector<int64_t>(
              {std::nullopt, std::nullopt}, DECIMAL(10, 2))}));
  assertWideningWithFilter(
      writeData,
      ROW({"col"}, {DECIMAL(10, 2)}),
      exec::isNotNull(),
      makeRowVector(
          {"col"}, {makeFlatVector<int64_t>({1'000, 5'000}, DECIMAL(10, 2))}));
}
