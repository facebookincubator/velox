/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <folly/container/F14Set.h>
#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <bit>

#include <fmt/format.h>
#include <gtest/gtest.h>

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

// Reinterprets a float as its raw bits so that -0.0f and +0.0f (which compare
// equal under float ==) can be distinguished in assertions.
uint32_t floatBits(float value) {
  return std::bit_cast<uint32_t>(value);
}

// Tests for FlatMapAsStructColumnReader, FlatMapAsMapColumnReader, and
// FlatMapColumnReader (native).
class FlatMapColumnReaderTest : public ::testing::TestWithParam<bool>,
                                public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::initializeMemoryManager(velox::memory::MemoryManager::Options{});
    registerSelectiveNimbleReaderFactory();
  }

  static void TearDownTestCase() {
    unregisterSelectiveNimbleReaderFactory();
  }

  memory::MemoryPool* rootPool() {
    return rootPool_.get();
  }

  struct Readers {
    std::unique_ptr<dwio::common::Reader> reader;
    std::unique_ptr<dwio::common::RowReader> rowReader;
  };

  std::string writeFlatMapFile(
      const RowVectorPtr& input,
      const std::string& flatMapColumn = "c0") {
    WriterOptions writerOptions;
    writerOptions.flatMapColumns = {{flatMapColumn, {}}};
    writerOptions.skipConstantFlatMapInMapStreams = GetParam();
    return test::createNimbleFile(*rootPool(), input, writerOptions);
  }

  // Standard makeReaders: uses expected's type as the requestedType.
  Readers makeReaders(
      const RowVectorPtr& expected,
      const std::string& file,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      bool preserveFlatMapsInMemory = false,
      bool lazyColumnIo = false,
      const folly::F14FastSet<std::string>& remainingFilterColumns = {}) {
    auto readFile = std::make_shared<InMemoryReadFile>(file);
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions options(pool());
    options.setDataIoStats(dataIoStats_);
    options.setMetadataIoStats(metadataIoStats_);
    options.setScanSpec(scanSpec);
    Readers readers;
    readers.reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
        options);
    auto type = asRowType(expected->type());
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(type);
    rowOptions.setPreserveFlatMapsInMemory(preserveFlatMapsInMemory);
    rowOptions.setLazyColumnIo(lazyColumnIo);
    if (!remainingFilterColumns.empty()) {
      rowOptions.setRemainingFilterColumns(remainingFilterColumns);
    }
    readers.rowReader = readers.reader->createRowReader(rowOptions);
    return readers;
  }

  void validate(
      const RowVector& expected,
      dwio::common::RowReader& rowReader,
      int batchSize) {
    auto result = BaseVector::create(asRowType(expected.type()), 0, pool());
    int numScanned = 0;
    int offset = 0;
    while (numScanned < expected.size()) {
      numScanned += rowReader.next(batchSize, result);
      result->validate();
      for (int j = 0; j < result->size(); ++j) {
        ASSERT_TRUE(result->equalValueAt(&expected, j, offset + j))
            << "Mismatch at row " << (offset + j) << ": expected "
            << expected.toString(offset + j) << " got " << result->toString(j);
      }
      offset += result->size();
    }
    ASSERT_EQ(numScanned, expected.size());
    ASSERT_EQ(0, rowReader.next(1, result));
  }

  const std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  const std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
};

// ----- FlatMapAsStruct tests -----
// Pattern: write input (MAP type) via writeFlatMapFile, create reader using
// input (MAP-typed) for requestedType, set setFlatMapAsStruct on scanSpec,
// then read into a batch created with outType (struct-typed).

TEST_P(FlatMapColumnReaderTest, asStructBasic) {
  auto input = makeRowVector({
      makeMapVector<int32_t, int64_t>({
          {{1, 10}, {2, 20}, {3, 30}},
          {{1, 40}, {3, 50}},
          {{2, 60}},
      }),
  });
  auto file = writeFlatMapFile(input);
  auto outType =
      ROW({"c0"}, {ROW({"1", "2", "3"}, {BIGINT(), BIGINT(), BIGINT()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, file, scanSpec);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 3);
  auto expected = makeRowVector({makeRowVector(
      {"1", "2", "3"},
      {
          makeNullableFlatVector<int64_t>({10, 40, std::nullopt}),
          makeNullableFlatVector<int64_t>({20, std::nullopt, 60}),
          makeNullableFlatVector<int64_t>({30, 50, std::nullopt}),
      })});
  velox::test::assertEqualVectors(expected, batch);
}

TEST_P(FlatMapColumnReaderTest, asStructInt8Keys) {
  auto input = makeRowVector({
      makeMapVector<int8_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{1, 300}},
      }),
  });
  auto file = writeFlatMapFile(input);
  auto outType = ROW({"c0"}, {ROW({"1", "2"}, {BIGINT(), BIGINT()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, file, scanSpec);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 2);
  auto expected = makeRowVector({makeRowVector(
      {"1", "2"},
      {
          makeFlatVector<int64_t>({100, 300}),
          makeNullableFlatVector<int64_t>({200, std::nullopt}),
      })});
  velox::test::assertEqualVectors(expected, batch);
}

TEST_P(FlatMapColumnReaderTest, asStructInt16Keys) {
  auto input = makeRowVector({
      makeMapVector<int16_t, int64_t>({
          {{10, 100}, {20, 200}},
          {{10, 300}, {20, 400}},
      }),
  });
  auto file = writeFlatMapFile(input);
  auto outType = ROW({"c0"}, {ROW({"10", "20"}, {BIGINT(), BIGINT()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, file, scanSpec);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 2);
  auto expected = makeRowVector({makeRowVector(
      {"10", "20"},
      {
          makeFlatVector<int64_t>({100, 300}),
          makeFlatVector<int64_t>({200, 400}),
      })});
  velox::test::assertEqualVectors(expected, batch);
}

TEST_P(FlatMapColumnReaderTest, asStructInt64Keys) {
  auto input = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{100, 1}, {200, 2}},
          {{200, 3}},
      }),
  });
  auto file = writeFlatMapFile(input);
  auto outType = ROW({"c0"}, {ROW({"100", "200"}, {BIGINT(), BIGINT()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, file, scanSpec);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 2);
  auto expected = makeRowVector({makeRowVector(
      {"100", "200"},
      {
          makeNullableFlatVector<int64_t>({1, std::nullopt}),
          makeFlatVector<int64_t>({2, 3}),
      })});
  velox::test::assertEqualVectors(expected, batch);
}

TEST_P(FlatMapColumnReaderTest, asStructStringKeys) {
  auto input = makeRowVector({
      makeMapVector<StringView, int64_t>({
          {{"a", 1}, {"b", 2}, {"c", 3}},
          {{"a", 4}, {"c", 5}},
          {{"b", 6}},
      }),
  });
  auto file = writeFlatMapFile(input);
  auto outType =
      ROW({"c0"}, {ROW({"a", "b", "c"}, {BIGINT(), BIGINT(), BIGINT()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, file, scanSpec);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 3);
  auto expected = makeRowVector({makeRowVector(
      {"a", "b", "c"},
      {
          makeNullableFlatVector<int64_t>({1, 4, std::nullopt}),
          makeNullableFlatVector<int64_t>({2, std::nullopt, 6}),
          makeNullableFlatVector<int64_t>({3, 5, std::nullopt}),
      })});
  velox::test::assertEqualVectors(expected, batch);
}

TEST_P(FlatMapColumnReaderTest, asStructMultiBatch) {
  // 50 rows, batch size 7 — multi-batch transitions.
  auto input = makeRowVector({
      makeMapVector<int32_t, int64_t>(
          50,
          [](auto i) { return 1 + i % 3; },
          [](auto j) { return static_cast<int32_t>(1 + j % 5); },
          [](auto j) { return static_cast<int64_t>(j * 10); }),
  });
  auto file = writeFlatMapFile(input);
  // Keys are 1-5.
  auto outType =
      ROW({"c0"},
          {ROW(
              {"1", "2", "3", "4", "5"},
              {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, file, scanSpec);
  auto result = BaseVector::create(outType, 0, pool());
  int totalRead = 0;
  while (totalRead < 50) {
    auto numRead = readers.rowReader->next(7, result);
    if (numRead == 0) {
      break;
    }
    result->validate();
    auto* row = result->asUnchecked<RowVector>();
    ASSERT_TRUE(row->childAt(0)->type()->isRow());
    totalRead += result->size();
  }
  ASSERT_EQ(totalRead, 50);
}

TEST_P(FlatMapColumnReaderTest, asStructSubsetOfKeys) {
  // Request only keys {"2", "4"} from {1, 2, 3, 4, 5} — subfield pruning.
  auto input = makeRowVector({
      makeMapVector<int32_t, int64_t>({
          {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}},
          {{1, 60}, {2, 70}, {3, 80}, {4, 90}, {5, 100}},
      }),
  });
  auto file = writeFlatMapFile(input);
  auto outType = ROW({"c0"}, {ROW({"2", "4"}, {BIGINT(), BIGINT()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, file, scanSpec);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 2);
  auto expected = makeRowVector({makeRowVector(
      {"2", "4"},
      {
          makeFlatVector<int64_t>({20, 70}),
          makeFlatVector<int64_t>({40, 90}),
      })});
  velox::test::assertEqualVectors(expected, batch);
}

TEST_P(FlatMapColumnReaderTest, asStructWithNulls) {
  // Some null map rows — null propagation to struct output.
  auto input = makeRowVector({
      makeNullableMapVector<int32_t, int64_t>({
          {{{{1, 10}, {2, 20}}}},
          std::nullopt,
          {{{{1, 30}}}},
          std::nullopt,
      }),
  });
  auto file = writeFlatMapFile(input);
  auto outType = ROW({"c0"}, {ROW({"1", "2"}, {BIGINT(), BIGINT()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, file, scanSpec);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 4);
  batch->validate();
  auto* row = batch->asUnchecked<RowVector>();
  auto& c0 = row->childAt(0);
  // Rows 1 and 3 should be null structs.
  ASSERT_FALSE(c0->isNullAt(0));
  ASSERT_TRUE(c0->isNullAt(1));
  ASSERT_FALSE(c0->isNullAt(2));
  ASSERT_TRUE(c0->isNullAt(3));
}

TEST_P(FlatMapColumnReaderTest, asStructAllEmpty) {
  // All rows have empty maps — all struct fields should be null.
  auto input = makeRowVector({
      makeMapVector<int32_t, int64_t>({{}, {}, {}}),
  });
  auto file = writeFlatMapFile(input);
  auto outType = ROW({"c0"}, {ROW({"1"}, {BIGINT()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, file, scanSpec);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 3);
  auto expected = makeRowVector({
      makeRowVector({"1"}, {makeNullConstant(TypeKind::BIGINT, 3)}),
  });
  velox::test::assertEqualVectors(expected, batch);
}

TEST_P(FlatMapColumnReaderTest, asStructComplexValues) {
  // MAP(INT, ARRAY(BIGINT)) as struct — complex value types.
  auto inner = makeArrayVector<int64_t>({{1, 2}, {3}, {4, 5, 6}});
  auto keys = makeFlatVector<int32_t>({1, 2, 1});
  auto mapVector = std::make_shared<MapVector>(
      pool(),
      MAP(INTEGER(), ARRAY(BIGINT())),
      nullptr,
      2,
      makeIndices({0, 2}),
      makeIndices({2, 1}),
      keys,
      inner);
  auto input = makeRowVector({mapVector});
  auto file = writeFlatMapFile(input);
  auto outType =
      ROW({"c0"}, {ROW({"1", "2"}, {ARRAY(BIGINT()), ARRAY(BIGINT())})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, file, scanSpec);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 2);
  batch->validate();
  auto* row = batch->asUnchecked<RowVector>();
  auto* structCol = row->childAt(0)->loadedVector()->asUnchecked<RowVector>();
  ASSERT_EQ(structCol->childrenSize(), 2);
}

// ----- FlatMapAsMap tests -----

TEST_P(FlatMapColumnReaderTest, asMapBasic) {
  auto input = makeRowVector({
      makeMapVector<int32_t, int64_t>({
          {{1, 10}, {2, 20}},
          {{1, 30}, {3, 40}},
          {{2, 50}},
      }),
  });
  auto file = writeFlatMapFile(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, file, scanSpec);
  validate(*input, *readers.rowReader, 10);
}

TEST_P(FlatMapColumnReaderTest, asMapMultiBatch) {
  // 30 rows, batch size 5.
  auto input = makeRowVector({
      makeMapVector<int32_t, int64_t>(
          30,
          [](auto i) { return 1 + i % 3; },
          [](auto j) { return static_cast<int32_t>(1 + j % 4); },
          [](auto j) { return static_cast<int64_t>(j * 10); }),
  });
  auto file = writeFlatMapFile(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, file, scanSpec);
  validate(*input, *readers.rowReader, 5);
}

TEST_P(FlatMapColumnReaderTest, asMapWithNulls) {
  auto input = makeRowVector({
      makeNullableMapVector<int32_t, int64_t>({
          {{{{1, 10}, {2, 20}}}},
          std::nullopt,
          {{{{3, 30}}}},
          std::nullopt,
          {{{{1, 40}, {2, 50}, {3, 60}}}},
      }),
  });
  auto file = writeFlatMapFile(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, file, scanSpec);
  validate(*input, *readers.rowReader, 3);
}

TEST_P(FlatMapColumnReaderTest, asMapKeyFilter) {
  auto input = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 100}, {5, 500}, {10, 1000}},
          {{2, 200}, {8, 800}},
          {{3, 300}, {7, 700}, {15, 1500}},
      }),
  });
  auto file = writeFlatMapFile(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")
      ->childByName(common::ScanSpec::kMapKeysFieldName)
      ->setFilter(std::make_unique<common::BigintRange>(5, 10, false));
  auto expected = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{5, 500}, {10, 1000}},
          {{8, 800}},
          {{7, 700}},
      }),
  });
  auto readers = makeReaders(expected, file, scanSpec);
  validate(*expected, *readers.rowReader, 10);
}

TEST_P(FlatMapColumnReaderTest, asMapComplexValues) {
  // MAP(INT, ROW({BIGINT})) — struct-valued flat map as map.
  auto values =
      makeRowVector({"val"}, {makeFlatVector<int64_t>({100, 200, 300, 400})});
  auto keys = makeFlatVector<int32_t>({1, 2, 1, 2});
  auto mapVector = std::make_shared<MapVector>(
      pool(),
      MAP(INTEGER(), ROW({"val"}, {BIGINT()})),
      nullptr,
      2,
      makeIndices({0, 2}),
      makeIndices({2, 2}),
      keys,
      values);
  auto input = makeRowVector({mapVector});
  auto file = writeFlatMapFile(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, file, scanSpec);
  validate(*input, *readers.rowReader, 10);
}

// ----- Native FlatMapColumnReader tests (preserveFlatMapsInMemory) -----

TEST_P(FlatMapColumnReaderTest, nativeMultiBatch) {
  // 40 rows, batch size 7 — FlatMapVector multi-batch.
  std::vector<std::vector<std::pair<int32_t, std::optional<int64_t>>>> data;
  for (int i = 0; i < 40; ++i) {
    std::vector<std::pair<int32_t, std::optional<int64_t>>> row;
    for (int j = 0; j <= i % 3; ++j) {
      row.emplace_back(
          static_cast<int32_t>(1 + j), static_cast<int64_t>(i * 10 + j));
    }
    data.push_back(std::move(row));
  }
  auto flatMap = makeFlatMapVector<int32_t, int64_t>(data);
  auto input = makeRowVector({flatMap, flatMap->toMapVector()});
  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c0", {}}};
  writerOptions.skipConstantFlatMapInMapStreams = GetParam();
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers =
      makeReaders(input, file, scanSpec, /*preserveFlatMapsInMemory=*/true);
  // Cross-compare: c0 (flat map on disk) is read as FlatMapVector,
  // c1 (regular map on disk) is read as MapVector.
  auto expected = makeRowVector({flatMap->toMapVector(), flatMap});
  validate(*expected, *readers.rowReader, 7);
}

TEST_P(FlatMapColumnReaderTest, nativeWithNulls) {
  auto flatMap = makeNullableFlatMapVector<int32_t, int64_t>({
      {{{1, 10}, {2, 20}}},
      std::nullopt,
      {{{1, 30}}},
      std::nullopt,
      {{{2, 40}, {3, 50}}},
  });
  auto input = makeRowVector({flatMap, flatMap->toMapVector()});
  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c0", {}}};
  writerOptions.skipConstantFlatMapInMapStreams = GetParam();
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers =
      makeReaders(input, file, scanSpec, /*preserveFlatMapsInMemory=*/true);
  auto expected = makeRowVector({flatMap->toMapVector(), flatMap});
  validate(*expected, *readers.rowReader, 3);
}

// Regression: float flat-map value streams with mixed sign bits (-0.0f/+0.0f)
// must round-trip bit-exactly through the native FlatMapVector path. The value
// stream is logically constant (all zeros) but physically distinct; without ALP
// the writer must not collapse it into a single ConstantEncoding value.
// Existing native FlatMapVector tests only cover int64, so this closes the
// float coverage gap.
TEST_P(FlatMapColumnReaderTest, nativeFloatMixedSignedZeroPreservesBits) {
  constexpr int kNumRows = 200;
  std::vector<std::vector<std::pair<int32_t, std::optional<float>>>> data;
  data.reserve(kNumRows);
  for (int i = 0; i < kNumRows; ++i) {
    // Single key per row; the key's value stream alternates -0.0f/+0.0f.
    data.push_back({{1, (i % 2 == 0) ? -0.0f : 0.0f}});
  }

  auto flatMap = makeFlatMapVector<int32_t, float>(data);
  auto input = makeRowVector({flatMap});
  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c0", {}}};
  writerOptions.skipConstantFlatMapInMapStreams = GetParam();
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);

  // Read c0 back as a native FlatMapVector (preserveFlatMapsInMemory).
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto firstReaders =
      makeReaders(input, file, scanSpec, /*preserveFlatMapsInMemory=*/true);
  auto roundTripped = BaseVector::create(asRowType(input->type()), 0, pool());
  ASSERT_EQ(firstReaders.rowReader->next(kNumRows, roundTripped), kNumRows);
  roundTripped->validate();

  // Write the read-back FlatMapVector out again and re-read it as a MapVector.
  auto file2 = test::createNimbleFile(*rootPool(), roundTripped, writerOptions);
  auto secondScanSpec = std::make_shared<common::ScanSpec>("root");
  secondScanSpec->addAllChildFields(*input->type());
  auto secondReaders = makeReaders(input, file2, secondScanSpec);
  auto result = BaseVector::create(asRowType(input->type()), 0, pool());
  ASSERT_EQ(secondReaders.rowReader->next(kNumRows, result), kNumRows);
  result->validate();

  // Bit-compare each value against the original input; -0.0f != +0.0f.
  auto* resultRow = result->asUnchecked<RowVector>();
  auto* mapVec =
      resultRow->childAt(0)->loadedVector()->asUnchecked<MapVector>();
  auto* values = mapVec->mapValues()->loadedVector()->as<SimpleVector<float>>();
  ASSERT_NE(values, nullptr);
  const auto* rawOffsets = mapVec->rawOffsets();
  const auto* rawSizes = mapVec->rawSizes();
  for (int i = 0; i < kNumRows; ++i) {
    ASSERT_EQ(rawSizes[i], 1) << "row " << i;
    EXPECT_EQ(
        floatBits(values->valueAt(rawOffsets[i])),
        floatBits(*data[i][0].second))
        << "row " << i;
  }
}

// ----- Lazy I/O tests -----

TEST_P(FlatMapColumnReaderTest, lazyIOFilterOnScalar) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{1, 300}},
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
  });
  auto file = writeFlatMapFile(input, "c1");
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(30, 100, false));
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 3);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
  });
  velox::test::assertEqualVectors(expected, result);
}

TEST_P(FlatMapColumnReaderTest, lazyIOFilterDropsAll) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}},
          {{2, 200}},
          {{3, 300}},
      }),
  });
  auto file = writeFlatMapFile(input, "c1");
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(999, 1000, false));
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 0);
}

TEST_P(FlatMapColumnReaderTest, lazyIOFilterOnScalarAsStruct) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{1, 300}},
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
      }),
  });
  auto file = writeFlatMapFile(input, "c1");
  auto outType =
      ROW({"c0", "c1"},
          {BIGINT(), ROW({"1", "2", "3"}, {BIGINT(), BIGINT(), BIGINT()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c1")->setFlatMapAsStruct(true);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(30, 100, false));
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  readers.rowReader->next(100, batch);
  ASSERT_EQ(batch->size(), 2);
  batch->validate();
  auto expected = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({30, 40}),
       makeRowVector(
           {"1", "2", "3"},
           {
               makeNullableFlatVector<int64_t>({std::nullopt, 600}),
               makeNullableFlatVector<int64_t>({400, 700}),
               makeNullableFlatVector<int64_t>({500, 800}),
           })});
  velox::test::assertEqualVectors(expected, batch);
}

TEST_P(FlatMapColumnReaderTest, lazyIOMultiBatch) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>(
          50, [](auto i) { return static_cast<int64_t>(i); }),
      makeMapVector<int32_t, int64_t>(
          50,
          [](auto i) { return 1 + i % 3; },
          [](auto j) { return static_cast<int32_t>(1 + j % 3); },
          [](auto j) { return static_cast<int64_t>(j * 10); }),
  });
  auto file = writeFlatMapFile(input, "c1");
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  // Keep rows where c0 >= 25 (25 of 50 rows survive).
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(25, 100, false));
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  auto result = BaseVector::create(asRowType(input->type()), 0, pool());
  int totalOutput = 0;
  int totalScanned = 0;
  while (totalScanned < 50) {
    auto n = readers.rowReader->next(7, result);
    if (n == 0) {
      break;
    }
    result->validate();
    totalOutput += result->size();
    totalScanned += n;
  }
  ASSERT_EQ(totalScanned, 50);
  ASSERT_EQ(totalOutput, 25);
}

TEST_P(FlatMapColumnReaderTest, lazyIOMultipleComplexColumns) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{1, 300}},
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
      makeMapVector<int32_t, int32_t>({
          {{10, 1}},
          {{20, 2}, {30, 3}},
          {{10, 4}},
          {{20, 5}},
          {{10, 6}, {30, 7}},
      }),
  });
  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c1", {}}, {"c2", {}}};
  writerOptions.skipConstantFlatMapInMapStreams = GetParam();
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(30, 100, false));
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 3);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
      makeMapVector<int32_t, int32_t>({
          {{10, 4}},
          {{20, 5}},
          {{10, 6}, {30, 7}},
      }),
  });
  velox::test::assertEqualVectors(expected, result);
}

TEST_P(FlatMapColumnReaderTest, noLazyIOByDefault) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{1, 300}},
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
  });
  auto file = writeFlatMapFile(input, "c1");
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(30, 100, false));
  auto readers = makeReaders(input, file, scanSpec);
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 3);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
  });
  velox::test::assertEqualVectors(expected, result);
}

// Remaining filter column should NOT be lazy — reads same data as eager.
TEST_P(FlatMapColumnReaderTest, lazyIORemainingFilterStaysEager) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{1, 300}},
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
  });
  auto file = writeFlatMapFile(input, "c1");
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(30, 100, false));
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true,
      /*remainingFilterColumns=*/{"c1"});
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 3);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
  });
  velox::test::assertEqualVectors(expected, result);
}

// No filter at all — hasFilterChild is false, nothing should be lazy.
TEST_P(FlatMapColumnReaderTest, lazyIONoFilter) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{1, 300}},
          {{2, 400}, {3, 500}},
      }),
  });
  auto file = writeFlatMapFile(input, "c1");
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  // No filter set on any column.
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 3);
  velox::test::assertEqualVectors(input, result);
}

// Remaining filter only — no pushdown filter on any column. c0 should be
// lazy, c1 should stay eager (in remainingFilterColumns).
TEST_P(FlatMapColumnReaderTest, lazyIORemainingFilterOnly) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{1, 300}},
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
  });
  auto file = writeFlatMapFile(input, "c1");
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  // No pushdown filter on any column. c1 is in remainingFilterColumns.
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true,
      /*remainingFilterColumns=*/{"c1"});
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 5);
  velox::test::assertEqualVectors(input, result);
}

// Only scalar columns — c1 (no filter) is lazy.
TEST_P(FlatMapColumnReaderTest, lazyIOScalarOnly) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
  });
  WriterOptions writerOptions;
  writerOptions.skipConstantFlatMapInMapStreams = GetParam();
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(30, 100, false));
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 3);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({30, 40, 50}),
      makeFlatVector<int32_t>({3, 4, 5}),
  });
  velox::test::assertEqualVectors(expected, result);
}

// Mixed scalar + FlatMap: scalar filter, scalar projected, FlatMap lazy.
// Verifies interleaved lazy/non-lazy children work correctly.
TEST_P(FlatMapColumnReaderTest, lazyIOMixedScalarAndFlatMap) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}},
          {{2, 200}},
          {{3, 300}},
          {{1, 400}},
          {{2, 500}},
      }),
      makeFlatVector<double>({1.1, 2.2, 3.3, 4.4, 5.5}),
  });
  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c2", {}}};
  writerOptions.skipConstantFlatMapInMapStreams = GetParam();
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(30, 100, false));
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 3);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({30, 40, 50}),
      makeFlatVector<int32_t>({3, 4, 5}),
      makeMapVector<int32_t, int64_t>({
          {{3, 300}},
          {{1, 400}},
          {{2, 500}},
      }),
      makeFlatVector<double>({3.3, 4.4, 5.5}),
  });
  velox::test::assertEqualVectors(expected, result);
}

// FlatMap key selection is the only filter in the scan. hasFilterInSubtree()
// detects it and enables lazy I/O; hasFilter() ignores it so the FlatMap
// itself is still lazy. The scalar column c0 has no filter and should
// also be lazy.
TEST_P(FlatMapColumnReaderTest, lazyIOFlatMapKeySelectionAsOnlyFilter) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}, {3, 300}},
          {{1, 400}, {2, 500}},
          {{1, 600}, {3, 700}},
          {{2, 800}, {3, 900}},
          {{1, 1000}},
      }),
  });
  auto file = writeFlatMapFile(input, "c1");
  auto outType =
      ROW({"c0", "c1"}, {BIGINT(), ROW({"1", "2"}, {BIGINT(), BIGINT()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c1")->setFlatMapAsStruct(true);
  // No filter on c0 — the only filter is implicit key selection on c1.
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  readers.rowReader->next(100, batch);
  ASSERT_EQ(batch->size(), 5);
  batch->validate();
  auto expected = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
       makeRowVector(
           {"1", "2"},
           {
               makeNullableFlatVector<int64_t>(
                   {100, 400, 600, std::nullopt, 1000}),
               makeNullableFlatVector<int64_t>(
                   {200, 500, std::nullopt, 800, std::nullopt}),
           })});
  velox::test::assertEqualVectors(expected, batch);
}

// Nested ROW column with a pushdown filter on a nested child. The parent
// column should NOT be lazy because hasFilter() recurses into non-map
// children and detects the nested filter.
TEST_P(FlatMapColumnReaderTest, lazyIONestedRowWithFilter) {
  auto rowType = ROW({"a", "b"}, {BIGINT(), BIGINT()});
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeRowVector(
          {"a", "b"},
          {
              makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
              makeFlatVector<int64_t>({100, 200, 300, 400, 500}),
          }),
      makeMapVector<int32_t, int64_t>({
          {{1, 10}},
          {{2, 20}},
          {{3, 30}},
          {{1, 40}},
          {{2, 50}},
      }),
  });
  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c2", {}}};
  writerOptions.skipConstantFlatMapInMapStreams = GetParam();
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  // Filter on nested field c1.a — parent c1 should NOT be lazy.
  scanSpec->childByName("c1")->childByName("a")->setFilter(
      std::make_unique<common::BigintRange>(3, 100, false));
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 3);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({30, 40, 50}),
      makeRowVector(
          {"a", "b"},
          {
              makeFlatVector<int64_t>({3, 4, 5}),
              makeFlatVector<int64_t>({300, 400, 500}),
          }),
      makeMapVector<int32_t, int64_t>({
          {{3, 30}},
          {{1, 40}},
          {{2, 50}},
      }),
  });
  velox::test::assertEqualVectors(expected, result);
}

// Nested ROW column with no filter — should use lazy I/O when a sibling has a
// pushdown filter. Exercises the makeChildParams() path: the nested
// StructColumnReader must not create LazyInput (guarded by NIMBLE_CHECK).
TEST_P(FlatMapColumnReaderTest, lazyIONestedRowNoFilter) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeRowVector(
          {"a", "b"},
          {
              makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
              makeFlatVector<int64_t>({100, 200, 300, 400, 500}),
          }),
  });
  WriterOptions writerOptions;
  writerOptions.skipConstantFlatMapInMapStreams = GetParam();
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(30, 100, false));
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 3);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({30, 40, 50}),
      makeRowVector(
          {"a", "b"},
          {
              makeFlatVector<int64_t>({3, 4, 5}),
              makeFlatVector<int64_t>({300, 400, 500}),
          }),
  });
  velox::test::assertEqualVectors(expected, result);
}

// A projected column with a post-read extraction transform, read with lazy
// column I/O enabled. The transform is still applied and the result must match
// the eager read.
TEST_P(FlatMapColumnReaderTest, lazyIOTransformColumn) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{1, 300}},
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
  });
  WriterOptions writerOptions;
  writerOptions.skipConstantFlatMapInMapStreams = GetParam();
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto outType =
      ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), input->type()->childAt(2)});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(30, 100, false));
  scanSpec->childByName("c1")->setTransform(
      [](const VectorPtr& v, memory::MemoryPool* pool) -> VectorPtr {
        auto flat = v->asFlatVector<int64_t>();
        auto result = BaseVector::create(BIGINT(), flat->size(), pool);
        auto* out = result->asFlatVector<int64_t>();
        for (auto i = 0; i < flat->size(); ++i) {
          out->set(i, flat->valueAt(i) * 10);
        }
        return result;
      },
      BIGINT());
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr result = BaseVector::create(outType, 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 3);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({30, 40, 50}),
      makeFlatVector<int64_t>({30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
  });
  velox::test::assertEqualVectors(expected, result);
}

// A transform column read lazily across three stripes, applying the transform
// at each stripe; the result must match the eager read.
TEST_P(FlatMapColumnReaderTest, lazyIOTransformColumnMultiStripe) {
  auto batch1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({100, 200, 300}),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int64_t>({4, 5, 6}),
      makeFlatVector<int64_t>({400, 500, 600}),
  });
  auto batch3 = makeRowVector({
      makeFlatVector<int64_t>({7, 8, 9}),
      makeFlatVector<int64_t>({700, 800, 900}),
  });
  WriterOptions writerOptions;
  writerOptions.skipConstantFlatMapInMapStreams = GetParam();
  auto file = test::createNimbleFile(
      *rootPool(),
      {batch1, batch2, batch3},
      writerOptions,
      /*flushAfterWrite=*/true);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*batch1->type());
  // Pass-all filter keeps c0 eager; c1 has a post-read transform (x10) and is
  // projected -> lazy-eligible.
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(0, 1000, false));
  scanSpec->childByName("c1")->setTransform(
      [](const VectorPtr& v, memory::MemoryPool* pool) -> VectorPtr {
        auto* flat = v->asFlatVector<int64_t>();
        auto result = BaseVector::create(BIGINT(), flat->size(), pool);
        auto* out = result->asFlatVector<int64_t>();
        for (auto i = 0; i < flat->size(); ++i) {
          out->set(i, flat->valueAt(i) * 10);
        }
        return result;
      },
      BIGINT());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}),
      makeFlatVector<int64_t>(
          {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000}),
  });
  auto readers = makeReaders(
      expected,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  // Batch size 2 (< 3 rows/stripe) crosses stripe boundaries while c1 is
  // materialized via the lazy TransformColumnLoader + hook.
  validate(*expected, *readers.rowReader, 2);
}

TEST_P(FlatMapColumnReaderTest, lazyIORegularMapColumn) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{3, 300}},
          {{4, 400}, {5, 500}},
          {{6, 600}},
      }),
  });
  WriterOptions writerOptions;
  writerOptions.skipConstantFlatMapInMapStreams = GetParam();
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(20, 100, false));
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 3);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({20, 30, 40}),
      makeMapVector<int32_t, int64_t>({
          {{3, 300}},
          {{4, 400}, {5, 500}},
          {{6, 600}},
      }),
  });
  velox::test::assertEqualVectors(expected, result);
}

// A lazy flat-map column read across three stripes, covering lazy loading at
// each stripe boundary.
TEST_P(FlatMapColumnReaderTest, lazyIOMultiStripe) {
  auto batch1 = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{1, 300}},
          {{2, 400}, {3, 500}},
      }),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int64_t>({40, 50, 60}),
      makeMapVector<int32_t, int64_t>({
          {{1, 600}},
          {{2, 700}, {3, 800}},
          {{1, 900}},
      }),
  });
  auto batch3 = makeRowVector({
      makeFlatVector<int64_t>({70, 80, 90}),
      makeMapVector<int32_t, int64_t>({
          {{3, 1000}},
          {{1, 1100}, {2, 1200}},
          {},
      }),
  });
  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c1", {}}};
  writerOptions.skipConstantFlatMapInMapStreams = GetParam();
  auto file = test::createNimbleFile(
      *rootPool(),
      {batch1, batch2, batch3},
      writerOptions,
      /*flushAfterWrite=*/true);

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60, 70, 80, 90}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{1, 300}},
          {{2, 400}, {3, 500}},
          {{1, 600}},
          {{2, 700}, {3, 800}},
          {{1, 900}},
          {{3, 1000}},
          {{1, 1100}, {2, 1200}},
          {},
      }),
  });

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*expected->type());
  // Pass-all filter keeps c0 eager (filter column) while the flatmap c1 stays
  // lazy; all 9 rows survive so validate() compares the full cross-stripe read.
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(0, 1000, false));
  auto readers = makeReaders(
      expected,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  // Batch size 2 (< 3 rows/stripe) forces next() calls that cross stripe
  // boundaries while c1 is materialized via the lazy hook.
  validate(*expected, *readers.rowReader, 2);
}

INSTANTIATE_TEST_SUITE_P(
    FlatMapColumnReaderTests,
    FlatMapColumnReaderTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return fmt::format("skipConstantInMap_{}", info.param);
    });

} // namespace
} // namespace facebook::nimble
