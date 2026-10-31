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

#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/selective/ChunkedDecoder.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>
#include <cmath>

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

WriterOptions makeForcedEncodingWriterOptions(EncodingType encodingType) {
  WriterOptions writerOptions;
  writerOptions.encodingSelectionPolicyCreator =
      [encodingType](DataType dataType) {
        ManualEncodingSelectionPolicyFactory factory{
            {{encodingType, 1.0}}, std::nullopt};
        return factory.createPolicy(dataType);
      };
  return writerOptions;
}

// Tests for ByteColumnReader, IntegerColumnReader, and
// FloatingPointColumnReader.
class ScalarColumnReaderTest : public ::testing::Test,
                               public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
    common::testutil::TestValue::enable();
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

  Readers makeReaders(
      const RowVectorPtr& expected,
      const std::string& file,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      bool scanFastPath = false) {
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
    rowOptions.setNimbleScanFastPath(scanFastPath);
    readers.rowReader = readers.reader->createRowReader(rowOptions);
    return readers;
  }

  Readers makeReaders(
      const RowVectorPtr& input,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      bool scanFastPath = false) {
    return makeReaders(
        input,
        test::createNimbleFile(*rootPool(), input),
        scanSpec,
        scanFastPath);
  }

  // Writes 'input' as one chunk per 'rowsPerChunk' rows in a single stripe.
  std::string createMultiChunkFile(
      const RowVectorPtr& input,
      vector_size_t rowsPerChunk) {
    std::vector<VectorPtr> chunks;
    for (vector_size_t offset = 0; offset < input->size();
         offset += rowsPerChunk) {
      chunks.push_back(
          input->slice(offset, std::min(rowsPerChunk, input->size() - offset)));
    }
    WriterOptions options;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    return test::createNimbleFile(
        *rootPool(), chunks, std::move(options), /*flushAfterWrite=*/false);
  }

  // Read with a given readType that may differ from the write type
  // (for schema evolution tests).
  Readers makeReadersWithEvolution(
      const RowVectorPtr& input,
      const RowTypePtr& readType,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      bool scanFastPath = false) {
    auto file = test::createNimbleFile(*rootPool(), input);
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
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(readType);
    rowOptions.setNimbleScanFastPath(scanFastPath);
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

  void validateWithFilter(
      const RowVector& input,
      dwio::common::RowReader& rowReader,
      int batchSize,
      const std::function<bool(int)>& filter) {
    auto result = BaseVector::create(asRowType(input.type()), 0, pool());
    int numScanned = 0;
    int i = 0;
    while (numScanned < input.size()) {
      numScanned += rowReader.next(batchSize, result);
      result->validate();
      for (int j = 0; j < result->size(); ++j) {
        for (;;) {
          ASSERT_LT(i, input.size());
          if (filter(i)) {
            break;
          }
          ++i;
        }
        ASSERT_TRUE(result->equalValueAt(&input, j, i))
            << "Mismatch at input row " << i;
        ++i;
      }
    }
    while (i < input.size()) {
      ASSERT_FALSE(filter(i));
      ++i;
    }
    ASSERT_EQ(numScanned, input.size());
    ASSERT_EQ(0, rowReader.next(1, result));
  }

  static int64_t deltaBlockFilterValue(int row) {
    return int64_t{10} + row / 8;
  }

  template <typename T>
  void testDeltaBlockWithFilter() {
    auto input = makeRowVector({
        makeFlatVector<T>(
            513,
            [](auto row) {
              return static_cast<T>(deltaBlockFilterValue(row));
            }),
    });

    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::BigintRange>(30, 60, false));

    auto file = test::createNimbleFile(
        *rootPool(),
        input,
        makeForcedEncodingWriterOptions(EncodingType::DeltaBlock));
    auto readers = makeReaders(input, file, scanSpec);
    validateWithFilter(*input, *readers.rowReader, 29, [](auto row) {
      const auto value = deltaBlockFilterValue(row);
      return value >= 30 && value <= 60;
    });
  }

  const std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  const std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
};

// ----- ByteColumnReader (BOOLEAN) tests -----

TEST_F(ScalarColumnReaderTest, booleanDenseRead) {
  auto input = makeRowVector({
      makeFlatVector<bool>(100, [](auto i) { return i % 3 == 0; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  validate(*input, *readers.rowReader, 100);
}

TEST_F(ScalarColumnReaderTest, booleanWithNulls) {
  auto input = makeRowVector({
      makeFlatVector<bool>(
          100, [](auto i) { return i % 2 == 0; }, nullEvery(5)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  validate(*input, *readers.rowReader, 13);
}

TEST_F(ScalarColumnReaderTest, booleanFilter) {
  auto c0 = makeFlatVector<bool>(100, [](auto i) { return i % 3 == 0; });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BoolValue>(true, false));
  auto readers = makeReaders(input, scanSpec);
  validateWithFilter(
      *input, *readers.rowReader, 23, [](auto i) { return i % 3 == 0; });
}

TEST_F(ScalarColumnReaderTest, tinyintWithFilter) {
  auto c0 = makeFlatVector<int8_t>(
      100, [](auto i) { return static_cast<int8_t>(i); });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(20, 50, false));
  auto readers = makeReaders(input, scanSpec);
  validateWithFilter(*input, *readers.rowReader, 17, [](auto i) {
    return i >= 20 && i <= 50;
  });
}

TEST_F(ScalarColumnReaderTest, tinyintToBigintEvolution) {
  auto input = makeRowVector({
      makeFlatVector<int8_t>(50, [](auto i) { return static_cast<int8_t>(i); }),
  });
  auto readType = ROW({"c0"}, {BIGINT()});
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(50, [](auto i) { return i; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  auto readers = makeReadersWithEvolution(input, readType, scanSpec);
  validate(*expected, *readers.rowReader, 50);
}

// ----- IntegerColumnReader tests -----

TEST_F(ScalarColumnReaderTest, smallintRead) {
  auto input = makeRowVector({
      makeFlatVector<int16_t>(
          100, [](auto i) { return static_cast<int16_t>(i * 3); }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  validate(*input, *readers.rowReader, 23);
}

TEST_F(ScalarColumnReaderTest, integerWithFilter) {
  auto c0 = makeFlatVector<int32_t>(200, [](auto i) { return i * 7; });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(100, 500, false));
  auto readers = makeReaders(input, scanSpec);
  validateWithFilter(*input, *readers.rowReader, 31, [](auto i) {
    return i * 7 >= 100 && i * 7 <= 500;
  });
}

TEST_F(ScalarColumnReaderTest, integerDeltaBlockWithFilter) {
  {
    SCOPED_TRACE("int8_t");
    testDeltaBlockWithFilter<int8_t>();
  }
  {
    SCOPED_TRACE("uint8_t");
    testDeltaBlockWithFilter<uint8_t>();
  }
  {
    SCOPED_TRACE("int16_t");
    testDeltaBlockWithFilter<int16_t>();
  }
  {
    SCOPED_TRACE("uint16_t");
    testDeltaBlockWithFilter<uint16_t>();
  }
  {
    SCOPED_TRACE("int32_t");
    testDeltaBlockWithFilter<int32_t>();
  }
  {
    SCOPED_TRACE("uint32_t");
    testDeltaBlockWithFilter<uint32_t>();
  }
  {
    SCOPED_TRACE("int64_t");
    testDeltaBlockWithFilter<int64_t>();
  }
  {
    SCOPED_TRACE("uint64_t");
    testDeltaBlockWithFilter<uint64_t>();
  }
}

TEST_F(ScalarColumnReaderTest, integerEliasFanoWithFilter) {
  auto input = makeRowVector({makeFlatVector<int64_t>(
      513, [](auto row) { return int64_t{-300} + row / 3; })});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(-200, -100, false));

  auto file = test::createNimbleFile(
      *rootPool(),
      input,
      makeForcedEncodingWriterOptions(EncodingType::EliasFano));
  auto readers = makeReaders(input, file, scanSpec);
  validateWithFilter(*input, *readers.rowReader, 29, [](auto row) {
    const auto value = int64_t{-300} + row / 3;
    return value >= -200 && value <= -100;
  });
}

TEST_F(ScalarColumnReaderTest, bigintWithNulls) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>(
          200, [](auto i) { return i * 100; }, nullEvery(7)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  validate(*input, *readers.rowReader, 19);
}

TEST_F(
    ScalarColumnReaderTest,
    dictionaryBigintAfterNullableChunkWithNullAllowedFilter) {
  auto makeBatch = [&](int32_t offset, int32_t size, bool hasNulls) {
    return makeRowVector({makeFlatVector<int64_t>(
        size,
        [offset](auto row) { return static_cast<int64_t>((offset + row) % 4); },
        hasNulls ? nullEvery(5) : nullptr)});
  };
  auto firstBatch = makeBatch(0, 33, true);
  auto secondBatch = makeBatch(33, 33, false);
  auto thirdBatch = makeBatch(66, 34, true);
  const std::vector<VectorPtr> batches{firstBatch, secondBatch, thirdBatch};

  auto input = makeRowVector({makeFlatVector<int64_t>(
      100,
      [](auto row) { return static_cast<int64_t>(row % 4); },
      [](auto row) {
        return row < 33 ? row % 5 == 0 : row >= 66 && (row - 66) % 5 == 0;
      })});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(1, 3, true));

  auto writerOptions =
      makeForcedEncodingWriterOptions(EncodingType::Dictionary);
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        [](const StripeProgress&) { return false; },
        [](const StripeProgress&) { return true; });
  };
  const auto file = test::createNimbleFile(
      *rootPool(), batches, std::move(writerOptions), false);
  auto readers = makeReaders(input, file, scanSpec);
  validateWithFilter(*input, *readers.rowReader, 100, [](auto row) {
    const auto value = row % 4;
    const bool isNull = row < 33 ? row % 5 == 0 : row >= 66 && row % 5 == 1;
    return isNull || (value >= 1 && value <= 3);
  });
}

TEST_F(ScalarColumnReaderTest, integerAllNulls) {
  auto input = makeRowVector({
      makeConstant<int32_t>(std::nullopt, 100),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(100, result), 100);
  auto* row = result->asUnchecked<RowVector>();
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(row->childAt(0)->isNullAt(i));
  }
}

TEST_F(ScalarColumnReaderTest, integerSkipAndRead) {
  auto input = makeRowVector({
      makeFlatIdentityVector<int32_t>(100),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  // Read in small batches to exercise skip between batches.
  validate(*input, *readers.rowReader, 7);
}

TEST_F(ScalarColumnReaderTest, smallintToBigintEvolution) {
  auto input = makeRowVector({
      makeFlatVector<int16_t>(
          50, [](auto i) { return static_cast<int16_t>(i * 10); }),
  });
  auto readType = ROW({"c0"}, {BIGINT()});
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(50, [](auto i) { return i * 10; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  auto readers = makeReadersWithEvolution(input, readType, scanSpec);
  validate(*expected, *readers.rowReader, 50);
}

TEST_F(ScalarColumnReaderTest, integerToBigintEvolution) {
  auto input = makeRowVector({
      makeFlatVector<int32_t>(50, [](auto i) { return i * 1000; }),
  });
  auto readType = ROW({"c0"}, {BIGINT()});
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(50, [](auto i) { return i * 1000; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  auto readers = makeReadersWithEvolution(input, readType, scanSpec);
  validate(*expected, *readers.rowReader, 50);
}

TEST_F(ScalarColumnReaderTest, integerToBigintEvolutionNonZeroBaseline) {
  // Minimum value is well above zero so FixedBitWidth encoding picks a non-zero
  // baseline; reading int32 as int64 (the upcast fast path) must add the
  // baseline back to every value.
  auto input = makeRowVector({
      makeFlatVector<int32_t>(
          50, [](auto i) { return 1000000 + static_cast<int32_t>(i) * 1000; }),
  });
  auto readType = ROW({"c0"}, {BIGINT()});
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(
          50, [](auto i) { return int64_t{1000000} + i * 1000; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  auto readers = makeReadersWithEvolution(input, readType, scanSpec);
  validate(*expected, *readers.rowReader, 50);
}

// ----- FloatingPointColumnReader tests -----

TEST_F(ScalarColumnReaderTest, floatRead) {
  auto input = makeRowVector({
      makeFlatVector<float>(
          100, [](auto i) { return static_cast<float>(i) * 0.5f; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  validate(*input, *readers.rowReader, 23);
}

TEST_F(ScalarColumnReaderTest, doubleWithFilter) {
  auto c0 = makeFlatVector<double>(100, [](auto i) { return sin(i); });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::FloatingPointRange<double>>(
          -INFINITY, true, false, 0.0, false, false, false));
  auto readers = makeReaders(input, scanSpec);
  validateWithFilter(
      *input, *readers.rowReader, 17, [](auto i) { return sin(i) <= 0.0; });
}

TEST_F(ScalarColumnReaderTest, floatWithNulls) {
  auto c0 = makeFlatVector<float>(
      100, [](auto i) { return static_cast<float>(i) * 1.5f; }, nullEvery(3));
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNotNull>());
  auto readers = makeReaders(input, scanSpec);
  validateWithFilter(
      *input, *readers.rowReader, 19, [](auto i) { return i % 3 != 0; });
}

TEST_F(ScalarColumnReaderTest, floatToDoubleEvolution) {
  auto input = makeRowVector({
      makeFlatVector<float>(
          50, [](auto i) { return static_cast<float>(i) * 0.1f; }),
  });
  auto readType = ROW({"c0"}, {DOUBLE()});
  auto expected = makeRowVector({
      makeFlatVector<double>(
          50,
          [](auto i) {
            return static_cast<double>(static_cast<float>(i) * 0.1f);
          }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  auto readers = makeReadersWithEvolution(input, readType, scanSpec);
  validate(*expected, *readers.rowReader, 50);
}

TEST_F(ScalarColumnReaderTest, specialFloatValues) {
  auto input = makeRowVector({
      makeFlatVector<double>({
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity(),
          0.0,
          -0.0,
          std::numeric_limits<double>::lowest(),
          std::numeric_limits<double>::max(),
          std::numeric_limits<double>::denorm_min(),
      }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(8, result), 8);
  auto* row = result->asUnchecked<RowVector>();
  auto child = row->childAt(0);
  DecodedVector decoded(*child);
  // NaN
  ASSERT_TRUE(std::isnan(decoded.valueAt<double>(0)));
  // +Inf
  ASSERT_TRUE(std::isinf(decoded.valueAt<double>(1)));
  ASSERT_GT(decoded.valueAt<double>(1), 0);
  // -Inf
  ASSERT_TRUE(std::isinf(decoded.valueAt<double>(2)));
  ASSERT_LT(decoded.valueAt<double>(2), 0);
  // 0.0
  ASSERT_EQ(decoded.valueAt<double>(3), 0.0);
  // -0.0
  ASSERT_EQ(decoded.valueAt<double>(4), -0.0);
  ASSERT_TRUE(std::signbit(decoded.valueAt<double>(4)));
  // lowest
  ASSERT_EQ(decoded.valueAt<double>(5), std::numeric_limits<double>::lowest());
  // max
  ASSERT_EQ(decoded.valueAt<double>(6), std::numeric_limits<double>::max());
  // denorm_min
  ASSERT_EQ(
      decoded.valueAt<double>(7), std::numeric_limits<double>::denorm_min());
}

// ----- Nulls-only reads -----

// IS NULL, and IS NOT NULL on a column that is not projected, need only the
// nulls. The chunks have some nulls, no nulls, some nulls, and only nulls.
TEST_F(ScalarColumnReaderTest, nullsOnlyFilter) {
  constexpr vector_size_t kSize = 400;
  const auto isNullAt = [](vector_size_t row) {
    return (row < 100 && row % 3 == 0) ||
        (row >= 200 && row < 300 && row % 5 == 0) || row >= 300;
  };
  const std::vector<VectorPtr> columns = {
      makeFlatVector<bool>(
          kSize, [](auto row) { return row % 2 == 0; }, isNullAt),
      makeFlatVector<int8_t>(
          kSize, [](auto row) { return static_cast<int8_t>(row); }, isNullAt),
      makeFlatVector<int16_t>(
          kSize, [](auto row) { return static_cast<int16_t>(row); }, isNullAt),
      makeFlatVector<int32_t>(
          kSize, [](auto row) { return row; }, isNullAt),
      makeFlatVector<int64_t>(
          kSize, [](auto row) { return int64_t{row} << 32; }, isNullAt),
      makeFlatVector<float>(
          kSize, [](auto row) { return row * 0.5f; }, isNullAt),
      makeFlatVector<double>(
          kSize, [](auto row) { return row * 0.25; }, isNullAt),
  };
  for (const auto& column : columns) {
    auto input =
        makeRowVector({column, makeFlatIdentityVector<int64_t>(kSize)});
    const auto file = createMultiChunkFile(input, /*rowsPerChunk=*/100);
    for (const bool isNull : {true, false}) {
      for (const bool projectOut : {true, false}) {
        SCOPED_TRACE(
            column->type()->toString() +
            (isNull ? " IS NULL" : " IS NOT NULL") +
            (projectOut ? "" : ", not projected"));
        auto scanSpec = std::make_shared<common::ScanSpec>("root");
        scanSpec->addAllChildFields(*input->type());
        auto* c0 = scanSpec->childByName("c0");
        if (isNull) {
          c0->setFilter(std::make_unique<common::IsNull>());
        } else {
          c0->setFilter(std::make_unique<common::IsNotNull>());
        }
        auto expected = input;
        if (!projectOut) {
          c0->setProjectOut(false);
          c0->setChannel(common::ScanSpec::kNoChannel);
          scanSpec->childByName("c1")->setChannel(0);
          expected = makeRowVector({"c1"}, {input->childAt(1)});
        }
        auto readers = makeReaders(
            input,
            file,
            scanSpec,
            /*scanFastPath=*/true);
        validateWithFilter(*expected, *readers.rowReader, 37, [&](auto row) {
          return isNullAt(row) == isNull;
        });
      }
    }
  }
}

// The nulls of a struct member include the rows where the struct is null.
TEST_F(ScalarColumnReaderTest, nullsOnlyFilterInStruct) {
  auto input = makeRowVector({makeRowVector(
      {makeFlatVector<int32_t>(
          300, [](auto row) { return row; }, nullEvery(2))},
      nullEvery(3))});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->childByName("c0")->setFilter(
      std::make_unique<common::IsNull>());
  auto readers = makeReaders(
      input,
      createMultiChunkFile(input, /*rowsPerChunk=*/100),
      scanSpec,
      /*scanFastPath=*/true);
  validateWithFilter(*input, *readers.rowReader, 37, [](auto row) {
    return row % 2 == 0 || row % 3 == 0;
  });
}

// A nulls-only read keeps the values at the file's width, and getValues widens
// the null rows to the requested type.
TEST_F(ScalarColumnReaderTest, nullsOnlyFilterWithEvolution) {
  constexpr vector_size_t kSize = 300;
  const std::vector<std::pair<VectorPtr, TypePtr>> columnsAndReadTypes = {
      {makeFlatVector<int16_t>(
           kSize,
           [](auto row) { return static_cast<int16_t>(row); },
           nullEvery(3)),
       BIGINT()},
      {makeFlatVector<int32_t>(
           kSize, [](auto row) { return row; }, nullEvery(3)),
       BIGINT()},
      {makeFlatVector<float>(
           kSize, [](auto row) { return row * 0.5f; }, nullEvery(3)),
       DOUBLE()},
  };
  for (const auto& [column, readType] : columnsAndReadTypes) {
    SCOPED_TRACE(column->type()->toString() + " as " + readType->toString());
    auto input =
        makeRowVector({column, makeFlatIdentityVector<int64_t>(kSize)});
    auto rowReadType = ROW({{"c0", readType}, {"c1", BIGINT()}});
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowReadType);
    scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNull>());
    auto readers = makeReadersWithEvolution(
        input,
        rowReadType,
        scanSpec,
        /*scanFastPath=*/true);
    auto expected = makeRowVector(
        {"c0", "c1"},
        {BaseVector::createNullConstant(readType, kSize, pool()),
         input->childAt(1)});
    validateWithFilter(*expected, *readers.rowReader, 37, [](auto row) {
      return row % 3 == 0;
    });
  }
}

// c0 drops row 3 and the last 3 rows of each batch. c1 then reads a row set
// with a gap, and skips the dropped tail before its next read, sometimes across
// a chunk boundary.
TEST_F(ScalarColumnReaderTest, nullsOnlyFilterSkipsRows) {
  const auto c0IsNullAt = [](vector_size_t row) {
    return row % 10 < 7 && row % 10 != 3;
  };
  auto input = makeRowVector({
      makeFlatVector<int64_t>(
          300, [](auto row) { return row; }, c0IsNullAt),
      makeFlatVector<int64_t>(
          300, [](auto row) { return row; }, nullEvery(3)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNull>());
  scanSpec->childByName("c1")->setFilter(std::make_unique<common::IsNull>());

  // Filters of the same kind run in name order, so c0 runs first.
  scanSpec->disableStatsBasedFilterReorder();
  auto readers = makeReaders(
      input,
      createMultiChunkFile(input, /*rowsPerChunk=*/98),
      scanSpec,
      /*scanFastPath=*/true);
  validateWithFilter(*input, *readers.rowReader, 10, [&](auto row) {
    return c0IsNullAt(row) && row % 3 == 0;
  });
}

// A dynamic filter can replace a column's filter between batches, so one chunk
// is read partly through the nulls-only path and partly through the value
// path. Each read must start where the other left the value stream.
TEST_F(ScalarColumnReaderTest, nullsOnlyFilterSwitchesMidStripe) {
  constexpr vector_size_t kSize = 400;
  constexpr vector_size_t kBatchSize = 37;
  // The switch lands inside the third chunk, whose nulls make it a
  // NullableEncoding chunk.
  constexpr vector_size_t kSwitchRow = 6 * kBatchSize;

  // The chunks have some nulls, no nulls, some nulls, and only nulls.
  const auto isNullAt = [](vector_size_t row) {
    return (row < 100 && row % 3 == 0) ||
        (row >= 200 && row < 300 && row % 5 == 0) || row >= 300;
  };
  auto input = makeRowVector({
      makeFlatVector<int64_t>(
          kSize, [](auto row) { return row; }, isNullAt),
      makeFlatIdentityVector<int64_t>(kSize),
  });
  const auto file = createMultiChunkFile(input, /*rowsPerChunk=*/100);

  // IS NOT NULL on the unprojected c0 needs only its nulls; the BigintRange
  // filter needs its values.
  const auto makeFilter =
      [](bool nullsOnly) -> std::unique_ptr<common::Filter> {
    if (nullsOnly) {
      return std::make_unique<common::IsNotNull>();
    }
    return std::make_unique<common::BigintRange>(
        50, 250, /*nullAllowed=*/false);
  };
  const auto passes = [&](vector_size_t row, bool nullsOnly) {
    return !isNullAt(row) && (nullsOnly || (row >= 50 && row <= 250));
  };
  for (const bool nullsOnlyFirst : {true, false}) {
    SCOPED_TRACE(nullsOnlyFirst ? "nulls-only first" : "values first");
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto* c0 = scanSpec->childByName("c0");
    c0->setProjectOut(false);
    c0->setChannel(common::ScanSpec::kNoChannel);
    scanSpec->childByName("c1")->setChannel(0);
    c0->setFilter(makeFilter(nullsOnlyFirst));
    auto readers = makeReaders(input, file, scanSpec, /*scanFastPath=*/true);

    VectorPtr result = BaseVector::create(ROW("c1", BIGINT()), 0, pool());
    std::vector<int64_t> actualRows;
    vector_size_t numScanned = 0;
    while (numScanned < kSize) {
      if (numScanned == kSwitchRow) {
        // Mirrors FileDataSource::addDynamicFilter.
        c0->setFilter(makeFilter(!nullsOnlyFirst));
        scanSpec->resetCachedValues(true);
        readers.rowReader->resetFilterCaches();
      }
      numScanned += readers.rowReader->next(kBatchSize, result);
      auto* c1 = result->asUnchecked<RowVector>()
                     ->childAt(0)
                     ->loadedVector()
                     ->as<SimpleVector<int64_t>>();
      ASSERT_NE(c1, nullptr);
      for (vector_size_t i = 0; i < result->size(); ++i) {
        actualRows.push_back(c1->valueAt(i));
      }
    }

    std::vector<int64_t> expectedRows;
    for (vector_size_t row = 0; row < kSize; ++row) {
      if (passes(row, (row < kSwitchRow) == nullsOnlyFirst)) {
        expectedRows.push_back(row);
      }
    }
    EXPECT_EQ(actualRows, expectedRows);
  }
}

DEBUG_ONLY_TEST_F(ScalarColumnReaderTest, nullsOnlyFilterSkipsValueDecoding) {
  int numNullsOnlyReads{0};
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::ChunkedDecoder::nextNulls",
      std::function<void(ChunkedDecoder*)>(
          [&](ChunkedDecoder*) { ++numNullsOnlyReads; }));

  // One column per reader: ByteColumnReader, IntegerColumnReader and
  // FloatingPointColumnReader.
  const std::vector<VectorPtr> columns = {
      makeFlatVector<bool>(
          100, [](auto row) { return row % 2 == 0; }, nullEvery(3)),
      makeFlatVector<int64_t>(
          100, [](auto row) { return row; }, nullEvery(3)),
      makeFlatVector<double>(
          100, [](auto row) { return row * 0.5; }, nullEvery(3)),
  };
  for (const auto& column : columns) {
    SCOPED_TRACE(column->type()->toString());
    auto input = makeRowVector({column, makeFlatIdentityVector<int64_t>(100)});
    const auto readsNullsOnly = [&](std::unique_ptr<common::Filter> filter,
                                    bool projectOut,
                                    bool scanFastPath) {
      auto scanSpec = std::make_shared<common::ScanSpec>("root");
      scanSpec->addAllChildFields(*input->type());
      auto* c0 = scanSpec->childByName("c0");
      c0->setFilter(std::move(filter));
      auto outputType = asRowType(input->type());
      if (!projectOut) {
        c0->setProjectOut(false);
        c0->setChannel(common::ScanSpec::kNoChannel);
        scanSpec->childByName("c1")->setChannel(0);
        outputType = ROW("c1", BIGINT());
      }
      auto readers = makeReaders(input, scanSpec, scanFastPath);
      VectorPtr result = BaseVector::create(outputType, 0, pool());
      numNullsOnlyReads = 0;
      readers.rowReader->next(input->size(), result);
      return numNullsOnlyReads > 0;
    };

    EXPECT_TRUE(readsNullsOnly(
        std::make_unique<common::IsNull>(),
        /*projectOut=*/true,
        /*scanFastPath=*/true));
    EXPECT_TRUE(readsNullsOnly(
        std::make_unique<common::IsNotNull>(),
        /*projectOut=*/false,
        /*scanFastPath=*/true));

    // A projected column needs its values.
    EXPECT_FALSE(readsNullsOnly(
        std::make_unique<common::IsNotNull>(),
        /*projectOut=*/true,
        /*scanFastPath=*/true));

    // Nulls-only reads stay off until scanFastPath is on.
    EXPECT_FALSE(readsNullsOnly(
        std::make_unique<common::IsNull>(),
        /*projectOut=*/true,
        /*scanFastPath=*/false));
  }
}

} // namespace
} // namespace facebook::nimble
