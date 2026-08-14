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

#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
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
                               public velox::test::VectorTestBase,
                               public ::testing::WithParamInterface<bool> {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
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
      bool stringDecoderZeroCopy) {
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
    rowOptions.setStringDecoderZeroCopy(stringDecoderZeroCopy);
    readers.rowReader = readers.reader->createRowReader(rowOptions);
    return readers;
  }

  Readers makeReaders(
      const RowVectorPtr& input,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      bool stringDecoderZeroCopy) {
    return makeReaders(
        input,
        test::createNimbleFile(*rootPool(), input),
        scanSpec,
        stringDecoderZeroCopy);
  }

  // Read with a given readType that may differ from the write type
  // (for schema evolution tests).
  Readers makeReadersWithEvolution(
      const RowVectorPtr& input,
      const RowTypePtr& readType,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      bool stringDecoderZeroCopy) {
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
    rowOptions.setStringDecoderZeroCopy(stringDecoderZeroCopy);
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
  void testDeltaBlockWithFilter(bool stringDecoderZeroCopy) {
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
    auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
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

TEST_P(ScalarColumnReaderTest, booleanDenseRead) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeFlatVector<bool>(100, [](auto i) { return i % 3 == 0; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 100);
}

TEST_P(ScalarColumnReaderTest, booleanWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeFlatVector<bool>(
          100, [](auto i) { return i % 2 == 0; }, nullEvery(5)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 13);
}

TEST_P(ScalarColumnReaderTest, booleanFilter) {
  const bool stringDecoderZeroCopy = GetParam();
  auto c0 = makeFlatVector<bool>(100, [](auto i) { return i % 3 == 0; });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BoolValue>(true, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(
      *input, *readers.rowReader, 23, [](auto i) { return i % 3 == 0; });
}

TEST_P(ScalarColumnReaderTest, tinyintWithFilter) {
  const bool stringDecoderZeroCopy = GetParam();
  auto c0 = makeFlatVector<int8_t>(
      100, [](auto i) { return static_cast<int8_t>(i); });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(20, 50, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(*input, *readers.rowReader, 17, [](auto i) {
    return i >= 20 && i <= 50;
  });
}

TEST_P(ScalarColumnReaderTest, tinyintToBigintEvolution) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeFlatVector<int8_t>(50, [](auto i) { return static_cast<int8_t>(i); }),
  });
  auto readType = ROW({"c0"}, {BIGINT()});
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(50, [](auto i) { return i; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  auto readers = makeReadersWithEvolution(
      input, readType, scanSpec, stringDecoderZeroCopy);
  validate(*expected, *readers.rowReader, 50);
}

// ----- IntegerColumnReader tests -----

TEST_P(ScalarColumnReaderTest, smallintRead) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeFlatVector<int16_t>(
          100, [](auto i) { return static_cast<int16_t>(i * 3); }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 23);
}

TEST_P(ScalarColumnReaderTest, integerWithFilter) {
  const bool stringDecoderZeroCopy = GetParam();
  auto c0 = makeFlatVector<int32_t>(200, [](auto i) { return i * 7; });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(100, 500, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(*input, *readers.rowReader, 31, [](auto i) {
    return i * 7 >= 100 && i * 7 <= 500;
  });
}

TEST_P(ScalarColumnReaderTest, integerDeltaBlockWithFilter) {
  const bool stringDecoderZeroCopy = GetParam();
  {
    SCOPED_TRACE("int8_t");
    testDeltaBlockWithFilter<int8_t>(stringDecoderZeroCopy);
  }
  {
    SCOPED_TRACE("uint8_t");
    testDeltaBlockWithFilter<uint8_t>(stringDecoderZeroCopy);
  }
  {
    SCOPED_TRACE("int16_t");
    testDeltaBlockWithFilter<int16_t>(stringDecoderZeroCopy);
  }
  {
    SCOPED_TRACE("uint16_t");
    testDeltaBlockWithFilter<uint16_t>(stringDecoderZeroCopy);
  }
  {
    SCOPED_TRACE("int32_t");
    testDeltaBlockWithFilter<int32_t>(stringDecoderZeroCopy);
  }
  {
    SCOPED_TRACE("uint32_t");
    testDeltaBlockWithFilter<uint32_t>(stringDecoderZeroCopy);
  }
  {
    SCOPED_TRACE("int64_t");
    testDeltaBlockWithFilter<int64_t>(stringDecoderZeroCopy);
  }
  {
    SCOPED_TRACE("uint64_t");
    testDeltaBlockWithFilter<uint64_t>(stringDecoderZeroCopy);
  }
}

TEST_P(ScalarColumnReaderTest, bigintWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeFlatVector<int64_t>(
          200, [](auto i) { return i * 100; }, nullEvery(7)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 19);
}

TEST_P(ScalarColumnReaderTest, integerAllNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeConstant<int32_t>(std::nullopt, 100),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(100, result), 100);
  auto* row = result->asUnchecked<RowVector>();
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(row->childAt(0)->isNullAt(i));
  }
}

TEST_P(ScalarColumnReaderTest, integerSkipAndRead) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeFlatVector<int32_t>(100, folly::identity),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  // Read in small batches to exercise skip between batches.
  validate(*input, *readers.rowReader, 7);
}

TEST_P(ScalarColumnReaderTest, smallintToBigintEvolution) {
  const bool stringDecoderZeroCopy = GetParam();
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
  auto readers = makeReadersWithEvolution(
      input, readType, scanSpec, stringDecoderZeroCopy);
  validate(*expected, *readers.rowReader, 50);
}

TEST_P(ScalarColumnReaderTest, integerToBigintEvolution) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeFlatVector<int32_t>(50, [](auto i) { return i * 1000; }),
  });
  auto readType = ROW({"c0"}, {BIGINT()});
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(50, [](auto i) { return i * 1000; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  auto readers = makeReadersWithEvolution(
      input, readType, scanSpec, stringDecoderZeroCopy);
  validate(*expected, *readers.rowReader, 50);
}

TEST_P(ScalarColumnReaderTest, integerToBigintEvolutionNonZeroBaseline) {
  const bool stringDecoderZeroCopy = GetParam();
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
  auto readers = makeReadersWithEvolution(
      input, readType, scanSpec, stringDecoderZeroCopy);
  validate(*expected, *readers.rowReader, 50);
}

// ----- FloatingPointColumnReader tests -----

TEST_P(ScalarColumnReaderTest, floatRead) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeFlatVector<float>(
          100, [](auto i) { return static_cast<float>(i) * 0.5f; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 23);
}

TEST_P(ScalarColumnReaderTest, doubleWithFilter) {
  const bool stringDecoderZeroCopy = GetParam();
  auto c0 = makeFlatVector<double>(100, [](auto i) { return sin(i); });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::FloatingPointRange<double>>(
          -INFINITY, true, false, 0.0, false, false, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(
      *input, *readers.rowReader, 17, [](auto i) { return sin(i) <= 0.0; });
}

TEST_P(ScalarColumnReaderTest, floatWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  auto c0 = makeFlatVector<float>(
      100, [](auto i) { return static_cast<float>(i) * 1.5f; }, nullEvery(3));
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNotNull>());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(
      *input, *readers.rowReader, 19, [](auto i) { return i % 3 != 0; });
}

TEST_P(ScalarColumnReaderTest, floatToDoubleEvolution) {
  const bool stringDecoderZeroCopy = GetParam();
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
  auto readers = makeReadersWithEvolution(
      input, readType, scanSpec, stringDecoderZeroCopy);
  validate(*expected, *readers.rowReader, 50);
}

TEST_P(ScalarColumnReaderTest, specialFloatValues) {
  const bool stringDecoderZeroCopy = GetParam();
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
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
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

INSTANTIATE_TEST_CASE_P(
    ScalarColumnReaderTestSuite,
    ScalarColumnReaderTest,
    testing::Values(false, true));

} // namespace
} // namespace facebook::nimble
