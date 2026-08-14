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
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

// Tests for TimestampColumnReader (dual-stream: micros + nanos).
class TimestampColumnReaderTest : public ::testing::Test,
                                  public velox::test::VectorTestBase {
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
      const std::shared_ptr<common::ScanSpec>& scanSpec) {
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
    readers.rowReader = readers.reader->createRowReader(rowOptions);
    return readers;
  }

  Readers makeReaders(
      const RowVectorPtr& input,
      const std::shared_ptr<common::ScanSpec>& scanSpec) {
    return makeReaders(
        input, test::createNimbleFile(*rootPool(), input), scanSpec);
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

  const std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  const std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
};

// Write and read timestamps with no filter.
TEST_F(TimestampColumnReaderTest, basicTimestamp) {
  auto input = makeRowVector({
      makeFlatVector<Timestamp>(
          100, [](auto i) { return Timestamp(1000 + i, i * 1000); }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  validate(*input, *readers.rowReader, 23);
}

// Timestamps with nulls — verify micros/nanos alignment.
TEST_F(TimestampColumnReaderTest, timestampWithNulls) {
  auto input = makeRowVector({
      makeFlatVector<Timestamp>(
          100,
          [](auto i) { return Timestamp(2000 + i * 10, i * 500); },
          nullEvery(7)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  validate(*input, *readers.rowReader, 13);
}

// All-null timestamp column.
TEST_F(TimestampColumnReaderTest, timestampAllNulls) {
  auto input = makeRowVector({
      makeConstant<Timestamp>(std::nullopt, 50),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(50, result), 50);
  auto* row = result->asUnchecked<RowVector>();
  for (int i = 0; i < 50; ++i) {
    ASSERT_TRUE(row->childAt(0)->isNullAt(i));
  }
}

// IsNull filter on timestamp column.
TEST_F(TimestampColumnReaderTest, timestampFilterIsNull) {
  auto input = makeRowVector({
      makeFlatVector<Timestamp>(
          100, [](auto i) { return Timestamp(1000 + i, 0); }, nullEvery(5)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNull>());
  auto readers = makeReaders(input, scanSpec);
  validateWithFilter(
      *input, *readers.rowReader, 17, [](auto i) { return i % 5 == 0; });
}

// IsNotNull filter on timestamp column.
TEST_F(TimestampColumnReaderTest, timestampFilterIsNotNull) {
  auto input = makeRowVector({
      makeFlatVector<Timestamp>(
          100, [](auto i) { return Timestamp(1000 + i, 0); }, nullEvery(5)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNotNull>());
  auto readers = makeReaders(input, scanSpec);
  validateWithFilter(
      *input, *readers.rowReader, 17, [](auto i) { return i % 5 != 0; });
}

// Read across multiple next() calls with small batch sizes.
TEST_F(TimestampColumnReaderTest, timestampMultiBatch) {
  auto input = makeRowVector({
      makeFlatVector<Timestamp>(
          100, [](auto i) { return Timestamp(i * 100, i * 999); }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  validate(*input, *readers.rowReader, 7);
}

// Skip + read to verify dual-stream synchronization.
TEST_F(TimestampColumnReaderTest, timestampSkipAndRead) {
  auto ts = makeFlatVector<Timestamp>(
      100, [](auto i) { return Timestamp(3000 + i, i * 123); }, nullEvery(11));
  // Add a filter column to force skipping of some rows.
  auto filterCol = makeFlatVector<int64_t>(100, folly::identity);
  auto input = makeRowVector({ts, filterCol});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  // Only keep rows where c1 >= 30 (skips the first 30 rows).
  scanSpec->childByName("c1")->setFilter(
      std::make_unique<common::BigintRange>(30, 99, false));
  auto readers = makeReaders(input, scanSpec);
  validateWithFilter(
      *input, *readers.rowReader, 11, [](auto i) { return i >= 30; });
}

// Verify nanosecond precision beyond microseconds is preserved.
TEST_F(TimestampColumnReaderTest, subMicrosecondPrecision) {
  // Create timestamps with sub-microsecond nanos.
  // Nimble stores timestamps as micros + sub-micros-nanos.
  // Velox Timestamp is (seconds, nanos).
  // For example, Timestamp(1, 500000123) means:
  //   1 second + 500000123 nanos = 1.500000123 seconds
  //   micros = 1500000, sub_micros_nanos = 123.
  auto input = makeRowVector({
      makeFlatVector<Timestamp>({
          Timestamp(0, 0),
          Timestamp(0, 1), // 1 nanosecond
          Timestamp(0, 999), // 999 nanoseconds
          Timestamp(0, 1000), // 1 microsecond exactly
          Timestamp(0, 1001), // 1 microsecond + 1 nano
          Timestamp(1, 500000123), // 1.500000123 seconds
          Timestamp(100, 999999999), // max nanos within a second
          Timestamp(-1, 0), // negative timestamp
          Timestamp(-1, 500000000), // negative with nanos
      }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(9, result), 9);
  auto* row = result->asUnchecked<RowVector>();
  auto child = row->childAt(0);
  DecodedVector decoded(*child);
  auto* expected = input->childAt(0)->asFlatVector<Timestamp>();
  for (int i = 0; i < 9; ++i) {
    ASSERT_EQ(decoded.valueAt<Timestamp>(i), expected->valueAt(i))
        << "Mismatch at row " << i;
  }
}

} // namespace
} // namespace facebook::nimble
