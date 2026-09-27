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

// Tests for NullColumnReader: exercises schema evolution where a column is
// missing from the file but present in the requested schema.
class NullColumnReaderTest : public ::testing::Test,
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

  // Write `input` to a Nimble file, but read back as `readType` (which may
  // have extra columns not in the file — triggering NullColumnReader).
  // The scanSpec must be set up for `readType` before calling this.
  Readers makeReaders(
      const RowVectorPtr& input,
      const RowTypePtr& readType,
      const std::shared_ptr<common::ScanSpec>& scanSpec) {
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
    dwio::common::typeutils::checkTypeCompatibility(
        *readers.reader->rowType(), *readType);
    // For root-level schema evolution: mark fields present in readType but
    // missing from the file as null constants in the ScanSpec. The root
    // struct's isChildMissing() does not detect missing fields (isRoot_ is
    // true), so we must mark them explicitly.
    const auto& fileRowType = readers.reader->rowType();
    for (auto i = 0; i < readType->size(); ++i) {
      const auto& fieldName = readType->nameOf(i);
      if (!fileRowType->containsChild(fieldName)) {
        auto* childSpec = scanSpec->childByName(fieldName);
        childSpec->setConstantValue(
            BaseVector::createNullConstant(readType->childAt(i), 1, pool()));
      }
    }
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(readType);
    readers.rowReader = readers.reader->createRowReader(rowOptions);
    return readers;
  }

  const std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  const std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
};

// Write ROW({c0: BIGINT}), read as ROW({c0: BIGINT, c1: BIGINT}).
// c1 should be all nulls.
TEST_F(NullColumnReaderTest, missingScalarColumn) {
  constexpr int kSize = 100;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  auto readType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  auto readers = makeReaders(input, readType, scanSpec);
  auto result = BaseVector::create(readType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(kSize, result), kSize);
  auto* row = result->asUnchecked<RowVector>();
  ASSERT_EQ(row->size(), kSize);
  // c0 should have real values.
  for (int i = 0; i < kSize; ++i) {
    ASSERT_FALSE(row->childAt(0)->isNullAt(i));
  }
  // c1 should be all nulls.
  for (int i = 0; i < kSize; ++i) {
    ASSERT_TRUE(row->childAt(1)->isNullAt(i));
  }
}

// Write ROW({c0: INT}), read as ROW({c0: INT, c1: ARRAY(BIGINT)}).
// c1 should be all null arrays.
TEST_F(NullColumnReaderTest, missingComplexColumn) {
  constexpr int kSize = 50;
  auto input = makeRowVector({
      makeFlatVector<int32_t>(kSize, folly::identity),
  });
  auto readType = ROW({"c0", "c1"}, {INTEGER(), ARRAY(BIGINT())});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  auto readers = makeReaders(input, readType, scanSpec);
  auto result = BaseVector::create(readType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(kSize, result), kSize);
  auto* row = result->asUnchecked<RowVector>();
  for (int i = 0; i < kSize; ++i) {
    ASSERT_TRUE(row->childAt(1)->isNullAt(i));
  }
}

// Write ROW({c0: INT}), read as ROW({c0: INT, c1: TIMESTAMP}).
// c1 should be all null timestamps.
TEST_F(NullColumnReaderTest, missingTimestampColumn) {
  constexpr int kSize = 50;
  auto input = makeRowVector({
      makeFlatVector<int32_t>(kSize, folly::identity),
  });
  auto readType = ROW({"c0", "c1"}, {INTEGER(), TIMESTAMP()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  auto readers = makeReaders(input, readType, scanSpec);
  auto result = BaseVector::create(readType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(kSize, result), kSize);
  auto* row = result->asUnchecked<RowVector>();
  for (int i = 0; i < kSize; ++i) {
    ASSERT_TRUE(row->childAt(1)->isNullAt(i));
  }
}

// IsNull filter on a missing column — all rows should pass.
TEST_F(NullColumnReaderTest, filterIsNullOnMissing) {
  constexpr int kSize = 50;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  auto readType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  scanSpec->childByName("c1")->setFilter(std::make_unique<common::IsNull>());
  auto readers = makeReaders(input, readType, scanSpec);
  auto result = BaseVector::create(readType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(kSize, result), kSize);
  ASSERT_EQ(result->size(), kSize);
  // Reader returns all rows; the missing column is all nulls.
  auto* row = result->asUnchecked<RowVector>();
  for (int i = 0; i < kSize; ++i) {
    ASSERT_TRUE(row->childAt(1)->isNullAt(i));
  }
}

// IsNotNull filter on a missing column.
// With runtime filtering enabled for missing constants, all rows are
// filtered out because the column materializes as all-null.
TEST_F(NullColumnReaderTest, filterIsNotNullOnMissing) {
  constexpr int kSize = 50;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  auto readType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  scanSpec->childByName("c1")->setFilter(std::make_unique<common::IsNotNull>());
  auto readers = makeReaders(input, readType, scanSpec);
  auto result = BaseVector::create(readType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(kSize, result), kSize);
  // Reader returns empty result; the missing column is all nulls.
  ASSERT_EQ(result->size(), 0);
}

// BigintRange filter on a missing column.
// Missing column materializes as null, which does not pass this filter
// (nullAllowed=false), so all rows are filtered out.
TEST_F(NullColumnReaderTest, filterValueOnMissing) {
  constexpr int kSize = 50;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  auto readType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  scanSpec->childByName("c1")->setFilter(
      std::make_unique<common::BigintRange>(0, 100, false));
  auto readers = makeReaders(input, readType, scanSpec);
  auto result = BaseVector::create(readType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(kSize, result), kSize);
  // Reader returns empty result; the missing column is all nulls.
  ASSERT_EQ(result->size(), 0);
}

// BigintRange(nullAllowed=true) on a missing column — all rows pass.
TEST_F(NullColumnReaderTest, filterValueAllowNullOnMissing) {
  constexpr int kSize = 50;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  auto readType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  scanSpec->childByName("c1")->setFilter(
      std::make_unique<common::BigintRange>(0, 100, true));
  auto readers = makeReaders(input, readType, scanSpec);
  auto result = BaseVector::create(readType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(kSize, result), kSize);
  ASSERT_EQ(result->size(), kSize);
  // Reader returns all rows; the missing column is all nulls.
  auto* row = result->asUnchecked<RowVector>();
  for (int i = 0; i < kSize; ++i) {
    ASSERT_TRUE(row->childAt(1)->isNullAt(i));
  }
}

// Read with small batch sizes to exercise skip on NullColumnReader.
TEST_F(NullColumnReaderTest, skipOnMissingColumn) {
  constexpr int kSize = 100;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  auto readType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  auto readers = makeReaders(input, readType, scanSpec);
  auto result = BaseVector::create(readType, 0, pool());
  int totalRead = 0;
  int batchSize = 7;
  while (totalRead < kSize) {
    auto numRead = readers.rowReader->next(batchSize, result);
    if (numRead == 0) {
      break;
    }
    auto* row = result->asUnchecked<RowVector>();
    for (int i = 0; i < row->size(); ++i) {
      ASSERT_TRUE(row->childAt(1)->isNullAt(i));
    }
    totalRead += numRead;
  }
  ASSERT_EQ(totalRead, kSize);
}

} // namespace
} // namespace facebook::nimble
