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

#include "velox/connectors/hive/iceberg/IcebergDeletionVectorSink.h"

#include <filesystem>

#include <folly/json.h>
#include <gtest/gtest.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/hive/iceberg/DeletionVectorReader.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive::iceberg;
using namespace facebook::velox::common::testutil;

namespace {

// Builds a minimal IcebergInsertTableHandle for use by
// IcebergDeletionVectorSink. The sink only consults locationHandle() (for
// the Puffin output directory) and writeKind(); inputColumns and partition
// spec are not exercised by the deletion-vector path.
IcebergInsertTableHandlePtr makeDeletionVectorHandle(
    const std::string& outputDirectory) {
  auto locationHandle = std::make_shared<connector::hive::LocationHandle>(
      outputDirectory,
      outputDirectory,
      connector::hive::LocationHandle::TableType::kExisting);
  return std::make_shared<const IcebergInsertTableHandle>(
      /*inputColumns=*/std::vector<IcebergColumnHandlePtr>{},
      locationHandle,
      /*tableStorageFormat=*/dwio::common::FileFormat::PARQUET,
      /*partitionSpec=*/nullptr,
      /*compressionKind=*/std::nullopt,
      /*serdeParameters=*/std::unordered_map<std::string, std::string>{},
      IcebergInsertTableHandle::WriteKind::kDeletionVector);
}

// Allocates a bit-packed buffer big enough to hold 'numBits' bits, zeroed.
BufferPtr allocateBitmap(uint64_t numBits, memory::MemoryPool* pool) {
  return AlignedBuffer::allocate<uint8_t>(bits::nbytes(numBits), pool, 0);
}

// Returns the indices of all set bits in 'bitmap[0..numBits)'.
std::vector<uint64_t> getSetBits(const BufferPtr& bitmap, uint64_t numBits) {
  const auto* raw = bitmap->as<uint8_t>();
  std::vector<uint64_t> result;
  for (uint64_t i = 0; i < numBits; ++i) {
    if (bits::isBitSet(raw, i)) {
      result.push_back(i);
    }
  }
  return result;
}

// Reads the puffin blob at 'puffinPath' between [offset, offset+length)
// through DeletionVectorReader and returns the set positions in [0, maxPos].
std::vector<uint64_t> readBackDeletedPositions(
    const std::string& puffinPath,
    uint64_t offset,
    uint64_t length,
    uint64_t recordCount,
    uint64_t maxPos,
    memory::MemoryPool* pool) {
  std::unordered_map<int32_t, std::string> lowerBounds;
  std::unordered_map<int32_t, std::string> upperBounds;
  lowerBounds[DeletionVectorReader::kDvOffsetFieldId] = std::to_string(offset);
  upperBounds[DeletionVectorReader::kDvLengthFieldId] = std::to_string(length);

  IcebergDeleteFile dvFile(
      FileContent::kDeletionVector,
      puffinPath,
      dwio::common::FileFormat::DWRF,
      recordCount,
      /*fileSizeInBytes=*/std::filesystem::file_size(puffinPath),
      /*equalityFieldIds=*/{},
      lowerBounds,
      upperBounds);

  DeletionVectorReader reader(dvFile, 0, pool);
  const uint64_t batchSize = 1024;
  const uint64_t totalRows = maxPos + batchSize;

  std::vector<uint64_t> deleted;
  for (uint64_t base = 0; base < totalRows; base += batchSize) {
    auto bitmap = allocateBitmap(batchSize, pool);
    reader.readDeletePositions(base, batchSize, bitmap);
    for (auto offsetInBatch : getSetBits(bitmap, batchSize)) {
      deleted.push_back(base + offsetInBatch);
    }
  }
  return deleted;
}

} // namespace

class IcebergDeletionVectorSinkTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    filesystems::registerLocalFileSystem();
    pool_ =
        memory::memoryManager()->addLeafPool("IcebergDeletionVectorSinkTest");
    vectorMaker_ = std::make_unique<test::VectorMaker>(pool_.get());
  }

  RowVectorPtr makePositionDeleteRows(
      const std::vector<std::string>& filePaths,
      const std::vector<int64_t>& positions) {
    VELOX_CHECK_EQ(filePaths.size(), positions.size());
    auto filePathVector = vectorMaker_->flatVector<StringView>(
        filePaths.size(),
        [&](vector_size_t i) { return StringView(filePaths[i]); });
    auto positionVector = vectorMaker_->flatVector<int64_t>(positions);
    return vectorMaker_->rowVector(
        {"file_path", "pos"}, {filePathVector, positionVector});
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<test::VectorMaker> vectorMaker_;
};

TEST_F(IcebergDeletionVectorSinkTest, singleFileSinglePosition) {
  auto tempDir = TempDirectoryPath::create();
  auto handle = makeDeletionVectorHandle(tempDir->getPath());

  IcebergDeletionVectorSink sink(
      ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()}),
      handle,
      /*connectorQueryCtx=*/nullptr,
      connector::CommitStrategy::kNoCommit);

  // Place the synthetic data file under tempDir so the puffin writer can
  // create its output next to it.
  const std::string dataFile = tempDir->getPath() + "/data-file.parquet";
  sink.appendData(makePositionDeleteRows({dataFile}, {42}));
  EXPECT_TRUE(sink.finish());
  auto commitMessages = sink.close();

  ASSERT_EQ(commitMessages.size(), 1);
  auto parsed = folly::parseJson(commitMessages[0]);
  EXPECT_EQ(parsed["content"].asString(), "POSITION_DELETES");
  EXPECT_EQ(parsed["fileFormat"].asString(), "PUFFIN");
  EXPECT_EQ(parsed["referencedDataFile"].asString(), dataFile);
  EXPECT_EQ(parsed["metrics"]["recordCount"].asInt(), 1);
  ASSERT_TRUE(parsed.find("path") != parsed.items().end());
  ASSERT_TRUE(parsed.find("contentOffset") != parsed.items().end());
  ASSERT_TRUE(parsed.find("contentSizeInBytes") != parsed.items().end());

  // Verify the emitted Puffin file is readable by DeletionVectorReader and
  // round-trips the position we wrote.
  const auto puffinPath = parsed["path"].asString();
  const auto offset = static_cast<uint64_t>(parsed["contentOffset"].asInt());
  const auto length =
      static_cast<uint64_t>(parsed["contentSizeInBytes"].asInt());
  const auto recordCount =
      static_cast<uint64_t>(parsed["metrics"]["recordCount"].asInt());
  auto deleted = readBackDeletedPositions(
      puffinPath, offset, length, recordCount, /*maxPos=*/42, pool_.get());
  EXPECT_EQ(deleted, std::vector<uint64_t>{42});
}

TEST_F(IcebergDeletionVectorSinkTest, multipleFilesDemultiplexed) {
  auto tempDir = TempDirectoryPath::create();
  auto handle = makeDeletionVectorHandle(tempDir->getPath());

  IcebergDeletionVectorSink sink(
      ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()}),
      handle,
      nullptr,
      connector::CommitStrategy::kNoCommit);

  const std::string fileA = tempDir->getPath() + "/A.parquet";
  const std::string fileB = tempDir->getPath() + "/B.parquet";
  sink.appendData(makePositionDeleteRows(
      {fileA, fileB, fileA, fileB, fileA}, {1, 2, 3, 5, 7}));
  EXPECT_TRUE(sink.finish());
  auto commitMessages = sink.close();

  // Two referenced data files -> two Puffin files -> two commit messages.
  ASSERT_EQ(commitMessages.size(), 2);

  // Insertion order is preserved: fileA appears first in the input page so
  // its commit message comes first.
  auto first = folly::parseJson(commitMessages[0]);
  auto second = folly::parseJson(commitMessages[1]);
  EXPECT_EQ(first["referencedDataFile"].asString(), fileA);
  EXPECT_EQ(second["referencedDataFile"].asString(), fileB);
  EXPECT_EQ(first["metrics"]["recordCount"].asInt(), 3);
  EXPECT_EQ(second["metrics"]["recordCount"].asInt(), 2);

  // Round-trip each Puffin to confirm the right positions landed under the
  // right file.
  auto fileAPositions = readBackDeletedPositions(
      first["path"].asString(),
      static_cast<uint64_t>(first["contentOffset"].asInt()),
      static_cast<uint64_t>(first["contentSizeInBytes"].asInt()),
      static_cast<uint64_t>(first["metrics"]["recordCount"].asInt()),
      /*maxPos=*/7,
      pool_.get());
  auto fileBPositions = readBackDeletedPositions(
      second["path"].asString(),
      static_cast<uint64_t>(second["contentOffset"].asInt()),
      static_cast<uint64_t>(second["contentSizeInBytes"].asInt()),
      static_cast<uint64_t>(second["metrics"]["recordCount"].asInt()),
      /*maxPos=*/7,
      pool_.get());

  EXPECT_EQ(fileAPositions, (std::vector<uint64_t>{1, 3, 7}));
  EXPECT_EQ(fileBPositions, (std::vector<uint64_t>{2, 5}));
}

TEST_F(IcebergDeletionVectorSinkTest, emptyInputProducesNoCommitMessages) {
  auto tempDir = TempDirectoryPath::create();
  auto handle = makeDeletionVectorHandle(tempDir->getPath());

  IcebergDeletionVectorSink sink(
      ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()}),
      handle,
      nullptr,
      connector::CommitStrategy::kNoCommit);

  // No appendData call -> no perFile state -> finish() must still be safe.
  EXPECT_TRUE(sink.finish());
  EXPECT_TRUE(sink.close().empty());
  auto stats = sink.stats();
  EXPECT_EQ(stats.numWrittenFiles, 0u);
  EXPECT_EQ(stats.numWrittenBytes, 0u);
}

TEST_F(IcebergDeletionVectorSinkTest, abortBeforeFinishDropsState) {
  auto tempDir = TempDirectoryPath::create();
  auto handle = makeDeletionVectorHandle(tempDir->getPath());

  IcebergDeletionVectorSink sink(
      ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()}),
      handle,
      nullptr,
      connector::CommitStrategy::kNoCommit);

  sink.appendData(makePositionDeleteRows({"/path/to/A.parquet"}, {100}));
  sink.abort();

  // After abort, no Puffin files should be flushed and close() returns an
  // empty commit-message list. We do not check the temp directory for
  // file presence because the sink never reached its writePuffinFile call.
  EXPECT_TRUE(sink.close().empty());
}

TEST_F(IcebergDeletionVectorSinkTest, rejectsWrongChannelCount) {
  auto tempDir = TempDirectoryPath::create();
  auto handle = makeDeletionVectorHandle(tempDir->getPath());

  IcebergDeletionVectorSink sink(
      ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()}),
      handle,
      nullptr,
      connector::CommitStrategy::kNoCommit);

  // Construct a 1-column page; the sink expects exactly 2 channels.
  auto badPage = vectorMaker_->rowVector(
      {"pos"}, {vectorMaker_->flatVector<int64_t>({1, 2, 3})});
  VELOX_ASSERT_USER_THROW(
      sink.appendData(badPage),
      "IcebergDeletionVectorSink expects 2-column input pages");
}
