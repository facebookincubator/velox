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
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/DeletionVectorReader.h"
#include "velox/core/QueryCtx.h"
#include "velox/dwio/common/FileSink.h"
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

// Like makeDeletionVectorHandle but carries a map of existing deletion vectors
// (data-file path -> descriptor) so the sink seeds each new DV with the prior
// DV's positions.
IcebergInsertTableHandlePtr makeDeletionVectorHandleWithExisting(
    const std::string& outputDirectory,
    std::unordered_map<
        std::string,
        IcebergInsertTableHandle::ExistingDeletionVector>
        existingDeletionVectors) {
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
      IcebergInsertTableHandle::WriteKind::kDeletionVector,
      std::move(existingDeletionVectors));
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

  DeletionVectorReader reader(dvFile, 0, pool, /*connectorConfig=*/nullptr);
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
    // The deletion-vector sink now opens puffin output through
    // dwio::common::FileSink::create, which dispatches by URI scheme to a
    // registered factory. Register the local-filesystem factory so test
    // paths under TempDirectoryPath (plain '/tmp/...' paths) land through
    // the same dispatch the production binary uses.
    dwio::common::LocalFileSink::registerFactory();

    pool_ =
        memory::memoryManager()->addLeafPool("IcebergDeletionVectorSinkTest");
    connectorPool_ = memory::memoryManager()->addRootPool(
        "IcebergDeletionVectorSinkTestConnector");
    vectorMaker_ = std::make_unique<test::VectorMaker>(pool_.get());

    sessionProperties_ = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>(), true);
    hiveConfig_ = std::make_shared<connector::hive::HiveConfig>(
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>()));

    queryCtx_ = core::QueryCtx::create(nullptr, core::QueryConfig({}));
    connectorQueryCtx_ = std::make_unique<connector::ConnectorQueryCtx>(
        pool_.get(),
        connectorPool_.get(),
        sessionProperties_.get(),
        /*spillConfig=*/nullptr,
        common::PrefixSortConfig(),
        /*expressionEvaluator=*/nullptr,
        /*cache=*/nullptr,
        /*queryId=*/"query.IcebergDeletionVectorSinkTest",
        /*taskId=*/"task.IcebergDeletionVectorSinkTest",
        /*planNodeId=*/"planNodeId.IcebergDeletionVectorSinkTest",
        /*driverId=*/0,
        /*sessionTimezone=*/"",
        /*adjustTimestampToTimezone=*/false);
  }

  RowVectorPtr makePositionDeleteRows(
      const std::vector<std::string>& filePaths,
      const std::vector<int64_t>& positions) {
    VELOX_CHECK_EQ(filePaths.size(), positions.size());
    auto filePathVector = vectorMaker_->flatVector<StringView>(
        static_cast<vector_size_t>(filePaths.size()),
        [&](vector_size_t i) { return StringView(filePaths[i]); });
    auto positionVector = vectorMaker_->flatVector<int64_t>(positions);
    return vectorMaker_->rowVector(
        {"file_path", "pos"}, {filePathVector, positionVector});
  }

  // Builds an input page whose row-id is a 4-field
  // ROW<file_path, pos, spec_id, partition> (the shape Presto's V3
  // getMergeTargetTableRowIdColumnHandle produces), dictionary-wrapped to
  // 'selected' base rows. This mirrors the delete operator, which selects the
  // deleted rows via a dictionary over the row-id column rather than
  // materializing a compact page. 'filePathConstant' toggles whether file_path
  // is a constant vector (single-file DELETE) or a flat per-row vector.
  // When 'withLeadingIdColumn' is true, a passthrough BIGINT column is placed
  // before the row-id ROW (the real V3 DELETE layout ROW<id,
  // $row_id:ROW<...>>).
  RowVectorPtr makeRowIdStructDeletePage(
      const std::string& dataFile,
      const std::vector<int64_t>& allPositions,
      const std::vector<vector_size_t>& selected,
      bool filePathConstant,
      bool withLeadingIdColumn = false) {
    const auto baseSize = static_cast<vector_size_t>(allPositions.size());
    VectorPtr filePathVector;
    if (filePathConstant) {
      filePathVector = BaseVector::createConstant(
          VARCHAR(), variant(dataFile), baseSize, pool_.get());
    } else {
      filePathVector = vectorMaker_->flatVector<StringView>(
          baseSize, [&](vector_size_t /*i*/) { return StringView(dataFile); });
    }
    auto positionVector = vectorMaker_->flatVector<int64_t>(allPositions);
    auto specIdVector = vectorMaker_->flatVector<int32_t>(
        baseSize, [](vector_size_t /*i*/) { return 0; });
    auto partitionVector = vectorMaker_->flatVector<StringView>(
        baseSize, [](vector_size_t /*i*/) { return StringView(""); });
    auto rowId = vectorMaker_->rowVector(
        {"_file", "_pos", "_spec_id", "partition_data"},
        {filePathVector, positionVector, specIdVector, partitionVector});

    const auto selectedSize = static_cast<vector_size_t>(selected.size());
    auto indices =
        AlignedBuffer::allocate<vector_size_t>(selectedSize, pool_.get());
    auto* rawIndices = indices->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < selectedSize; ++i) {
      rawIndices[i] = selected[i];
    }
    auto wrappedRowId = BaseVector::wrapInDictionary(
        /*nulls=*/nullptr, indices, selectedSize, rowId);

    if (!withLeadingIdColumn) {
      return vectorMaker_->rowVector({"$row_id"}, {wrappedRowId});
    }
    // Leading passthrough column aligned to the selected (top-level) rows.
    auto idVector = vectorMaker_->flatVector<int64_t>(
        selectedSize, [](vector_size_t i) { return i; });
    return vectorMaker_->rowVector({"id", "$row_id"}, {idVector, wrappedRowId});
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<memory::MemoryPool> connectorPool_;
  std::shared_ptr<config::ConfigBase> sessionProperties_;
  std::shared_ptr<connector::hive::HiveConfig> hiveConfig_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<connector::ConnectorQueryCtx> connectorQueryCtx_;
  std::unique_ptr<test::VectorMaker> vectorMaker_;
};

TEST_F(IcebergDeletionVectorSinkTest, singleFileSinglePosition) {
  auto tempDir = TempDirectoryPath::create();
  auto handle = makeDeletionVectorHandle(tempDir->getPath());

  IcebergDeletionVectorSink sink(
      ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()}),
      handle,
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      hiveConfig_);

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
  ASSERT_NE(parsed.find("path"), parsed.items().end());
  ASSERT_NE(parsed.find("contentOffset"), parsed.items().end());
  ASSERT_NE(parsed.find("contentSizeInBytes"), parsed.items().end());

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

// Writes a first DV for a data file, then runs a second sink that DELETEs a
// different position for the SAME data file while carrying the first DV as an
// existing DV. The emitted DV must be the union of the prior and new
// positions (Iceberg V3 allows at most one DV per data file).
TEST_F(IcebergDeletionVectorSinkTest, seedFromExistingDeletionVectorUnions) {
  auto tempDir = TempDirectoryPath::create();
  const std::string dataFile = tempDir->getPath() + "/data-file.parquet";

  // First mutation: DELETE positions {1, 3}.
  folly::dynamic firstCommit;
  {
    auto handle = makeDeletionVectorHandle(tempDir->getPath());
    IcebergDeletionVectorSink sink(
        ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()}),
        handle,
        connectorQueryCtx_.get(),
        connector::CommitStrategy::kNoCommit,
        hiveConfig_);
    sink.appendData(makePositionDeleteRows({dataFile, dataFile}, {1, 3}));
    EXPECT_TRUE(sink.finish());
    auto messages = sink.close();
    ASSERT_EQ(messages.size(), 1);
    firstCommit = folly::parseJson(messages[0]);
  }

  // Build the existing-DV descriptor from the first commit message.
  IcebergInsertTableHandle::ExistingDeletionVector existing;
  existing.puffinPath = firstCommit["path"].asString();
  existing.contentOffset = firstCommit["contentOffset"].asInt();
  existing.contentLength = firstCommit["contentSizeInBytes"].asInt();
  existing.recordCount = firstCommit["metrics"]["recordCount"].asInt();
  existing.fileSizeInBytes = firstCommit["fileSizeInBytes"].asInt();

  // Second mutation: DELETE position {5} for the same data file, seeded with
  // the existing DV.
  auto handle = makeDeletionVectorHandleWithExisting(
      tempDir->getPath(), {{dataFile, existing}});
  IcebergDeletionVectorSink sink(
      ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()}),
      handle,
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      hiveConfig_);
  sink.appendData(makePositionDeleteRows({dataFile}, {5}));
  EXPECT_TRUE(sink.finish());
  auto messages = sink.close();
  ASSERT_EQ(messages.size(), 1);
  auto parsed = folly::parseJson(messages[0]);

  // The emitted DV holds the union {1, 3, 5}.
  EXPECT_EQ(parsed["metrics"]["recordCount"].asInt(), 3);
  auto deleted = readBackDeletedPositions(
      parsed["path"].asString(),
      static_cast<uint64_t>(parsed["contentOffset"].asInt()),
      static_cast<uint64_t>(parsed["contentSizeInBytes"].asInt()),
      static_cast<uint64_t>(parsed["metrics"]["recordCount"].asInt()),
      /*maxPos=*/5,
      pool_.get());
  EXPECT_EQ(deleted, (std::vector<uint64_t>{1, 3, 5}));
}

// Verifies the DeletionVectorReader::deletedPositions() accessor returns the
// sorted positions of a known Puffin file (used by the sink to seed a new DV).
TEST_F(
    IcebergDeletionVectorSinkTest,
    deletionVectorReaderExposesSortedPositions) {
  auto tempDir = TempDirectoryPath::create();
  const std::string dataFile = tempDir->getPath() + "/data-file.parquet";

  auto handle = makeDeletionVectorHandle(tempDir->getPath());
  IcebergDeletionVectorSink sink(
      ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()}),
      handle,
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      hiveConfig_);
  // Append out of order to confirm the accessor returns sorted positions.
  sink.appendData(
      makePositionDeleteRows({dataFile, dataFile, dataFile}, {7, 2, 4}));
  EXPECT_TRUE(sink.finish());
  auto messages = sink.close();
  ASSERT_EQ(messages.size(), 1);
  auto parsed = folly::parseJson(messages[0]);

  IcebergDeleteFile dvFile(
      FileContent::kDeletionVector,
      parsed["path"].asString(),
      dwio::common::FileFormat::PARQUET,
      static_cast<uint64_t>(parsed["metrics"]["recordCount"].asInt()),
      static_cast<uint64_t>(parsed["fileSizeInBytes"].asInt()),
      /*equalityFieldIds=*/{},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/0,
      parsed["contentOffset"].asInt(),
      parsed["contentSizeInBytes"].asInt(),
      dataFile);
  DeletionVectorReader reader(
      dvFile, /*splitOffset=*/0, pool_.get(), hiveConfig_->config());
  EXPECT_EQ(reader.deletedPositions(), (std::vector<int64_t>{2, 4, 7}));
}

TEST_F(IcebergDeletionVectorSinkTest, multipleFilesDemultiplexed) {
  auto tempDir = TempDirectoryPath::create();
  auto handle = makeDeletionVectorHandle(tempDir->getPath());

  IcebergDeletionVectorSink sink(
      ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()}),
      handle,
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      hiveConfig_);

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
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      hiveConfig_);

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
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      hiveConfig_);

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
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      hiveConfig_);

  // Construct a 1-column page; the sink expects exactly 2 channels.
  auto badPage = vectorMaker_->rowVector(
      {"pos"}, {vectorMaker_->flatVector<int64_t>({1, 2, 3})});
  VELOX_ASSERT_USER_THROW(
      sink.appendData(badPage),
      "IcebergDeletionVectorSink expects 2-column input pages");
}

// Regression guard for the warm-storage (ws://) write path. The puffin
// writer used to hand-roll std::ofstream which silently failed on any
// non-local URI scheme. Today the writer routes through
// dwio::common::FileSink::create, which dispatches by URI scheme. We
// exercise that dispatch by passing a 'file:' prefixed path — the
// LocalFileSink factory strips the prefix and routes the bytes through
// the same Velox FileSink machinery the production binary uses. A future
// refactor that re-hard-codes std::ofstream will break this test because
// 'file:' is not a valid POSIX filesystem path.
TEST_F(IcebergDeletionVectorSinkTest, writesThroughRegisteredFileSinkFactory) {
  auto tempDir = TempDirectoryPath::create();
  const std::string schemePrefix = "file:";
  auto handle = makeDeletionVectorHandle(schemePrefix + tempDir->getPath());

  IcebergDeletionVectorSink sink(
      ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()}),
      handle,
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      hiveConfig_);

  const std::string dataFile =
      schemePrefix + tempDir->getPath() + "/data-file.parquet";
  sink.appendData(
      makePositionDeleteRows({dataFile, dataFile, dataFile}, {7, 13, 21}));
  EXPECT_TRUE(sink.finish());
  auto commitMessages = sink.close();

  ASSERT_EQ(commitMessages.size(), 1);
  auto parsed = folly::parseJson(commitMessages[0]);
  // The puffin output path keeps the 'file:' scheme prefix because the
  // sink computes it from the data file path. Strip the prefix when we
  // read back through DeletionVectorReader, which expects a local path.
  const auto puffinPath = parsed["path"].asString();
  EXPECT_TRUE(puffinPath.starts_with(schemePrefix))
      << "puffin path lost its scheme: " << puffinPath;
  const auto localPuffinPath = puffinPath.substr(schemePrefix.size());

  const auto offset = static_cast<uint64_t>(parsed["contentOffset"].asInt());
  const auto length =
      static_cast<uint64_t>(parsed["contentSizeInBytes"].asInt());
  const auto recordCount =
      static_cast<uint64_t>(parsed["metrics"]["recordCount"].asInt());
  auto deleted = readBackDeletedPositions(
      localPuffinPath, offset, length, recordCount, /*maxPos=*/21, pool_.get());
  EXPECT_EQ(deleted, (std::vector<uint64_t>{7, 13, 21}));
}

// Reproduces the pure-DELETE (non-merge) path: the row-id arrives as a single
// 4-field ROW column, dictionary-wrapped to the deleted rows. Regression guard
// for a worker SIGSEGV where appendData read the base ROW's children ignoring
// the outer dictionary. A single-file DELETE delivers file_path as a CONSTANT
// vector inside the ROW.
TEST_F(
    IcebergDeletionVectorSinkTest,
    dictionaryWrappedRowIdStructConstantPath) {
  auto tempDir = TempDirectoryPath::create();
  auto handle = makeDeletionVectorHandle(tempDir->getPath());

  IcebergDeletionVectorSink sink(
      ROW({"$row_id"},
          {ROW(
              {"file_path", "pos", "spec_id", "partition"},
              {VARCHAR(), BIGINT(), INTEGER(), VARCHAR()})}),
      handle,
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      hiveConfig_);

  const std::string dataFile = tempDir->getPath() + "/data-file.parquet";
  // Table had rows at positions 0..4; DELETE removed ids 2 and 4 -> positions
  // 1 and 3. The delete operator wraps the full row-id in a dictionary that
  // selects only those two base rows.
  sink.appendData(makeRowIdStructDeletePage(
      dataFile,
      /*allPositions=*/{0, 1, 2, 3, 4},
      /*selected=*/{1, 3},
      /*filePathConstant=*/true));
  EXPECT_TRUE(sink.finish());
  auto commitMessages = sink.close();

  ASSERT_EQ(commitMessages.size(), 1);
  auto parsed = folly::parseJson(commitMessages[0]);
  EXPECT_EQ(parsed["referencedDataFile"].asString(), dataFile);
  EXPECT_EQ(parsed["metrics"]["recordCount"].asInt(), 2);

  auto deleted = readBackDeletedPositions(
      parsed["path"].asString(),
      static_cast<uint64_t>(parsed["contentOffset"].asInt()),
      static_cast<uint64_t>(parsed["contentSizeInBytes"].asInt()),
      static_cast<uint64_t>(parsed["metrics"]["recordCount"].asInt()),
      /*maxPos=*/4,
      pool_.get());
  EXPECT_EQ(deleted, (std::vector<uint64_t>{1, 3}));
}

// Reproduces the exact production V3 DELETE layout captured from the worker:
// the sink input is ROW<id:BIGINT, $row_id:ROW<_file, _pos, _spec_id,
// partition_data>> — a passthrough data column PLUS the row-id ROW. Regression
// guard for a worker SIGSEGV where the sink counted 2 columns and mis-took the
// page for the legacy flat (file_path, pos) layout, then read the ROW column
// as BIGINT positions.
TEST_F(IcebergDeletionVectorSinkTest, rowIdStructWithLeadingDataColumn) {
  auto tempDir = TempDirectoryPath::create();
  auto handle = makeDeletionVectorHandle(tempDir->getPath());

  IcebergDeletionVectorSink sink(
      ROW({"id", "$row_id"},
          {BIGINT(),
           ROW({"_file", "_pos", "_spec_id", "partition_data"},
               {VARCHAR(), BIGINT(), INTEGER(), VARCHAR()})}),
      handle,
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      hiveConfig_);

  const std::string dataFile = tempDir->getPath() + "/data-file.parquet";
  sink.appendData(makeRowIdStructDeletePage(
      dataFile,
      /*allPositions=*/{0, 1, 2, 3, 4},
      /*selected=*/{1, 3},
      /*filePathConstant=*/true,
      /*withLeadingIdColumn=*/true));
  EXPECT_TRUE(sink.finish());
  auto commitMessages = sink.close();

  ASSERT_EQ(commitMessages.size(), 1);
  auto parsed = folly::parseJson(commitMessages[0]);
  EXPECT_EQ(parsed["referencedDataFile"].asString(), dataFile);
  EXPECT_EQ(parsed["metrics"]["recordCount"].asInt(), 2);

  auto deleted = readBackDeletedPositions(
      parsed["path"].asString(),
      static_cast<uint64_t>(parsed["contentOffset"].asInt()),
      static_cast<uint64_t>(parsed["contentSizeInBytes"].asInt()),
      static_cast<uint64_t>(parsed["metrics"]["recordCount"].asInt()),
      /*maxPos=*/4,
      pool_.get());
  EXPECT_EQ(deleted, (std::vector<uint64_t>{1, 3}));
}
TEST_F(IcebergDeletionVectorSinkTest, dictionaryWrappedRowIdStructFlatPath) {
  auto tempDir = TempDirectoryPath::create();
  auto handle = makeDeletionVectorHandle(tempDir->getPath());

  IcebergDeletionVectorSink sink(
      ROW({"$row_id"},
          {ROW(
              {"file_path", "pos", "spec_id", "partition"},
              {VARCHAR(), BIGINT(), INTEGER(), VARCHAR()})}),
      handle,
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      hiveConfig_);

  const std::string dataFile = tempDir->getPath() + "/data-file.parquet";
  sink.appendData(makeRowIdStructDeletePage(
      dataFile,
      /*allPositions=*/{0, 1, 2, 3, 4},
      /*selected=*/{1, 3},
      /*filePathConstant=*/false));
  EXPECT_TRUE(sink.finish());
  auto commitMessages = sink.close();

  ASSERT_EQ(commitMessages.size(), 1);
  auto parsed = folly::parseJson(commitMessages[0]);
  auto deleted = readBackDeletedPositions(
      parsed["path"].asString(),
      static_cast<uint64_t>(parsed["contentOffset"].asInt()),
      static_cast<uint64_t>(parsed["contentSizeInBytes"].asInt()),
      static_cast<uint64_t>(parsed["metrics"]["recordCount"].asInt()),
      /*maxPos=*/4,
      pool_.get());
  EXPECT_EQ(deleted, (std::vector<uint64_t>{1, 3}));
}
