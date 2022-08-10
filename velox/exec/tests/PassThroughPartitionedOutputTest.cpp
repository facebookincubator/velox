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
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/DataSink.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/serializers/UnsafeRowSerde.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::connector::hive;

using facebook::velox::test::BatchMaker;

class PassThroughPartitionOutputTest : public HiveConnectorTestBase {
 protected:
  static std::string makeTaskId(const std::string& prefix, int num) {
    return fmt::format("local://{}-{}", prefix, num);
  }

  std::shared_ptr<Task> makeTask(
      const std::string& taskId,
      std::shared_ptr<const core::PlanNode> planNode,
      int destination) {
    auto queryCtx = core::QueryCtx::createForTest(
        std::make_shared<core::MemConfig>(configSettings_));
    core::PlanFragment planFragment{planNode};
    return std::make_shared<Task>(
        taskId, std::move(planFragment), destination, std::move(queryCtx));
  }

  std::vector<RowVectorPtr> makeVectors(int count, int rowsPerVector) {
    std::vector<RowVectorPtr> vectors;
    for (int i = 0; i < count; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          BatchMaker::createBatch(rowType_, rowsPerVector, *pool_));
      vectors.push_back(vector);
    }
    return vectors;
  }

  void addHiveSplits(
      std::shared_ptr<Task> task,
      const std::vector<std::shared_ptr<TempFilePath>>& filePaths) {
    for (auto& filePath : filePaths) {
      auto split = exec::Split(
          std::make_shared<HiveConnectorSplit>(
              kHiveConnectorId,
              "file:" + filePath->path,
              facebook::velox::dwio::common::FileFormat::DWRF),
          -1);
      task->addSplit("0", std::move(split));
      VLOG(1) << filePath->path << "\n";
    }
    task->noMoreSplits("0");
  }

  void setupSources(int filePathCount, int rowsPerVector) {
    filePaths_ = makeFilePaths(filePathCount);
    vectors_ = makeVectors(filePaths_.size(), rowsPerVector);
    for (int i = 0; i < filePaths_.size(); i++) {
      writeToFile(filePaths_[i]->path, vectors_[i]);
    }
    createDuckDbTable(vectors_);
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5"},
          {BIGINT(), INTEGER(), SMALLINT(), REAL(), DOUBLE(), VARCHAR()})};
  std::unordered_map<std::string, std::string> configSettings_;
  std::vector<std::shared_ptr<TempFilePath>> filePaths_;
  std::vector<RowVectorPtr> vectors_;
};

// Instead of comparing input and shuffled vectors directly, this function
// makes sure their difference is zero
void assertEquivalentVectors(
    const VectorPtr& expected,
    const VectorPtr& actual) {
  ASSERT_EQ(expected->size(), actual->size());
  ASSERT_TRUE(expected->type()->equivalent(*actual->type()))
      << "Expected " << expected->type()->toString() << ", but got "
      << actual->type()->toString();
  std::vector<bool> checkedActualValues;
  checkedActualValues.resize(expected->size(), false);
  for (auto i = 0; i < expected->size(); i++) {
    bool found = false;
    for (auto j = 0; j < expected->size(); j++) {
      if (!checkedActualValues[j]) {
        if (expected->equalValueAt(actual.get(), i, j)) {
          found = true;
          checkedActualValues[j] = true;
          break;
        }
      }
    }
    ASSERT_TRUE(found) << "Value expected " << expected->toString(i)
                       << " in position " << i << " was not found!"
                       << " expected "
                       << expected->toString(0, expected->size(), ",", true)
                       << " actual "
                       << actual->toString(0, expected->size(), ",", true);
  }
}

// A shared buffer to mimic an external shuffle model for the result of
// partitioning
class UnsaferowBuffer {
 public:
  UnsaferowBuffer(
      const std::optional<std::vector<uint64_t>>& keys,
      memory::MemoryPool* pool)
      : keys_(keys), pool_(pool) {
    keySerializer_ = keys.has_value()
        ? std::make_unique<batch::UnsafeRowKeySerializer>()
        : nullptr;
    serde_ = std::make_unique<batch::UnsafeRowVectorSerde>(
        pool_, keySerializer_, keys);
  }

  void resetBuffer(const RowVectorPtr& nextVector) {
    serde_->reset(nextVector);
    blockUsedSize_ = 0;
    rowType_ = std::dynamic_pointer_cast<const RowType>(nextVector->type());
  }

  size_t blockUsedSize() {
    return blockUsedSize_;
  }

  void collectVector(RowVectorPtr& outputVector) {
    // Collect row pointers
    std::vector<std::optional<std::string_view>> rowPointers;
    auto collected = collectRowPointers(
        std::string_view(buffer_, blockUsedSize_), true, rowPointers);
    ASSERT_TRUE(collected);

    // Use block deserializer to deserialize all the rows
    auto err = serde_->deserializeVector(rowPointers, rowType_, &outputVector);
    ASSERT_EQ(err, batch::BatchSerdeStatus::Success);
  }

  size_t serializeRow(const RowVectorPtr& vector, vector_size_t index) {
    // Serializing row and keys
    std::string_view serializedKeys;
    std::string_view serializedRow;
    // Serialize one row at a time and add to block
    auto err =
        serde_->serializeRow(vector, index, serializedKeys, serializedRow);
    VELOX_CHECK_EQ(err, batch::BatchSerdeStatus::Success);
    auto prevBlockUsedSize = blockUsedSize();
    addRowToBuffer(serializedKeys, serializedRow);
    return blockUsedSize() - prevBlockUsedSize;
  }

 private:
  void addRowToBuffer(
      const std::string_view& serializedKeys,
      const std::string_view& serializedRow) {
    // The key value pairs (keys, row) are added to the buffer fashion
    // <keys size> <serialized keys> <row size> <serialized row>
    if (blockUsedSize_ + serializedKeys.size() + serializedRow.size() +
            2 * sizeof(size_t) >
        blockSize_) {
      // Reallocate block buffer if the rows are overflowing
      size_t newBlockSize =
          (blockSize_ + serializedKeys.size() + serializedRow.size()) * 2;
      AlignedBuffer::reallocate<char>(&bufferPtr_, newBlockSize, 0);
      buffer_ = bufferPtr_->asMutable<char>();
      blockSize_ = newBlockSize;
    }
    // Add the row (and keys) to the block
    auto buffer = buffer_ + blockUsedSize_;
    *((size_t*)buffer) = serializedKeys.size();
    // Counting for the size value
    buffer += sizeof(size_t);
    std::memcpy(buffer, serializedKeys.data(), serializedKeys.size());
    buffer += serializedKeys.size();
    *((size_t*)buffer) = serializedRow.size();
    buffer += sizeof(size_t);
    std::memcpy(buffer, serializedRow.data(), serializedRow.size());
    blockUsedSize_ +=
        serializedKeys.size() + serializedRow.size() + 2 * sizeof(size_t);
  }

  bool collectRowPointers(
      const std::string_view& block,
      bool includesKeys,
      std::vector<std::optional<std::string_view>>& rowPointers) {
    char* buffer = const_cast<char*>(block.data());
    char* originalBuffer = buffer;

    // Precess until the end of block
    ssize_t remainingSize = block.size();
    while (remainingSize > 0) {
      // Skip keys if any
      if (includesKeys) {
        size_t keysSize = *((size_t*)buffer);
        if (keysSize < 0 || keysSize > remainingSize) {
          return false;
        }
        buffer += sizeof(keysSize);
        buffer += keysSize;
      }
      // Read the row size
      size_t rowSize = *((size_t*)buffer);
      if (rowSize < 0 || rowSize > remainingSize) {
        return false;
      }
      buffer += sizeof(rowSize);
      rowPointers.emplace_back(std::string_view(buffer, rowSize));
      buffer += rowSize;
      remainingSize = block.size() - (buffer - originalBuffer);
      if (remainingSize < 0) {
        return false;
      }
    }
    return true;
  }

 private:
  RowTypePtr rowType_;
  const std::optional<std::vector<uint64_t>>& keys_;
  std::unique_ptr<batch::VectorKeySerializer> keySerializer_;
  std::unique_ptr<batch::UnsafeRowVectorSerde> serde_;
  memory::MemoryPool* pool_ = nullptr;
  BufferPtr bufferPtr_ =
      AlignedBuffer::allocate<char>(kInitialBlockSize, pool_, true);
  char* buffer_ = bufferPtr_->asMutable<char>();
  size_t blockSize_ = kInitialBlockSize;
  size_t blockUsedSize_ = 0;
  static constexpr size_t kInitialBlockSize = 10 << 10; // 10k
};

// A simple pass throw destination that writes unsaferows to a shared
// buffer provided by the factory
class UnsafeRowTestDestination : public PassThroughDestination {
 public:
  UnsafeRowTestDestination(
      const std::string& taskId,
      int destination,
      memory::MappedMemory* FOLLY_NONNULL memory,
      UnsaferowBuffer& rowBuffer)
      : PassThroughDestination(taskId, destination, memory),
        rowBuffer_(rowBuffer) {}

  BlockingReason advance(
      uint64_t maxBytes,
      const std::vector<vector_size_t>& sizes,
      const RowVectorPtr& output,
      PartitionedOutputBufferManager& bufferManager,
      bool* FOLLY_NONNULL atEnd,
      ContinueFuture* FOLLY_NONNULL future) {
    if (row_ >= rows_.size()) {
      *atEnd = true;
      return BlockingReason::kNotBlocked;
    }
    // Simply serialize all rows in one go into the row buffer
    for (; row_ < rows_.size(); ++row_) {
      for (vector_size_t i = 0; i < rows_[row_].size; i++) {
        auto rowIndex = rows_[row_].begin + i;
        bytesInCurrent_ += rowBuffer_.serializeRow(output, rowIndex);
      }
    }
    *atEnd = true;
    return BlockingReason::kNotBlocked;
  };

 private:
  UnsaferowBuffer& rowBuffer_;
};

class UnsafeRowTestDestinationFactory : public DestinationFactory {
 public:
  UnsafeRowTestDestinationFactory(
      const std::optional<std::vector<uint64_t>>& keys,
      memory::MemoryPool* pool,
      const std::vector<RowVectorPtr>& referenceVectors)
      : DestinationFactory(), rowBuffer_(keys, pool) {}

  std::unique_ptr<Destination> createDestination(
      const std::string& taskId,
      int destination,
      memory::MappedMemory* FOLLY_NONNULL memory) override {
    return std::make_unique<UnsafeRowTestDestination>(
        taskId, destination, memory, rowBuffer_);
  }

  void beginBatch(const RowVectorPtr& vector) override {
    rowBuffer_.resetBuffer(vector);
    latestVector_ = vector;
  }

  void outputReady() override {
    if (rowBuffer_.blockUsedSize() > 0) {
      RowVectorPtr outputVector;
      rowBuffer_.collectVector(outputVector);
      // Compare the input and output vectors
      assertEquivalentVectors(latestVector_, outputVector);
      checkedBatches_++;
    }
  }

  virtual ~UnsafeRowTestDestinationFactory() {}

 private:
  // The buffer shared across the destination in each partitioned output object
  UnsaferowBuffer rowBuffer_;
  // The latest input vector seen by beginBatch
  RowVectorPtr latestVector_ = nullptr;
  // Some stats
  int checkedBatches_ = 0;
};

static void waitForFinishedDrivers(
    const std::shared_ptr<Task>& task,
    uint32_t n) {
  while (task->numFinishedDrivers() < n) {
    /* sleep override */
    usleep(100'000); // 0.1 second.
  }
  ASSERT_EQ(n, task->numFinishedDrivers());
}

TEST_F(PassThroughPartitionOutputTest, unsaferowDestinations) {
  // Initializing factory generator for UnsafeRowTestDestinationFactory
  // which serializes the first two keys and the rows
  auto factoryGenerator = [this]() {
    return std::make_unique<UnsafeRowTestDestinationFactory>(
        std::vector<uint64_t>({0, 1}), pool_.get(), vectors_);
  };
  PartitionedOutput::registerDestinationFactory(factoryGenerator, false);

  setupSources(3, 2);
  // Run the table scan several times to test the caching.
  for (int i = 0; i < 3; ++i) {
    auto taskId = makeTaskId("task", 0);

    // Partitioning on the first two columns with 1000 destinations
    auto plan = PlanBuilder()
                    .tableScan(rowType_)
                    .project({"c0 % 10", "c1 % 2", "c2"})
                    .partitionedOutput({"p0", "p1"}, 1000, {"c2", "p1", "p0"})
                    .planNode();

    auto task = makeTask(taskId, plan, 0);
    Task::start(task, 4);
    addHiveSplits(task, filePaths_);

    task->noMoreSplits("0");
    waitForFinishedDrivers(task, 4);
  }
}
