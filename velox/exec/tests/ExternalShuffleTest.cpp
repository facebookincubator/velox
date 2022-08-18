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

class ExternalShuffleTest : public HiveConnectorTestBase {
 protected:
  static std::string makeTaskId(const std::string& prefix, int num) {
    return fmt::format("external_shuffle://{}-{}", prefix, num);
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

  void assertQuery(
      const std::shared_ptr<const core::PlanNode>& plan,
      const std::vector<std::string>& remoteTaskIds,
      const std::string& duckDbSql,
      std::optional<std::vector<uint32_t>> sortingKeys = std::nullopt) {
    std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;
    for (auto& taskId : remoteTaskIds) {
      splits.push_back(std::make_shared<RemoteConnectorSplit>(taskId));
    }
    OperatorTestBase::assertQuery(plan, splits, duckDbSql, sortingKeys);
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5"},
          {BIGINT(), INTEGER(), SMALLINT(), REAL(), DOUBLE(), VARCHAR()})};
  std::unordered_map<std::string, std::string> configSettings_;
  std::vector<std::shared_ptr<TempFilePath>> filePaths_;
  std::vector<RowVectorPtr> vectors_;
};

// A buffer holding the data per for each partition. This class represents a
// simulation of a buffer used by external shuffle service.
class UnsaferowBuffer {
 public:
  UnsaferowBuffer(memory::MemoryPool* pool, bool hasKeys)
      : pool_(pool), hasKeys_(hasKeys) {
    serde_ = std::make_unique<batch::UnsafeRowVectorSerde>(
        pool, nullptr, std::nullopt);
  }

  void resetBuffer(const RowVectorPtr& nextVector) {
    serde_->reset(nextVector);
    blockUsedSize_ = 0;
    rowType_ = std::dynamic_pointer_cast<const RowType>(nextVector->type());
  }

  size_t blockUsedSize() {
    return blockUsedSize_;
  }

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

  const char* buffer() {
    return buffer_;
  }

 private:
  RowTypePtr rowType_;
  std::unique_ptr<batch::UnsafeRowVectorSerde> serde_;
  memory::MemoryPool* pool_ = nullptr;
  BufferPtr bufferPtr_ =
      AlignedBuffer::allocate<char>(kInitialBlockSize, pool_, true);
  char* buffer_ = bufferPtr_->asMutable<char>();
  size_t blockSize_ = kInitialBlockSize;
  size_t blockUsedSize_ = 0;
  bool hasKeys_;
  static constexpr size_t kInitialBlockSize = 10 << 10; // 10k
};

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

void CovertRowBlockToVector(
    std::string_view rowBlock,
    RowTypePtr rowType,
    memory::MemoryPool* pool,
    bool hasKeys,
    RowVectorPtr& outputVector) {
  // Collect row pointers
  std::vector<std::optional<std::string_view>> rowPointers;
  auto collected = collectRowPointers(rowBlock, hasKeys, rowPointers);
  ASSERT_TRUE(collected);

  // Use block deserializer to deserialize all the rows
  batch::UnsafeRowVectorSerde serde(pool, nullptr, std::nullopt);
  auto err = serde.deserializeVector(rowPointers, rowType, &outputVector);
  ASSERT_EQ(err, batch::BatchSerdeStatus::Success);
}

class ShuffleService {
 public:
  using DataAvailCallbackType =
      std::function<bool()>;
  ShuffleService(int partitions, memory::MemoryPool* pool, bool hasKeys) {
    partitionBuffers_.reserve(partitions);
    for (int i = 0; i < partitions; i++) {
      partitionBuffers_.emplace_back(pool, hasKeys);
    }
  }
  size_t collect(
      int partition,
      const std::string_view& serializedKeys,
      const std::string_view& serializedRow) {
    VELOX_CHECK_LT(partition, partitionBuffers_.size());
    auto& destination = partitionBuffers_[partition];
    auto currentSize = destination.blockUsedSize();
    destination.addRowToBuffer(serializedKeys, serializedRow);
    return destination.blockUsedSize() - currentSize;
  }

  std::unique_ptr<folly::IOBuf> getPartition(int destination) {
    VELOX_CHECK_LT(destination, partitionBuffers_.size());
    return folly::IOBuf::wrapBuffer(
        partitionBuffers_[destination].buffer(),
        partitionBuffers_[destination].blockUsedSize());
  }

  void registerDataCallback(const DataAvailCallbackType& callback) {
    callBacks_.push_back(callback);
  }

  bool ready() {
    return allPartitionsReady_;
  }

  void setReady() {
    allPartitionsReady_ = true;
    for (auto callback: callBacks_) {
      callback();
    }
  }

 private:
  std::vector<UnsaferowBuffer> partitionBuffers_;
  bool allPartitionsReady_ = false;
  std::vector<DataAvailCallbackType> callBacks_;
};

// A simple pass throw destination that writes unsaferows to a shared
// buffer provided by the factory
class UnsafeRowTestDestination : public PassThroughDestination {
 public:
  UnsafeRowTestDestination(
      const std::string& taskId,
      int destination,
      memory::MappedMemory* FOLLY_NONNULL memory,
      const std::optional<std::vector<uint64_t>>& keys,
      memory::MemoryPool* pool,
      ShuffleService& shuffleService)
      : PassThroughDestination(taskId, destination, memory),
        shuffle_(shuffleService) {
    keySerializer_ = keys.has_value()
        ? std::make_unique<batch::UnsafeRowKeySerializer>()
        : nullptr;
    serde_ = std::make_unique<batch::UnsafeRowVectorSerde>(
        pool, keySerializer_, keys);
  }

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
        // Serializing row and keys
        std::string_view serializedKeys;
        std::string_view serializedRow;
        // Serialize one row at a time and add to block
        auto err = serde_->serializeRow(
            output, rowIndex, serializedKeys, serializedRow);
        VELOX_CHECK_EQ(err, batch::BatchSerdeStatus::Success);
        bytesInCurrent_ +=
            shuffle_.collect(destination_, serializedKeys, serializedRow);
      }
    }
    *atEnd = true;
    return BlockingReason::kNotBlocked;
  };

 private:
  std::unique_ptr<batch::UnsafeRowVectorSerde> serde_;
  std::unique_ptr<batch::VectorKeySerializer> keySerializer_;
  ShuffleService& shuffle_;
};

class UnsafeRowTestDestinationFactory : public DestinationFactory {
 public:
  UnsafeRowTestDestinationFactory(
      const std::optional<std::vector<uint64_t>>& keys,
      memory::MemoryPool* pool,
      ShuffleService& shuffleService,
      int expectedVectorInputs)
      : DestinationFactory(),
        pool_(pool),
        keys_(keys),
        shuffle_(shuffleService),
        expectedInputs_(expectedVectorInputs) {}

  std::unique_ptr<Destination> createDestination(
      const std::string& taskId,
      int destination,
      memory::MappedMemory* FOLLY_NONNULL memory) override {
    return std::make_unique<UnsafeRowTestDestination>(
        taskId, destination, memory, keys_, pool_, shuffle_);
  }

  void beginBatch(const RowVectorPtr& vector) override {
    latestVector_ = vector;
  }

  void outputReady() override {
    checkedBatches_++;
    if (checkedBatches_ >= expectedInputs_) {
      shuffle_.setReady();
    }
  }

  virtual ~UnsafeRowTestDestinationFactory() {}

 private:
  // The latest input vector seen by beginBatch
  RowVectorPtr latestVector_ = nullptr;
  // Some stats
  int checkedBatches_ = 0;

  memory::MemoryPool* pool_;

  std::optional<std::vector<uint64_t>> keys_;

  ShuffleService& shuffle_;

  int expectedInputs_ = 0;
};

class UnsafeRowExchangeSource : public ExchangeSource {
  using shuffleMetaDataType = long;

 public:
  UnsafeRowExchangeSource(
      const std::string& taskId,
      int destination,
      std::shared_ptr<ExchangeQueue> queue,
      ShuffleService& shuffle)
      : ExchangeSource(taskId, destination, queue), shuffle_(shuffle) {}

  bool shouldRequestLocked() override {
    /*if (!shuffle_.ready()) {
      return false;
    }*/
    if (atEnd_) {
      return false;
    }
    return true;
  }

  void request() override {
    auto dataCallBack = [this]() {
      std::lock_guard<std::mutex> l(queue_->mutex());
      auto block = shuffle_.getPartition(destination_);
      queue_->enqueue(
          std::move(std::make_unique<SerializedPage>(std::move(block))));
      finished_ = true;
      atEnd_ = true;
      return true;
    };
    if (!shuffle_.ready()) {
      shuffle_.registerDataCallback(dataCallBack);
      return;
    }
    VELOX_CHECK(requestPending_);
    if (!finished_) {
      // When it is the first time, we grab the block iterator for the given
      // shuffle partition and metadata
      dataCallBack();
    }
  }

  void close() override {}

 private:
  ShuffleService& shuffle_;
  bool finished_ = false;
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

TEST_F(ExternalShuffleTest, unsaferowDestinationExchange) {
  static int numPartitions_s = 2;
  static std::optional<std::vector<uint64_t>> keys =
      std::vector<uint64_t>({0, 1});

  ShuffleService shuffle(numPartitions_s, pool_.get(), keys.has_value());
  // Initializing factory generator for UnsafeRowTestDestinationFactory
  // which serializes the first two keys and the rows
  auto factoryGenerator = [this, &shuffle]() {
    return std::make_unique<UnsafeRowTestDestinationFactory>(
        keys, pool_.get(), shuffle, filePaths_.size());
  };
  PartitionedOutput::registerDestinationFactory(factoryGenerator, false);

  auto createUnsafeRowExchangeSource = [&shuffle](
                                           const std::string& taskId,
                                           int destination,
                                           std::shared_ptr<ExchangeQueue> queue)
      -> std::unique_ptr<UnsafeRowExchangeSource> {
    if (strncmp(taskId.c_str(), "external_shuffle://", 19) == 0) {
      return std::make_unique<UnsafeRowExchangeSource>(
          taskId, destination, std::move(queue), shuffle);
    }
    return nullptr;
  };

  ExchangeSource::registerFactory(createUnsafeRowExchangeSource);

  auto externalBlockConvertor = [this](
                                    const std::unique_ptr<SerializedPage>& page,
                                    RowVectorPtr& output) {
    std::string_view block(
        (char*)page->getIOBuf()->data(), page->getIOBuf()->length());
    CovertRowBlockToVector(
        block,
        ROW({"p0", "p1", "c2"}, {SMALLINT(), BIGINT(), BIGINT()}),
        pool_.get(),
        true,
        output);
    return true;
  };

  Exchange::setExternalDataConvertor(externalBlockConvertor);

  setupSources(1, 1);
  // Run the table scan several times to test the caching.
  for (int i = 0; i < 1; ++i) {
    auto taskId = makeTaskId("task", 0);

    // Partitioning on the first two columns with 1000 destinations
    auto plan = PlanBuilder()
                    .tableScan(rowType_)
                    .project({"c0 % 10", "c1 % 2", "c2"})
                    .partitionedOutput(
                        {"p0", "p1"}, numPartitions_s, {"c2", "p1", "p0"})
                    .planNode();

    auto task = makeTask(taskId, plan, 0);
    Task::start(task, 1);
    addHiveSplits(task, filePaths_);

    auto op = PlanBuilder().exchange(plan->outputType()).planNode();
    assertQuery(op, {taskId}, "SELECT c2, c1 % 2, c0 % 10 FROM tmp");

    ASSERT_TRUE(waitForTaskCompletion(task.get())) << task->taskId();
  }
}
