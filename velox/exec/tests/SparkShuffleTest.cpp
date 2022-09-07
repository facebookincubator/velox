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
#include "velox/exec/Exchange.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/tests/PartitionAndSerialize.h"
#include "velox/exec/tests/ShuffleWrite.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/VectorFunction.h"
#include "velox/serializers/UnsafeRowSerde.h"
#include "velox/serializers/UnsafeRowSerializer.h"

using namespace facebook::velox;

namespace facebook::velox::exec::test {

namespace {
class PartitionFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx* context,
      VectorPtr* result) const override {
    // TODO Verify that partition count is constant.
    auto numPartitions =
        args[0]->as<SimpleVector<int32_t>>()->valueAt(rows.begin());

    auto rowType = makeRowType(args);

    auto argsCopy = args;
    auto input = std::make_shared<RowVector>(
        context->pool(), rowType, nullptr, rows.size(), std::move(argsCopy));

    std::vector<column_index_t> keyChannels(args.size() - 1);
    std::iota(keyChannels.begin(), keyChannels.end(), 1);

    auto partitionFunction = std::make_unique<HashPartitionFunction>(
        numPartitions, rowType, keyChannels);

    std::vector<uint32_t> partitions(rows.size());
    partitionFunction->partition(*input, partitions);

    context->ensureWritable(rows, INTEGER(), *result);
    auto flatVector = (*result)->asFlatVector<int32_t>();
    rows.applyToSelected(
        [&](auto row) { flatVector->set(row, partitions[row]); });
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("integer")
                .argumentType("any")
                .variableArity()
                .returnType("integer")
                .build()};
  }

 private:
  RowTypePtr makeRowType(const std::vector<VectorPtr>& args) const {
    std::vector<std::string> inputNames;
    std::vector<TypePtr> inputTypes;
    inputNames.reserve(args.size());
    inputTypes.reserve(args.size());

    for (auto i = 0; i < args.size(); ++i) {
      inputNames.push_back(fmt::format("c{}", i));
      inputTypes.push_back(args[i]->type());
    }

    return ROW(std::move(inputNames), std::move(inputTypes));
  }
};

class SerializeToUnsafeRowFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx* context,
      VectorPtr* result) const override {
    auto serde = std::make_unique<batch::UnsafeRowVectorSerde>(context->pool());

    context->ensureWritable(rows, VARBINARY(), *result);
    auto flatVector = (*result)->asFlatVector<StringView>();

    auto argsCopy = args;
    auto input = std::make_shared<RowVector>(
        context->pool(),
        makeRowType(args),
        nullptr,
        rows.size(),
        std::move(argsCopy));

    rows.applyToSelected([&](auto row) {
      // TODO Handle errors. Avoid extra copy.
      std::string_view unused;
      std::string_view serialized;
      serde->serializeRow(input, row, unused, serialized);
      flatVector->set(row, StringView(serialized));
    });
  }

  RowTypePtr makeRowType(const std::vector<VectorPtr>& args) const {
    std::vector<std::string> inputNames;
    std::vector<TypePtr> inputTypes;
    inputNames.reserve(args.size());
    inputTypes.reserve(args.size());

    for (auto i = 0; i < args.size(); ++i) {
      inputNames.push_back(fmt::format("c{}", i));
      inputTypes.push_back(args[i]->type());
    }

    return ROW(std::move(inputNames), std::move(inputTypes));
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("any")
                .variableArity()
                .returnType("varbinary")
                .build()};
  }
};

class TestShuffle : public exec::Shuffle {
 public:
  TestShuffle(
      memory::MemoryPool* pool,
      uint32_t numPartitions,
      uint32_t maxBytesPerPartition)
      : pool_{pool},
        numPartitions_{numPartitions},
        maxBytesPerPartition_{maxBytesPerPartition},
        inProgressSizes_(numPartitions, 0) {
    inProgressPartitions_.resize(numPartitions_);
    readyPartitions_.resize(numPartitions_);
  }

  void collect(int32_t partition, std::string_view data) {
    auto& buffer = inProgressPartitions_[partition];

    // Check if there is enough space in the buffer.
    if (buffer &&
        inProgressSizes_[partition] + data.size() + sizeof(size_t) >=
            maxBytesPerPartition_) {
      buffer->setSize(inProgressSizes_[partition]);
      readyPartitions_[partition].emplace_back(std::move(buffer));
      inProgressPartitions_[partition].reset();
    }

    // Allocate buffer if needed.
    if (!buffer) {
      buffer = AlignedBuffer::allocate<char>(maxBytesPerPartition_, pool_);
      inProgressSizes_[partition] = 0;
    }

    // Copy data.
    auto rawBuffer = buffer->asMutable<char>();
    auto offset = inProgressSizes_[partition];

    *(size_t*)(rawBuffer + offset) = data.size();

    offset += sizeof(size_t);
    memcpy(rawBuffer + offset, data.data(), data.size());

    inProgressSizes_[partition] += sizeof(size_t) + data.size();
  }

  void noMoreData() {
    for (auto i = 0; i < numPartitions_; ++i) {
      if (inProgressSizes_[i] > 0) {
        auto& buffer = inProgressPartitions_[i];
        buffer->setSize(inProgressSizes_[i]);
        readyPartitions_[i].emplace_back(std::move(buffer));
        inProgressPartitions_[i].reset();
      }
    }
  }

  bool hasNext(int32_t partition) const {
    return !readyPartitions_[partition].empty();
  }

  BufferPtr next(int32_t partition) {
    VELOX_CHECK(!readyPartitions_[partition].empty());

    auto buffer = readyPartitions_[partition].back();
    readyPartitions_[partition].pop_back();
    return buffer;
  }

 private:
  memory::MemoryPool* pool_;
  const uint32_t numPartitions_;
  const uint32_t maxBytesPerPartition_;
  std::vector<BufferPtr> inProgressPartitions_;
  std::vector<size_t> inProgressSizes_;
  std::vector<std::vector<BufferPtr>> readyPartitions_;
};

class UnsafeRowExchangeSource : public ExchangeSource {
 public:
  UnsafeRowExchangeSource(
      const std::string& taskId,
      int destination,
      std::shared_ptr<exec::ExchangeQueue> queue,
      Shuffle* shuffle,
      memory::MemoryPool* pool)
      : ExchangeSource(taskId, destination, queue, pool), shuffle_(shuffle) {}

  bool shouldRequestLocked() override {
    return !atEnd_;
  }

  void request() override {
    std::lock_guard<std::mutex> l(queue_->mutex());

    if (!shuffle_->hasNext(destination_)) {
      atEnd_ = true;
      queue_->enqueue(nullptr);
      return;
    }

    auto buffer = shuffle_->next(destination_);

    auto ioBuf = folly::IOBuf::wrapBuffer(buffer->as<char>(), buffer->size());
    queue_->enqueue(std::make_unique<SerializedPage>(
        std::move(ioBuf), pool_, [buffer](auto&) { buffer->release(); }));
  }

  void close() override {}

 private:
  Shuffle* shuffle_;
};

void registerExchangeSource(Shuffle* shuffle) {
  ExchangeSource::registerFactory(
      [shuffle](
          const std::string& taskId,
          int destination,
          std::shared_ptr<ExchangeQueue> queue,
          memory::MemoryPool* FOLLY_NONNULL pool)
          -> std::unique_ptr<ExchangeSource> {
        if (strncmp(taskId.c_str(), "spark://", 8) == 0) {
          return std::make_unique<UnsafeRowExchangeSource>(
              taskId, destination, std::move(queue), shuffle, pool);
        }
        return nullptr;
      });
}

auto addPartitionAndSerializeNode(int numPartitions) {
  return [numPartitions](
             core::PlanNodeId nodeId,
             core::PlanNodePtr source) -> core::PlanNodePtr {
    auto outputType = ROW({"p", "d"}, {INTEGER(), VARBINARY()});

    std::vector<core::TypedExprPtr> keys;
    keys.push_back(
        std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0"));

    return std::make_shared<PartitionAndSerializeNode>(
        nodeId, keys, numPartitions, outputType, std::move(source));
  };
}

auto addShuffleWriteNode(Shuffle* shuffle) {
  return [shuffle](
             core::PlanNodeId nodeId,
             core::PlanNodePtr source) -> core::PlanNodePtr {
    return std::make_shared<ShuffleWriteNode>(
        nodeId, shuffle, std::move(source));
  };
}
} // namespace

class SparkShuffleTest : public OperatorTestBase {
 protected:
  static std::string makeTaskId(const std::string& prefix, int num) {
    return fmt::format("spark://{}-{}", prefix, num);
  }

  std::shared_ptr<Task> makeTask(
      const std::string& taskId,
      core::PlanNodePtr planNode,
      int destination) {
    auto queryCtx =
        core::QueryCtx::createForTest(std::make_shared<core::MemConfig>());
    core::PlanFragment planFragment{planNode};
    return std::make_shared<Task>(
        taskId, std::move(planFragment), destination, std::move(queryCtx));
  }

  void addRemoteSplits(
      Task* task,
      const std::vector<std::string>& remoteTaskIds) {
    for (auto& taskId : remoteTaskIds) {
      auto split =
          exec::Split(std::make_shared<RemoteConnectorSplit>(taskId), -1);
      task->addSplit("0", std::move(split));
    }
    task->noMoreSplits("0");
  }

  RowVectorPtr deserialize(
      const RowVectorPtr& serializedResult,
      const RowTypePtr& rowType) {
    auto serializedData =
        serializedResult->childAt(1)->as<FlatVector<StringView>>();

    // Serialize data into a single block.

    // Calculate total size.
    size_t totalSize = 0;
    for (auto i = 0; i < serializedData->size(); ++i) {
      totalSize += serializedData->valueAt(i).size();
    }

    // Allocate the block. Add an extra sizeof(size_t) bytes for each row to
    // hold row size.
    BufferPtr buffer = AlignedBuffer::allocate<char>(
        totalSize + sizeof(size_t) * serializedData->size(), pool());
    auto rawBuffer = buffer->asMutable<char>();

    // Copy data.
    size_t offset = 0;
    for (auto i = 0; i < serializedData->size(); ++i) {
      auto value = serializedData->valueAt(i);

      *(size_t*)(rawBuffer + offset) = value.size();
      offset += sizeof(size_t);

      memcpy(rawBuffer + offset, value.data(), value.size());
      offset += value.size();
    }

    // Deserialize the block.
    return deserialize(buffer, rowType);
  }

  RowVectorPtr deserialize(BufferPtr& serialized, const RowTypePtr& rowType) {
    auto serializer =
        std::make_unique<serializer::spark::UnsafeRowVectorSerde>();

    ByteRange byteRange = {
        serialized->asMutable<uint8_t>(), (int32_t)serialized->size(), 0};

    auto input = std::make_unique<ByteStream>();
    input->resetInput({byteRange});

    RowVectorPtr result;
    serializer->deserialize(input.get(), pool(), rowType, &result, nullptr);
    return result;
  }
};

// Use custom functions to partition and serialize data.
TEST_F(SparkShuffleTest, functions) {
  exec::registerVectorFunction(
      "serialize_to_unsaferow",
      SerializeToUnsafeRowFunction::signatures(),
      std::make_unique<SerializeToUnsafeRowFunction>());

  exec::registerVectorFunction(
      "partition",
      PartitionFunction::signatures(),
      std::make_unique<PartitionFunction>());

  Operator::registerOperator(std::make_unique<ShuffleWriteTranslator>());

  TestShuffle shuffle(pool(), 4, 1 << 20 /* 1MB */);

  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
  });

  auto plan =
      PlanBuilder()
          .values({data}, true)
          .project(
              {"partition(4::integer, c0)", "serialize_to_unsaferow(c0, c1)"})
          .localPartition({})
          .addNode(
              [&](core::PlanNodeId nodeId,
                  core::PlanNodePtr source) -> core::PlanNodePtr {
                return std::make_shared<ShuffleWriteNode>(
                    nodeId, &shuffle, std::move(source));
              })
          .planNode();

  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 2;

  auto [taskCursor, serializedResults] =
      readCursor(params, [](auto /*task*/) {});
  ASSERT_EQ(serializedResults.size(), 0);
}

TEST_F(SparkShuffleTest, functionAndSerializeFunctions) {
  exec::registerVectorFunction(
      "serialize_to_unsaferow",
      SerializeToUnsafeRowFunction::signatures(),
      std::make_unique<SerializeToUnsafeRowFunction>());

  exec::registerVectorFunction(
      "partition",
      PartitionFunction::signatures(),
      std::make_unique<PartitionFunction>());

  Operator::registerOperator(std::make_unique<ShuffleWriteTranslator>());

  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
  });

  auto plan =
      PlanBuilder()
          .values({data}, true)
          .project(
              {"partition(4::integer, c0)", "serialize_to_unsaferow(c0, c1)"})
          .planNode();

  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 2;

  auto [taskCursor, serializedResults] =
      readCursor(params, [](auto /*task*/) {});
  ASSERT_EQ(serializedResults.size(), 2);

  for (auto& serializedResult : serializedResults) {
    // Print out partition numbers.
    std::cout << serializedResult->childAt(0)->toString(0, 100) << std::endl;

    // Verify that serialized data can be deserialized successfully into the
    // original data.
    auto deserialized = deserialize(serializedResult, asRowType(data->type()));
    velox::test::assertEqualVectors(data, deserialized);
  }
}

// Use custom operator to partition and serialize data.
TEST_F(SparkShuffleTest, operators) {
  Operator::registerOperator(
      std::make_unique<PartitionAndSerializeTranslator>());
  Operator::registerOperator(std::make_unique<ShuffleWriteTranslator>());

  TestShuffle shuffle(pool(), 4, 1 << 20 /* 1MB */);

  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
  });

  auto plan = PlanBuilder()
                  .values({data}, true)
                  .addNode(addPartitionAndSerializeNode(4))
                  .localPartition({})
                  .addNode(addShuffleWriteNode(&shuffle))
                  .planNode();

  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 2;

  auto [taskCursor, serializedResults] =
      readCursor(params, [](auto /*task*/) {});
  ASSERT_EQ(serializedResults.size(), 0);

  for (auto i = 0; i < 4; ++i) {
    while (shuffle.hasNext(i)) {
      auto buffer = shuffle.next(i);
      auto vector = deserialize(buffer, asRowType(data->type()));
      std::cout << "Partition " << i << ": " << vector->toString() << std::endl;
      std::cout << vector->toString(0, 10) << std::endl;
    }
  }
}

TEST_F(SparkShuffleTest, endToEnd) {
  Operator::registerOperator(
      std::make_unique<PartitionAndSerializeTranslator>());
  Operator::registerOperator(std::make_unique<ShuffleWriteTranslator>());

  serializer::spark::UnsafeRowVectorSerde::registerVectorSerde();

  size_t numPartitions = 5;
  TestShuffle shuffle(pool(), numPartitions, 1 << 20 /* 1MB */);

  registerExchangeSource(&shuffle);

  // Create and run single leaf task to partition data and write it to shuffle.
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5, 6}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
  });

  auto dataType = asRowType(data->type());

  auto leafPlan = PlanBuilder()
                      .values({data}, true)
                      .addNode(addPartitionAndSerializeNode(numPartitions))
                      .localPartition({})
                      .addNode(addShuffleWriteNode(&shuffle))
                      .planNode();

  auto leafTaskId = makeTaskId("leaf", 0);
  auto leafTask = makeTask(leafTaskId, leafPlan, 0);
  Task::start(leafTask, 2);
  ASSERT_TRUE(waitForTaskCompletion(leafTask.get()));

  // Create and run multiple downstream tasks, one per partition, to read data
  // from shuffle.
  for (auto i = 0; i < numPartitions; ++i) {
    auto plan =
        PlanBuilder().exchange(dataType).project(dataType->names()).planNode();

    CursorParameters params;
    params.planNode = plan;
    params.destination = i;

    bool noMoreSplits = false;
    auto [taskCursor, serializedResults] = readCursor(params, [&](auto* task) {
      if (noMoreSplits) {
        return;
      }
      addRemoteSplits(task, {leafTaskId});
      noMoreSplits = true;
    });

    if (serializedResults.empty()) {
      std::cout << "No results for partition " << i << std::endl;
    }

    for (auto& result : serializedResults) {
      std::cout << "Partition: " << i << ": " << result->toString()
                << std::endl;
      std::cout << result->toString(0, 10) << std::endl;
    }

    ASSERT_FALSE(shuffle.hasNext(i)) << i;
  }
}

TEST_F(SparkShuffleTest, partitionAndSerializeOperator) {
  Operator::registerOperator(
      std::make_unique<PartitionAndSerializeTranslator>());
  Operator::registerOperator(std::make_unique<ShuffleWriteTranslator>());

  auto data = makeRowVector({
      makeFlatVector<int32_t>(1'000, [](auto row) { return row; }),
      makeFlatVector<int64_t>(1'000, [](auto row) { return row * 10; }),
  });

  auto plan =
      PlanBuilder()
          .values({data}, true)
          .addNode(
              [](core::PlanNodeId nodeId,
                 core::PlanNodePtr source) -> core::PlanNodePtr {
                auto outputType = ROW({"p", "d"}, {INTEGER(), VARBINARY()});

                std::vector<core::TypedExprPtr> keys;
                keys.push_back(std::make_shared<core::FieldAccessTypedExpr>(
                    INTEGER(), "c0"));

                return std::make_shared<PartitionAndSerializeNode>(
                    nodeId, keys, 4, outputType, std::move(source));
              })
          .planNode();

  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 2;

  auto [taskCursor, serializedResults] =
      readCursor(params, [](auto /*task*/) {});
  ASSERT_EQ(serializedResults.size(), 2);

  for (auto& serializedResult : serializedResults) {
    // Print out partition numbers.
    std::cout << serializedResult->childAt(0)->toString(0, 100) << std::endl;

    // Verify that serialized data can be deserialized successfully into the
    // original data.
    auto deserialized = deserialize(serializedResult, asRowType(data->type()));
    velox::test::assertEqualVectors(data, deserialized);
  }
}
} // namespace facebook::velox::exec::test
