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

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/MaterializedOutputBufferManager.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/vector/VectorStream.h"

using namespace facebook::velox;

namespace facebook::velox::exec::test {
namespace {

constexpr std::string_view kTransport{"materialized-test"};

struct RecordingSinkState {
  explicit RecordingSinkState(int32_t numPartitions, bool failFinish = false)
      : partitionData(numPartitions), failFinish(failFinish) {}

  std::vector<std::string> snapshotData() const {
    std::lock_guard<std::mutex> lock(mutex);
    return partitionData;
  }

  std::vector<int64_t> snapshotRowGroupSizes() const {
    std::lock_guard<std::mutex> lock(mutex);
    return rowGroupSizes;
  }

  mutable std::mutex mutex;
  std::vector<std::string> partitionData;
  std::vector<int64_t> rowGroupSizes;
  const bool failFinish;
  std::atomic_bool finished{false};
  std::atomic_bool aborted{false};
};

class RecordingExchangeSink : public ExchangeSink {
 public:
  explicit RecordingExchangeSink(std::shared_ptr<RecordingSinkState> state)
      : state_(std::move(state)) {}

  void append(int32_t partition, std::string_view data) override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->partitionData.at(partition).append(data);
  }

  void append(int32_t partition, std::unique_ptr<folly::IOBuf> data) override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto* current = data.get();
    do {
      state_->rowGroupSizes.push_back(current->length());
      current = current->next();
    } while (current != data.get());
    data->coalesce();
    state_->partitionData.at(partition).append(
        reinterpret_cast<const char*>(data->data()), data->length());
  }

  CommittedExchangeOutput finish() override {
    if (state_->failFinish) {
      VELOX_FAIL("Simulated sink finish failure");
    }
    state_->finished = true;
    CommittedExchangeOutput output;
    for (int32_t partition = 0;
         partition < static_cast<int32_t>(state_->partitionData.size());
         ++partition) {
      output.locations.emplace(
          partition, fmt::format("partition:{}", partition));
    }
    return output;
  }

  void abort() override {
    state_->aborted = true;
  }

  folly::F14FastMap<std::string, int64_t> stats() const override {
    return {{"sharedSinkCount", 7}};
  }

 private:
  const std::shared_ptr<RecordingSinkState> state_;
};

class MaterializedPartitionedOutputTest : public OperatorTestBase {
 protected:
  struct RunResult {
    std::shared_ptr<Task> task;
    std::shared_ptr<RecordingSinkState> sink;
  };

  RunResult runPlan(
      core::PlanNodePtr plan,
      int32_t numPartitions,
      int32_t numDrivers = 1,
      bool failFinish = false,
      bool expectFailure = false) {
    auto sink = std::make_shared<RecordingSinkState>(numPartitions, failFinish);
    auto manager = std::make_shared<MaterializedOutputBufferManager>(
        [sink](const std::string&, const std::string&, memory::MemoryPool*)
            -> std::shared_ptr<ExchangeSink> {
          return std::make_shared<RecordingExchangeSink>(sink);
        },
        64L << 20);
    auto registry = OutputTransportRegistry::create(nullptr);
    registry->insert(std::string(kTransport), manager->transportEntry());

    auto queryCtx = core::QueryCtx::create(driverExecutor_.get());
    queryCtx->setRegistry(OutputTransportRegistry::kRegistryKey, registry);
    auto task = Task::create(
        fmt::format("materialized-output-test-{}", nextTaskId_++),
        core::PlanFragment{std::move(plan)},
        0,
        std::move(queryCtx),
        Task::ExecutionMode::kParallel,
        Consumer{});
    task->start(numDrivers);
    if (expectFailure) {
      EXPECT_TRUE(waitForTaskFailure(task.get(), 10'000'000));
    } else {
      EXPECT_TRUE(waitForTaskCompletion(task.get(), 10'000'000));
    }
    return {std::move(task), std::move(sink)};
  }

  RunResult runExchangeWrite(
      const std::vector<RowVectorPtr>& data,
      int32_t numPartitions,
      int32_t numDrivers = 1,
      bool replicateNullsAndAny = false) {
    const std::vector<std::string> keys = numPartitions == 1
        ? std::vector<std::string>{}
        : std::vector<std::string>{"c0"};
    auto plan = PlanBuilder()
                    .values(data, true)
                    .partitionedOutput(
                        keys,
                        numPartitions,
                        replicateNullsAndAny,
                        {},
                        "CompactRow",
                        std::string(kTransport))
                    .planNode();
    return runPlan(std::move(plan), numPartitions, numDrivers);
  }

  std::vector<RowVectorPtr> deserialize(
      const std::shared_ptr<RecordingSinkState>& sink,
      const RowTypePtr& rowType) {
    std::vector<RowVectorPtr> results;
    for (auto& data : sink->snapshotData()) {
      if (data.empty()) {
        continue;
      }
      ByteRange range{
          reinterpret_cast<uint8_t*>(data.data()),
          static_cast<int32_t>(data.size()),
          0};
      BufferInputStream stream({range});
      RowVectorPtr result;
      getNamedVectorSerde("CompactRow")
          ->deserialize(&stream, pool(), rowType, &result, nullptr);
      results.push_back(std::move(result));
    }
    return results;
  }

  static inline std::atomic_int64_t nextTaskId_{0};
};

TEST_F(MaterializedPartitionedOutputTest, basicEndToEnd) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5, 6}),
      makeFlatVector<std::string>({"a", "bb", "ccc", "dddd", "eeeee", "f"}),
  });

  auto result = runExchangeWrite({data}, 4, 2);
  EXPECT_TRUE(result.sink->finished);
  assertEqualResults(
      {data, data}, deserialize(result.sink, asRowType(data->type())));
}

TEST_F(MaterializedPartitionedOutputTest, outputBufferDescription) {
  auto data = makeRowVector({makeFlatVector<int32_t>({1, 2, 3, 4})});
  auto sinkState = std::make_shared<RecordingSinkState>(4);
  auto manager = std::make_shared<MaterializedOutputBufferManager>(
      [sinkState](const std::string&, const std::string&, memory::MemoryPool*)
          -> std::shared_ptr<ExchangeSink> {
        return std::make_shared<RecordingExchangeSink>(sinkState);
      },
      64L << 20);
  auto queryCtx = core::QueryCtx::create(driverExecutor_.get());
  auto plan =
      PlanBuilder()
          .values({data})
          .partitionedOutput(
              {"c0"}, 4, false, {}, "CompactRow", std::string(kTransport))
          .planNode();
  auto task = Task::create(
      "materialized-output-description-test",
      core::PlanFragment{std::move(plan)},
      0,
      std::move(queryCtx),
      Task::ExecutionMode::kParallel,
      Consumer{});
  manager->initializeTask(
      task, core::PartitionedOutputNode::Kind::kPartitioned, 4, 2);

  const auto outputBufferDescription = manager->toString(task->taskId());

  EXPECT_NE(
      outputBufferDescription.find("partitionCount=4"), std::string::npos);
  EXPECT_NE(
      outputBufferDescription.find("maxBufferedBytes=67108864"),
      std::string::npos);
  EXPECT_NE(
      outputBufferDescription.find("partitionDrainThresholdBytes=131072"),
      std::string::npos);
  EXPECT_NE(
      outputBufferDescription.find("highWatermarkBytes=60397977"),
      std::string::npos);
  EXPECT_NE(
      outputBufferDescription.find("lowWatermarkBytes=46976204"),
      std::string::npos);
  EXPECT_NE(
      outputBufferDescription.find("outputBatchSizeBytes=1048576"),
      std::string::npos);
  EXPECT_NE(
      outputBufferDescription.find("minOutputBatchBytes=1048576"),
      std::string::npos);
  EXPECT_NE(
      outputBufferDescription.find("maxOutputBatchBytes=16777216"),
      std::string::npos);
  EXPECT_NE(
      outputBufferDescription.find("estimatedRowBytes=1024"),
      std::string::npos);

  manager->removeTask(task->taskId());
}

TEST_F(MaterializedPartitionedOutputTest, largeDataEndToEnd) {
  constexpr int32_t kNumRows = 10'000;
  auto data = makeRowVector({
      makeFlatVector<int32_t>(kNumRows, [](auto row) { return row; }),
      makeFlatVector<int64_t>(kNumRows, [](auto row) { return row * 100; }),
      makeFlatVector<std::string>(
          kNumRows, [](auto row) { return fmt::format("str_{}", row); }),
      makeFlatVector<double>(kNumRows, [](auto row) { return row * 1.5; }),
  });

  auto result = runExchangeWrite({data}, 8, 4);
  assertEqualResults(
      {data, data, data, data},
      deserialize(result.sink, asRowType(data->type())));
}

TEST_F(MaterializedPartitionedOutputTest, preservesManySmallInputBatches) {
  constexpr int32_t kNumBatches = 4'096;
  std::vector<RowVectorPtr> data;
  data.reserve(kNumBatches);
  for (int32_t batch = 0; batch < kNumBatches; ++batch) {
    data.push_back(makeRowVector({
        makeFlatVector<int64_t>({batch}),
        makeFlatVector<std::string>({fmt::format("Customer#{:09}", batch)}),
    }));
  }

  auto result = runExchangeWrite(data, 1);
  assertEqualResults(
      data, deserialize(result.sink, asRowType(data.front()->type())));
}

TEST_F(MaterializedPartitionedOutputTest, recordsSharedSinkStatsOnce) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4}),
      makeFlatVector<std::string>({"a", "b", "c", "d"}),
  });

  auto result = runExchangeWrite({data}, 2, 4);
  int64_t sharedSinkCount = 0;
  int32_t materializedOutputDrivers = 0;
  for (const auto& pipeline : result.task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      const auto stat = op.runtimeStats.find("sharedSinkCount");
      if (stat != op.runtimeStats.end()) {
        sharedSinkCount += stat->second.sum;
        materializedOutputDrivers += op.numDrivers;
      }
    }
  }

  EXPECT_EQ(materializedOutputDrivers, 4);
  EXPECT_EQ(sharedSinkCount, 7);
}

TEST_F(MaterializedPartitionedOutputTest, boundsRowGroupsForLargeInputBatch) {
  constexpr int32_t kNumRows = 128;
  constexpr int32_t kValueBytes = 8 * 1024;
  auto data = makeRowVector({
      makeFlatVector<int32_t>(kNumRows, [](auto) { return 0; }),
      makeFlatVector<std::string>(
          kNumRows,
          [](auto row) {
            return std::string(kValueBytes, static_cast<char>('a' + row % 26));
          }),
  });

  auto result = runExchangeWrite({data}, 1);
  const auto rowGroupSizes = result.sink->snapshotRowGroupSizes();
  ASSERT_GT(rowGroupSizes.size(), 1);
  for (const auto size : rowGroupSizes) {
    EXPECT_LE(size, MaterializedOutputBuffer::kDefaultDrainThreshold);
  }
  assertEqualResults({data}, deserialize(result.sink, asRowType(data->type())));
}

TEST_F(MaterializedPartitionedOutputTest, singlePartition) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({10, 20, 30}),
      makeFlatVector<std::string>({"x", "y", "z"}),
  });

  auto result = runExchangeWrite({data}, 1);
  assertEqualResults({data}, deserialize(result.sink, asRowType(data->type())));
}

TEST_F(MaterializedPartitionedOutputTest, manyPartitions) {
  constexpr int32_t kNumRows = 500;
  auto data = makeRowVector({
      makeFlatVector<int32_t>(kNumRows, [](auto row) { return row; }),
      makeFlatVector<int64_t>(kNumRows, [](auto row) { return row * 7; }),
  });

  auto result = runExchangeWrite({data}, 16);
  assertEqualResults({data}, deserialize(result.sink, asRowType(data->type())));
}

TEST_F(MaterializedPartitionedOutputTest, emptyInput) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>(0, [](auto row) { return row; }),
      makeFlatVector<std::string>(0, [](auto) { return ""; }),
  });

  auto result = runExchangeWrite({data}, 4);
  EXPECT_TRUE(result.sink->finished);
  EXPECT_TRUE(deserialize(result.sink, asRowType(data->type())).empty());
}

TEST_F(MaterializedPartitionedOutputTest, zeroColumnOutput) {
  auto data = makeRowVector({makeFlatVector<int32_t>({1, 2, 3, 4, 5, 6})});
  auto sourceType = asRowType(data->type());
  auto source = PlanBuilder().values({data}).planNode();
  std::vector<core::TypedExprPtr> keys{
      std::make_shared<core::FieldAccessTypedExpr>(
          sourceType->childAt(0), sourceType->nameOf(0))};
  auto plan = std::make_shared<core::PartitionedOutputNode>(
      "materialized-output",
      core::PartitionedOutputNode::Kind::kPartitioned,
      keys,
      4,
      false,
      std::make_shared<HashPartitionFunctionSpec>(
          sourceType, std::vector<column_index_t>{0}),
      ROW({}, {}),
      "CompactRow",
      std::string(kTransport),
      std::move(source));

  auto result = runPlan(std::move(plan), 4);
  auto rows = deserialize(result.sink, ROW({}, {}));
  int64_t numRows = 0;
  for (const auto& vector : rows) {
    EXPECT_EQ(vector->childrenSize(), 0);
    numRows += vector->size();
  }
  EXPECT_EQ(numRows, data->size());
}

TEST_F(MaterializedPartitionedOutputTest, replicateNullsAndAny) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>({1, 2, std::nullopt, 4, std::nullopt, 6}),
      makeFlatVector<std::string>({"a", "b", "c", "d", "e", "f"}),
  });

  auto result = runExchangeWrite({data}, 4, 1, true);
  std::map<std::string, int32_t> counts;
  for (const auto& vector : deserialize(result.sink, asRowType(data->type()))) {
    auto* values = vector->childAt(1)->as<SimpleVector<StringView>>();
    for (vector_size_t row = 0; row < vector->size(); ++row) {
      ++counts[values->valueAt(row).str()];
    }
  }

  EXPECT_EQ(counts["a"], 4);
  EXPECT_EQ(counts["c"], 4);
  EXPECT_EQ(counts["e"], 4);
  EXPECT_EQ(counts["b"], 1);
  EXPECT_EQ(counts["d"], 1);
  EXPECT_EQ(counts["f"], 1);
}

TEST_F(MaterializedPartitionedOutputTest, replicateNullsAndAnyDisabled) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>({1, 2, std::nullopt, 4, std::nullopt, 6}),
      makeFlatVector<std::string>({"a", "b", "c", "d", "e", "f"}),
  });

  auto result = runExchangeWrite({data}, 4, 1, false);
  std::map<std::string, int32_t> counts;
  for (const auto& vector : deserialize(result.sink, asRowType(data->type()))) {
    auto* values = vector->childAt(1)->as<SimpleVector<StringView>>();
    for (vector_size_t row = 0; row < vector->size(); ++row) {
      ++counts[values->valueAt(row).str()];
    }
  }

  for (const auto* value : {"a", "b", "c", "d", "e", "f"}) {
    EXPECT_EQ(counts[value], 1);
  }
}

TEST_F(MaterializedPartitionedOutputTest, propagatesSinkFinishFailure) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<std::string>({"a", "b", "c"}),
  });
  auto plan = PlanBuilder()
                  .values({data})
                  .partitionedOutput(
                      {"c0"}, 4, {}, "CompactRow", std::string(kTransport))
                  .planNode();

  auto result = runPlan(
      std::move(plan), 4, 1, /*failFinish=*/true, /*expectFailure=*/true);
  EXPECT_NE(
      result.task->errorMessage().find("Simulated sink finish failure"),
      std::string::npos);
  EXPECT_TRUE(result.sink->aborted);
}

} // namespace
} // namespace facebook::velox::exec::test
