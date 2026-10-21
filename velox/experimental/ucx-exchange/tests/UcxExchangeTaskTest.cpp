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

// Two-fragment plans over the UCX transport, run through real exec::Task
// instances. The cases in UcxExchangeTest.cpp drive UcxPartitionedOutput and
// UcxExchange from SourceDriverMock/SinkDriverMock; these go through Task and
// LocalPlanner, so both operators are resolved from the transport registries.

#include <cudf/utilities/memory_resource.hpp>
#include <folly/Synchronized.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "velox/common/memory/MemoryPool.h"
#include "velox/core/QueryConfig.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/ExchangeTransportRegistry.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Task.h"
#include "velox/exec/TaskStats.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/tests/UcxExchangeTestBase.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::ucx_exchange {

// Each case registers cuDF, and with it both UCX transports, for its own
// duration, through CudfRegistration.
class UcxExchangeTaskTest : public UcxExchangeTestBase {};

namespace {

// Row type for the real-Task cases below. Two fixed-width columns keep the
// host/device round trip cheap while still exercising a multi-column table.
const RowTypePtr& taskShuffleRowType() {
  static const RowTypePtr rowType = ROW({{"c0", INTEGER()}, {"c1", BIGINT()}});
  return rowType;
}

// Serde named on both plan nodes of the two-fragment plans. The UCX transport
// ignores it because it ships packed cudf tables, but PlanBuilder requires a
// registered name on the nodes.
const std::string& taskShuffleSerdeKind() {
  static const std::string kind =
      VectorSerde::kindName(VectorSerde::Kind::kPresto);
  return kind;
}

// Registers what a serialized-page transport needs. OperatorTestBase does this
// for the tests derived from it; this suite has its own fixture.
void registerSerdes() {
  if (!isRegisteredVectorSerde()) {
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde(taskShuffleSerdeKind())) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
}

// Drives every Task built below. QueryCtx keeps a raw pointer to the executor,
// so it has to outlive the Tasks. It is deliberately never destroyed: folly
// joins its threads in the destructor, and doing that at static destruction
// races with the communicator shutdown this binary performs at exit.
folly::CPUThreadPoolExecutor* taskShuffleExecutor() {
  static auto* executor = new folly::CPUThreadPoolExecutor(8);
  return executor;
}

std::shared_ptr<exec::Task> makeTaskShuffleTask(
    const std::string& taskId,
    core::PlanFragment planFragment,
    exec::Consumer consumer) {
  auto queryCtx = core::QueryCtx::create(
      taskShuffleExecutor(),
      core::QueryConfig(std::unordered_map<std::string, std::string>{}));
  return exec::Task::create(
      taskId,
      std::move(planFragment),
      /*destination=*/0,
      std::move(queryCtx),
      exec::Task::ExecutionMode::kParallel,
      std::move(consumer));
}

// Builds one deterministic batch. c0 runs 'firstValue', 'firstValue + stride',
// ... so callers can interleave the ranges of several producers, and c1 is
// derived from c0 so a column mix-up cannot go unnoticed.
RowVectorPtr makeTaskShuffleBatch(
    memory::MemoryPool* pool,
    int32_t firstValue,
    int32_t stride,
    vector_size_t numRowsInBatch) {
  auto keys =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), numRowsInBatch, pool);
  auto payload =
      BaseVector::create<FlatVector<int64_t>>(BIGINT(), numRowsInBatch, pool);
  for (vector_size_t i = 0; i < numRowsInBatch; ++i) {
    const int32_t value = firstValue + i * stride;
    keys->set(i, value);
    payload->set(i, static_cast<int64_t>(value) * 10);
  }
  return std::make_shared<RowVector>(
      pool,
      taskShuffleRowType(),
      BufferPtr(nullptr),
      numRowsInBatch,
      std::vector<VectorPtr>{keys, payload});
}

// Brings a batch the consumer saw into 'pool'. CallbackSinkAdapter declares it
// does not accept GPU input, so CompileState splices a CudfToVelox in front of
// the sink and the batches arrive on the host. The CudfVector branch only keeps
// a run without that splice from crashing; the cases that care assert the
// consumer's pipeline shape, which is what catches it.
// Either way the result is a fresh vector owned by 'pool': what the consumer
// hands over is allocated from its operator pools, which report a leak and fail
// the arbitrator's reservation check if anything still references them when the
// Task is destroyed.
RowVectorPtr toHost(const RowVectorPtr& batch, memory::MemoryPool* pool) {
  auto cudfVector = std::dynamic_pointer_cast<cudf_velox::CudfVector>(batch);
  if (cudfVector != nullptr) {
    auto stream = cudfVector->stream();
    auto host = cudf_velox::with_arrow::toVeloxColumn(
        cudfVector->getTableView(),
        pool,
        taskShuffleRowType(),
        "",
        stream,
        cudf::get_current_device_resource_ref());
    stream.sync();
    return host;
  }
  auto copy =
      BaseVector::create<RowVector>(taskShuffleRowType(), batch->size(), pool);
  copy->copy(batch.get(), 0, 0, batch->size());
  return copy;
}

// Collects a consumer Task's output. The Consumer callback runs on driver
// threads, so the batches need their own lock. The callback holds the store
// rather than 'this': a Task can outlive a case that failed early.
class TaskShuffleResults {
 public:
  exec::Consumer consumer() {
    return
        [batches = batches_](
            RowVectorPtr batch, bool /*drained*/, ContinueFuture* /*future*/) {
          if (batch != nullptr && batch->size() > 0) {
            batches->wlock()->push_back(std::move(batch));
          }
          return exec::BlockingReason::kNotBlocked;
        };
  }

  std::vector<RowVectorPtr> batches() const {
    return batches_->copy();
  }

  void clear() {
    batches_->wlock()->clear();
  }

 private:
  const std::shared_ptr<folly::Synchronized<std::vector<RowVectorPtr>>>
      batches_{
          std::make_shared<folly::Synchronized<std::vector<RowVectorPtr>>>()};
};

// The UCX round trip goes through the communicator's event loop, so allow well
// over waitForTaskCompletion's 1s default.
constexpr uint64_t kTaskShuffleMaxWaitMicros = 60'000'000;

// Cancels the Tasks a case started, and waits for their drivers to stop, when
// it goes out of scope. A case that bails out early -- a failed ASSERT or
// VELOX_CHECK -- would otherwise leave producers blocked in the process-wide
// UcxOutputQueueManager for every later case, running over cuDF state that
// CudfRegistration has already torn down. Then it drops 'results', whose
// vectors come from the Tasks' operator pools, before the Tasks themselves go.
// The wait is bounded: a Task that does not stop in time is reported as a test
// failure rather than hanging the binary, which would hide which case failed.
// Declare it after the CudfRegistration and TaskShuffleResults it protects.
class StartedTasks {
 public:
  explicit StartedTasks(TaskShuffleResults* results) : results_(results) {}

  StartedTasks(const StartedTasks&) = delete;
  StartedTasks& operator=(const StartedTasks&) = delete;

  ~StartedTasks() {
    for (const auto& task : tasks_) {
      if (!task->requestCancel().wait(
              std::chrono::microseconds(kTaskShuffleMaxWaitMicros))) {
        ADD_FAILURE() << "Task did not stop after cancellation: "
                      << task->taskId();
      }
    }
    if (results_ != nullptr) {
      results_->clear();
    }
  }

  void add(std::shared_ptr<exec::Task> task) {
    tasks_.push_back(std::move(task));
  }

  void add(const std::vector<std::shared_ptr<exec::Task>>& tasks) {
    tasks_.insert(tasks_.end(), tasks.begin(), tasks.end());
  }

 private:
  TaskShuffleResults* const results_;
  std::vector<std::shared_ptr<exec::Task>> tasks_;
};

// Pairs registerCudf() with unregisterCudf() so the cuDF driver adapter and the
// transport registrations it installs do not outlive the case even if the body
// throws. unregisterCudf() empties output_mr_, which main() seeded for the
// whole binary, so the previous value is put back for the cases that run later
// and still allocate through get_output_mr().
class CudfRegistration {
 public:
  CudfRegistration() : previousOutputMr_(cudf_velox::output_mr_) {
    cudf_velox::registerCudf();
  }

  ~CudfRegistration() {
    cudf_velox::unregisterCudf();
    cudf_velox::output_mr_ = std::move(previousOutputMr_);
  }

  CudfRegistration(const CudfRegistration&) = delete;
  CudfRegistration& operator=(const CudfRegistration&) = delete;

 private:
  std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
      previousOutputMr_;
};

// A producer Task and the batches its Values node emits.
struct TaskShuffleProducer {
  std::string taskId;
  std::vector<RowVectorPtr> batches;
};

// Sums the row counts of 'batches'.
vector_size_t totalRows(const std::vector<RowVectorPtr>& batches) {
  vector_size_t total = 0;
  for (const auto& batch : batches) {
    total += batch->size();
  }
  return total;
}

// Starts one 'Values -> PartitionedOutput' Task per entry in 'producers', all
// on 'transportKind'. The Tasks are returned rather than waited for: a producer
// only completes once whatever reads its output buffer has drained it.
std::vector<std::shared_ptr<exec::Task>> startTaskShuffleProducers(
    const std::string& transportKind,
    const std::vector<TaskShuffleProducer>& producers) {
  std::vector<std::shared_ptr<exec::Task>> producerTasks;
  for (const auto& producer : producers) {
    auto plan = exec::test::PlanBuilder()
                    .values(producer.batches)
                    .partitionedOutput(
                        /*keys=*/{},
                        /*numPartitions=*/1,
                        /*outputLayout=*/{},
                        taskShuffleSerdeKind(),
                        transportKind)
                    .planFragment();
    auto task = makeTaskShuffleTask(producer.taskId, std::move(plan), nullptr);
    task->start(1);
    producerTasks.push_back(std::move(task));
  }
  return producerTasks;
}

// The operator types of 'pipeline', in pipeline order. Read from TaskStats
// rather than from the Driver because TaskStats is the vector
// Driver::initializeOperatorStats() indexes by operatorId: a repeated id lands
// here as an overwritten entry plus a nameless leftover.
std::vector<std::string> operatorTypesOf(const exec::PipelineStats& pipeline) {
  std::vector<std::string> types;
  types.reserve(pipeline.operatorStats.size());
  for (const auto& stats : pipeline.operatorStats) {
    types.push_back(stats.operatorType);
  }
  return types;
}

// The operator ids of 'pipeline', in pipeline order.
std::vector<int32_t> operatorIdsOf(const exec::PipelineStats& pipeline) {
  std::vector<int32_t> ids;
  ids.reserve(pipeline.operatorStats.size());
  for (const auto& stats : pipeline.operatorStats) {
    ids.push_back(stats.operatorId);
  }
  return ids;
}

// The plan-node boundary each operator of 'pipeline' reported, in pipeline
// order.
std::vector<core::PlanNode::Boundary> planNodeBoundariesOf(
    const exec::PipelineStats& pipeline) {
  std::vector<core::PlanNode::Boundary> boundaries;
  boundaries.reserve(pipeline.operatorStats.size());
  for (const auto& stats : pipeline.operatorStats) {
    boundaries.push_back(stats.planNodeBoundary);
  }
  return boundaries;
}

// The c0 column of 'batches' flattened into one sequence, so that a single
// assertion can report the first ordering violation rather than one per row.
std::vector<int32_t> sortKeysOf(const std::vector<RowVectorPtr>& batches) {
  std::vector<int32_t> sortKeys;
  for (const auto& batch : batches) {
    auto* keyColumn = batch->childAt(0)->as<SimpleVector<int32_t>>();
    VELOX_CHECK_NOT_NULL(keyColumn, "Sort key column is not a flat INTEGER");
    for (vector_size_t i = 0; i < batch->size(); ++i) {
      VELOX_CHECK(!keyColumn->isNullAt(i), "Sort key is null at row: {}", i);
      sortKeys.push_back(keyColumn->valueAt(i));
    }
  }
  return sortKeys;
}

// Index of the first key that is not strictly greater than its predecessor, or
// 'sortKeys.size()' when the whole sequence ascends.
int64_t firstUnorderedIndex(const std::vector<int32_t>& sortKeys) {
  return std::distance(
      sortKeys.begin(),
      std::adjacent_find(
          sortKeys.begin(), sortKeys.end(), [](int32_t left, int32_t right) {
            return left >= right;
          }));
}

// Builds 'numProducers' producers whose c0 ranges interleave -- producer i
// emits i, i + numProducers, ... -- so that no producer's output is globally
// ordered on its own. The batches stay on the host: CompileState puts a
// CudfFromVelox between Values and UcxPartitionedOutput, see
// taskShuffleOverUcx.
std::vector<TaskShuffleProducer> makeInterleavedProducers(
    memory::MemoryPool* pool,
    const std::string& taskIdPrefix,
    int numProducers,
    vector_size_t numRowsPerProducer) {
  std::vector<TaskShuffleProducer> producers;
  producers.reserve(numProducers);
  for (int i = 0; i < numProducers; ++i) {
    producers.push_back(
        TaskShuffleProducer{
            taskIdPrefix + std::to_string(i),
            {makeTaskShuffleBatch(
                pool,
                /*firstValue=*/i,
                /*stride=*/numProducers,
                numRowsPerProducer)},
        });
  }
  return producers;
}

// The batches of every producer in 'producers', in producer order.
std::vector<RowVectorPtr> allBatchesOf(
    const std::vector<TaskShuffleProducer>& producers) {
  std::vector<RowVectorPtr> batches;
  for (const auto& producer : producers) {
    batches.insert(
        batches.end(), producer.batches.begin(), producer.batches.end());
  }
  return batches;
}

// Fails the current test unless the c0 column of 'batches' strictly ascends,
// reporting the first violation.
void expectGloballyOrdered(const std::vector<RowVectorPtr>& batches) {
  const auto sortKeys = sortKeysOf(batches);
  const auto firstUnordered = firstUnorderedIndex(sortKeys);
  EXPECT_EQ(firstUnordered, static_cast<int64_t>(sortKeys.size()))
      << "Merge over UCX produced unordered output at index: "
      << firstUnordered;
}

// Starts one 'Values -> PartitionedOutput' producer Task per entry in
// 'producers', all on 'transportKind', then runs 'consumerPlan' against them,
// feeding it one split per producer built by 'splitFor'. Returns the
// consumer's output in host memory. Starts the consumer with
// 'numConsumerDrivers' drivers; a MergeExchangeNode pipeline runs
// single-threaded regardless. Copies the consumer's stats into 'consumerStats'
// when it is not null, which is the only way to see the consumer's operator
// pipeline after its Task is gone.
std::vector<RowVectorPtr> runTaskShuffle(
    const std::string& transportKind,
    const std::vector<TaskShuffleProducer>& producers,
    const core::PlanFragment& consumerPlan,
    const core::PlanNodeId& consumerNodeId,
    const std::string& consumerTaskId,
    const std::function<exec::Split(const std::string&)>& splitFor,
    memory::MemoryPool* pool,
    uint32_t numConsumerDrivers,
    exec::TaskStats* consumerStats) {
  TaskShuffleResults results;
  StartedTasks startedTasks(&results);
  auto producerTasks = startTaskShuffleProducers(transportKind, producers);
  startedTasks.add(producerTasks);
  auto consumerTask =
      makeTaskShuffleTask(consumerTaskId, consumerPlan, results.consumer());
  startedTasks.add(consumerTask);
  consumerTask->start(numConsumerDrivers);
  for (const auto& producer : producers) {
    consumerTask->addSplit(consumerNodeId, splitFor(producer.taskId));
  }
  consumerTask->noMoreSplits(consumerNodeId);

  VELOX_CHECK(
      exec::test::waitForTaskCompletion(
          consumerTask.get(), kTaskShuffleMaxWaitMicros),
      "Consumer task did not complete for transport: {}",
      transportKind);
  for (const auto& task : producerTasks) {
    VELOX_CHECK(
        exec::test::waitForTaskCompletion(
            task.get(), kTaskShuffleMaxWaitMicros),
        "Producer task did not complete: {}",
        task->taskId());
  }

  if (consumerStats != nullptr) {
    *consumerStats = consumerTask->taskStats();
  }

  std::vector<RowVectorPtr> hostBatches;
  for (const auto& batch : results.batches()) {
    hostBatches.push_back(toHost(batch, pool));
  }
  // Drop the consumer's own vectors before the Tasks go out of scope below.
  // They are allocated from operator pools that check for leaks on destruction,
  // and 'results' outlives 'consumerTask' by declaration order.
  results.clear();
  return hostBatches;
}

// Repeats the producer's one batch often enough that the producer cannot finish
// within a case; the case ends it by failing or cancelling it.
constexpr size_t kEndlessRepeatTimes = 1'000'000;

// Starts a 'Values -> PartitionedOutput' producer over UCX that keeps sending
// 512-row batches until it is stopped.
std::shared_ptr<exec::Task> startEndlessProducer(
    const std::string& taskId,
    memory::MemoryPool* pool) {
  auto plan = exec::test::PlanBuilder()
                  .values(
                      {makeTaskShuffleBatch(
                          pool,
                          /*firstValue=*/0,
                          /*stride=*/1,
                          /*numRowsInBatch=*/512)},
                      /*parallelizable=*/false,
                      kEndlessRepeatTimes)
                  .partitionedOutput(
                      /*keys=*/{},
                      /*numPartitions=*/1,
                      /*outputLayout=*/{},
                      taskShuffleSerdeKind(),
                      std::string{core::TransportKind::kUcx})
                  .planFragment();
  auto task = makeTaskShuffleTask(taskId, std::move(plan), nullptr);
  task->start(1);
  return task;
}

// Waits until the consumer behind 'results' received a batch. Returns false if
// none arrived within 'maxWait'.
bool waitForBatches(
    const TaskShuffleResults& results,
    std::chrono::microseconds maxWait) {
  const auto deadline = std::chrono::steady_clock::now() + maxWait;
  while (results.batches().empty()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return true;
}

// Selects intra-node or remote transfer for the exchanges a case starts, and
// restores the process-wide setting when it goes out of scope.
class ScopedIntraNodeExchange {
 public:
  explicit ScopedIntraNodeExchange(bool intraNode)
      : previous_(cudf_velox::CudfConfig::getInstance().intraNodeExchange) {
    cudf_velox::CudfConfig::getInstance().intraNodeExchange = intraNode;
  }

  ~ScopedIntraNodeExchange() {
    cudf_velox::CudfConfig::getInstance().intraNodeExchange = previous_;
  }

  ScopedIntraNodeExchange(const ScopedIntraNodeExchange&) = delete;
  ScopedIntraNodeExchange& operator=(const ScopedIntraNodeExchange&) = delete;

 private:
  const bool previous_;
};

// The plan of a consumer that reads one UCX exchange, optionally followed by
// a final LIMIT of 'limit' rows. Sets 'exchangeNodeId'.
core::PlanFragment ucxConsumerPlan(
    std::optional<int64_t> limit,
    core::PlanNodeId& exchangeNodeId) {
  auto builder = exec::test::PlanBuilder()
                     .exchange(
                         taskShuffleRowType(),
                         taskShuffleSerdeKind(),
                         std::string{core::TransportKind::kUcx})
                     .capturePlanNodeId(exchangeNodeId);
  if (limit.has_value()) {
    builder.limit(/*offset=*/0, *limit, /*isPartial=*/false);
  }
  return builder.planFragment();
}

} // namespace

// Runs a two-fragment plan over UCX through real Tasks. The exchange operator
// is resolved through ExchangeTransportRegistry and the output buffer through
// OutputTransportRegistry, both keyed by the transportKind on the plan nodes.
// It is the non-merge control for mergeExchangeOverUcxIsGloballyOrdered: the
// producers are the same, so if only the merge case fails, the merge expansion
// is at fault.
TEST_F(UcxExchangeTaskTest, taskShuffleOverUcx) {
  registerSerdes();
  const auto taskPrefix = getUniqueTaskPrefix();

  // registerCudf() is what seeds both transport registries, so it is a
  // prerequisite of the Task-level path and not just of the merge case.
  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  CudfRegistration cudfRegistration;

  // The batches stay on the host. UcxPartitionedOutput requires CudfVector
  // input, but UcxPartitionedOutputAdapter declares acceptsGpuInput(), so
  // CompileState splices a CudfFromVelox between Values and it. Staging the
  // batches on the device first makes that conversion operator call childAt()
  // on a childless CudfVector and the producer task dies with "Trying to access
  // non-existing child in RowVector". The producers are the same as in
  // mergeExchangeOverUcxIsGloballyOrdered, so that the two cases differ only in
  // plain versus merge exchange.
  constexpr vector_size_t kNumRowsPerProducer = 512;
  constexpr int kNumProducers = 2;
  const auto producers = makeInterleavedProducers(
      pool_.get(),
      taskPrefix + "ucxProducer",
      kNumProducers,
      kNumRowsPerProducer);
  const auto expected = allBatchesOf(producers);

  core::PlanNodeId exchangeNodeId;
  auto consumerPlan = exec::test::PlanBuilder()
                          .exchange(
                              taskShuffleRowType(),
                              taskShuffleSerdeKind(),
                              std::string{core::TransportKind::kUcx})
                          .capturePlanNodeId(exchangeNodeId)
                          .planFragment();

  // Two consumer drivers share the one UcxExchangeClient the factory builds
  // for them, which is what its numberOfConsumers forwarding is for.
  exec::TaskStats consumerStats;
  auto actual = runTaskShuffle(
      std::string{core::TransportKind::kUcx},
      producers,
      consumerPlan,
      exchangeNodeId,
      taskPrefix + "ucxConsumer",
      [this](const std::string& taskId) { return remoteSplit(taskId, 0); },
      pool_.get(),
      /*numConsumerDrivers=*/2,
      &consumerStats);

  EXPECT_EQ(totalRows(actual), kNumProducers * kNumRowsPerProducer);
  EXPECT_TRUE(exec::test::assertEqualResults(expected, actual));

  // The cuDF pass keeps the registry-built UcxExchange as a GPU source and
  // converts its output for the CPU sink. A missing adapter would leave the
  // CudfToVelox out, and CPU fallback would hide it from the row checks above.
  EXPECT_EQ(consumerStats.numTotalDrivers, 2);
  ASSERT_EQ(consumerStats.pipelineStats.size(), 1);
  EXPECT_THAT(
      operatorTypesOf(consumerStats.pipelineStats[0]),
      testing::ElementsAre("UcxExchange", "CudfToVelox", "CallbackSink"));
  // A plain exchange is its node's only operator and counts both sides; only
  // the merge expansion narrows it.
  const auto boundaries = planNodeBoundariesOf(consumerStats.pipelineStats[0]);
  ASSERT_FALSE(boundaries.empty());
  EXPECT_EQ(boundaries[0], core::PlanNode::Boundary::kBoth);
}

// A producer task that fails after it started sending must fail its consumer
// with the producer's reason. Ending the consumer's stream as complete would
// return a result with the rest of the producer's rows missing.
TEST_F(UcxExchangeTaskTest, producerFailureMidStreamFailsConsumer) {
  registerSerdes();
  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  CudfRegistration cudfRegistration;

  for (const bool intraNode : {false, true}) {
    SCOPED_TRACE(intraNode ? "intra-node" : "remote");
    ScopedIntraNodeExchange intraNodeExchange(intraNode);
    const auto taskPrefix = getUniqueTaskPrefix();
    TaskShuffleResults results;
    StartedTasks startedTasks(&results);

    const auto producerId = taskPrefix + "failingProducer";
    auto producer = startEndlessProducer(producerId, pool_.get());
    startedTasks.add(producer);

    core::PlanNodeId exchangeNodeId;
    auto consumer = makeTaskShuffleTask(
        taskPrefix + "consumer",
        ucxConsumerPlan(/*limit=*/std::nullopt, exchangeNodeId),
        results.consumer());
    startedTasks.add(consumer);
    consumer->start(1);
    consumer->addSplit(exchangeNodeId, remoteSplit(producerId, 0));
    consumer->noMoreSplits(exchangeNodeId);

    ASSERT_TRUE(waitForBatches(
        results, std::chrono::microseconds(kTaskShuffleMaxWaitMicros)));
    producer->setError("Injected producer failure");

    ASSERT_TRUE(
        exec::test::waitForTaskFailure(
            consumer.get(), kTaskShuffleMaxWaitMicros))
        << exec::taskStateString(consumer->state());
    EXPECT_THAT(
        consumer->errorMessage(),
        testing::AllOf(
            testing::HasSubstr("Producer failed before sending all its data"),
            testing::HasSubstr("Injected producer failure")));
  }
}

// A producer's error can be far larger than the end-of-stream message that
// carries its reason, for instance a cast error that quotes its whole input.
// The consumer must still fail with that reason, and the worker must survive:
// a message over kMaxMetaBufSize cannot be serialized on the Communicator
// thread.
TEST_F(UcxExchangeTaskTest, oversizedProducerErrorFailsConsumer) {
  registerSerdes();
  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  CudfRegistration cudfRegistration;
  ScopedIntraNodeExchange intraNodeExchange(/*intraNode=*/false);
  const auto taskPrefix = getUniqueTaskPrefix();
  TaskShuffleResults results;
  StartedTasks startedTasks(&results);

  const auto producerId = taskPrefix + "verboseProducer";
  auto producer = startEndlessProducer(producerId, pool_.get());
  startedTasks.add(producer);

  core::PlanNodeId exchangeNodeId;
  auto consumer = makeTaskShuffleTask(
      taskPrefix + "consumer",
      ucxConsumerPlan(/*limit=*/std::nullopt, exchangeNodeId),
      results.consumer());
  startedTasks.add(consumer);
  consumer->start(1);
  consumer->addSplit(exchangeNodeId, remoteSplit(producerId, 0));
  consumer->noMoreSplits(exchangeNodeId);

  ASSERT_TRUE(waitForBatches(
      results, std::chrono::microseconds(kTaskShuffleMaxWaitMicros)));
  producer->setError(
      "Injected producer failure: " + std::string(2'000'000, 'x'));

  ASSERT_TRUE(
      exec::test::waitForTaskFailure(consumer.get(), kTaskShuffleMaxWaitMicros))
      << exec::taskStateString(consumer->state());
  EXPECT_THAT(
      consumer->errorMessage(),
      testing::AllOf(
          testing::HasSubstr("Producer failed before sending all its data"),
          testing::HasSubstr("Injected producer failure")));
}

// A consumer whose producer already failed and was removed before the
// consumer connected must fail as well, rather than read an empty stream.
// Remote path only: Acceptor refuses intra-node transfer for a removed task.
TEST_F(UcxExchangeTaskTest, lateConsumerOfFailedProducerFails) {
  registerSerdes();
  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  CudfRegistration cudfRegistration;
  ScopedIntraNodeExchange intraNodeExchange(/*intraNode=*/false);
  const auto taskPrefix = getUniqueTaskPrefix();
  TaskShuffleResults results;
  StartedTasks startedTasks(&results);

  const auto producerId = taskPrefix + "failedProducer";
  auto producer = startEndlessProducer(producerId, pool_.get());
  startedTasks.add(producer);
  producer->setError("Injected producer failure");
  ASSERT_TRUE(
      exec::test::waitForTaskFailure(producer.get(), kTaskShuffleMaxWaitMicros))
      << exec::taskStateString(producer->state());

  core::PlanNodeId exchangeNodeId;
  auto consumer = makeTaskShuffleTask(
      taskPrefix + "lateConsumer",
      ucxConsumerPlan(/*limit=*/std::nullopt, exchangeNodeId),
      results.consumer());
  startedTasks.add(consumer);
  consumer->start(1);
  consumer->addSplit(exchangeNodeId, remoteSplit(producerId, 0));
  consumer->noMoreSplits(exchangeNodeId);

  ASSERT_TRUE(
      exec::test::waitForTaskFailure(consumer.get(), kTaskShuffleMaxWaitMicros))
      << exec::taskStateString(consumer->state());
  EXPECT_THAT(
      consumer->errorMessage(),
      testing::AllOf(
          testing::HasSubstr("Producer failed before sending all its data"),
          testing::HasSubstr("Injected producer failure")));
}

// A consumer that stops reading early, here after a LIMIT, has all it needs.
// It must finish successfully, and cancelling its producer afterwards, as the
// coordinator does, must leave the Communicator able to serve later cases.
TEST_F(UcxExchangeTaskTest, limitBeforeProducerFinishesStillSucceeds) {
  registerSerdes();
  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  CudfRegistration cudfRegistration;
  const auto taskPrefix = getUniqueTaskPrefix();
  TaskShuffleResults results;
  StartedTasks startedTasks(&results);

  const auto producerId = taskPrefix + "endlessProducer";
  auto producer = startEndlessProducer(producerId, pool_.get());
  startedTasks.add(producer);

  core::PlanNodeId exchangeNodeId;
  auto consumer = makeTaskShuffleTask(
      taskPrefix + "limitedConsumer",
      ucxConsumerPlan(/*limit=*/10, exchangeNodeId),
      results.consumer());
  startedTasks.add(consumer);
  consumer->start(1);
  consumer->addSplit(exchangeNodeId, remoteSplit(producerId, 0));
  consumer->noMoreSplits(exchangeNodeId);

  ASSERT_TRUE(
      exec::test::waitForTaskCompletion(
          consumer.get(), kTaskShuffleMaxWaitMicros))
      << exec::taskStateString(consumer->state()) << ": "
      << consumer->errorMessage();
  EXPECT_EQ(totalRows(results.batches()), 10);

  EXPECT_TRUE(producer->requestCancel().wait(
      std::chrono::microseconds(kTaskShuffleMaxWaitMicros)));
}

// A fragment whose output layout is empty ships rows that carry no columns --
// the build side of a cross join that projects nothing, for instance. A cuDF
// table derives num_rows() from its columns, so once such a payload is packed
// it can no longer report its own cardinality: the count only survives if it
// travels beside the data, in MetadataMsg on the remote path and in the
// registry entry on the intra-node one.
//
// The row count is the entire observable result here, which is the point. If
// UcxExchange derived the count from the packed table instead, it would rebuild
// a 0-row vector and the consumer task would die on "Operator::getOutput() must
// return nullptr or a non-empty vector"; a downstream global aggregation would
// turn that into a wrong scalar rather than an error -- SELECT count(*) over
// such a plan would return 0.
TEST_F(UcxExchangeTaskTest, zeroColumnPayloadKeepsItsRowCount) {
  registerSerdes();
  const auto taskPrefix = getUniqueTaskPrefix();

  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  CudfRegistration cudfRegistration;

  // Zero columns, non-zero rows, staged on the host so that CudfFromVelox takes
  // its zero-column path and hands UcxPartitionedOutput a CudfVector whose
  // size() is the only remaining record of the cardinality.
  const RowTypePtr zeroColumnRowType = ROW({});
  constexpr vector_size_t kRowsPerBatch = 125;
  constexpr int kNumBatches = 2;
  constexpr int64_t kExpectedRows =
      static_cast<int64_t>(kNumBatches) * kRowsPerBatch;

  std::vector<RowVectorPtr> batches;
  batches.reserve(kNumBatches);
  for (int i = 0; i < kNumBatches; ++i) {
    batches.push_back(
        std::make_shared<RowVector>(
            pool_.get(),
            zeroColumnRowType,
            BufferPtr(nullptr),
            kRowsPerBatch,
            std::vector<VectorPtr>{}));
  }

  core::PlanNodeId exchangeNodeId;
  auto consumerPlan = exec::test::PlanBuilder()
                          .exchange(
                              zeroColumnRowType,
                              taskShuffleSerdeKind(),
                              std::string{core::TransportKind::kUcx})
                          .capturePlanNodeId(exchangeNodeId)
                          .planFragment();

  const auto producerTaskId = taskPrefix + "zeroColumnProducer";
  // Not runTaskShuffle(): its toHost() converts through taskShuffleRowType(),
  // and there is nothing to convert here. The counts are read directly instead.
  TaskShuffleResults results;
  StartedTasks startedTasks(&results);
  auto producerTasks = startTaskShuffleProducers(
      std::string{core::TransportKind::kUcx}, {{producerTaskId, batches}});
  startedTasks.add(producerTasks);
  auto consumerTask = makeTaskShuffleTask(
      taskPrefix + "zeroColumnConsumer", consumerPlan, results.consumer());
  startedTasks.add(consumerTask);
  consumerTask->start(1);
  consumerTask->addSplit(exchangeNodeId, remoteSplit(producerTaskId, 0));
  consumerTask->noMoreSplits(exchangeNodeId);

  ASSERT_TRUE(
      exec::test::waitForTaskCompletion(
          consumerTask.get(), kTaskShuffleMaxWaitMicros))
      << "Consumer task did not complete";
  for (const auto& task : producerTasks) {
    ASSERT_TRUE(
        exec::test::waitForTaskCompletion(
            task.get(), kTaskShuffleMaxWaitMicros))
        << "Producer task did not complete: " << task->taskId();
  }

  int64_t receivedRows = 0;
  for (const auto& batch : results.batches()) {
    EXPECT_EQ(batch->type()->size(), 0) << "expected a column-less batch";
    receivedRows += batch->size();
  }
  // Drop the consumer's vectors before the Tasks go out of scope, for the
  // reason given in runTaskShuffle().
  results.clear();

  EXPECT_EQ(receivedRows, kExpectedRows);
}

// End-to-end check of the UCX merge path: a kUcx MergeExchangeNode runs as
// UcxExchange followed by CudfOrderBy, because UcxExchangeClient multiplexes
// every source into one queue and so destroys the per-source orderings
// exec::MergeExchange relies on. The transport builds only the exchange; the
// sort is spliced in behind it by the cuDF driver-adaptation pass, which is
// also what renumbers the operator ids.
//
// Two properties are asserted, and only two: global ordering, since nothing
// about merge internals, batch boundaries or per-source runs survives that
// substitution, and the shape of the resulting operator pipeline, since that is
// where the expansion and the renumbering are observable.
TEST_F(UcxExchangeTaskTest, mergeExchangeOverUcxIsGloballyOrdered) {
  registerSerdes();
  const auto taskPrefix = getUniqueTaskPrefix();

  // Start from an empty pair of registries so the assertions below prove that
  // registerCudf() -- the path a real worker takes with cudf.exchange enabled
  // -- is what seeds the kUcx transport on both sides of the edge.
  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  CudfRegistration cudfRegistration;
  auto exchangeEntry = exec::ExchangeTransportRegistry::tryGet(
      std::string{core::TransportKind::kUcx});
  ASSERT_NE(exchangeEntry, nullptr);
  // Without this, Task fails the MergeExchangeNode before any driver is built.
  ASSERT_NE(exchangeEntry->makeMergeExchangeOperator, nullptr);
  ASSERT_NE(
      exec::OutputTransportRegistry::tryGet(
          std::string{core::TransportKind::kUcx}),
      nullptr);

  // Two producers with interleaved key ranges: neither is globally ordered on
  // its own, so an ordered result can only come from ordering across both.
  // CudfOrderBy accumulates every batch on the device before sorting, so the
  // row counts stay small.
  constexpr vector_size_t kNumRowsPerProducer = 512;
  constexpr int kNumProducers = 2;
  const auto producers = makeInterleavedProducers(
      pool_.get(),
      taskPrefix + "mergeProducer",
      kNumProducers,
      kNumRowsPerProducer);
  const auto expected = allBatchesOf(producers);

  core::PlanNodeId mergeNodeId;
  auto consumerPlan = exec::test::PlanBuilder()
                          .mergeExchange(
                              taskShuffleRowType(),
                              {"c0"},
                              taskShuffleSerdeKind(),
                              std::string{core::TransportKind::kUcx})
                          .capturePlanNodeId(mergeNodeId)
                          .planFragment();

  exec::TaskStats consumerStats;
  auto actual = runTaskShuffle(
      std::string{core::TransportKind::kUcx},
      producers,
      consumerPlan,
      mergeNodeId,
      taskPrefix + "mergeConsumer",
      [this](const std::string& taskId) { return remoteSplit(taskId, 0); },
      pool_.get(),
      /*numConsumerDrivers=*/1,
      &consumerStats);

  ASSERT_EQ(totalRows(actual), kNumProducers * kNumRowsPerProducer);
  EXPECT_TRUE(exec::test::assertEqualResults(expected, actual));

  // One plan node became four operators: the exchange the transport built, the
  // sort the cuDF pass spliced in behind it, the conversion back to host
  // vectors that the CallbackSink needs, and the sink. The ids have to be
  // consecutive from zero -- Driver::initializeOperatorStats indexes the stats
  // by operatorId, so a repeated id would silently overwrite a slot and leave a
  // nameless one behind.
  ASSERT_EQ(consumerStats.pipelineStats.size(), 1);
  EXPECT_THAT(
      operatorTypesOf(consumerStats.pipelineStats[0]),
      testing::ElementsAre(
          "UcxExchange", "CudfOrderBy", "CudfToVelox", "CallbackSink"));
  EXPECT_THAT(
      operatorIdsOf(consumerStats.pipelineStats[0]),
      testing::ElementsAre(0, 1, 2, 3));

  // UcxExchange and CudfOrderBy share the merge node's id, so the exchange
  // reports only the node's input and the sort only its output
  // (Operator::planNodeBoundary()).
  const auto boundaries = planNodeBoundariesOf(consumerStats.pipelineStats[0]);
  ASSERT_GE(boundaries.size(), 2);
  EXPECT_EQ(boundaries[0], core::PlanNode::Boundary::kInput);
  EXPECT_EQ(boundaries[1], core::PlanNode::Boundary::kOutput);

  expectGloballyOrdered(actual);
}

// The same merge expansion as above, but in the driver shape where the operator
// ids are not repaired by anything else: every operator of the pipeline runs on
// the GPU, so the cuDF pass splices in no conversion operator and the merge
// expansion is the only replacement made.
//
// That distinction is the whole point of the case.
// DriverFactory::replaceOperators renumbers the entire driver, so a single
// CudfToVelox in front of a CPU sink is enough to make the ids come out
// consecutive whatever produced them, and
// mergeExchangeOverUcxIsGloballyOrdered, which ends in a CallbackSink, cannot
// see a repeated id. This pipeline ends in UcxPartitionedOutput, which consumes
// CudfVectors, so nothing else is spliced and the ids are whatever the merge
// expansion produced: [0, 1, 2] when it goes through replaceOperators.
//
// Three Tasks are needed to reach that shape: the two producers feed the merge,
// and a third Task drains the sorted output so the middle Task can finish and
// report its stats.
TEST_F(UcxExchangeTaskTest, mergeExchangeOverUcxNumbersOperatorsConsecutively) {
  registerSerdes();
  const auto taskPrefix = getUniqueTaskPrefix();
  const auto ucx = std::string{core::TransportKind::kUcx};

  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  CudfRegistration cudfRegistration;

  // Interleaved key ranges again, so the output cannot be ordered by accident.
  constexpr vector_size_t kNumRowsPerProducer = 512;
  constexpr int kNumProducers = 2;
  const auto producers = makeInterleavedProducers(
      pool_.get(),
      taskPrefix + "gpuMergeProducer",
      kNumProducers,
      kNumRowsPerProducer);
  const auto expected = allBatchesOf(producers);
  TaskShuffleResults results;
  StartedTasks startedTasks(&results);
  auto producerTasks = startTaskShuffleProducers(ucx, producers);
  startedTasks.add(producerTasks);

  // The Task under test: a kUcx merge exchange straight into a kUcx partitioned
  // output, which is the all-GPU driver.
  core::PlanNodeId mergeNodeId;
  auto sorterPlan =
      exec::test::PlanBuilder()
          .mergeExchange(
              taskShuffleRowType(), {"c0"}, taskShuffleSerdeKind(), ucx)
          .capturePlanNodeId(mergeNodeId)
          .partitionedOutput(
              /*keys=*/{},
              /*numPartitions=*/1,
              /*outputLayout=*/{},
              taskShuffleSerdeKind(),
              ucx)
          .planFragment();
  const auto sorterTaskId = taskPrefix + "gpuMergeSorter";
  auto sorterTask = makeTaskShuffleTask(sorterTaskId, sorterPlan, nullptr);
  startedTasks.add(sorterTask);
  sorterTask->start(1);
  for (const auto& producer : producers) {
    sorterTask->addSplit(mergeNodeId, remoteSplit(producer.taskId, 0));
  }
  sorterTask->noMoreSplits(mergeNodeId);

  // A plain exchange to drain the sorted output. Its own pipeline is not under
  // test; it exists so the sorter's output buffer is consumed.
  core::PlanNodeId drainNodeId;
  auto drainPlan =
      exec::test::PlanBuilder()
          .exchange(taskShuffleRowType(), taskShuffleSerdeKind(), ucx)
          .capturePlanNodeId(drainNodeId)
          .planFragment();
  auto drainTask = makeTaskShuffleTask(
      taskPrefix + "gpuMergeDrain", drainPlan, results.consumer());
  startedTasks.add(drainTask);
  drainTask->start(1);
  drainTask->addSplit(drainNodeId, remoteSplit(sorterTaskId, 0));
  drainTask->noMoreSplits(drainNodeId);

  ASSERT_TRUE(
      exec::test::waitForTaskCompletion(
          drainTask.get(), kTaskShuffleMaxWaitMicros))
      << "Drain task did not complete";
  ASSERT_TRUE(
      exec::test::waitForTaskCompletion(
          sorterTask.get(), kTaskShuffleMaxWaitMicros))
      << "Sorter task did not complete";
  for (const auto& task : producerTasks) {
    ASSERT_TRUE(
        exec::test::waitForTaskCompletion(
            task.get(), kTaskShuffleMaxWaitMicros))
        << "Producer task did not complete: " << task->taskId();
  }

  const auto sorterStats = sorterTask->taskStats();

  std::vector<RowVectorPtr> actual;
  for (const auto& batch : results.batches()) {
    actual.push_back(toHost(batch, pool_.get()));
  }
  // The consumer's vectors come from its operator pools, which check for leaks
  // when the Task is destroyed; 'results' outlives 'drainTask' by declaration
  // order, so let them go here. See runTaskShuffle.
  results.clear();

  ASSERT_EQ(sorterStats.pipelineStats.size(), 1);
  EXPECT_THAT(
      operatorTypesOf(sorterStats.pipelineStats[0]),
      testing::ElementsAre(
          "UcxExchange", "CudfOrderBy", "cudfPartitionedOutput"));
  EXPECT_THAT(
      operatorIdsOf(sorterStats.pipelineStats[0]),
      testing::ElementsAre(0, 1, 2));

  // Nothing but the merge expansion shares the merge node's id here, so its
  // totals are UcxExchange's input and CudfOrderBy's output, each counted once.
  const auto planStats = exec::toPlanStats(sorterStats);
  const auto& mergeStats = planStats.at(mergeNodeId);
  const uint64_t numMergedRows = kNumProducers * kNumRowsPerProducer;
  EXPECT_EQ(mergeStats.inputRows, numMergedRows);
  EXPECT_EQ(mergeStats.outputRows, numMergedRows);

  // The consequence of a repeated id, asserted directly: stats.resize() default
  // constructs OperatorStats(0, 0, "", ""), and every slot has to be claimed by
  // an operator afterwards.
  for (const auto& pipeline : sorterStats.pipelineStats) {
    for (const auto& stats : pipeline.operatorStats) {
      EXPECT_FALSE(stats.operatorType.empty())
          << "Operator stats slot left unwritten at operatorId: "
          << stats.operatorId << ", planNodeId: " << stats.planNodeId;
    }
  }

  // The sort still has to be a sort, or the assertions above would be happy
  // with a pipeline that merely has the right shape.
  ASSERT_EQ(totalRows(actual), kNumProducers * kNumRowsPerProducer);
  EXPECT_TRUE(exec::test::assertEqualResults(expected, actual));
  expectGloballyOrdered(actual);
}

} // namespace facebook::velox::ucx_exchange
