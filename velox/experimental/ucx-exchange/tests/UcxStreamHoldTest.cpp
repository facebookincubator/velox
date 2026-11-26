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

// A receive runs on the Communicator thread, which drives every channel on the
// worker, so it must never wait for GPU work that a downstream operator queued
// on the stream of a table received earlier.

#include <cuda_runtime.h>
#include <cudf/contiguous_split.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <folly/ScopeGuard.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/ExchangeTransportRegistry.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TaskMonitor.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/tests/UcxExchangeTestBase.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestData.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestHelpers.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::ucx_exchange {
namespace {

// Blocks the stream of every vector it holds until open(), like a downstream
// operator whose kernels still run on its input's stream.
//
// No CUDA call may run under 'mutex_'. The host function that blocks a stream
// needs 'mutex_' to return, and a CUDA call can wait for that host function,
// for example while another thread's cudaFree synchronizes the device.
class StreamHoldGate {
 public:
  // Queues a host function on the stream of 'input' that blocks the stream
  // until open(). Keeps 'input' alive, so that nothing frees its memory while
  // the stream is blocked.
  void hold(const RowVectorPtr& input) {
    auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
    std::optional<std::string> error;
    if (cudfInput == nullptr) {
      error = "StreamHold input is not a CudfVector";
    } else {
      const auto status = cudaLaunchHostFunc(
          cudfInput->stream().get(), &StreamHoldGate::waitForOpen, this);
      if (status != cudaSuccess) {
        error = std::string("cudaLaunchHostFunc failed: ") +
            cudaGetErrorString(status);
      }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (error.has_value()) {
      errors_.push_back(std::move(*error));
    }
    held_.push_back(input);
    ++numHeld_;
    changed_.notify_all();
  }

  void open() {
    std::lock_guard<std::mutex> lock(mutex_);
    open_ = true;
    changed_.notify_all();
  }

  // Waits until 'numVectors' vectors were held. Returns false on timeout.
  bool waitForHeld(int numVectors, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(
        lock, timeout, [&] { return numHeld_ >= numVectors; });
  }

  int numHeld() {
    std::lock_guard<std::mutex> lock(mutex_);
    return numHeld_;
  }

  std::vector<std::string> errors() {
    std::lock_guard<std::mutex> lock(mutex_);
    return errors_;
  }

  // Frees the held vectors outside 'mutex_'.
  void releaseHeld() {
    std::vector<RowVectorPtr> held;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      held.swap(held_);
    }
  }

 private:
  // Runs on a CUDA host-function thread, so it must not call CUDA. The bound
  // lets the stream go on if a failing case never opens the gate.
  static void CUDART_CB waitForOpen(void* gate) {
    auto* self = static_cast<StreamHoldGate*>(gate);
    std::unique_lock<std::mutex> lock(self->mutex_);
    self->changed_.wait_for(
        lock, std::chrono::seconds(30), [self] { return self->open_; });
  }

  std::mutex mutex_;
  std::condition_variable changed_;
  bool open_{false};
  int numHeld_{0};
  std::vector<RowVectorPtr> held_;
  std::vector<std::string> errors_;
};

// Hands its input to a StreamHoldGate.
class StreamHoldNode : public core::PlanNode {
 public:
  StreamHoldNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      std::shared_ptr<StreamHoldGate> gate)
      : PlanNode(id), sources_{std::move(source)}, gate_{std::move(gate)} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "StreamHold";
  }

  const std::shared_ptr<StreamHoldGate>& gate() const {
    return gate_;
  }

 private:
  void addDetails(std::stringstream& /*stream*/) const override {}

  const std::vector<core::PlanNodePtr> sources_;
  const std::shared_ptr<StreamHoldGate> gate_;
};

// A CudfOperator, so that the cuDF driver adapter hands it the received
// CudfVectors as they are. It emits no rows: a CudfToVelox after it would sync
// on the held stream and block the consumer's driver.
class StreamHoldOperator : public exec::Operator,
                           public cudf_velox::CudfOperator {
 public:
  StreamHoldOperator(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const StreamHoldNode>& node)
      : exec::Operator(
            driverCtx,
            node->outputType(),
            operatorId,
            node->id(),
            "StreamHold"),
        cudf_velox::CudfOperator(operatorId, node->id()),
        gate_(node->gate()) {}

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override {
    gate_->hold(input);
  }

  RowVectorPtr getOutput() override {
    return nullptr;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_;
  }

 private:
  const std::shared_ptr<StreamHoldGate> gate_;
};

class StreamHoldTranslator : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator> toOperator(
      exec::DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override {
    if (auto holdNode = std::dynamic_pointer_cast<const StreamHoldNode>(node)) {
      return std::make_unique<StreamHoldOperator>(id, ctx, holdNode);
    }
    return nullptr;
  }
};

// Serde named on the exchange node. The UCX transport ships packed cuDF tables
// and ignores it, but PlanBuilder requires a registered name.
const std::string& serdeKind() {
  static const std::string kind =
      VectorSerde::kindName(VectorSerde::Kind::kPresto);
  return kind;
}

void registerSerdes() {
  if (!isRegisteredVectorSerde()) {
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde(serdeKind())) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
}

// Drives the consumer Task. QueryCtx keeps a raw pointer to the executor, so it
// has to outlive the Task. It is never destroyed: folly joins its threads in
// the destructor, and doing that at static destruction races with the
// communicator shutdown this binary performs at exit.
folly::CPUThreadPoolExecutor* consumerExecutor() {
  static auto* executor = new folly::CPUThreadPoolExecutor(4);
  return executor;
}

// Registers cuDF with 'memoryResource' for the scope of one case, and then puts
// back the memory resources and the setting it changed. registerCudf()
// replaces the current device resource and drops the previous one, so the
// constructor takes the previous one out first. Every allocation from cuDF's
// resource must be freed before the destructor runs.
class CudfRegistration {
 public:
  explicit CudfRegistration(const std::string& memoryResource)
      : previousMemoryResource_(
            cudf_velox::CudfConfig::getInstance().memoryResource),
        previousOutputMr_(cudf_velox::output_mr_),
        previousDeviceResource_(cudf::reset_current_device_resource()) {
    cudf_velox::CudfConfig::getInstance().memoryResource = memoryResource;
    cudf_velox::registerCudf();
  }

  ~CudfRegistration() {
    cudf_velox::unregisterCudf();
    cudf_velox::output_mr_ = std::move(previousOutputMr_);
    cudf::set_current_device_resource(std::move(previousDeviceResource_));
    cudf_velox::CudfConfig::getInstance().memoryResource =
        previousMemoryResource_;
  }

  CudfRegistration(const CudfRegistration&) = delete;
  CudfRegistration& operator=(const CudfRegistration&) = delete;

 private:
  const std::string previousMemoryResource_;
  std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
      previousOutputMr_;
  cuda::mr::any_resource<cuda::mr::device_accessible> previousDeviceResource_;
};

// Posts an empty action to the Communicator loop and returns how long it took
// to run, or nullopt if it did not run within 'timeout'.
std::optional<std::chrono::milliseconds> postedActionLatency(
    std::chrono::milliseconds timeout) {
  auto ran = std::make_shared<std::promise<void>>();
  auto future = ran->get_future();
  const auto start = std::chrono::steady_clock::now();
  Communicator::getInstance()->postToLoop([ran]() { ran->set_value(); });
  if (future.wait_for(timeout) != std::future_status::ready) {
    return std::nullopt;
  }
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
}

// Returns 'numChunks' packed tables of 'numRowsPerChunk' rows each, ready to
// enqueue.
std::vector<std::unique_ptr<cudf::packed_columns>> makeChunks(
    int numChunks,
    vector_size_t numRowsPerChunk) {
  std::vector<std::unique_ptr<cudf::packed_columns>> chunks;
  cuda::stream_ref stream{cudaStream_t{cudaStreamDefault}};
  for (int i = 0; i < numChunks; ++i) {
    auto table = makeTable(numRowsPerChunk, UcxTestData::kTestRowType, stream);
    chunks.push_back(
        std::make_unique<cudf::packed_columns>(
            cudf::pack(table->view(), stream)));
  }
  stream.sync();
  return chunks;
}

class UcxStreamHoldTest : public UcxExchangeTestBase {
 protected:
  static void SetUpTestSuite() {
    exec::Operator::registerOperator(std::make_unique<StreamHoldTranslator>());
  }
};

// Sends chunks one at a time to a consumer whose downstream operator blocks the
// stream of every chunk it receives. Each chunk must still arrive, and the
// Communicator loop must stay responsive, however many streams are blocked.
TEST_F(UcxStreamHoldTest, receivesProceedWhileDownstreamWorkHoldsStreams) {
  constexpr int kNumChunks{8};
  constexpr vector_size_t kNumRowsPerChunk{10'000};
  constexpr std::chrono::milliseconds kArrivalTimeout{1'000};
  constexpr std::chrono::milliseconds kPostedActionTimeout{200};

  registerSerdes();
  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  // The async resource frees on a stream. A plain cudaFree synchronizes the
  // device, so with the cuda resource the producer's free after each send would
  // wait for the blocked streams, whatever stream a receive uses.
  CudfRegistration cudfRegistration("async");

  // Packed before the consumer starts, so that the case's own CUDA work is done
  // before any stream is blocked.
  auto chunks = makeChunks(kNumChunks, kNumRowsPerChunk);

  const auto taskPrefix = getUniqueTaskPrefix();
  const auto producerId = taskPrefix + "producer";
  auto producer = makeProducerTask(producerId, pool_.get());
  auto gate = std::make_shared<StreamHoldGate>();
  std::shared_ptr<exec::Task> consumer;
  // Runs before 'cudfRegistration' restores the memory resources, so it frees
  // everything that was allocated from cuDF's resource. Opening the gate first
  // lets the blocked streams, and with them the Communicator thread, go on.
  SCOPE_EXIT {
    gate->open();
    queueManager()->removeTask(producerId);
    producer.reset();
    if (consumer != nullptr) {
      if (!consumer->requestCancel().wait(std::chrono::seconds(30))) {
        ADD_FAILURE() << "Task did not stop after cancellation: "
                      << consumer->taskId();
      }
      consumer.reset();
    }
    gate->releaseHeld();
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    exec::test::TaskMonitor::waitForAllTasksDeleted();
  };
  queueManager()->initializeTask(
      producer,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);

  core::PlanNodeId exchangeNodeId;
  auto plan = exec::test::PlanBuilder()
                  .exchange(
                      UcxTestData::kTestRowType,
                      serdeKind(),
                      std::string{core::TransportKind::kUcx})
                  .capturePlanNodeId(exchangeNodeId)
                  .addNode([gate](std::string id, core::PlanNodePtr source) {
                    return std::make_shared<StreamHoldNode>(
                        id, std::move(source), gate);
                  })
                  .planFragment();
  consumer = exec::Task::create(
      taskPrefix + "consumer",
      std::move(plan),
      /*destination=*/0,
      core::QueryCtx::create(
          consumerExecutor(),
          core::QueryConfig(std::unordered_map<std::string, std::string>{})),
      exec::Task::ExecutionMode::kParallel,
      [](RowVectorPtr /*batch*/, bool /*drained*/, ContinueFuture* /*future*/) {
        return exec::BlockingReason::kNotBlocked;
      });
  consumer->start(1);
  consumer->addSplit(exchangeNodeId, remoteSplit(producerId, 0));
  consumer->noMoreSplits(exchangeNodeId);

  for (int chunk = 1; chunk <= kNumChunks; ++chunk) {
    queueManager()->enqueue(
        producerId, 0, std::move(chunks[chunk - 1]), kNumRowsPerChunk);
    const bool arrived = gate->waitForHeld(chunk, kArrivalTimeout);
    const auto latency = postedActionLatency(kPostedActionTimeout);
    EXPECT_TRUE(latency.has_value())
        << "The Communicator did not run a posted action within "
        << kPostedActionTimeout.count() << " ms after chunk: " << chunk;
    if (!arrived) {
      // A chunk that arrives once the gate opens was held up by the blocked
      // streams; one that does not was lost for another reason.
      gate->open();
      const bool arrivedAfterOpen =
          gate->waitForHeld(chunk, std::chrono::seconds(10));
      FAIL() << "Chunk " << chunk << " did not arrive within "
             << kArrivalTimeout.count()
             << " ms while downstream work blocked the streams of earlier "
                "chunks. Once the gate opened, it "
             << (arrivedAfterOpen ? "arrived." : "was still missing.");
    }
  }

  gate->open();
  queueManager()->noMoreData(producerId);
  EXPECT_TRUE(exec::test::TaskMonitor::waitForCompletion(*consumer, 30'000'000))
      << exec::taskStateString(consumer->state());
  EXPECT_EQ(gate->numHeld(), kNumChunks);
  EXPECT_THAT(gate->errors(), testing::IsEmpty());
}

} // namespace
} // namespace facebook::velox::ucx_exchange
