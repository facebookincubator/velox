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

#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

#include "velox/common/memory/MemoryPool.h"
#include "velox/exec/Driver.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/RoundRobinPartitionFunction.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/tests/UcxExchangeTestBase.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestHelpers.h"

namespace facebook::velox::ucx_exchange {
namespace {

class UcxPartitionedOutputFairnessTest
    : public testing::TestWithParam<std::tuple<int, bool>> {
 protected:
  static constexpr int64_t kWindow = 1 << 20;
  static constexpr int32_t kRows = 4096;
  const RowTypePtr rowType_{ROW({"c0"}, {INTEGER()})};

  struct FetchedPage {
    bool called{false};
    std::shared_ptr<UcxOutputQueue> queue;
    std::shared_ptr<cudf::packed_columns> payload;
    vector_size_t rows{0};
  };

  void SetUp() override {
    std::tie(numPartitions_, hashPartitioned_) = GetParam();
    auto& config = cudf_velox::CudfConfig::getInstance();
    previousExchange_ = config.exchange;
    config.exchange = true;
    pool_ = memory::memoryManager()->addLeafPool();
    manager_ = UcxOutputQueueManager::getInstanceRef();
    task_ = createPartitionedOutputTask(
        taskId_,
        pool_,
        rowType_,
        numPartitions_,
        hashPartitioned_ ? std::vector<std::string>{"c0"}
                         : std::vector<std::string>{},
        kWindow,
        {{cudf_velox::CudfConfig::kUcxPartitionedOutputBatchRows, "1"}},
        false,
        hashPartitioned_
            ? nullptr
            : std::make_shared<exec::RoundRobinPartitionFunctionSpec>());
    manager_->initializeTask(
        task_,
        core::PartitionedOutputNode::Kind::kPartitioned,
        numPartitions_,
        1);
    context_ = std::make_shared<exec::DriverCtx>(
        task_, 0, 0, exec::kUngroupedGroupId, 0);
    auto node = std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
        task_->planFragment().planNode);
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->kind(), core::PartitionedOutputNode::Kind::kPartitioned);
    // Empty keys normally select Gather. Explicitly require the equal/hash
    // paths so every destination has pending rows in this regression.
    if (hashPartitioned_) {
      ASSERT_NE(
          dynamic_cast<const exec::HashPartitionFunctionSpec*>(
              &node->partitionFunctionSpec()),
          nullptr);
      ASSERT_EQ(node->keys().size(), 1);
    } else {
      ASSERT_NE(
          dynamic_cast<const exec::RoundRobinPartitionFunctionSpec*>(
              &node->partitionFunctionSpec()),
          nullptr);
    }
    op_ = std::make_unique<UcxPartitionedOutput>(
        0, context_.get(), node, /*eagerFlush=*/true, manager_);
  }

  void TearDown() override {
    if (op_) {
      op_->close();
    }
    op_.reset();
    if (manager_) {
      for (int destination = 0; destination < numPartitions_; ++destination) {
        manager_->deleteResults(taskId_, destination);
      }
      manager_->removeTask(taskId_);
    }
    cudf_velox::CudfConfig::getInstance().exchange = previousExchange_;
  }

  std::unique_ptr<cudf::table> makeTable(int32_t rows, int32_t start) {
    std::vector<int32_t> values(rows);
    std::iota(values.begin(), values.end(), start);
    auto column = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT32},
        rows,
        cudf::mask_state::UNALLOCATED,
        stream_);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        column->mutable_view().data<int32_t>(),
        values.data(),
        values.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream_.get()));
    stream_.sync();
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(column));
    return std::make_unique<cudf::table>(std::move(columns));
  }

  std::shared_ptr<FetchedPage> fetch(int destination) {
    auto result = std::make_shared<FetchedPage>();
    manager_->getDataWithQueue(
        taskId_,
        destination,
        [result](
            std::shared_ptr<UcxOutputQueue> queue,
            std::shared_ptr<cudf::packed_columns> payload,
            vector_size_t rows,
            std::vector<int64_t>) {
          result->called = true;
          result->queue = std::move(queue);
          result->payload = std::move(payload);
          result->rows = rows;
        });
    return result;
  }

  void complete(int destination, const std::shared_ptr<FetchedPage>& page) {
    const auto bytes = page->payload->gpu_data->size();
    page->payload.reset();
    stream_.sync();
    page->queue->releaseInFlightBytes(destination, bytes, 1);
  }

  void appendRows(
      int destination,
      const std::shared_ptr<FetchedPage>& page,
      std::vector<int32_t>& rows) {
    ASSERT_TRUE(page->called);
    ASSERT_NE(page->payload, nullptr);
    ASSERT_GT(page->rows, 0);
    if (!hashPartitioned_) {
      ASSERT_EQ(page->rows, kRows / numPartitions_);
    }
    ASSERT_LE(page->payload->gpu_data->size(), kRows * sizeof(int32_t));
    const auto view = cudf::unpack(*page->payload);
    auto values = getColVector<int32_t>(view.column(0), page->rows, stream_);
    rows.insert(rows.end(), values.begin(), values.end());
    complete(destination, page);
  }

  void blockProducer(ContinueFuture& blocked) {
    // Model retained output from another producer, without exhausting VRAM.
    // Public congestion accounting selects the chunk-drain path.
    for (int destination = 0; destination < numPartitions_; ++destination) {
      auto retained = makeTable(kWindow / sizeof(int32_t), -1'000'000);
      auto packed = std::make_unique<cudf::packed_columns>(
          cudf::pack(retained->view(), stream_));
      stream_.sync();
      ASSERT_EQ(packed->gpu_data->size(), kWindow);
      manager_->enqueue(
          taskId_, destination, std::move(packed), retained->num_rows());
    }
    manager_->recordFullTransferCongestion(taskId_);

    auto input = std::make_shared<cudf_velox::CudfVector>(
        op_->pool(), rowType_, kRows, makeTable(kRows, 0), stream_);
    op_->addInput(input);
    ASSERT_EQ(op_->isBlocked(&blocked), exec::BlockingReason::kWaitForConsumer);
    ASSERT_TRUE(blocked.valid());
    ASSERT_FALSE(blocked.isReady());
    for (int destination = 0; destination < numPartitions_; ++destination) {
      ASSERT_TRUE(manager_->checkTransferCapacity(
          taskId_, destination, kWindow, nullptr));
    }
    ASSERT_EQ(manager_->stats(taskId_)->bufferedPages, numPartitions_);
  }

  void verifyProgressWhileOtherDestinationsAreFull(int releasedDestination) {
    ContinueFuture blocked;
    ASSERT_NO_FATAL_FAILURE(blockProducer(blocked));
    auto previousOutput = fetch(releasedDestination);
    ASSERT_TRUE(previousOutput->called);
    ASSERT_NE(previousOutput->payload, nullptr);
    ASSERT_EQ(previousOutput->rows, kWindow / sizeof(int32_t));
    // Dequeue transfers ownership to the send; it does not return capacity.
    EXPECT_FALSE(blocked.isReady());
    complete(releasedDestination, previousOutput);
    const bool resumedByCompletion = blocked.isReady();

    // Verify actual admission, not just an empty queue: the aggregate budget
    // can hold even the entire small input while other destinations stay full.
    constexpr int64_t kPendingBytes = kRows * sizeof(int32_t);
    ASSERT_FALSE(manager_->reserveTransferBytes(
        taskId_, releasedDestination, kPendingBytes, kWindow, nullptr));
    manager_->releaseTransferReservation(
        taskId_, releasedDestination, kPendingBytes);
    ASSERT_TRUE(resumedByCompletion);
    std::move(blocked).get();

    op_->getOutput();
    EXPECT_FALSE(op_->isFinished());
    for (int destination = 0; destination < numPartitions_; ++destination) {
      if (destination != releasedDestination) {
        ASSERT_TRUE(manager_->checkTransferCapacity(
            taskId_, destination, kWindow, nullptr));
      }
    }
    // The freed destination must publish its pending payload before any other
    // retained payload is released. Waking the driver alone is insufficient.
    std::vector<int32_t> received;
    ASSERT_NO_FATAL_FAILURE(
        appendRows(releasedDestination, fetch(releasedDestination), received));

    ContinueFuture pending;
    const auto reason = op_->isBlocked(&pending);
    ASSERT_EQ(reason, exec::BlockingReason::kWaitForConsumer);
    ASSERT_TRUE(pending.valid());
    for (int destination = 0; destination < numPartitions_; ++destination) {
      if (destination == releasedDestination) {
        continue;
      }
      auto previous = fetch(destination);
      ASSERT_TRUE(previous->called);
      ASSERT_NE(previous->payload, nullptr);
      ASSERT_EQ(previous->rows, kWindow / sizeof(int32_t));
      complete(destination, previous);
    }
    ASSERT_TRUE(pending.isReady());
    std::move(pending).get();
    op_->noMoreInput();
    for (int attempt = 0; attempt < 8 && !op_->isFinished(); ++attempt) {
      op_->getOutput();
    }
    ASSERT_TRUE(op_->isFinished());

    for (int destination = 0; destination < numPartitions_; ++destination) {
      if (destination != releasedDestination) {
        ASSERT_NO_FATAL_FAILURE(
            appendRows(destination, fetch(destination), received));
      }
      auto end = fetch(destination);
      ASSERT_TRUE(end->called);
      ASSERT_EQ(end->payload, nullptr);
    }
    std::sort(received.begin(), received.end());
    ASSERT_EQ(received.size(), kRows);
    for (int32_t row = 0; row < kRows; ++row) {
      ASSERT_EQ(received[row], row);
    }
    EXPECT_EQ(manager_->stats(taskId_)->bufferedBytes, 0);
  }

  const std::string taskId_{
      UcxExchangeTestEnvironment::nextTaskPrefix() + "fairness"};
  cuda::stream_ref stream_{cudaStream_t{cudaStreamDefault}};
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<UcxOutputQueueManager> manager_;
  std::shared_ptr<exec::Task> task_;
  std::shared_ptr<exec::DriverCtx> context_;
  std::unique_ptr<UcxPartitionedOutput> op_;
  int numPartitions_{0};
  bool hashPartitioned_{false};
  bool previousExchange_{false};
};

TEST_P(UcxPartitionedOutputFairnessTest, laterDestinationMakesProgress) {
  verifyProgressWhileOtherDestinationsAreFull(1);
}

TEST_P(UcxPartitionedOutputFairnessTest, firstDestinationMakesProgress) {
  verifyProgressWhileOtherDestinationsAreFull(0);
}

INSTANTIATE_TEST_SUITE_P(
    PartitionModes,
    UcxPartitionedOutputFairnessTest,
    testing::Combine(testing::Values(2, 4), testing::Bool()),
    [](const testing::TestParamInfo<std::tuple<int, bool>>& info) {
      return "N" + std::to_string(std::get<0>(info.param)) +
          (std::get<1>(info.param) ? "Hash" : "Equal");
    });

} // namespace
} // namespace facebook::velox::ucx_exchange
