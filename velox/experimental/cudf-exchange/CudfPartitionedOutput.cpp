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
#include "velox/experimental/cudf-exchange/CudfPartitionedOutput.h"
#include <fmt/format.h>
#include "velox/core/PlanNode.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>

using namespace facebook::velox::cudf_velox;
using facebook::velox::exec::Task;
namespace facebook::velox::cudf_exchange {

// Computes a mapping from names in n2 to names in n1
// and returns that mapping in remap.
// Names in n2 must occurs in n1.
static void getRemapping(
    const std::vector<std::string>& n1,
    const std::vector<std::string>& n2,
    std::vector<uint32_t>& remap) {
  std::unordered_map<std::string, uint32_t> lookup;
  for (uint32_t i = 0; i < n1.size(); ++i) {
    lookup[n1[i]] = i;
  }

  remap.clear();
  remap.reserve(n2.size());
  for (const auto& key : n2) {
    remap.push_back(lookup.at(key));
  }
}

CudfPartitionedOutput::CudfPartitionedOutput(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    bool eagerFlush)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "cudfPartitionedOutput"),
      NvtxHelper(
          nvtx3::rgb{255, 215, 0}, // Gold
          operatorId,
          fmt::format("[{}]", planNode->id())),
      queueManager_(CudfOutputQueueManager::getInstanceRef()),
      numPartitions_(planNode->numPartitions()),
      pipelineId_(ctx->pipelineId),
      driverId_(ctx->driverId) {
  this->initPartitionKeys(planNode);
  auto sources = planNode->sources();
  std::vector<std::string> inNames, outNames;
  inNames.reserve(planNode->inputType()->size());
  for (int i = 0; i < planNode->inputType()->size(); ++i) {
    inNames.push_back(planNode->inputType()->nameOf(i));
  }
  outNames.reserve(planNode->outputType()->size());
  for (int i = 0; i < planNode->outputType()->size(); ++i) {
    outNames.push_back(planNode->outputType()->nameOf(i));
  }
  if (inNames != outNames) {
    getRemapping(inNames, outNames, remap_);
  }
}

void CudfPartitionedOutput::addInput(RowVectorPtr input) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " addInput";
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK(cudfVector, "Input must be a CudfVector");
  VELOX_CHECK(
      !future_.valid() || future_.hasValue(),
      "addInput with outstanding future!");
  try {
    auto stream = cudfVector->stream();

    cudf::table_view tableView;
    bool blocked;
    if (remap_.empty()) {
      // input and output column order is the same.
      tableView = cudfVector->getTableView();
    } else {
      // input and output column order needs re-mapping.
      tableView =
          cudfVector->getTableView().select(remap_.begin(), remap_.end());
    }

    auto queueManager = sharedQueueManager();
    if (numPartitions_ > 1) {
      if (partitionKeyIndices_.size() > 0 || spec_ == "gather") {
        hashPartition(tableView, stream);
      } else {
        equalPartition(tableView, stream);
      }
    } else {
      // Single partition case. No need to hash, assume queue zero
      auto packedCols = cudf::pack(tableView, stream);
      // Sync the stream since UCXX/UCX is not stream oriented and without
      // syncing, data could get lost. Syncing here is  easy but notthe most
      // efficient. A better approach is to create an event and pass it along
      // the data through the queue and synchronize on the event before calling
      // into UCXX.
      // TODO: change stream sync and move to event sync
      // Thanks to Lawrence Mitchel for pointing this out!
      stream.synchronize();
      std::unique_ptr<cudf::packed_columns> packedColsPtr =
          std::make_unique<cudf::packed_columns>(
              std::move(packedCols.metadata), std::move(packedCols.gpu_data));
      queueManager->enqueue(
          this->taskId(), 0, std::move(packedColsPtr), tableView.num_rows());
    }
    // Check once after all enqueues if we're blocked
    blocked = queueManager->checkBlocked(this->taskId(), &future_);
    // record the statistics.
    {
      auto lockedStats = stats_.wlock();
      lockedStats->addOutputVector(input->estimateFlatSize(), input->size());
    }
    if (blocked) {
      VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
              << " is blocked, can no longer write to output!";
    }
    blockingReason_ = blocked ? exec::BlockingReason::kWaitForConsumer
                              : exec::BlockingReason::kNotBlocked;

  } catch (const rmm::bad_alloc& e) {
    VLOG(1)
        << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
        << " caught memory alloc error, removing all memory in output queues";
    for (int i = 0; i < numPartitions_; i++) {
      sharedQueueManager()->deleteResults(this->taskId(), i);
    }
    throw; // Let the driver know we have failed
  }
}

exec::BlockingReason CudfPartitionedOutput::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    blockingReason_ = exec::BlockingReason::kNotBlocked;
    return exec::BlockingReason::kWaitForConsumer;
  }
  return exec::BlockingReason::kNotBlocked;
}

RowVectorPtr CudfPartitionedOutput::getOutput() {
  if (finished_) {
    return nullptr;
  }
  if (noMoreInput_) {
    // Tell the queue manager there is nothing more to come
    sharedQueueManager()->noMoreData(this->taskId());
    finished_ = true;
  }
  return nullptr;
}

bool CudfPartitionedOutput::isFinished() {
  return finished_;
}

std::shared_ptr<facebook::velox::cudf_exchange::CudfOutputQueueManager>
CudfPartitionedOutput::sharedQueueManager() {
  auto shared_queueManager = queueManager_.lock();
  VELOX_CHECK_NOT_NULL(
      shared_queueManager, "OutputQueueManager was already destructed");
  return shared_queueManager;
}

void CudfPartitionedOutput::initPartitionKeys(
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode) {
  // Following Logic copied direcly from CudLocalPartition (!)

  // Following is IMO a hacky way to get the partition key indices. It is to
  // workaround the fact that the partition spec constructs the hash function
  // directly and has no public methods to get the partition key indices.

  // When the operator is of type kRepartition, the partition spec is a string
  // in the format "HASH(key1, key2, ...)"
  // We're going to extract the keys between HASH( and ) and find their indices
  // in the output row type.

  // When operator is of type kGather, we don't need to store any partition key
  // indices because we're going to merge all the incoming streams together.

  // Get partition function specification string
  spec_ = planNode->partitionFunctionSpec().toString();

  // Only parse keys if it's a hash function
  if (spec_.find("HASH(") != std::string::npos) {
    // Extract keys between HASH( and )
    size_t start = spec_.find("HASH(") + 5;
    size_t end = spec_.find(")", start);
    if (start != std::string::npos && end != std::string::npos) {
      std::string keysStr = spec_.substr(start, end - start);

      // Split by comma to get individual keys.
      std::vector<std::string> keys;
      size_t pos = 0;
      while ((pos = keysStr.find(",")) != std::string::npos) {
        std::string key = keysStr.substr(0, pos);
        keys.push_back(key);
        keysStr.erase(0, pos + 1);
      }
      keys.push_back(keysStr); // Add the last key.

      // Find field indices for each key.
      const auto& rowType = planNode->outputType();
      for (const auto& key : keys) {
        auto trimmedKey = key;
        // Trim whitespace
        trimmedKey.erase(0, trimmedKey.find_first_not_of(" "));
        trimmedKey.erase(trimmedKey.find_last_not_of(" ") + 1);

        auto fieldIndex = rowType->getChildIdx(trimmedKey);
        partitionKeyIndices_.push_back(fieldIndex);
      }
    }
  }
}

void CudfPartitionedOutput::hashPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Hashing and partitioning into " << numPartitions_ << " chunks";

  // Use cudf hash partitioning
  std::vector<cudf::size_type> partitionKeyIndices;
  for (const auto& idx : partitionKeyIndices_) {
    partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
  }

  auto [partitionedTable, partitionOffsets] = cudf::hash_partition(
      tableView,
      partitionKeyIndices,
      numPartitions_,
      cudf::hash_id::HASH_MURMUR3,
      cudf::DEFAULT_HASH_SEED,
      stream);

  VELOX_CHECK(partitionOffsets.size() == numPartitions_);
  VELOX_CHECK(partitionOffsets[0] == 0);

  // Erase first element since it's always 0 and we don't need it.
  partitionOffsets.erase(partitionOffsets.begin());

  splitAndEnqueue(partitionedTable->view(), partitionOffsets, stream);
}

void CudfPartitionedOutput::equalPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Splitting into " << numPartitions_ << " chunks";
  std::vector<cudf::size_type> offsets;
  cudf::size_type size = tableView.num_rows();
  for (int i = 1; i < numPartitions_; ++i) {
    cudf::size_type idx = size / (numPartitions_ / (double)i);
    offsets.push_back(idx);
  }
  splitAndEnqueue(tableView, offsets, stream);
}

void CudfPartitionedOutput::splitAndEnqueue(
    cudf::table_view tableView,
    std::vector<cudf::size_type> offsets,
    rmm::cuda_stream_view stream) {
  auto contiguousTables = cudf::contiguous_split(tableView, offsets, stream);

  VELOX_CHECK_EQ(
      offsets.size() + 1, numPartitions_, "mismatch in numPartitions_");
  auto queueManager = sharedQueueManager();
  for (int i = 0; i < numPartitions_; ++i) {
    auto const& partitionTable = contiguousTables[i];
    if (partitionTable.table.num_rows() == 0) {
      // Skip empty partitions.
      continue;
    }

    auto packedColsPtr = std::make_unique<cudf::packed_columns>(
        std::move(contiguousTables[i].data.metadata),
        std::move(contiguousTables[i].data.gpu_data));

    // enqueue partition data on Cudf Output Buffer
    queueManager->enqueue(
        this->taskId(),
        i,
        std::move(packedColsPtr),
        partitionTable.table.num_rows());
  }
}

} // namespace facebook::velox::cudf_exchange
