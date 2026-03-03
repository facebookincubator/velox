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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfLocalPartition.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/RoundRobinPartitionFunction.h"
#include "velox/exec/Task.h"

#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>

namespace facebook::velox::cudf_velox {

namespace {
template <class... Deriveds, class Base>
bool isAnyOf(const Base* p) {
  return ((dynamic_cast<const Deriveds*>(p) != nullptr) || ...);
}
} // namespace

bool CudfLocalPartition::shouldReplace(
    const std::shared_ptr<const core::LocalPartitionNode>& planNode) {
  // Only replace for Hash, Round Robin, and Round Robin Row-Wise Partitioning.
  if (isAnyOf<
          exec::HashPartitionFunctionSpec,
          exec::RoundRobinPartitionFunctionSpec,
          core::GatherPartitionFunctionSpec>(
          &planNode->partitionFunctionSpec())) {
    return true;
  }
  std::string spec = planNode->partitionFunctionSpec().toString();
  if (spec.find("ROUND ROBIN ROW") != std::string::npos) {
    return true;
  }
  LOG(WARNING) << "CudfLocalPartition unsupported spec = " << spec;
  return false;
}

CudfLocalPartition::CudfLocalPartition(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::LocalPartitionNode>& planNode)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "CudfLocalPartition"),
      NvtxHelper(
          nvtx3::rgb{255, 215, 0}, // Gold
          operatorId,
          fmt::format("[{}]", planNode->id())),
      queues_{
          ctx->task->getLocalExchangeQueues(ctx->splitGroupId, planNode->id())},
      numPartitions_{queues_.size()} {
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
  std::string spec = planNode->partitionFunctionSpec().toString();
  auto* hashFunctionSpec = dynamic_cast<const exec::HashPartitionFunctionSpec*>(
      &planNode->partitionFunctionSpec());

  // Only parse keys if it's a hash function
  if (hashFunctionSpec) {
    // Extract keys between HASH( and )
    size_t start = spec.find("HASH(") + 5;
    size_t end = spec.find(")", start);
    if (start != std::string::npos && end != std::string::npos) {
      std::string keysStr = spec.substr(start, end - start);

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
    partitionFunctionType_ = PartitionFunctionType::kHash;
  } else if (
      dynamic_cast<const exec::RoundRobinPartitionFunctionSpec*>(
          &planNode->partitionFunctionSpec()) &&
      numPartitions_ > 1) {
    partitionFunctionType_ = PartitionFunctionType::kRoundRobin;
  } else if (
      numPartitions_ > 1 && spec.find("ROUND ROBIN ROW") != std::string::npos) {
    partitionFunctionType_ = PartitionFunctionType::kRoundRobinRow;
  }
  VELOX_CHECK(
      numPartitions_ == 1 || partitionKeyIndices_.size() > 0 ||
      partitionFunctionType_ == PartitionFunctionType::kRoundRobin ||
      partitionFunctionType_ == PartitionFunctionType::kRoundRobinRow);

  // Since we're replacing the LocalPartition with CudfLocalPartition, the
  // number of producers is already set. Adding producer only adds to a counter
  // which we don't have to do again.
  // Normally, this is what we'd have to do:
  // for (auto& queue : queues_) {
  //   queue->addProducer();
  // }
}

void CudfLocalPartition::recordOutputStats(RowVectorPtr& input) {
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(input->estimateFlatSize(), input->size());
  }
}

void CudfLocalPartition::flushVectorPool() {
  // We reuse the LocalExchangeQueue from the CPU implementation. That impl
  // stores used vectors in a vector pool for the CPU LocalPartition to re-use.
  // CudfLocalPartition does not need it and does not extract it. This results
  // in unnecessary extension of the lifetimes of vectors that were exchanged,
  // resulting in kind of a memory leak.
  // This is a hack to forcefully flush the vector pools.

  for (auto& queue : queues_) {
    queue->getVector();
  }
}

void CudfLocalPartition::enqueuePartition(
    int partitionIndex,
    const CudfVectorPtr& cudfVector) {
  if (cudfVector->size() == 0) {
    // Skip empty partitions.
    return;
  }

  ContinueFuture future;
  auto blockingReason =
      queues_[partitionIndex]->enqueue(cudfVector, cudfVector->size(), &future);
  if (blockingReason != exec::BlockingReason::kNotBlocked) {
    blockingReasons_.push_back(blockingReason);
    futures_.push_back(std::move(future));
  }
}

void CudfLocalPartition::addInput(RowVectorPtr input) {
  flushVectorPool();
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  recordOutputStats(input);
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK(cudfVector, "Input must be a CudfVector");
  auto stream = cudfVector->stream();

  if (numPartitions_ > 1) {
    if (partitionFunctionType_ == PartitionFunctionType::kRoundRobin) {
      const auto partition = counter_ % numPartitions_;
      ++counter_;
      enqueuePartition(partition, cudfVector);
      return;
    }
    auto [partitionedTable, partitionOffsets] = [&]() {
      auto tableView = cudfVector->getTableView();
      // Use cudf hash partitioning
      if (partitionFunctionType_ == PartitionFunctionType::kHash) {
        std::vector<cudf::size_type> partitionKeyIndices;
        for (const auto& idx : partitionKeyIndices_) {
          partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
        }

        return cudf::hash_partition(
            tableView,
            partitionKeyIndices,
            numPartitions_,
            cudf::hash_id::HASH_MURMUR3,
            cudf::DEFAULT_HASH_SEED,
            stream,
            get_temp_mr());
      } else if (
          partitionFunctionType_ == PartitionFunctionType::kRoundRobinRow) {
        auto result = cudf::round_robin_partition(
            tableView, numPartitions_, counter_, stream, get_temp_mr());
        counter_ = (counter_ + cudfVector->size()) % numPartitions_;
        return result;
      }
      VELOX_FAIL("Unsupported partition function");
    }();

    // cuDF partitioning APIs return num_partitions + 1 offsets where:
    // - offsets[i] is the starting row index for partition i
    // - offsets[num_partitions] is the total row count
    VELOX_CHECK(partitionOffsets.size() == numPartitions_ + 1);
    VELOX_CHECK(partitionOffsets[0] == 0);

    // cudf::split expects split points (excluding first 0 and last totalRows).
    // Erase first element (always 0) and last element (total row count).
    partitionOffsets.erase(partitionOffsets.begin());
    partitionOffsets.pop_back();

    auto partitionedTables =
        cudf::split(partitionedTable->view(), partitionOffsets, stream);

    // DM: We should investigate if keeping partitionedTables alive and using
    // the table view in partitionData is more efficient than creating a new
    // table each time. Currently out of scope because it would need a new
    // type of RowVector that can hold a table view and shared_ptr to the
    // table.
    for (int i = 0; i < numPartitions_; ++i) {
      auto partitionData = partitionedTables[i];
      if (partitionData.num_rows() == 0) {
        continue;
      }

      auto partitionCudfVector = std::make_shared<CudfVector>(
          pool(),
          outputType_,
          partitionData.num_rows(),
          std::make_unique<cudf::table>(partitionData, stream, get_output_mr()),
          stream);
      enqueuePartition(i, partitionCudfVector);
    }
  } else {
    // Single partition case.
    ContinueFuture future;
    auto blockingReason =
        queues_[0]->enqueue(input, input->retainedSize(), &future);
    if (blockingReason != exec::BlockingReason::kNotBlocked) {
      blockingReasons_.push_back(blockingReason);
      futures_.push_back(std::move(future));
    }
  }
}

exec::BlockingReason CudfLocalPartition::isBlocked(ContinueFuture* future) {
  if (!futures_.empty()) {
    auto blockingReason = blockingReasons_.front();
    *future = folly::collectAll(futures_.begin(), futures_.end()).unit();
    futures_.clear();
    blockingReasons_.clear();
    return blockingReason;
  }

  return exec::BlockingReason::kNotBlocked;
}

void CudfLocalPartition::noMoreInput() {
  Operator::noMoreInput();
  for (const auto& queue : queues_) {
    queue->noMoreData();
  }
}

bool CudfLocalPartition::isFinished() {
  if (!futures_.empty() || !noMoreInput_) {
    return false;
  }
  flushVectorPool();

  return true;
}

} // namespace facebook::velox::cudf_velox
