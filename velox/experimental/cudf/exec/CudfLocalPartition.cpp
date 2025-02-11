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

#include "CudfLocalPartition.h"
#include <experimental/cudf/vector/CudfVector.h>
#include "velox/exec/Task.h"

namespace facebook::velox::cudf_velox {

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
      queues_{
          ctx->task->getLocalExchangeQueues(ctx->splitGroupId, planNode->id())},
      numPartitions_{queues_.size()}
// partitionFunction_(
//     numPartitions_ == 1 ? nullptr
//                         : planNode->partitionFunctionSpec().create(
//                               numPartitions_,
//                               /*localExchange=*/true))
{
  // VELOX_CHECK(numPartitions_ == 1 || partitionFunction_ != nullptr);

  // DM: Since we're replacing the LocalPartition with CudfLocalPartition, the
  // number of producers is already set. Adding producer only adds to a counter
  // which we don't have to do again.

  // for (auto& queue : queues_) {
  //   queue->addProducer();
  // }
  // if (numPartitions_ > 0) {
  //   indexBuffers_.resize(numPartitions_);
  //   rawIndices_.resize(numPartitions_);
  // }
}

// void CudfLocalPartition::allocateIndexBuffers(
//     const std::vector<vector_size_t>& sizes) {
//   VELOX_CHECK_EQ(indexBuffers_.size(), sizes.size());
//   VELOX_CHECK_EQ(rawIndices_.size(), sizes.size());
//
//   for (auto i = 0; i < sizes.size(); ++i) {
//     const auto indicesBufferBytes = sizes[i] * sizeof(vector_size_t);
//     if ((indexBuffers_[i] == nullptr) ||
//         (indexBuffers_[i]->capacity() < indicesBufferBytes) ||
//         !indexBuffers_[i]->unique()) {
//       indexBuffers_[i] = allocateIndices(sizes[i], pool());
//     } else {
//       const auto indicesBufferBytes = sizes[i] * sizeof(vector_size_t);
//       indexBuffers_[i]->setSize(indicesBufferBytes);
//     }
//     rawIndices_[i] = indexBuffers_[i]->asMutable<vector_size_t>();
//   }
// }

// RowVectorPtr CudfLocalPartition::wrapChildren(
//     const RowVectorPtr& input,
//     vector_size_t size,
//     const BufferPtr& indices,
//     RowVectorPtr reusable) {
//   RowVectorPtr result;
//   if (!reusable) {
//     result = std::make_shared<RowVector>(
//         pool(),
//         input->type(),
//         nullptr,
//         size,
//         std::vector<VectorPtr>(input->childrenSize()));
//   } else {
//     VELOX_CHECK(!reusable->mayHaveNulls());
//     VELOX_CHECK_EQ(reusable.use_count(), 1);
//     reusable->unsafeResize(size);
//     result = std::move(reusable);
//   }
//   VELOX_CHECK_NOT_NULL(result);
//
//   for (auto i = 0; i < input->childrenSize(); ++i) {
//     auto& child = result->childAt(i);
//     if (child && child->encoding() == VectorEncoding::Simple::DICTIONARY &&
//         child.use_count() == 1) {
//       child->BaseVector::resize(size);
//       child->setWrapInfo(indices);
//       child->setValueVector(input->childAt(i));
//     } else {
//       child = BaseVector::wrapInDictionary(
//           nullptr, indices, size, input->childAt(i));
//     }
//   }
//
//   result->updateContainsLazyNotLoaded();
//   return result;
// }

void CudfLocalPartition::addInput(RowVectorPtr input) {
  prepareForInput(input);
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK(cudfVector, "Input must be a CudfVector");

  // const auto singlePartition = numPartitions_ == 1
  //     ? 0
  //     : partitionFunction_->partition(*input, partitions_);
  // if (singlePartition.has_value()) {
  ContinueFuture future;
  // auto blockingReason = queues_[singlePartition.value()]->enqueue(
  auto blockingReason =
      queues_[0]->enqueue(input, input->retainedSize(), &future);
  if (blockingReason != exec::BlockingReason::kNotBlocked) {
    blockingReasons_.push_back(blockingReason);
    futures_.push_back(std::move(future));
  }
  return;
  // }

  // const auto numInput = input->size();
  // std::vector<vector_size_t> maxIndex(numPartitions_, 0);
  // for (auto i = 0; i < numInput; ++i) {
  //   ++maxIndex[partitions_[i]];
  // }
  // allocateIndexBuffers(maxIndex);

  // std::fill(maxIndex.begin(), maxIndex.end(), 0);
  // for (auto i = 0; i < numInput; ++i) {
  //   auto partition = partitions_[i];
  //   rawIndices_[partition][maxIndex[partition]] = i;
  //   ++maxIndex[partition];
  // }

  // const int64_t totalSize = input->retainedSize();
  // for (auto i = 0; i < numPartitions_; i++) {
  //   auto partitionSize = maxIndex[i];
  //   if (partitionSize == 0) {
  //     // Do not enqueue empty partitions.
  //     continue;
  //   }
  //   auto partitionData = wrapChildren(
  //       input, partitionSize, indexBuffers_[i], queues_[i]->getVector());
  //   ContinueFuture future;
  //   auto reason = queues_[i]->enqueue(
  //       partitionData, totalSize * partitionSize / numInput, &future);
  //   if (reason != exec::BlockingReason::kNotBlocked) {
  //     blockingReasons_.push_back(reason);
  //     futures_.push_back(std::move(future));
  //   }
  // }
}

void CudfLocalPartition::prepareForInput(RowVectorPtr& input) {
  // DM: This might not do anything because CudfVector sets children to nullptr.
  // eh, whatever :shrug:
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(input->estimateFlatSize(), input->size());
  }

  // Lazy vectors must be loaded or processed to ensure the late materialized in
  // order.
  // DM: We don't have to do this because we're expecting cudf tables which are
  // already loaded.
  // for (auto& child : input->children()) {
  //   child->loadedVector();
  // }
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

  return true;
}

} // namespace facebook::velox::cudf_velox
