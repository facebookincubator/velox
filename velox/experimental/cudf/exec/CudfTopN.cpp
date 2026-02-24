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
#include "velox/experimental/cudf/CudfQueryConfig.h"
#include "velox/experimental/cudf/exec/CudfTopN.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/merge.hpp>
#include <cudf/sorting.hpp>

namespace facebook::velox::cudf_velox {
CudfTopN::CudfTopN(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::TopNNode>& topNNode)
    : exec::Operator(
          driverCtx,
          topNNode->outputType(),
          operatorId,
          topNNode->id(),
          "CudfTopN"),
      NvtxHelper(
          nvtx3::rgb{175, 238, 238}, // Pale Turquoise
          operatorId,
          fmt::format("[{}]", topNNode->id())),
      count_(topNNode->count()),
      topNNode_(topNNode) {
  kBatchSize_ = driverCtx->queryConfig().get<int32_t>(
      CudfQueryConfig::kCudfTopNBatchSize, kBatchSize_);
  const auto numColumns{outputType_->children().size()};
  const auto numSortingKeys{topNNode->sortingKeys().size()};
  std::vector<bool> isSortingKey(numColumns);
  sortKeys_.reserve(numSortingKeys);
  columnOrder_.reserve(numSortingKeys);
  nullOrder_.reserve(numSortingKeys);

  for (int i = 0; i < numSortingKeys; ++i) {
    const auto channel =
        exec::exprToChannel(topNNode->sortingKeys()[i].get(), outputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "TopN doesn't allow constant sorting keys");
    sortKeys_.push_back(channel);
    isSortingKey[channel] = true;
    auto const& sortingOrder = topNNode->sortingOrders()[i];
    columnOrder_.push_back(
        sortingOrder.isAscending() ? cudf::order::ASCENDING
                                   : cudf::order::DESCENDING);
    nullOrder_.push_back(
        (sortingOrder.isNullsFirst() ^ !sortingOrder.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

CudfVectorPtr CudfTopN::mergeTopK(
    std::vector<CudfVectorPtr> topNBatches,
    int32_t k,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<cudf::table_view> tableViews;
  for (const auto& batch : topNBatches) {
    tableViews.push_back(batch->getTableView());
  }
  auto mergedTable =
      cudf::merge(tableViews, sortKeys_, columnOrder_, nullOrder_, stream, mr);
  // slice it
  auto topk =
      cudf::split(
          mergedTable->view(), {std::min(k, mergedTable->num_rows())}, stream)
          .front();
  auto const size = topk.num_rows();
  return std::make_shared<CudfVector>(
      topNBatches[0]->pool(),
      outputType_,
      size,
      std::make_unique<cudf::table>(topk, stream, mr),
      stream);
}

std::unique_ptr<cudf::table> CudfTopN::getTopK(
    cudf::table_view const& values,
    int32_t k,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto keys = values.select(sortKeys_);
  auto const indices =
      cudf::stable_sorted_order(keys, columnOrder_, nullOrder_, stream, mr);
  auto const kIndices =
      cudf::split(indices->view(), {std::min(k, indices->size())}, stream)
          .front();
  return cudf::detail::gather(
      values,
      kIndices,
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::detail::negative_index_policy::NOT_ALLOWED,
      stream,
      mr);
}

// helper to get topk of a table
CudfVectorPtr CudfTopN::getTopKBatch(CudfVectorPtr cudfInput, int32_t k) {
  if (k == 0 || cudfInput->size() == 0) {
    return nullptr;
  }
  auto stream = cudfInput->stream();
  auto mr = get_output_mr();
  auto values = cudfInput->getTableView();
  auto result = getTopK(values, k, stream, mr);
  auto const size = result->num_rows();
  return std::make_shared<CudfVector>(
      cudfInput->pool(), cudfInput->type(), size, std::move(result), stream);
}

void CudfTopN::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (count_ == 0 || input->size() == 0) {
    return;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  // Take topk of each input, add to batch.
  // If got kBatchSize_ batches, concat batches and topk once.
  // During getOutput, concat batches and topk once.
  topNBatches_.push_back(getTopKBatch(cudfInput, count_));
  // sum of sizes of topNBatches_ >= count_, then concat and topk once.
  auto totalSize = std::accumulate(
      topNBatches_.begin(),
      topNBatches_.end(),
      0,
      [](int32_t sum, const auto& batch) {
        return sum + (batch ? batch->size() : 0);
      });
  if (topNBatches_.size() >= kBatchSize_ and totalSize >= count_) {
    auto stream = cudfGlobalStreamPool().get_stream();
    auto mr = get_output_mr();

    auto result = mergeTopK(topNBatches_, count_, stream, mr);
    topNBatches_.clear();
    topNBatches_.push_back(std::move(result));
  }
}

RowVectorPtr CudfTopN::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  if (topNBatches_.empty()) {
    finished_ = noMoreInput_;
    return nullptr;
  }

  auto stream = topNBatches_[0]->stream();
  auto mr = get_output_mr();
  auto result = mergeTopK(topNBatches_, count_, stream, mr);
  topNBatches_.clear();
  finished_ = noMoreInput_ && topNBatches_.empty();
  return result;
}

void CudfTopN::noMoreInput() {
  Operator::noMoreInput();
  if (topNBatches_.empty()) {
    finished_ = true;
    return;
  }
}

bool CudfTopN::isFinished() {
  return finished_;
}
} // namespace facebook::velox::cudf_velox
