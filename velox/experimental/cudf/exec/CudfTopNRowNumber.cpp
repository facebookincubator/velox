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
#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/merge.hpp>
#include <cudf/rolling.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

namespace facebook::velox::cudf_velox {

namespace {

cudf::table_view makePartitionKeys(
    cudf::table_view sortedView,
    const std::vector<cudf::size_type>& partitionKeyIndices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::unique_ptr<cudf::column>& singlePartitionCol) {
  if (!partitionKeyIndices.empty()) {
    return sortedView.select(partitionKeyIndices);
  }
  auto zero = cudf::numeric_scalar<int8_t>(0, true, stream, mr);
  singlePartitionCol =
      cudf::make_column_from_scalar(zero, sortedView.num_rows(), stream, mr);
  return cudf::table_view{{singlePartitionCol->view()}};
}

// Computes row_number() over `view` grouped by the columns at
// `partitionKeyIndices` (or a single implicit partition if empty), assuming
// `view` is already sorted by partition+ordering keys. The value column fed
// to grouped_rolling_window is arbitrary (row_number ignores it); column(0)
// is used for convenience.
std::unique_ptr<cudf::column> computeRowNumbers(
    cudf::table_view view,
    const std::vector<cudf::size_type>& partitionKeyIndices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::unique_ptr<cudf::column> singlePartitionCol;
  auto partKeys = makePartitionKeys(
      view, partitionKeyIndices, stream, mr, singlePartitionCol);
  auto firstCol = view.column(0);
  auto rowNumberAgg =
      cudf::make_row_number_aggregation<cudf::rolling_aggregation>();
  auto unbounded = cudf::window_bounds::unbounded();
  auto currentRow = cudf::window_bounds::get(0);
  return cudf::grouped_rolling_window(
      partKeys, firstCol, unbounded, currentRow, 1, *rowNumberAgg, stream, mr);
}

// Filters `rowNums` to <= limit and returns the resulting boolean mask.
std::unique_ptr<cudf::column> makeLimitMask(
    const cudf::column& rowNums,
    int32_t limit,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto limitScalar = cudf::numeric_scalar<int64_t>(limit, true, stream, mr);
  return cudf::binary_operation(
      rowNums.view(),
      limitScalar,
      cudf::binary_operator::LESS_EQUAL,
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      mr);
}

} // namespace

CudfTopNRowNumber::CudfTopNRowNumber(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::TopNRowNumberNode>& node)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          node->outputType(),
          node->id(),
          "CudfTopNRowNumber",
          nvtx3::rgb{255, 200, 100},
          NvtxMethodFlag::kAll,
          std::nullopt,
          node),
      node_(node),
      limit_(node->limit()),
      generateRowNumber_(node->generateRowNumber()),
      inputType_(node->sources()[0]->outputType()),
      cudaEvent_(std::make_unique<CudaEvent>(cudaEventDisableTiming)) {
  VELOX_CHECK_EQ(
      node->rankFunction(),
      core::TopNRowNumberNode::RankFunction::kRowNumber,
      "CudfTopNRowNumber only supports row_number");

  partitionKeyIndices_.reserve(node->partitionKeys().size());
  for (const auto& key : node->partitionKeys()) {
    partitionKeyIndices_.push_back(exec::exprToChannel(key.get(), inputType_));
  }

  sortKeyIndices_.reserve(node->sortingKeys().size());
  sortOrders_.reserve(node->sortingKeys().size());
  nullOrders_.reserve(node->sortingKeys().size());
  for (size_t i = 0; i < node->sortingKeys().size(); ++i) {
    sortKeyIndices_.push_back(
        exec::exprToChannel(node->sortingKeys()[i].get(), inputType_));
    const auto& order = node->sortingOrders()[i];
    sortOrders_.push_back(
        order.isAscending() ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    nullOrders_.push_back(
        (order.isNullsFirst() ^ !order.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }

  allSortKeys_.reserve(partitionKeyIndices_.size() + sortKeyIndices_.size());
  allOrders_.reserve(partitionKeyIndices_.size() + sortKeyIndices_.size());
  allNullOrders_.reserve(partitionKeyIndices_.size() + sortKeyIndices_.size());
  localPartitionKeyIndices_.reserve(partitionKeyIndices_.size());

  for (size_t i = 0; i < partitionKeyIndices_.size(); ++i) {
    allSortKeys_.push_back(partitionKeyIndices_[i]);
    allOrders_.push_back(cudf::order::ASCENDING);
    allNullOrders_.push_back(cudf::null_order::BEFORE);
    localPartitionKeyIndices_.push_back(static_cast<cudf::size_type>(i));
  }
  for (size_t i = 0; i < sortKeyIndices_.size(); ++i) {
    allSortKeys_.push_back(sortKeyIndices_[i]);
    allOrders_.push_back(sortOrders_[i]);
    allNullOrders_.push_back(nullOrders_[i]);
  }
}

CudfVectorPtr CudfTopNRowNumber::reduceBatchToLocalCandidates(
    const CudfVectorPtr& cudfInput,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto inputView = cudfInput->getTableView();
  auto keyTable = inputView.select(allSortKeys_);
  auto indices = cudf::stable_sorted_order(
      keyTable, allOrders_, allNullOrders_, stream, mr);
  auto sortedKeyTable = cudf::gather(
      keyTable,
      indices->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::negative_index_policy::NOT_ALLOWED,
      stream,
      mr);
  auto rowNums = computeRowNumbers(
      sortedKeyTable->view(), localPartitionKeyIndices_, stream, mr);
  auto mask = makeLimitMask(*rowNums, limit_, stream, mr);

  // Filter the sort permutation to the surviving rows before gathering the
  // full payload, so batches with many rows per partition don't pay for
  // materializing rows that will be pruned immediately after.
  auto filteredIndicesTable = cudf::apply_boolean_mask(
      cudf::table_view{{indices->view()}}, mask->view(), stream, mr);
  auto filteredIndices = filteredIndicesTable->view().column(0);

  auto localCandidatesTable = cudf::gather(
      inputView,
      filteredIndices,
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::negative_index_policy::NOT_ALLOWED,
      stream,
      mr);
  auto const size = localCandidatesTable->num_rows();
  return std::make_shared<CudfVector>(
      cudfInput->pool(),
      inputType_,
      size,
      std::move(localCandidatesTable),
      stream);
}

CudfVectorPtr CudfTopNRowNumber::mergeAndPruneCandidates(
    const CudfVectorPtr& previous,
    const CudfVectorPtr& incoming,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<rmm::cuda_stream_view> inputStreams{
      previous->stream(), incoming->stream()};
  cudf::detail::join_streams(inputStreams, stream);

  std::vector<cudf::table_view> tableViews{
      previous->getTableView(), incoming->getTableView()};
  auto merged = cudf::merge(
      tableViews, allSortKeys_, allOrders_, allNullOrders_, stream, mr);

  // Ensure input-stream deallocations don't race with the merge kernel.
  streamsWaitForStream(*cudaEvent_, inputStreams, stream);

  auto rowNums =
      computeRowNumbers(merged->view(), partitionKeyIndices_, stream, mr);
  auto mask = makeLimitMask(*rowNums, limit_, stream, mr);
  auto pruned =
      cudf::apply_boolean_mask(merged->view(), mask->view(), stream, mr);

  auto const size = pruned->num_rows();
  return std::make_shared<CudfVector>(
      previous->pool(), inputType_, size, std::move(pruned), stream);
}

void CudfTopNRowNumber::doAddInput(RowVectorPtr input) {
  if (limit_ == 0 || input->size() == 0) {
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfTopNRowNumber expects CudfVector");

  auto mr = get_output_mr();
  auto localCandidates =
      reduceBatchToLocalCandidates(cudfInput, cudfInput->stream(), mr);

  if (candidates_ == nullptr) {
    candidates_ = std::move(localCandidates);
    return;
  }

  // Merge on a fresh stream (rather than either input's stream) so the
  // merge/prune work can be scheduled independently of both producers; see
  // CudfTopN::mergeTopK for the same pattern.
  auto mergeStream = cudfGlobalStreamPool().get_stream();
  candidates_ =
      mergeAndPruneCandidates(candidates_, localCandidates, mergeStream, mr);
}

void CudfTopNRowNumber::doNoMoreInput() {
  Operator::noMoreInput();
  if (candidates_ == nullptr) {
    finished_ = true;
  }
}

bool CudfTopNRowNumber::isFinished() {
  return finished_;
}

RowVectorPtr CudfTopNRowNumber::doGetOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  finished_ = true;
  if (candidates_ == nullptr) {
    return nullptr;
  }

  if (!generateRowNumber_) {
    return std::move(candidates_);
  }

  auto stream = candidates_->stream();
  auto mr = get_output_mr();
  auto rowNums = computeRowNumbers(
      candidates_->getTableView(), partitionKeyIndices_, stream, mr);
  // cuDF row_number is int32; Velox expects bigint.
  const auto rowNumberCudfType = cudf_velox::veloxToCudfDataType(
      outputType_->childAt(outputType_->size() - 1));
  if (rowNums->type() != rowNumberCudfType) {
    rowNums = cudf::cast(*rowNums, rowNumberCudfType, stream, mr);
  }

  auto pool = candidates_->pool();
  auto const size = candidates_->size();
  auto cols = candidates_->release()->release();
  cols.push_back(std::move(rowNums));
  auto finalTable = std::make_unique<cudf::table>(std::move(cols));

  return std::make_shared<CudfVector>(
      pool, outputType_, size, std::move(finalTable), stream);
}

} // namespace facebook::velox::cudf_velox
