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
#include <cudf/groupby.hpp>
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
    cuda::stream_ref stream,
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

// Rows must be sorted by partition and ordering keys. Rank scans only need
// adjacent peer equality, so the presorted order also handles mixed directions.
std::unique_ptr<cudf::column> computeRanks(
    cudf::table_view view,
    const std::vector<cudf::size_type>& partitionKeyIndices,
    const std::vector<cudf::size_type>& sortKeyIndices,
    core::TopNRowNumberNode::RankFunction rankFunction,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::unique_ptr<cudf::column> singlePartitionCol;
  auto partKeys = makePartitionKeys(
      view, partitionKeyIndices, stream, mr, singlePartitionCol);
  if (rankFunction == core::TopNRowNumberNode::RankFunction::kRowNumber) {
    auto rowNumberAggregation =
        cudf::make_row_number_aggregation<cudf::rolling_aggregation>();
    return cudf::grouped_rolling_window(
        partKeys,
        view.column(0),
        cudf::window_bounds::unbounded(),
        cudf::window_bounds::get(0),
        1,
        *rowNumberAggregation,
        stream,
        mr);
  }

  auto sortingKeys = view.select(sortKeyIndices);
  auto rankValues = sortingKeys.column(0);
  if (sortKeyIndices.size() > 1) {
    rankValues = cudf::column_view{
        cudf::data_type{cudf::type_id::STRUCT},
        view.num_rows(),
        nullptr,
        nullptr,
        0,
        0,
        std::vector<cudf::column_view>{sortingKeys.begin(), sortingKeys.end()}};
  }

  cudf::groupby::groupby grouper(
      partKeys,
      cudf::null_policy::INCLUDE,
      cudf::sorted::YES,
      std::vector<cudf::order>(partKeys.num_columns(), cudf::order::ASCENDING),
      std::vector<cudf::null_order>(
          partKeys.num_columns(), cudf::null_order::BEFORE));
  std::vector<cudf::groupby::scan_request> requests(1);
  requests[0].values = rankValues;
  requests[0].aggregations.push_back(
      cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
          rankFunction == core::TopNRowNumberNode::RankFunction::kRank
              ? cudf::rank_method::MIN
              : cudf::rank_method::DENSE,
          cudf::order::ASCENDING,
          cudf::null_policy::INCLUDE,
          cudf::null_order::BEFORE));
  auto result = grouper.scan(requests, stream, mr);
  return std::move(result.second[0].results[0]);
}

// Keep every peer whose rank is within the limit.
std::unique_ptr<cudf::column> makeLimitMask(
    const cudf::column& ranks,
    int32_t limit,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto limitScalar = cudf::numeric_scalar<int64_t>(limit, true, stream, mr);
  return cudf::binary_operation(
      ranks.view(),
      limitScalar,
      cudf::binary_operator::LESS_EQUAL,
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      mr);
}

bool supportsRankKey(const TypePtr& type, bool sortingKey) {
  if (type->providesCustomComparison() ||
      (sortingKey && (type->isArray() || type->isMap()))) {
    return false;
  }
  for (uint32_t i = 0; i < type->size(); ++i) {
    if (!supportsRankKey(type->childAt(i), sortingKey)) {
      return false;
    }
  }
  return true;
}

} // namespace

bool CudfTopNRowNumber::canRunOnGPU(const core::TopNRowNumberNode& node) {
  switch (node.rankFunction()) {
    case core::TopNRowNumberNode::RankFunction::kRowNumber:
      return true;
    case core::TopNRowNumberNode::RankFunction::kRank:
    case core::TopNRowNumberNode::RankFunction::kDenseRank:
      break;
    default:
      return false;
  }
  for (const auto& key : node.partitionKeys()) {
    if (!supportsRankKey(key->type(), /*sortingKey=*/false)) {
      return false;
    }
  }
  for (const auto& key : node.sortingKeys()) {
    if (!supportsRankKey(key->type(), /*sortingKey=*/true)) {
      return false;
    }
  }
  return true;
}

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
  VELOX_CHECK(canRunOnGPU(*node), "Unsupported CudfTopNRowNumber keys");

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
  localSortKeyIndices_.reserve(sortKeyIndices_.size());

  for (size_t i = 0; i < partitionKeyIndices_.size(); ++i) {
    allSortKeys_.push_back(partitionKeyIndices_[i]);
    allOrders_.push_back(cudf::order::ASCENDING);
    allNullOrders_.push_back(cudf::null_order::BEFORE);
    localPartitionKeyIndices_.push_back(static_cast<cudf::size_type>(i));
  }
  for (size_t i = 0; i < sortKeyIndices_.size(); ++i) {
    localSortKeyIndices_.push_back(allSortKeys_.size());
    allSortKeys_.push_back(sortKeyIndices_[i]);
    allOrders_.push_back(sortOrders_[i]);
    allNullOrders_.push_back(nullOrders_[i]);
  }
}

CudfVectorPtr CudfTopNRowNumber::reduceBatchToLocalCandidates(
    const CudfVectorPtr& cudfInput,
    cuda::stream_ref stream,
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
      cudf::memory_resources{mr, get_temp_mr()});
  auto ranks = computeRanks(
      sortedKeyTable->view(),
      localPartitionKeyIndices_,
      localSortKeyIndices_,
      node_->rankFunction(),
      stream,
      mr);
  auto mask = makeLimitMask(*ranks, limit_, stream, mr);

  // Filter the sort permutation to the surviving rows before gathering the
  // full payload, so batches with many rows per partition don't pay for
  // materializing rows that will be pruned immediately after.
  auto filteredIndicesTable = cudf::apply_retention_mask(
      cudf::table_view{{indices->view()}}, mask->view(), stream, mr);
  auto filteredIndices = filteredIndicesTable->view().column(0);

  auto localCandidatesTable = cudf::gather(
      inputView,
      filteredIndices,
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::negative_index_policy::NOT_ALLOWED,
      stream,
      cudf::memory_resources{mr, get_temp_mr()});
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
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<cuda::stream_ref> inputStreams{
      previous->stream(), incoming->stream()};
  cudf::detail::join_streams(inputStreams, stream);

  std::vector<cudf::table_view> tableViews{
      previous->getTableView(), incoming->getTableView()};
  auto merged = cudf::merge(
      tableViews, allSortKeys_, allOrders_, allNullOrders_, stream, mr);

  // Ensure input-stream deallocations don't race with the merge kernel.
  streamsWaitForStream(*cudaEvent_, inputStreams, stream);

  auto ranks = computeRanks(
      merged->view(),
      partitionKeyIndices_,
      sortKeyIndices_,
      node_->rankFunction(),
      stream,
      mr);
  auto mask = makeLimitMask(*ranks, limit_, stream, mr);
  auto pruned =
      cudf::apply_retention_mask(merged->view(), mask->view(), stream, mr);

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
  auto ranks = computeRanks(
      candidates_->getTableView(),
      partitionKeyIndices_,
      sortKeyIndices_,
      node_->rankFunction(),
      stream,
      mr);
  // cuDF ranks are int32; Velox expects bigint.
  const auto rowNumberCudfType = cudf_velox::veloxToCudfDataType(
      outputType_->childAt(outputType_->size() - 1));
  if (ranks->type() != rowNumberCudfType) {
    ranks = cudf::cast(*ranks, rowNumberCudfType, stream, mr);
  }

  auto pool = candidates_->pool();
  auto const size = candidates_->size();
  auto cols = candidates_->release()->release();
  cols.push_back(std::move(ranks));
  auto finalTable = std::make_unique<cudf::table>(std::move(cols));

  return std::make_shared<CudfVector>(
      pool, outputType_, size, std::move(finalTable), stream);
}

} // namespace facebook::velox::cudf_velox
