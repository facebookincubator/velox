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
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/core/PlanNode.h"

#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/merge.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>

#include <limits>
#include <numeric>

namespace facebook::velox::cudf_velox {

CudfOrderBy::CudfOrderBy(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::OrderByNode>& orderByNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          orderByNode->outputType(),
          orderByNode->id(),
          "CudfOrderBy",
          nvtx3::rgb{64, 224, 208}, // Turquoise
          NvtxMethodFlag::kAll,
          std::nullopt,
          orderByNode),
      planNodeBoundary_{core::PlanNode::Boundary::kBoth},
      inputsAreSorted_{false} {
  initializeSortKeys(orderByNode->sortingKeys(), orderByNode->sortingOrders());
}

CudfOrderBy::CudfOrderBy(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::MergeExchangeNode>& mergeExchangeNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          mergeExchangeNode->outputType(),
          mergeExchangeNode->id(),
          "CudfOrderBy",
          nvtx3::rgb{64, 224, 208}, // Turquoise
          NvtxMethodFlag::kAll,
          std::nullopt,
          mergeExchangeNode),
      planNodeBoundary_{core::PlanNode::Boundary::kOutput},
      inputsAreSorted_{true} {
  initializeSortKeys(
      mergeExchangeNode->sortingKeys(), mergeExchangeNode->sortingOrders());
}

void CudfOrderBy::initializeSortKeys(
    const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys,
    const std::vector<core::SortOrder>& sortingOrders) {
  sortKeys_.reserve(sortingKeys.size());
  columnOrder_.reserve(sortingKeys.size());
  nullOrder_.reserve(sortingKeys.size());
  for (size_t i = 0; i < sortingKeys.size(); ++i) {
    const auto channel = exec::exprToChannel(sortingKeys[i].get(), outputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "OrderBy doesn't allow constant sorting keys");
    sortKeys_.push_back(channel);
    const auto& sortingOrder = sortingOrders[i];
    columnOrder_.push_back(
        sortingOrder.isAscending() ? cudf::order::ASCENDING
                                   : cudf::order::DESCENDING);
    nullOrder_.push_back(
        (sortingOrder.isNullsFirst() ^ !sortingOrder.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

void CudfOrderBy::doAddInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    inputs_.push_back(std::move(cudfInput));
  }
}

void CudfOrderBy::doNoMoreInput() {
  Operator::noMoreInput();

  if (inputs_.empty()) {
    return;
  }

  if (inputsAreSorted_ && inputs_.size() == 1) {
    // One sorted input is the result; it keeps its own stream. It is not
    // checked: unsorted, it would come out unsorted, as from
    // exec::MergeExchange, not corrupted as from cudf::merge.
    outputTable_ = std::move(inputs_.front());
    inputs_.clear();
    return;
  }

  // Where the inputs meet in the concatenated table, when they are sorted
  // runs to merge rather than rows to sort. cudf::merge runs k - 1 pairwise
  // merges of the two smallest tables, so its cost grows with the page count
  // while the sort's does not: on 16.8 M rows it was still ahead at 64 pages
  // with two long string keys, and behind with cheap keys from somewhere
  // between 4 and 64 pages on, depending on the keys (at worst 1.4x the sort
  // at 16 pages and 1.9x at 64, with five mixed keys). The cap bounds that
  // loss; the usual one page per upstream task keeps the win.
  const bool mergeRuns = inputsAreSorted_ && inputs_.size() <= kMaxMergeInputs;
  std::vector<cudf::size_type> splitPoints;
  int64_t numRows = 0;
  if (mergeRuns) {
    splitPoints.reserve(inputs_.size() - 1);
    for (size_t i = 0; i < inputs_.size(); ++i) {
      numRows += inputs_[i]->size();
      VELOX_CHECK_LE(
          numRows,
          std::numeric_limits<cudf::size_type>::max(),
          "Merge exchange input exceeds the cuDF row limit");
      if (i + 1 < inputs_.size()) {
        splitPoints.push_back(static_cast<cudf::size_type>(numRows));
      }
    }
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  // Using the output memory resource to allow spilling to CPU memory.
  auto tbl = getConcatenatedTable(
      std::exchange(inputs_, {}), outputType_, stream, get_output_mr());

  VELOX_CHECK_NOT_NULL(tbl);
  if (mergeRuns) {
    VELOX_CHECK_EQ(
        tbl->num_rows(),
        numRows,
        "The concatenated table does not hold the merge exchange input rows");
  }

  auto result = mergeRuns ? mergeSortedRuns(tbl->view(), splitPoints, stream)
                          : sortRows(tbl->view(), stream);
  auto const size = result->num_rows();
  outputTable_ = std::make_shared<CudfVector>(
      pool(), outputType_, size, std::move(result), stream);
}

std::unique_ptr<cudf::table> CudfOrderBy::sortRows(
    const cudf::table_view& rows,
    cuda::stream_ref stream) {
  auto keys = rows.select(sortKeys_);
  return cudf::sort_by_key(
      rows, keys, columnOrder_, nullOrder_, stream, get_output_mr());
}

std::unique_ptr<cudf::table> CudfOrderBy::mergeSortedRuns(
    const cudf::table_view& rows,
    const std::vector<cudf::size_type>& splitPoints,
    cuda::stream_ref stream) {
  // Merge the sort keys with a row index rather than the rows: cudf::merge
  // materializes every pairwise merge, so this keeps the other columns out of
  // its copies; the concatenation and the final gather copy them once each,
  // as the sort did.
  cudf::numeric_scalar<cudf::size_type> zero(0, true, stream, get_temp_mr());
  auto rowIndex = cudf::sequence(rows.num_rows(), zero, stream, get_temp_mr());
  std::vector<cudf::column_view> keyColumns;
  keyColumns.reserve(sortKeys_.size() + 1);
  for (const auto key : sortKeys_) {
    keyColumns.push_back(rows.column(key));
  }
  keyColumns.push_back(rowIndex->view());
  const auto runs =
      cudf::split(cudf::table_view{keyColumns}, splitPoints, stream);
  std::vector<cudf::size_type> keyIndices(sortKeys_.size());
  std::iota(keyIndices.begin(), keyIndices.end(), 0);

  // cudf::merge requires sorted inputs, and duplicates and drops rows rather
  // than returning them unsorted otherwise; nothing upstream enforces the
  // node's ordering: exec::MergeExchange trusts it, and the sort this replaces
  // did not need it. One host sync per input, once per query, keeps the sort's
  // guarantee for the merged inputs. The row index takes part in the check
  // because with a nested key cudf::is_sorted compares whole rows strictly and
  // would take a tie on every key for a descent; being strictly increasing,
  // the index breaks every tie.
  auto checkOrder = columnOrder_;
  checkOrder.push_back(cudf::order::ASCENDING);
  auto checkNullOrder = nullOrder_;
  checkNullOrder.push_back(cudf::null_order::AFTER);
  for (const auto& run : runs) {
    if (!cudf::is_sorted(run, checkOrder, checkNullOrder, stream)) {
      LOG(WARNING) << "CudfOrderBy: an input of the merge exchange is not "
                      "sorted by its keys; sorting the rows instead";
      stats_.wlock()->addRuntimeStat(kSortFallbacks, RuntimeCounter(1));
      return sortRows(rows, stream);
    }
  }

  auto merged = cudf::merge(
      runs, keyIndices, columnOrder_, nullOrder_, stream, get_temp_mr());
  stats_.wlock()->addRuntimeStat(
      kMergedInputs, RuntimeCounter(static_cast<int64_t>(runs.size())));

  return cudf::gather(
      rows,
      merged->view().column(sortKeys_.size()),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      cudf::memory_resources{get_output_mr(), get_temp_mr()});
}

RowVectorPtr CudfOrderBy::doGetOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  finished_ = true;
  return outputTable_;
}

void CudfOrderBy::doClose() {
  Operator::close();
  // Release stored inputs. close() can run before noMoreInput() (task
  // cancellation), in which case inputs_ still holds CudfVectors and this
  // clear is real work. After noMoreInput(), inputs_ was std::exchange'd
  // away and this is a no-op.
  //
  // outputTable_ is allocated by the sort or the merge's gather on the same
  // stream its kernels run on, or is an input handed on with its own stream;
  // rmm frees on the allocation stream, so the drop is already ordered behind
  // the work. No host sync.
  inputs_.clear();
  outputTable_.reset();
}
} // namespace facebook::velox::cudf_velox
