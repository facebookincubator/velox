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
#include "velox/experimental/cudf/exec/CudfRowNumber.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <utility>

namespace facebook::velox::cudf_velox {

CudfRowNumber::CudfRowNumber(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::RowNumberNode>& planNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          planNode->outputType(),
          planNode->id(),
          "CudfRowNumber",
          nvtx3::rgb{255, 140, 0}, // Dark orange
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput,
          std::nullopt,
          planNode),
      limit_(planNode->limit()),
      generateRowNumber_(planNode->generateRowNumber()),
      cudaEvent_(std::make_unique<CudaEvent>(cudaEventDisableTiming)) {
  const auto& inputType = planNode->sources()[0]->outputType();
  for (const auto& key : planNode->partitionKeys()) {
    const auto channel =
        static_cast<cudf::size_type>(exec::exprToChannel(key.get(), inputType));
    if (std::find(
            partitionKeyIndices_.begin(),
            partitionKeyIndices_.end(),
            channel) == partitionKeyIndices_.end()) {
      partitionKeyIndices_.push_back(channel);
    }
  }
}

void CudfRowNumber::doAddInput(RowVectorPtr input) {
  VELOX_CHECK_NULL(input_);
  input_ = std::move(input);
}

void CudfRowNumber::doClose() {
  // stateJoin_ references stateKeys_. State reads are ordered back onto the
  // state streams in computePartitionedRowNumbers(), so stream-ordered
  // destruction is safe.
  stateJoin_.reset();
  stateKeys_.reset();
  stateCounts_.reset();
  keysStream_.reset();
  countsStream_.reset();
  cudaEvent_.reset();
  Operator::close();
}

std::unique_ptr<cudf::column> CudfRowNumber::computePartitionedRowNumbers(
    cudf::table_view keys,
    cuda::stream_ref stream) {
  auto tempMr = get_temp_mr();
  const auto numKeys = keys.num_columns();
  const std::vector<cudf::order> columnOrder(numKeys, cudf::order::ASCENDING);
  const std::vector<cudf::null_order> nullOrder(
      numKeys, cudf::null_order::BEFORE);

  // Stable sorting keeps each partition's rows in arrival order.
  auto order =
      cudf::stable_sorted_order(keys, columnOrder, nullOrder, stream, tempMr);
  std::unique_ptr<cudf::table> groupKeys;
  std::unique_ptr<cudf::column> groupSizes;
  {
    auto sortedKeys = cudf::gather(
        keys,
        order->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        cudf::memory_resources{tempMr, tempMr});
    cudf::groupby::groupby groups(
        sortedKeys->view(),
        cudf::null_policy::INCLUDE,
        cudf::sorted::YES,
        columnOrder,
        nullOrder);
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values = sortedKeys->get_column(0).view();
    requests[0].aggregations.push_back(
        cudf::make_count_aggregation<cudf::groupby_aggregation>(
            cudf::null_policy::INCLUDE));
    // Sort aggregation returns keys and counts in consecutive group order.
    auto [uniqueKeys, results] = groups.aggregate(requests, stream, tempMr);
    groupKeys = std::move(uniqueKeys);
    groupSizes = std::move(results[0].results[0]);
  }

  const std::vector<cuda::stream_ref> stateStreams = stateJoin_
      ? std::vector<cuda::stream_ref>{*keysStream_, *countsStream_}
      : std::vector<cuda::stream_ref>{};
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> groupStateIndices;
  std::unique_ptr<cudf::column> sortedRowNumbers;
  std::vector<std::unique_ptr<cudf::column>> groupColumns;
  {
    auto sum = cudf::make_sum_aggregation<cudf::scan_aggregation>();
    // COUNT_ALL and its prefix sums fit in size_type: their total is N.
    auto groupStarts = cudf::scan(
        groupSizes->view(),
        *sum,
        cudf::scan_type::EXCLUSIVE,
        cudf::null_policy::EXCLUDE,
        stream,
        tempMr);
    cudf::numeric_scalar<int64_t> zero(0, true, stream, tempMr);
    std::unique_ptr<cudf::column> priorCounts;
    if (stateJoin_) {
      cudf::detail::join_streams(stateStreams, stream);
      groupStateIndices =
          stateJoin_->left_join(groupKeys->view(), stream, tempMr);
      auto gatheredCounts = cudf::gather(
          cudf::table_view{{stateCounts_->view()}},
          cudf::column_view{
              cudf::device_span<cudf::size_type const>{*groupStateIndices}},
          cudf::out_of_bounds_policy::NULLIFY,
          cudf::negative_index_policy::NOT_ALLOWED,
          stream,
          cudf::memory_resources{tempMr, tempMr});
      priorCounts = cudf::replace_nulls(
          gatheredCounts->get_column(0).view(), zero, stream, tempMr);
    }
    groupColumns = groupKeys->release();
    groupKeys.reset();
    const cudf::data_type int64Type{cudf::type_id::INT64};
    groupColumns.push_back(
        priorCounts
            ? cudf::binary_operation(
                  priorCounts->view(),
                  groupSizes->view(),
                  cudf::binary_operator::ADD,
                  int64Type,
                  stream,
                  tempMr)
            : cudf::cast(groupSizes->view(), int64Type, stream, tempMr));
    // Mixed INT64/INT32 operations promote before arithmetic. Adjustments
    // can be negative when a group's start exceeds its prior count.
    auto adjustment = priorCounts ? cudf::binary_operation(
                                        priorCounts->view(),
                                        groupStarts->view(),
                                        cudf::binary_operator::SUB,
                                        int64Type,
                                        stream,
                                        tempMr)
                                  : cudf::binary_operation(
                                        zero,
                                        groupStarts->view(),
                                        cudf::binary_operator::SUB,
                                        int64Type,
                                        stream,
                                        tempMr);
    priorCounts.reset();
    groupStarts.reset();
    auto repeatedAdjustment = cudf::repeat(
        cudf::table_view{{adjustment->view()}},
        groupSizes->view(),
        stream,
        tempMr);
    adjustment.reset();
    groupSizes.reset();
    cudf::numeric_scalar<cudf::size_type> positionOne(1, true, stream, tempMr);
    auto positions = cudf::sequence(keys.num_rows(), positionOne, stream, tempMr);
    sortedRowNumbers = cudf::binary_operation(
        positions->view(),
        repeatedAdjustment->get_column(0).view(),
        cudf::binary_operator::ADD,
        int64Type,
        stream,
        tempMr);
  }

  // Restore the arrival order. The row numbers are only output when they are
  // generated and there is no limit; otherwise they are a temporary.
  auto rowNumbers = std::move(
      cudf::scatter(
          cudf::table_view{{sortedRowNumbers->view()}},
          order->view(),
          cudf::table_view{{sortedRowNumbers->view()}},
          stream,
          generateRowNumber_ && !limit_.has_value() ? get_output_mr() : tempMr)
          ->release()[0]);
  sortedRowNumbers.reset();
  order.reset();

  // Split the partitions into seen ones, whose counts are updated in place,
  // and new ones, which are appended to the state.
  auto groups = std::make_unique<cudf::table>(std::move(groupColumns));
  std::unique_ptr<cudf::table> matchedGroups;
  if (groupStateIndices) {
    const cudf::column_view indices{
        cudf::device_span<cudf::size_type const>{*groupStateIndices}};
    cudf::numeric_scalar<cudf::size_type> noMatch(
        cudf::JoinNoMatch, true, stream, tempMr);
    auto matched = cudf::binary_operation(
        indices,
        noMatch,
        cudf::binary_operator::NOT_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        tempMr);
    matchedGroups = cudf::apply_retention_mask(
        cudf::table_view{{groups->get_column(numKeys).view(), indices}},
        matched->view(),
        stream,
        tempMr);
    groups = cudf::apply_deletion_mask(
        groups->view(), matched->view(), stream, tempMr);
  }

  groupStateIndices.reset();
  auto newColumns = groups->release();
  auto newCounts = std::move(newColumns.back());
  newColumns.pop_back();
  auto newKeys = std::make_unique<cudf::table>(std::move(newColumns));
  if (matchedGroups) {
    if (newKeys->num_rows() > 0) {
      std::vector<cudf::table_view> keysToConcat{
          stateKeys_->view(), newKeys->view()};
      newKeys = cudf::concatenate(keysToConcat, stream, tempMr);
      std::vector<cudf::column_view> countsToConcat{
          stateCounts_->view(), newCounts->view()};
      auto allCounts = cudf::concatenate(countsToConcat, stream, tempMr);
      newCounts.reset();
      // Free the old counts before the scatter below, so at most two count
      // columns of D rows coexist.
      const std::vector<cuda::stream_ref> countsStreams{*countsStream_};
      streamsWaitForStream(*cudaEvent_, countsStreams, stream);
      stateCounts_ = std::move(allCounts);
      countsStream_ = stream;
    } else {
      newKeys.reset();
    }
    if (matchedGroups->num_rows() > 0) {
      newCounts = std::move(
          cudf::scatter(
              cudf::table_view{{matchedGroups->get_column(0).view()}},
              matchedGroups->get_column(1).view(),
              cudf::table_view{{stateCounts_->view()}},
              stream,
              tempMr)
              ->release()[0]);
    } else {
      newCounts = std::move(stateCounts_);
    }
  }

  matchedGroups.reset();

  // Make the old state streams wait for the last state read on this stream
  // before the old state is freed or read on another input stream.
  streamsWaitForStream(*cudaEvent_, stateStreams, stream);
  stateCounts_ = std::move(newCounts);
  countsStream_ = stream;
  if (newKeys) {
    auto newJoin = std::make_unique<cudf::distinct_hash_join>(
        newKeys->view(), cudf::null_equality::EQUAL, 0.5, stream, tempMr);
    stateJoin_ = std::move(newJoin);
    stateKeys_ = std::move(newKeys);
    keysStream_ = stream;
  }
  return rowNumbers;
}

RowVectorPtr CudfRowNumber::doGetOutput() {
  if (input_ == nullptr) {
    return nullptr;
  }

  if (input_->size() == 0) {
    input_ = nullptr;
    return nullptr;
  }
  if (!generateRowNumber_ && !limit_) {
    return std::exchange(input_, nullptr);
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfRowNumber expects CudfVector input");
  input_ = nullptr;

  auto* pool = cudfInput->pool();
  const auto stream = cudfInput->stream();
  auto outputMr = get_output_mr();
  auto tableView = cudfInput->getTableView();
  // A zero-column cuDF table has no rows, so take the row count from the
  // vector.
  const auto numRows = static_cast<cudf::size_type>(cudfInput->size());

  std::unique_ptr<cudf::table> outputTable;
  vector_size_t outputSize;
  if (partitionKeyIndices_.empty()) {
    auto numOutput = numRows;
    if (limit_.has_value()) {
      VELOX_CHECK_LT(numTotalInput_, limit_.value());
      numOutput = static_cast<cudf::size_type>(
          std::min<int64_t>(numRows, limit_.value() - numTotalInput_));
      finishedEarly_ = numTotalInput_ + numOutput == limit_.value();
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    if (numOutput < numRows) {
      auto slice = cudf::slice(tableView, {0, numOutput}, stream)[0];
      columns = cudf::table(slice, stream, outputMr).release();
      VELOX_CHECK(cudfInput->rebindStream(stream));
      cudfInput.reset();
    } else {
      columns = cudfInput->release()->release();
    }
    if (generateRowNumber_) {
      cudf::numeric_scalar<int64_t> first(
          numTotalInput_ + 1, true, stream, get_temp_mr());
      columns.push_back(cudf::sequence(numOutput, first, stream, outputMr));
    }
    numTotalInput_ += numOutput;
    outputTable = std::make_unique<cudf::table>(std::move(columns));
    outputSize = numOutput;
  } else {
    auto rowNumbers = computePartitionedRowNumbers(
        tableView.select(partitionKeyIndices_), stream);
    if (limit_.has_value()) {
      cudf::numeric_scalar<int64_t> limit(
          limit_.value(), true, stream, get_temp_mr());
      auto mask = cudf::binary_operation(
          rowNumbers->view(),
          limit,
          cudf::binary_operator::LESS_EQUAL,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          get_temp_mr());
      std::vector<cudf::column_view> columns(
          tableView.begin(), tableView.end());
      if (generateRowNumber_) {
        columns.push_back(rowNumbers->view());
      }
      outputTable = cudf::apply_retention_mask(
          cudf::table_view{columns}, mask->view(), stream, outputMr);
      if (outputTable->num_rows() == 0) {
        return nullptr;
      }
    } else {
      auto columns = cudfInput->release()->release();
      if (generateRowNumber_) {
        columns.push_back(std::move(rowNumbers));
      }
      outputTable = std::make_unique<cudf::table>(std::move(columns));
    }
    // The table has at least the partition key columns.
    outputSize = outputTable->num_rows();
  }

  return std::make_shared<CudfVector>(
      pool, outputType_, outputSize, std::move(outputTable), stream);
}

} // namespace facebook::velox::cudf_velox
