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
#include "velox/experimental/cudf/exec/CudfMarkDistinct.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/error.hpp>

#include <fmt/format.h>

namespace facebook::velox::cudf_velox {

CudfMarkDistinct::CudfMarkDistinct(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::MarkDistinctNode>& planNode)
    : exec::Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "CudfMarkDistinct"),
      NvtxHelper(
          nvtx3::rgb{255, 165, 0}, // Orange
          operatorId,
          fmt::format("[{}]", planNode->id())) {
  const auto& inputType = planNode->sources()[0]->outputType();
  for (const auto& key : planNode->distinctKeys()) {
    auto idx = inputType->getChildIdx(key->name());
    distinctKeyIndices_.push_back(static_cast<cudf::size_type>(idx));
  }
}

void CudfMarkDistinct::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  VELOX_CHECK_NULL(input_);
  input_ = std::move(input);
}

RowVectorPtr CudfMarkDistinct::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (input_ == nullptr) {
    return nullptr;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfMarkDistinct expects CudfVector input");

  auto stream = cudfInput->stream();
  auto outputMr = get_output_mr();
  auto tempMr = get_temp_mr();
  auto tableView = cudfInput->getTableView();
  auto numRows = static_cast<cudf::size_type>(tableView.num_rows());

  if (numRows == 0) {
    input_ = nullptr;
    return nullptr;
  }

  // Extract key columns from the input batch.
  auto batchKeys = tableView.select(distinctKeyIndices_);

  // Create marker column (all false initially).
  cudf::numeric_scalar<bool> falseScalar(false, true, stream, tempMr);
  auto markerCol =
      cudf::make_column_from_scalar(falseScalar, numRows, stream, outputMr);

  // Find first occurrences within this batch (returns a column of indices).
  auto batchDistinctIdxCol = cudf::distinct_indices(
      batchKeys,
      cudf::duplicate_keep_option::KEEP_FIRST,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream,
      tempMr);

  // column_view used to pass indices to gather/scatter.
  std::unique_ptr<cudf::column> newRowIndicesCol;

  if (seenFilter_ == nullptr) {
    // First batch: all first-occurrence rows are new.
    newRowIndicesCol = std::move(batchDistinctIdxCol);

    // Initialize seenKeys_ by gathering the unique key rows using the indices
    // we already computed.
    seenKeys_ = cudf::gather(
        batchKeys,
        newRowIndicesCol->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        tempMr);
    seenFilter_ = std::make_unique<cudf::filtered_join>(
        seenKeys_->view(),
        cudf::null_equality::EQUAL,
        cudf::set_as_build_table::RIGHT,
        stream);

  } else {
    // Subsequent batch: probe the persistent filter — no hash table rebuild.

    // Gather the unique keys from this batch.
    auto uniqueBatchKeys = cudf::gather(
        batchKeys,
        batchDistinctIdxCol->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        tempMr);

    // Anti-join against the persistent seenFilter_ to find new keys.
    auto newKeyLocalIndices =
        seenFilter_->anti_join(uniqueBatchKeys->view(), stream, tempMr);

    if (!newKeyLocalIndices->is_empty()) {
      // Map local indices back to original batch row indices via gather.
      // Wrap device_uvector as column_view (same pattern as CudfHashJoin).
      auto localSpan =
          cudf::device_span<cudf::size_type const>{*newKeyLocalIndices};
      auto localCol = cudf::column_view{localSpan};
      auto remappedCol = cudf::gather(
          cudf::table_view{{batchDistinctIdxCol->view()}},
          localCol,
          cudf::out_of_bounds_policy::DONT_CHECK,
          stream,
          tempMr);
      newRowIndicesCol = std::move(remappedCol->release()[0]);

      // Append only the new unique keys to seenKeys_ and rebuild the filter.
      auto newKeys = cudf::gather(
          uniqueBatchKeys->view(),
          localCol,
          cudf::out_of_bounds_policy::DONT_CHECK,
          stream,
          tempMr);

      // Append new keys and rebuild the filter. This concatenates all seen
      // keys on every batch that introduces new keys, which is O(D) per such
      // batch. An amortized-doubling scheme (accumulate in a pending table,
      // consolidate when it reaches the size of seenKeys_) would reduce total
      // copy work to O(D), but that pattern does not yet exist in this
      // codebase. CudfHashAggregation::computePartialGroupbyStreaming and
      // computePartialDistinctStreaming use the same unconditional
      // concatenate-per-batch idiom.
      std::vector<cudf::table_view> seenPlusNew = {
          seenKeys_->view(), newKeys->view()};
      seenKeys_ = cudf::concatenate(seenPlusNew, stream, tempMr);
      seenFilter_ = std::make_unique<cudf::filtered_join>(
          seenKeys_->view(),
          cudf::null_equality::EQUAL,
          cudf::set_as_build_table::RIGHT,
          stream);
    }
  }

  // Scatter TRUE at new row positions.
  if (newRowIndicesCol && newRowIndicesCol->size() > 0) {
    cudf::numeric_scalar<bool> trueScalar(true, true, stream, tempMr);
    std::vector<std::reference_wrapper<const cudf::scalar>> sources = {
        trueScalar};
    cudf::table_view markerTableView({markerCol->view()});
    auto scatteredTable = cudf::scatter(
        sources, newRowIndicesCol->view(), markerTableView, stream, outputMr);
    markerCol = std::move(scatteredTable->release()[0]);
  }

  // Append marker column to output.
  auto pool = cudfInput->pool();
  auto size = cudfInput->size();
  auto columns = cudfInput->release()->release();
  columns.push_back(std::move(markerCol));
  cudfInput.reset();
  input_ = nullptr;
  return std::make_shared<CudfVector>(
      pool,
      outputType_,
      size,
      std::make_unique<cudf::table>(std::move(columns)),
      stream);
}

} // namespace facebook::velox::cudf_velox
