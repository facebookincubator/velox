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
#include "velox/experimental/cudf/exec/CudfJoin.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <optional>

namespace facebook::velox::cudf_velox {

namespace {

void scatterColumns(
    const std::vector<exec::IdentityProjection>& projections,
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    std::vector<std::unique_ptr<cudf::column>>& cols) {
  for (std::size_t i = 0; i < projections.size(); ++i) {
    outCols[projections[i].outputChannel] = std::move(cols[i]);
  }
}

void scatterColumns(
    const std::vector<exec::IdentityProjection>& projections,
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    std::vector<std::unique_ptr<cudf::column>>& cols,
    std::size_t srcOffset) {
  for (const auto& proj : projections) {
    outCols[proj.outputChannel] =
        std::move(cols[srcOffset + proj.inputChannel]);
  }
}

void fillNullColumns(
    const std::vector<exec::IdentityProjection>& projections,
    const RowTypePtr& inputType,
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) {
  for (const auto& proj : projections) {
    auto cudfDataType =
        veloxToCudfDataType(inputType->childAt(proj.inputChannel));
    auto nullScalar = cudf::make_default_constructed_scalar(
        cudfDataType, stream, get_temp_mr());
    outCols[proj.outputChannel] = cudf::make_column_from_scalar(
        *nullScalar, numRows, stream, get_output_mr());
  }
}

} // namespace

CudfJoinOutputLayout::CudfJoinOutputLayout(
    const RowTypePtr& probeType,
    const RowTypePtr& buildType,
    const RowTypePtr& outputType,
    core::JoinType joinType)
    : probeType_(probeType), buildType_(buildType) {
  // For kLeftSemiProject, the last output column is a BOOLEAN match flag
  // that doesn't exist in probe or build types — skip it during resolution.
  std::optional<std::size_t> syntheticOutputPosition;
  if (core::isLeftSemiProjectJoin(joinType)) {
    VELOX_CHECK_GT(outputType->size(), 0);
    VELOX_CHECK_EQ(
        outputType->childAt(outputType->size() - 1)->kind(),
        TypeKind::BOOLEAN,
        "Trailing output column of a LEFT SEMI PROJECT join must be BOOLEAN");
    syntheticOutputPosition = outputType->size() - 1;
  }

  for (std::size_t outputPosition = 0; outputPosition < outputType->size();
       ++outputPosition) {
    if (syntheticOutputPosition == outputPosition) {
      continue;
    }

    const auto& outputName = outputType->nameOf(outputPosition);
    if (auto probeIndex = probeType->getChildIdxIfExists(outputName)) {
      probeProjections_.emplace_back(
          *probeIndex, static_cast<column_index_t>(outputPosition));
      continue;
    }
    if (auto buildIndex = buildType->getChildIdxIfExists(outputName)) {
      buildProjections_.emplace_back(
          *buildIndex, static_cast<column_index_t>(outputPosition));
      continue;
    }
    VELOX_FAIL("Join field {} not in probe or build input", outputName);
  }

  probeColumnIndices.reserve(probeProjections_.size());
  for (const auto& proj : probeProjections_) {
    probeColumnIndices.push_back(
        static_cast<cudf::size_type>(proj.inputChannel));
  }
  buildColumnIndices.reserve(buildProjections_.size());
  for (const auto& proj : buildProjections_) {
    buildColumnIndices.push_back(
        static_cast<cudf::size_type>(proj.inputChannel));
  }
}

void CudfJoinOutputLayout::scatterProbeColumns(
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    std::vector<std::unique_ptr<cudf::column>>& cols) const {
  scatterColumns(probeProjections_, outCols, cols);
}

void CudfJoinOutputLayout::scatterBuildColumns(
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    std::vector<std::unique_ptr<cudf::column>>& cols) const {
  scatterColumns(buildProjections_, outCols, cols);
}

void CudfJoinOutputLayout::scatterProbeColumns(
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    std::vector<std::unique_ptr<cudf::column>>& cols,
    std::size_t srcOffset) const {
  scatterColumns(probeProjections_, outCols, cols, srcOffset);
}

void CudfJoinOutputLayout::scatterBuildColumns(
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    std::vector<std::unique_ptr<cudf::column>>& cols,
    std::size_t srcOffset) const {
  scatterColumns(buildProjections_, outCols, cols, srcOffset);
}

void CudfJoinOutputLayout::fillNullProbeColumns(
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const {
  fillNullColumns(probeProjections_, probeType_, outCols, numRows, stream);
}

void CudfJoinOutputLayout::fillNullBuildColumns(
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const {
  fillNullColumns(buildProjections_, buildType_, outCols, numRows, stream);
}

cudf::table_view makeExtendedTableView(
    cudf::table_view originalView,
    std::vector<ColumnOrView>& precomputedColumns) {
  if (precomputedColumns.empty()) {
    return originalView;
  }

  std::vector<cudf::column_view> allViews;
  allViews.reserve(originalView.num_columns() + precomputedColumns.size());
  for (cudf::size_type i = 0; i < originalView.num_columns(); ++i) {
    allViews.push_back(originalView.column(i));
  }
  for (auto& column : precomputedColumns) {
    allViews.push_back(asView(column));
  }
  return cudf::table_view(allViews);
}

} // namespace facebook::velox::cudf_velox
