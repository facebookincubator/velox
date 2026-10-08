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

void scatterGatheredColumns(
    const std::vector<exec::IdentityProjection>& projections,
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    std::vector<std::unique_ptr<cudf::column>>& gatheredCols) {
  VELOX_CHECK_EQ(gatheredCols.size(), projections.size());
  for (const auto& projection : projections) {
    VELOX_CHECK_LT(projection.outputChannel, outCols.size());
  }
  for (std::size_t i = 0; i < projections.size(); ++i) {
    outCols[projections[i].outputChannel] = std::move(gatheredCols[i]);
  }
}

void scatterInputColumns(
    const std::vector<exec::IdentityProjection>& projections,
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    std::vector<std::unique_ptr<cudf::column>>& inputCols,
    std::size_t srcOffset) {
  VELOX_CHECK_LE(srcOffset, inputCols.size());
  for (const auto& projection : projections) {
    VELOX_CHECK_LT(projection.inputChannel, inputCols.size() - srcOffset);
    VELOX_CHECK_LT(projection.outputChannel, outCols.size());
  }
  for (const auto& projection : projections) {
    outCols[projection.outputChannel] = std::move(
        inputCols
            [srcOffset + static_cast<std::size_t>(projection.inputChannel)]);
  }
}

void fillNullColumns(
    const std::vector<exec::IdentityProjection>& projections,
    const RowTypePtr& inputType,
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    cudf::size_type numRows,
    cuda::stream_ref stream) {
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

  probeColumnIndices_.reserve(probeProjections_.size());
  for (const auto& proj : probeProjections_) {
    probeColumnIndices_.push_back(
        static_cast<cudf::size_type>(proj.inputChannel));
  }
  buildColumnIndices_.reserve(buildProjections_.size());
  for (const auto& proj : buildProjections_) {
    buildColumnIndices_.push_back(
        static_cast<cudf::size_type>(proj.inputChannel));
  }
}

void CudfJoinOutputLayout::scatterGatheredProbeColumns(
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    std::vector<std::unique_ptr<cudf::column>>& gatheredCols) const {
  scatterGatheredColumns(probeProjections_, outCols, gatheredCols);
}

void CudfJoinOutputLayout::scatterGatheredBuildColumns(
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    std::vector<std::unique_ptr<cudf::column>>& gatheredCols) const {
  scatterGatheredColumns(buildProjections_, outCols, gatheredCols);
}

void CudfJoinOutputLayout::scatterProbeInputColumns(
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    std::vector<std::unique_ptr<cudf::column>>& inputCols,
    std::size_t srcOffset) const {
  scatterInputColumns(probeProjections_, outCols, inputCols, srcOffset);
}

void CudfJoinOutputLayout::scatterBuildInputColumns(
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    std::vector<std::unique_ptr<cudf::column>>& inputCols,
    std::size_t srcOffset) const {
  scatterInputColumns(buildProjections_, outCols, inputCols, srcOffset);
}

void CudfJoinOutputLayout::fillNullProbeColumns(
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    cudf::size_type numRows,
    cuda::stream_ref stream) const {
  fillNullColumns(probeProjections_, probeType_, outCols, numRows, stream);
}

void CudfJoinOutputLayout::fillNullBuildColumns(
    std::vector<std::unique_ptr<cudf::column>>& outCols,
    cudf::size_type numRows,
    cuda::stream_ref stream) const {
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
