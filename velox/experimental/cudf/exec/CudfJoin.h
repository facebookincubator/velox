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

#pragma once

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/type/Type.h"

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Maps a join's probe and build input columns to their positions in the
/// output row. The mapping is stored as IdentityProjection pairs
/// ({inputChannel, outputChannel}) per side, so an input index and its output
/// position cannot drift apart.
struct CudfJoinOutputLayout {
  CudfJoinOutputLayout() = default;

  /// Resolves each column of outputType to a probe or build input by name.
  /// joinType determines whether outputType has a synthetic trailing column
  /// present in neither input (the boolean match column of a LEFT SEMI
  /// PROJECT), which is skipped so the caller can fill it separately.
  CudfJoinOutputLayout(
      const RowTypePtr& probeType,
      const RowTypePtr& buildType,
      const RowTypePtr& outputType,
      core::JoinType joinType);

  /// Places cols[i] at the output position of probe/build projection i.
  /// cols must hold exactly the gathered columns of that side, in projection
  /// order (e.g. the result of gathering a select() of that side's input).
  void scatterProbeColumns(
      std::vector<std::unique_ptr<cudf::column>>& outCols,
      std::vector<std::unique_ptr<cudf::column>>& cols) const;
  void scatterBuildColumns(
      std::vector<std::unique_ptr<cudf::column>>& outCols,
      std::vector<std::unique_ptr<cudf::column>>& cols) const;

  /// Places cols[srcOffset + inputChannel] at the output position of each
  /// probe/build projection. Use when cols is a combined table of both
  /// sides' input columns (e.g. [probe inputs..., build inputs...]).
  void scatterProbeColumns(
      std::vector<std::unique_ptr<cudf::column>>& outCols,
      std::vector<std::unique_ptr<cudf::column>>& cols,
      std::size_t srcOffset) const;
  void scatterBuildColumns(
      std::vector<std::unique_ptr<cudf::column>>& outCols,
      std::vector<std::unique_ptr<cudf::column>>& cols,
      std::size_t srcOffset) const;

  /// Fills this side's output positions with all-null columns of numRows
  /// rows, with dtypes taken from the input type. Used to emit unmatched rows
  /// of the opposite side for outer joins.
  void fillNullProbeColumns(
      std::vector<std::unique_ptr<cudf::column>>& outCols,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream) const;
  void fillNullBuildColumns(
      std::vector<std::unique_ptr<cudf::column>>& outCols,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream) const;

  /// Read-only access to the probe-side projections, for call sites that
  /// copy from column_views instead of moving gathered columns.
  const std::vector<exec::IdentityProjection>& probeProjections() const {
    return probeProjections_;
  }

  /// Probe/build input column indices derived from the projections, in
  /// projection order, for cudf table_view::select().
  std::vector<cudf::size_type> probeColumnIndices;
  std::vector<cudf::size_type> buildColumnIndices;

 private:
  // Source of truth for the input-to-output mapping of each side.
  std::vector<exec::IdentityProjection> probeProjections_;
  std::vector<exec::IdentityProjection> buildProjections_;
  // Kept for the null-fill column dtypes.
  RowTypePtr probeType_;
  RowTypePtr buildType_;
};

/// Appends precomputed columns to a table view. The returned view is valid only
/// while the original columns and precomputed columns remain alive.
cudf::table_view makeExtendedTableView(
    cudf::table_view originalView,
    std::vector<ColumnOrView>& precomputedColumns);

} // namespace facebook::velox::cudf_velox
