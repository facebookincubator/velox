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

#include "velox/type/Type.h"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox {

struct CudfJoinOutputLayout {
  CudfJoinOutputLayout() = default;

  CudfJoinOutputLayout(
      const RowTypePtr& probeType,
      const RowTypePtr& buildType,
      const RowTypePtr& outputType,
      std::optional<std::size_t> syntheticOutputPosition = std::nullopt);

  std::vector<cudf::size_type> probeColumnIndices;
  std::vector<cudf::size_type> buildColumnIndices;
  std::vector<std::size_t> probeColumnOutputPositions;
  std::vector<std::size_t> buildColumnOutputPositions;
};

/// Appends precomputed columns to a table view. The returned view is valid only
/// while the original columns and precomputed columns remain alive.
cudf::table_view makeExtendedTableView(
    cudf::table_view originalView,
    std::vector<ColumnOrView>& precomputedColumns);

} // namespace facebook::velox::cudf_velox
