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

#include "velox/experimental/cudf/exec/CudfJoin.h"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::cudf_velox {

CudfJoinOutputLayout::CudfJoinOutputLayout(
    const RowTypePtr& probeType,
    const RowTypePtr& buildType,
    const RowTypePtr& outputType,
    std::optional<std::size_t> syntheticOutputPosition) {
  if (syntheticOutputPosition.has_value()) {
    VELOX_CHECK_LT(*syntheticOutputPosition, outputType->size());
  }

  for (std::size_t outputPosition = 0; outputPosition < outputType->size();
       ++outputPosition) {
    if (syntheticOutputPosition == outputPosition) {
      continue;
    }

    const auto& outputName = outputType->nameOf(outputPosition);
    if (auto probeIndex = probeType->getChildIdxIfExists(outputName)) {
      probeColumnIndices.push_back(
          static_cast<cudf::size_type>(*probeIndex));
      probeColumnOutputPositions.push_back(outputPosition);
      continue;
    }
    if (auto buildIndex = buildType->getChildIdxIfExists(outputName)) {
      buildColumnIndices.push_back(
          static_cast<cudf::size_type>(*buildIndex));
      buildColumnOutputPositions.push_back(outputPosition);
      continue;
    }
    VELOX_FAIL("Join field {} not in probe or build input", outputName);
  }
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
