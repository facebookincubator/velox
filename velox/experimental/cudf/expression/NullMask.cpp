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
#include "velox/experimental/cudf/expression/NullMask.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>

#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {

void mergeNullSourceNullsIntoResult(
    cudf::column& result,
    cudf::column_view nullSourceColumn,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  mergeNullSourceNullsIntoResult(
      result, std::vector<cudf::column_view>{nullSourceColumn}, stream, mr);
}

void mergeNullSourceNullsIntoResult(
    cudf::column& result,
    const std::vector<cudf::column_view>& nullSourceColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<cudf::column_view> nullableSourceColumns;
  nullableSourceColumns.reserve(nullSourceColumns.size());
  for (auto nullSourceColumn : nullSourceColumns) {
    VELOX_DCHECK_EQ(result.size(), nullSourceColumn.size());
    if (nullSourceColumn.has_nulls()) {
      nullableSourceColumns.push_back(nullSourceColumn);
    }
  }
  if (nullableSourceColumns.empty()) {
    return;
  }

  if (!result.nullable() && nullableSourceColumns.size() == 1) {
    auto nullSourceColumn = nullableSourceColumns.front();
    result.set_null_mask(
        cudf::copy_bitmask(nullSourceColumn, stream, mr),
        nullSourceColumn.null_count());
    return;
  }

  std::vector<cudf::bitmask_type const*> masks;
  std::vector<cudf::size_type> beginBits;
  masks.reserve(nullableSourceColumns.size() + result.nullable());
  beginBits.reserve(nullableSourceColumns.size() + result.nullable());
  if (result.nullable()) {
    masks.push_back(result.view().null_mask());
    beginBits.push_back(0);
  }
  for (auto nullSourceColumn : nullableSourceColumns) {
    masks.push_back(nullSourceColumn.null_mask());
    beginBits.push_back(nullSourceColumn.offset());
  }
  auto [nullMask, nullCount] =
      cudf::bitmask_and(masks, beginBits, result.size(), stream, mr);
  result.set_null_mask(std::move(nullMask), nullCount);
}

} // namespace facebook::velox::cudf_velox
