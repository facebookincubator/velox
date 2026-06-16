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
  VELOX_DCHECK_EQ(result.size(), nullSourceColumn.size());
  if (!nullSourceColumn.has_nulls()) {
    return;
  }

  if (!result.nullable()) {
    result.set_null_mask(
        cudf::copy_bitmask(nullSourceColumn, stream, mr),
        nullSourceColumn.null_count());
    return;
  }

  std::vector<cudf::bitmask_type const*> masks{
      result.view().null_mask(),
      nullSourceColumn.null_mask(),
  };
  std::vector<cudf::size_type> beginBits{0, nullSourceColumn.offset()};
  auto [nullMask, nullCount] =
      cudf::bitmask_and(masks, beginBits, result.size(), stream, mr);
  result.set_null_mask(std::move(nullMask), nullCount);
}

} // namespace facebook::velox::cudf_velox
