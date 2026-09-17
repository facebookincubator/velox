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

#include "velox/type/Type.h"

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// A complete key view with TIMESTAMP WITH TIME ZONE columns normalized.
struct NormalizedKeys {
  /// Unchanged columns refer to the caller's input.
  cudf::table_view view;

  /// Newly allocated normalized columns. This is not a complete key table.
  std::vector<std::unique_ptr<cudf::column>> owned;

  /// True when at least one column was rewritten. Lets a caller skip any
  /// bookkeeping it only needs on the normalized path.
  bool normalizedAny() const {
    return !owned.empty();
  }
};

/// Clears the zone-key bits of TIMESTAMP WITH TIME ZONE columns so cuDF hashes
/// and compares them by instant, matching Velox semantics. The transformation
/// preserves ordering, including for pre-epoch instants.
///
/// Normalized values are for key comparison only. Operators must emit the
/// original packed values to preserve their zone keys.
///
/// @param keys       the key columns, already selected in key order
/// @param rowType    the Velox row type the key columns came from
/// @param keyChannels index into `rowType` for each column of `keys`, in the
/// same
///                   order; must be the same length as `keys`
/// @param stream, mr  the stream and resource any rewritten column is built on
NormalizedKeys normalizeKeyColumns(
    cudf::table_view keys,
    const RowTypePtr& rowType,
    const std::vector<column_index_t>& keyChannels,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Overload for callers that already know which columns are TIMESTAMP WITH TIME
/// ZONE, e.g. because they cached the flags at operator construction. `isTswtz`
/// must be the same length as `keys`.
NormalizedKeys normalizeKeyColumns(
    cudf::table_view keys,
    const std::vector<bool>& isTswtz,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Whether any column of `rowType` named by `keyChannels` is a TIMESTAMP WITH
/// TIME ZONE. Lets an operator decide at construction time whether it will ever
/// need to normalize, so the common case costs one bool rather than a scan per
/// batch.
bool anyKeyNeedsNormalization(
    const RowTypePtr& rowType,
    const std::vector<column_index_t>& keyChannels);

} // namespace facebook::velox::cudf_velox
