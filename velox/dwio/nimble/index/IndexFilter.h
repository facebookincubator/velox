/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <optional>
#include <vector>

#include "velox/common/memory/MemoryPool.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/nimble/index/SortOrder.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/type/Filter.h"
#include "velox/type/Type.h"

namespace facebook::nimble {

/// Result of converting filters to index bounds.
struct FilterToIndexBoundsResult {
  /// The converted index bounds.
  velox::serializer::IndexBounds indexBounds;

  /// Filters that were removed from the ScanSpec during conversion.
  /// These can be used to restore the filters when the row reader completes,
  /// which is needed when the ScanSpec is shared across multiple split readers.
  std::vector<std::pair<std::string, std::shared_ptr<velox::common::Filter>>>
      removedFilters;
};

/// Converts filters from ScanSpec to IndexBounds for index-based filtering.
///
/// This utility examines the ScanSpec and index column names to extract filters
/// that can be converted to IndexBounds for efficient index-based pruning.
/// Filters that are fully captured by the index bounds are removed from the
/// ScanSpec directly.
///
/// IMPORTANT: The index columns provided can be a prefix of the file's index
/// columns. For example, if the file has index columns [A, B, C], you can pass
/// [A] or [A, B] to create bounds for prefix scans. The filters must still form
/// a prefix of the provided index columns.
///
/// For example, with provided index columns [A, B, C]:
/// - Filter on A only: valid, creates bounds for A
/// - Filters on A and B: valid, creates bounds for A and B
/// - Filters on A, B, and C: valid, creates bounds for all
/// - Filter on B only: invalid, cannot skip A (returns empty)
/// - Filters on A and C: creates bounds for A only (stops at missing B)
///
/// Sort Order Handling:
/// For DESC columns, the bounds are inverted when creating index bounds:
/// - Filter "x >= 10" on DESC column becomes upper bound (values decrease)
/// - Filter "x <= 100" on DESC column becomes lower bound (values decrease)
/// This ensures correct range scanning regardless of column sort order.
///
/// Supported filter types:
/// - kBigintRange: Integer range filters (e.g., x >= 10 AND x <= 100)
/// - kDoubleRange: Double range filters
/// - kFloatRange: Float range filters
/// - kBytesRange: String/bytes range filters
/// - kBytesValues: String/bytes values filter (single value only)
/// - kBoolValue: Boolean filters
/// - kTimestampRange: Timestamp range filters
///
/// @param indexColumns Ordered list of index column names. Can be a prefix of
///        the file's index columns. Must be in index definition order.
/// @param sortOrders Sort order for each index column. Must have the same size
///        as indexColumns.
/// @param rowType The row type containing column type information.
/// @param scanSpec The ScanSpec containing column filters. Filters that are
///        fully captured by the index bounds will be removed from this
///        ScanSpec.
/// @param pool Memory pool for allocations.
/// @return The result containing the converted IndexBounds and the removed
///         filters if filters can be converted, std::nullopt otherwise.
std::optional<FilterToIndexBoundsResult> convertFilterToIndexBounds(
    const std::vector<std::string>& indexColumns,
    const std::vector<SortOrder>& sortOrders,
    const velox::RowTypePtr& rowType,
    velox::common::ScanSpec& scanSpec,
    velox::memory::MemoryPool* pool);

} // namespace facebook::nimble
