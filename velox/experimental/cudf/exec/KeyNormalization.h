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

/// A key table whose TIMESTAMP WITH TIME ZONE columns have had their zone key
/// cleared, plus the columns backing the rewritten ones.
///
/// `owned` holds only the columns this normalization created; every other
/// column in `view` still refers to the caller's input. So `owned` must outlive
/// `view`, and `view` must not outlive the table it was built from.
struct NormalizedKeys {
  cudf::table_view view;
  std::vector<std::unique_ptr<cudf::column>> owned;

  /// True when at least one column was rewritten. Lets a caller skip any
  /// bookkeeping it only needs on the normalized path.
  bool normalizedAny() const {
    return !owned.empty();
  }
};

/// Rewrites the TIMESTAMP WITH TIME ZONE columns of a key table so that
/// hashing, equality and ordering over the result match Velox's semantics for
/// the type.
///
/// Why this is needed. A TIMESTAMP WITH TIME ZONE is physically one int64,
/// (millis << kMillisShift) | zone_key, and Velox compares it on the INSTANT
/// alone -- TimestampWithTimeZoneType::compare and ::hash both read
/// unpackMillisUtc, reached through the type's ProvideCustomComparison hook.
/// Two values for the same moment in different zones are therefore EQUAL, and
/// must group together, deduplicate to one, join to each other and hash to the
/// same partition. cuDF has no type-level hook: veloxToCudfDataType maps the
/// type to INT64, so every cuDF primitive that hashes or compares the column
/// uses all 64 bits and the zone key decides.
///
/// What it does. Clears the low kMillisShift bits, which computes
/// (v >> kMillisShift) << kMillisShift. That is monotone and collapses exactly
/// the values sharing an instant, so it preserves both equality and ordering --
/// everything a hash, a comparison and a sort need. It is also correct for
/// pre-epoch instants: two's-complement AND floors toward negative infinity,
/// matching unpackMillisUtc's arithmetic shift, where a divide by 4096 would
/// truncate toward zero and land a millisecond late.
///
/// Masking rather than shifting is deliberate. cudf::ast::ast_operator has
/// BITWISE_AND but no shift operator, so the expression-level fix in
/// AstExpressionUtils.h had to mask; this uses the same operation so the two
/// halves of the fix cannot drift.
///
/// WHAT THIS IS NOT FOR. A normalized value is equivalent for EQUALITY and
/// ORDERING ONLY. It must never be emitted as an operator's output: the packed
/// value is what a projection has to carry and what to_iso8601 and
/// timezone_hour have to read, so a normalized value flowing downstream would
/// silently rewrite every zone key to UTC. Callers whose primitive RETURNS its
/// key columns (cudf::groupby, and cudf::distinct as it is called in
/// CudfDistinct) must therefore not hand the result of this function on as
/// output -- they need to recover the original packed value separately.
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
