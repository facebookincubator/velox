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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Column-level primitives for the packed Presto TIMESTAMP WITH TIME ZONE
/// representation (upper 52 bits UTC millis, lower 12 bits time-zone key). All
/// operate per row and support columns that mix zone keys, matching the rest of
/// the GPU TSWTZ family. Shared by the timezone functions and the
/// date_trunc/date_add TSWTZ overloads.

/// Returns the per-row zone key (INT64, nulls preserved) of a packed column:
/// packed & kTimezoneMask. A null packed row yields a null key.
std::unique_ptr<cudf::column> tswtzZoneKey(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Returns the distinct non-null zone keys present in a per-row zone-key column
/// (as produced by tswtzZoneKey). Performs one device-to-host synchronization.
std::vector<int16_t> tswtzDistinctZoneKeys(
    const cudf::column_view& perRowZoneKey,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Returns the UTC instant (TIMESTAMP_MILLISECONDS) of a packed column:
/// arithmetic (packed >> 12) bit-cast to a timestamp column.
std::unique_ptr<cudf::column> tswtzUtcInstant(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Returns the per-row UT offset in whole seconds (INT64) for a packed column
/// that may mix zone keys. Null rows stay null. O(number of distinct zones)
/// device passes.
std::unique_ptr<cudf::column> tswtzOffsetSeconds(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Returns the per-row local wall clock (TIMESTAMP_MILLISECONDS) of a packed
/// column, applying each row's own zone offset (multi-zone).
std::unique_ptr<cudf::column> tswtzLocalWallClock(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Converts a per-row local wall-clock column back to UTC instants
/// (TIMESTAMP_MILLISECONDS), applying each row's own zone. localMillisTs is a
/// TIMESTAMP_MILLISECONDS local column; perRowZoneKey/distinctKeys identify
/// each row's zone (from tswtzZoneKey/tswtzDistinctZoneKeys). When
/// correctForward is false a spring-forward-gap local time throws (matches
/// Timestamp::toGMT); when true it snaps forward to the post-transition instant
/// (matches addToTimestampWithTimezone). Overlaps always resolve to the
/// earliest instant.
std::unique_ptr<cudf::column> tswtzLocalToUtc(
    const cudf::column_view& localMillisTs,
    const cudf::column_view& perRowZoneKey,
    const std::vector<int16_t>& distinctKeys,
    bool correctForward,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Repacks a UTC-instant column (any timestamp resolution; cast to millis) plus
/// a per-row zone-key column into a packed TSWTZ INT64 column. Throws if any
/// non-null instant falls outside the representable millis range.
std::unique_ptr<cudf::column> tswtzPack(
    const cudf::column_view& utcInstant,
    const cudf::column_view& perRowZoneKey,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
