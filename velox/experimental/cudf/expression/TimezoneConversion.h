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
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string_view>

namespace facebook::velox::cudf_velox {

/// Converts a column of UTC timestamps to the local wall-clock instants of
/// `timezoneName`, DST-aware. The returned column has the same timestamp type
/// (and resolution) as the input; each value is shifted by that instant's UTC
/// offset so that a subsequent cudf::datetime::extract_datetime_component or
/// cudf::strings::from_timestamps reads the local calendar fields -- matching
/// the Velox CPU path, which converts the instant to the session timezone
/// before extracting.
///
/// The offset comes from a per-zone transition table built from Velox's own
/// time zone database (the same source the CPU path uses) and cached for the
/// process lifetime; a sorted search (cudf::upper_bound) + cudf::gather selects
/// each row's offset and cudf::binary_operation adds it. Instants after the last
/// codified transition reuse its offset.
///
/// Null rows propagate. This is the inverse of toUtcTimestamp.
std::unique_ptr<cudf::column> toLocalTimestamp(
    const cudf::column_view& utcTimestamps,
    std::string_view timezoneName,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Converts a column of wall-clock local timestamps in `timezoneName` to the
/// UTC instants they denote, DST-aware and matching the Velox CPU path
/// (Timestamp::toGMT). A local time that falls in a spring-forward gap does not
/// exist, so the conversion raises a user error; a local time in a fall-back
/// overlap is ambiguous and resolves to the earliest instant. The returned
/// column keeps the input's timestamp resolution and null mask. Null rows are
/// never treated as gaps, so a caller that converts only some rows can null out
/// the rest to exclude them from the gap check.
///
/// This is the inverse of toLocalTimestamp and reads the same cached,
/// tzdb-sourced transition table, in its local-keyed form with a gap flag per
/// breakpoint. Building the table from Velox's own time zone database -- the
/// source the CPU path uses -- makes the gap and overlap boundaries match
/// exactly. The conversion is a sorted search (cudf::upper_bound) plus an offset
/// subtract.
std::unique_ptr<cudf::column> toUtcTimestamp(
    const cudf::column_view& localTimestamps,
    std::string_view timezoneName,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Returns the per-row UT offset (DURATION_SECONDS), DST-aware, for the given
/// timezone at each UTC instant -- i.e. local = utc + offset. This is the
/// primitive behind toLocalTimestamp; it is also used directly to render
/// timezone offsets (timezone_hour/minute, to_iso8601, format_datetime). Reads
/// the same cached, tzdb-sourced transition table as toLocalTimestamp. Null
/// rows in the input propagate to the result.
std::unique_ptr<cudf::column> utcOffsetSeconds(
    const cudf::column_view& utcTimestamps,
    std::string_view timezoneName,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
