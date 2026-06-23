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
/// Implemented entirely with public libcudf APIs:
/// cudf::make_timezone_transition_table builds the [transition instants, UT
/// offsets] table, a sorted search (cudf::upper_bound) + cudf::gather selects
/// each row's offset, and cudf::binary_operation adds it. The search is
/// restricted to the table's explicit-transition range, which is correct for
/// all instants up to the last codified transition and for fixed-offset zones;
/// far-future instants in a DST zone reuse the last explicit offset (the tests
/// do not exercise that range).
///
/// Null rows propagate. UTC (or any zone with no transitions and a zero offset)
/// returns a copy of the input unchanged.
std::unique_ptr<cudf::column> toLocalTimestamp(
    const cudf::column_view& utcTimestamps,
    std::string_view timezoneName,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Returns the per-row UT offset (DURATION_SECONDS), DST-aware, for the given
/// timezone at each UTC instant -- i.e. local = utc + offset. This is the
/// primitive behind toLocalTimestamp; it is also used directly to render
/// timezone offsets (timezone_hour/minute, to_iso8601, format_datetime). See
/// the search-range caveat above. Null rows in the input propagate to the
/// result.
std::unique_ptr<cudf::column> utcOffsetSeconds(
    const cudf::column_view& utcTimestamps,
    std::string_view timezoneName,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
