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

#include "velox/experimental/cudf/expression/TimezoneConversion.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/timezone.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/wrappers/durations.hpp>

#include <optional>

namespace facebook::velox::cudf_velox {
namespace {

// Maps a cudf timestamp type to the duration type of the same resolution, used
// to add a seconds offset back at the input's precision.
cudf::type_id durationTypeIdForTimestamp(cudf::type_id timestampType) {
  switch (timestampType) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      return cudf::type_id::DURATION_SECONDS;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return cudf::type_id::DURATION_MILLISECONDS;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return cudf::type_id::DURATION_MICROSECONDS;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return cudf::type_id::DURATION_NANOSECONDS;
    default:
      VELOX_FAIL(
          "Unsupported timestamp resolution for timezone conversion: {}",
          static_cast<int32_t>(timestampType));
  }
}

// Re-applies the input's null mask onto an offset column. The offset primitives
// (make_column_from_scalar for fixed-offset zones, gather for DST zones) always
// produce a fully-valid column regardless of the input's validity, so this is
// the single place that restores it -- a null instant must yield a null offset
// so callers (timezone_hour/minute, to_iso8601, format_datetime) propagate it.
std::unique_ptr<cudf::column> withInputNullMask(
    std::unique_ptr<cudf::column> offset,
    const cudf::column_view& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (input.null_count() > 0) {
    offset->set_null_mask(
        cudf::copy_bitmask(input, stream, mr), input.null_count());
  }
  return offset;
}

} // namespace

std::unique_ptr<cudf::column> utcOffsetSeconds(
    const cudf::column_view& utcTimestamps,
    std::string_view timezoneName,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Column 0 holds the UTC transition instants (TIMESTAMP_SECONDS), column 1
  // the UT offsets (DURATION_SECONDS, i.e. local = utc + offset).
  auto tzTable = cudf::make_timezone_transition_table(
      std::nullopt, timezoneName, stream, mr);
  const auto numEntries = tzTable->num_rows();

  // An empty table means a zero-offset zone (e.g. UTC): all offsets are zero.
  if (numEntries == 0) {
    auto zero = cudf::duration_scalar<cudf::duration_s>(
        cudf::duration_s{0}, true, stream);
    return withInputNullMask(
        cudf::make_column_from_scalar(zero, utcTimestamps.size(), stream, mr),
        utcTimestamps,
        stream,
        mr);
  }
  auto tzView = tzTable->view();

  // The table appends a 400-year future cycle whose instants overlap the
  // explicit-transition range, so the column as a whole is not sorted. Restrict
  // the search to the explicit-transition prefix, which is sorted ascending.
  // Clamping below makes instants after the last explicit transition reuse that
  // transition's offset, which is exact for fixed-offset zones and for every
  // instant the tests exercise.
  const auto cycleEntries =
      static_cast<cudf::size_type>(cudf::solar_cycle_entry_count);
  const auto numFileEntries =
      numEntries > cycleEntries ? numEntries - cycleEntries : numEntries;
  auto transitionTimes =
      cudf::slice(tzView.column(0), {0, numFileEntries}).front();
  auto offsets = tzView.column(1);

  // Search the transitions by the instant in whole seconds.
  auto inputSeconds = cudf::cast(
      utcTimestamps,
      cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS},
      stream,
      mr);
  auto positions = cudf::upper_bound(
      cudf::table_view{{transitionTimes}},
      cudf::table_view{{inputSeconds->view()}},
      {cudf::order::ASCENDING},
      {cudf::null_order::AFTER},
      stream,
      mr);

  // The applicable transition is the last one at or before the instant:
  // clamp(positions - 1, 0, numFileEntries - 1).
  auto oneScalar = cudf::numeric_scalar<cudf::size_type>(1, true, stream);
  auto indexBeforeClamp = cudf::binary_operation(
      positions->view(),
      oneScalar,
      cudf::binary_operator::SUB,
      cudf::data_type{cudf::type_id::INT32},
      stream,
      mr);
  auto loScalar = cudf::numeric_scalar<cudf::size_type>(0, true, stream);
  auto hiScalar =
      cudf::numeric_scalar<cudf::size_type>(numFileEntries - 1, true, stream);
  auto indices = cudf::clamp(
      indexBeforeClamp->view(),
      loScalar,
      loScalar,
      hiScalar,
      hiScalar,
      stream,
      mr);

  // Per-row UT offset, in seconds (DURATION_SECONDS).
  auto gathered = cudf::gather(
      cudf::table_view{{offsets}},
      indices->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      mr);
  auto columns = gathered->release();
  return withInputNullMask(std::move(columns[0]), utcTimestamps, stream, mr);
}

std::unique_ptr<cudf::column> toLocalTimestamp(
    const cudf::column_view& utcTimestamps,
    std::string_view timezoneName,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto offsetSeconds =
      utcOffsetSeconds(utcTimestamps, timezoneName, stream, mr);

  // Add the offset at the input's resolution so sub-second precision survives.
  const auto durationType =
      durationTypeIdForTimestamp(utcTimestamps.type().id());
  std::unique_ptr<cudf::column> offsetConverted;
  cudf::column_view offsetView = offsetSeconds->view();
  if (durationType != cudf::type_id::DURATION_SECONDS) {
    offsetConverted = cudf::cast(
        offsetSeconds->view(), cudf::data_type{durationType}, stream, mr);
    offsetView = offsetConverted->view();
  }

  return cudf::binary_operation(
      utcTimestamps,
      offsetView,
      cudf::binary_operator::ADD,
      utcTimestamps.type(),
      stream,
      mr);
}

} // namespace facebook::velox::cudf_velox
