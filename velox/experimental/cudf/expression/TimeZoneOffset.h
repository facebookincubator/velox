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
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace facebook::velox::tz {
class TimeZone;
}

namespace facebook::velox::cudf_velox {

/// Device-resident lookup tables that map between UTC instants and wall-clock
/// local times for a single time zone, entirely on the GPU.
///
/// A time zone is a step function from a UTC instant to a UTC offset that is
/// constant between daylight-savings transitions. The transitions are
/// enumerated once on the host from Velox's own tz database (the same source
/// the CPU path uses) and uploaded to the device as two small sorted tables:
/// a forward table keyed by UTC instant (for UTC->local) and an inverse table
/// keyed by local instant (for local->UTC) that bakes in the daylight-savings
/// ambiguity policy of Timestamp::toGMT. Each conversion is then a vectorized
/// table lookup plus an offset add, with no per-row host round-trip.
class TimeZoneOffsetTable {
 public:
  /// Returns the table for `timeZone`, building it on first use and caching it
  /// for the process lifetime (keyed by time zone id). Thread-safe.
  static std::shared_ptr<const TimeZoneOffsetTable> get(
      const tz::TimeZone* timeZone);

  /// Converts a column of wall-clock local timestamps to UTC, matching
  /// Timestamp::toGMT: ambiguous (fall-back overlap) local times resolve to the
  /// earliest instant, and nonexistent (spring-forward gap) local times throw a
  /// user error. Preserves the input column's resolution and null mask.
  std::unique_ptr<cudf::column> toUtc(
      cudf::column_view localTimestamps,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  /// Converts a column of UTC timestamps to wall-clock local times, matching
  /// Timestamp::toTimezone. Always well-defined. Preserves the input column's
  /// resolution and null mask.
  std::unique_ptr<cudf::column> toLocal(
      cudf::column_view utcTimestamps,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  TimeZoneOffsetTable(
      std::unique_ptr<cudf::table> forward,
      std::unique_ptr<cudf::table> inverse);

 private:
  // Column layout shared by both tables: column 0 holds the sorted transition
  // instants (TIMESTAMP_SECONDS), column 1 the UTC offset that takes effect at
  // each (DURATION_SECONDS).
  static constexpr int kKeyColumn{0};
  static constexpr int kOffsetColumn{1};
  // Inverse table only: column 2 flags local instants that fall in a
  // spring-forward gap and therefore do not exist (BOOL8).
  static constexpr int kGapColumn{2};

  // Keyed by UTC instant; offsets are added to convert UTC to local.
  std::unique_ptr<cudf::table> forward_;
  // Keyed by local instant; offsets are subtracted to convert local to UTC.
  // Carries the gap flag column used to reject nonexistent local times.
  std::unique_ptr<cudf::table> inverse_;
};

} // namespace facebook::velox::cudf_velox
