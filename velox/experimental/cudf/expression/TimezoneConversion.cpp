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
#include "velox/external/date/date.h"
#include "velox/external/tzdb/time_zone.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/durations.hpp>

#include <cuda_runtime.h>

#include <folly/Synchronized.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

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

// Re-applies the input's null mask onto an offset column. The gather that
// produces the offset yields a fully-valid column regardless of the input's
// validity, so this is the single place that restores it -- a null instant must
// yield a null offset so callers (timezone_hour/minute, to_iso8601,
// format_datetime) propagate it.
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

// A single offset interval: at `instant` (UTC seconds) the zone's UTC offset
// becomes `offset` (seconds) and stays constant until the next transition.
struct Transition {
  int64_t instant;
  int64_t offset;
};

// Bounds of the materialized transition window. The end is exclusive, so 10000
// covers all of year 9999 -- the upper bound of the year range the rest of this
// module accepts. A batch reaching past it is answered on the host rather than
// widening the table, which keeps the table's size fixed and independent of the
// data.
constexpr int kTransitionTableStartYear = 1700;
constexpr int kTransitionTableEndYear = 10000;

int64_t yearStartSeconds(int year) {
  return std::chrono::duration_cast<std::chrono::seconds>(
             date::sys_days{date::year{year} / date::January / 1}
                 .time_since_epoch())
      .count();
}

// UTC second at which the materialized window ends. An instant at or past this
// point is not reliably covered by the table, so its batch is answered on the
// host instead.
int64_t transitionWindowEnd() {
  return yearStartSeconds(kTransitionTableEndYear);
}

// Largest instant in the column as UTC seconds, used to decide whether the
// batch stays on the device path. Returns 0 for an empty or all-null column,
// which always does. Reading the reduced scalar synchronizes the stream; the
// per-row zone handling in this module already pays a device-to-host sync, so
// this adds one reduction rather than a new class of stall.
int64_t maxInstantSeconds(
    const cudf::column_view& timestamps,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (timestamps.size() == timestamps.null_count()) {
    return 0;
  }
  const auto secondsType = cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS};
  auto asSeconds = cudf::cast(timestamps, secondsType, stream, mr);
  auto maxInstant = cudf::reduce(
      asSeconds->view(),
      *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
      secondsType,
      stream,
      mr);
  using ScalarType = cudf::scalar_type_t<cudf::timestamp_s>;
  auto* typed = static_cast<ScalarType*>(maxInstant.get());
  if (!typed->is_valid(stream)) {
    return 0;
  }
  return typed->value(stream).time_since_epoch().count();
}

// Walks the zone's daylight-savings transitions from 1700 up to
// `horizonSeconds`.
//
// An instant past the last materialized transition would fold onto its offset
// and silently lose daylight saving -- America/Los_Angeles in July 2401
// reported -08:00 where the correct offset is -07:00, back when the window
// ended at 2400.
//
// Only a TIMESTAMP WITH TIME ZONE can get out that far. It stores 52-bit millis
// and so reaches roughly year 71,000, whereas a plain TIMESTAMP cannot cross
// into the GPU past about 2262 at all: Timestamp::toNanos multiplies seconds by
// 1e9 and raises on overflow, which is where the original 2400 bound came from.
// Ending the window at 9999 therefore keeps every plain timestamp and any
// plausible zoned one on the device path; the rest is answered on the host.
//
// A zone whose rules recur forever never reports a final interval, which is why
// enumeration needs a stopping point at all. One whose rules do end reports
// sys_seconds::max() and terminates early, whatever horizon was asked for.
std::vector<Transition> enumerateTransitions(
    const tz::TimeZone* timeZone,
    int64_t horizonSeconds) {
  using std::chrono::seconds;

  // Offset-only zones (e.g. "+05:30") have one constant offset that applies for
  // all time, so its interval starts at the earliest representable instant.
  //
  // Keying it at the epoch instead left the lookup depending on undefined
  // behaviour. activeIntervalIndices computes upper_bound - 1, so any pre-1970
  // instant produced -1, which then indexed a gather running with
  // out_of_bounds_policy::DONT_CHECK. It happens to return the right answer
  // today only because the table has exactly one row, so an out-of-range index
  // cannot reach a different one. A second row would make it read some other
  // interval's offset.
  if (auto fixed = timeZone->offset(); fixed.has_value()) {
    return {Transition{
        std::numeric_limits<int64_t>::min(),
        std::chrono::duration_cast<seconds>(*fixed).count()}};
  }

  const auto* zone = timeZone->tz();
  VELOX_CHECK_NOT_NULL(
      zone,
      "Time zone has neither a fixed offset nor a database entry: {}",
      timeZone->name());

  std::vector<Transition> transitions;
  int64_t probe = yearStartSeconds(kTransitionTableStartYear);
  while (true) {
    auto info = zone->get_info(date::sys_seconds{seconds{probe}});
    transitions.push_back(
        {info.begin.time_since_epoch().count(), info.offset.count()});

    const int64_t endSeconds = info.end.time_since_epoch().count();
    if (info.end == date::sys_seconds::max() || endSeconds >= horizonSeconds ||
        endSeconds <= probe) {
      break;
    }
    probe = endSeconds;
  }
  return transitions;
}

// Copies a host vector to a new device column of the given type. The element
// type T must match the column's physical representation (int64_t for
// TIMESTAMP_SECONDS/DURATION_SECONDS, int8_t for BOOL8).
template <typename T>
std::unique_ptr<cudf::column> makeDeviceColumn(
    const std::vector<T>& host,
    cudf::type_id typeId,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto size = static_cast<cudf::size_type>(host.size());
  auto column = cudf::make_fixed_width_column(
      cudf::data_type{typeId}, size, cudf::mask_state::UNALLOCATED, stream, mr);
  if (size > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        column->mutable_view().data<T>(),
        host.data(),
        host.size() * sizeof(T),
        cudaMemcpyHostToDevice,
        stream.value()));
  }
  return column;
}

// Builds the forward (UTC-keyed) table [instant (TIMESTAMP_SECONDS), offset
// (DURATION_SECONDS)] from the zone's transitions. The first key is forced to
// INT64_MIN so the active-interval index (upper_bound - 1) is never out of
// range; instants after the last transition reuse its offset. Synchronizes the
// stream before returning so the host vectors outlive the async uploads.
std::unique_ptr<cudf::table> buildForwardTable(
    const std::vector<Transition>& transitions,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<int64_t> keys;
  std::vector<int64_t> offsets;
  keys.reserve(transitions.size());
  offsets.reserve(transitions.size());
  for (const auto& transition : transitions) {
    keys.push_back(transition.instant);
    offsets.push_back(transition.offset);
  }
  keys.front() = std::numeric_limits<int64_t>::min();

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
      makeDeviceColumn(keys, cudf::type_id::TIMESTAMP_SECONDS, stream, mr));
  columns.push_back(
      makeDeviceColumn(offsets, cudf::type_id::DURATION_SECONDS, stream, mr));
  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(columns));
}

// Builds the local-keyed inverse table [localInstant (TIMESTAMP_SECONDS),
// offset (DURATION_SECONDS), gap (BOOL8)] from the zone's transitions. A
// transition from prevOffset to curOffset at UTC instant `inst` shifts the wall
// clock between inst+prevOffset and inst+curOffset. A forward shift (curOffset
// > prevOffset, spring forward) makes that local range nonexistent, so it is
// flagged as a gap; a backward shift (fall back) makes it ambiguous, and
// keeping the pre-transition offset over the overlap matches toGMT's kEarliest
// choice (so only the later local boundary needs a breakpoint). Synchronizes
// the stream before returning so the host vectors outlive the async uploads.
std::unique_ptr<cudf::table> buildInverseTable(
    const std::vector<Transition>& transitions,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Force the first key below every representable input so the active-interval
  // index (upper_bound - 1) is never out of range.
  constexpr int64_t kFloor = std::numeric_limits<int64_t>::min();

  struct Breakpoint {
    int64_t key;
    int64_t offset;
    int8_t gap;
  };
  std::vector<Breakpoint> breakpoints;
  breakpoints.push_back({kFloor, transitions.front().offset, 0});
  for (size_t i = 1; i < transitions.size(); ++i) {
    const int64_t prevOffset = transitions[i - 1].offset;
    const int64_t curOffset = transitions[i].offset;
    const int64_t localPrev = transitions[i].instant + prevOffset;
    const int64_t localCur = transitions[i].instant + curOffset;
    if (curOffset > prevOffset) {
      breakpoints.push_back({localPrev, prevOffset, 1});
      breakpoints.push_back({localCur, curOffset, 0});
    } else if (curOffset < prevOffset) {
      breakpoints.push_back({localPrev, curOffset, 0});
    }
  }
  std::stable_sort(
      breakpoints.begin(),
      breakpoints.end(),
      [](const Breakpoint& a, const Breakpoint& b) { return a.key < b.key; });

  std::vector<int64_t> keys;
  std::vector<int64_t> offsets;
  std::vector<int8_t> gaps;
  for (const auto& breakpoint : breakpoints) {
    // On duplicate local keys keep the last; the stable sort preserves the
    // emission order, which is the intended precedence.
    if (!keys.empty() && keys.back() == breakpoint.key) {
      offsets.back() = breakpoint.offset;
      gaps.back() = breakpoint.gap;
    } else {
      keys.push_back(breakpoint.key);
      offsets.push_back(breakpoint.offset);
      gaps.push_back(breakpoint.gap);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
      makeDeviceColumn(keys, cudf::type_id::TIMESTAMP_SECONDS, stream, mr));
  columns.push_back(
      makeDeviceColumn(offsets, cudf::type_id::DURATION_SECONDS, stream, mr));
  columns.push_back(makeDeviceColumn(gaps, cudf::type_id::BOOL8, stream, mr));
  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(columns));
}

// Returns the active-interval row index (upper_bound - 1, INT32) for each
// timestamp against a table's sorted key column. The key is truncated to whole
// seconds because offsets only change on second boundaries. Both tables force
// the first key to INT64_MIN, so the result is always a valid index.
std::unique_ptr<cudf::column> activeIntervalIndices(
    const cudf::column_view& transitionKeys,
    const cudf::column_view& timestamps,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto key = cudf::cast(
      timestamps,
      cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS},
      stream,
      mr);
  auto positions = cudf::upper_bound(
      cudf::table_view{{transitionKeys}},
      cudf::table_view{{key->view()}},
      {cudf::order::ASCENDING},
      {cudf::null_order::AFTER},
      stream,
      mr);
  auto one = cudf::numeric_scalar<cudf::size_type>(1, true, stream);
  return cudf::binary_operation(
      positions->view(),
      one,
      cudf::binary_operator::SUB,
      cudf::data_type{cudf::type_id::INT32},
      stream,
      mr);
}

// Per-zone forward (UTC-keyed) and inverse (local-keyed, gap-flagged) offset
// tables, built once from Velox's time zone database and cached for the process
// lifetime. The forward table answers UTC->local and per-row offset queries;
// the inverse table answers local->UTC with the daylight-savings policy of
// Timestamp::toGMT baked in.
class OffsetTable {
 public:
  OffsetTable(
      const tz::TimeZone* timeZone,
      std::unique_ptr<cudf::table> forward,
      std::unique_ptr<cudf::table> inverse)
      : timeZone_(timeZone),
        forward_(std::move(forward)),
        inverse_(std::move(inverse)) {}

  // Returns the table for `timeZone`, building it on first use and caching it
  // by zone id for the process lifetime. Thread-safe.
  //
  // The window has no lower bound to manage: tzdb reports the earliest interval
  // as beginning at the start of time, so the first key is at or below any
  // representable instant and a past instant always lands on a real interval.
  // Only the upper end is bounded, because a zone with recurring
  // daylight-saving rules never reports a final interval, so enumeration has to
  // stop somewhere. A batch reaching past that end is answered on the host
  // instead; see utcOffsetOnHost.
  static std::shared_ptr<const OffsetTable> get(const tz::TimeZone* timeZone);

  // Per-row UT offset (DURATION_SECONDS) at each UTC instant; the input null
  // mask is re-applied so a null instant yields a null offset.
  std::unique_ptr<cudf::column> utcOffset(
      const cudf::column_view& utcTimestamps,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // utc + offset, at the input's resolution.
  std::unique_ptr<cudf::column> toLocal(
      const cudf::column_view& utcTimestamps,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // local - offset; raises a user error on a nonexistent (spring-forward gap)
  // local time and resolves an ambiguous (fall-back overlap) one to the
  // earliest instant. Null rows are never treated as gaps.
  std::unique_ptr<cudf::column> toUtc(
      const cudf::column_view& localTimestamps,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

 private:
  // Offsets for a batch that reaches past the materialized window, computed on
  // the host from the same tzdb lookup the table itself is built from. tzdb
  // evaluates recurring rules arithmetically, so it answers any instant without
  // a horizon.
  //
  // The whole batch is routed rather than only the offending rows: such a batch
  // is rare by construction, and one loop over the column is far less machinery
  // -- no mask, gather, scatter or index bookkeeping -- on a path that little
  // else exercises.
  std::unique_ptr<cudf::column> utcOffsetOnHost(
      const cudf::column_view& utcTimestamps,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Borrowed from the process-wide zone registry, which outlives the cache.
  const tz::TimeZone* timeZone_;
  // [instant (TIMESTAMP_SECONDS), offset (DURATION_SECONDS)], UTC-keyed.
  std::unique_ptr<cudf::table> forward_;
  // [instant (TIMESTAMP_SECONDS), offset (DURATION_SECONDS), gap (BOOL8)],
  // local-keyed.
  std::unique_ptr<cudf::table> inverse_;
};

// static
std::shared_ptr<const OffsetTable> OffsetTable::get(
    const tz::TimeZone* timeZone) {
  VELOX_CHECK_NOT_NULL(timeZone, "Time zone must not be null");
  // Cached for the process and keyed only by timezone id, which assumes one
  // device per process. An OffsetTable owns CUDA device allocations, so an
  // entry built while one device was current is only usable on that device:
  // were a worker ever able to move between devices, a cache hit would hand it
  // memory belonging to another. The assumption holds for the current
  // deployment model of one worker process per GPU; a worker that switches
  // devices needs the key widened to include the device.
  static folly::Synchronized<
      std::unordered_map<int16_t, std::shared_ptr<const OffsetTable>>>
      cache;

  const auto id = timeZone->id();
  {
    auto locked = cache.rlock();
    if (auto it = locked->find(id); it != locked->end()) {
      return it->second;
    }
  }
  // Build on the default stream and current resource so the cached device
  // tables do not depend on any caller's stream or memory resource.
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto transitions = enumerateTransitions(timeZone, transitionWindowEnd());
  VELOX_CHECK(!transitions.empty());
  auto table = std::make_shared<const OffsetTable>(
      timeZone,
      buildForwardTable(transitions, stream, mr),
      buildInverseTable(transitions, stream, mr));
  // Another thread may have inserted the same zone meanwhile; emplace keeps the
  // existing entry and discards this build.
  return cache.wlock()->emplace(id, std::move(table)).first->second;
}

// Copies a column of instants to the host as whole seconds, with a validity
// flag per row. Shared by both host paths.
std::pair<std::vector<int64_t>, std::vector<int8_t>> instantsToHost(
    const cudf::column_view& timestamps,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto asSeconds = cudf::cast(
      timestamps,
      cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS},
      stream,
      mr);
  auto valid = cudf::is_valid(asSeconds->view(), stream, mr);

  const auto size = static_cast<size_t>(timestamps.size());
  std::vector<int64_t> seconds(size);
  std::vector<int8_t> isValid(size);
  if (size > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        seconds.data(),
        asSeconds->view().data<int64_t>(),
        size * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        isValid.data(),
        valid->view().data<int8_t>(),
        size * sizeof(int8_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
  }
  return {std::move(seconds), std::move(isValid)};
}

std::unique_ptr<cudf::column> OffsetTable::utcOffsetOnHost(
    const cudf::column_view& utcTimestamps,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  const auto* zone = timeZone_->tz();
  VELOX_CHECK_NOT_NULL(
      zone,
      "Instant outside the transition window in a zone with no database entry: {}",
      timeZone_->name());

  auto [seconds, isValid] = instantsToHost(utcTimestamps, stream, mr);
  std::vector<int64_t> offsets(seconds.size(), 0);
  for (size_t i = 0; i < seconds.size(); ++i) {
    if (!isValid[i]) {
      continue;
    }
    offsets[i] =
        zone->get_info(date::sys_seconds{std::chrono::seconds{seconds[i]}})
            .offset.count();
  }

  auto offsetColumn =
      makeDeviceColumn(offsets, cudf::type_id::DURATION_SECONDS, stream, mr);
  return withInputNullMask(std::move(offsetColumn), utcTimestamps, stream, mr);
}

std::unique_ptr<cudf::column> OffsetTable::utcOffset(
    const cudf::column_view& utcTimestamps,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  if (maxInstantSeconds(utcTimestamps, stream, mr) >= transitionWindowEnd()) {
    return utcOffsetOnHost(utcTimestamps, stream, mr);
  }
  auto indices = activeIntervalIndices(
      forward_->view().column(0), utcTimestamps, stream, mr);
  auto gathered = cudf::gather(
      cudf::table_view{{forward_->view().column(1)}},
      indices->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      mr);
  auto columns = gathered->release();
  return withInputNullMask(std::move(columns[0]), utcTimestamps, stream, mr);
}

std::unique_ptr<cudf::column> OffsetTable::toLocal(
    const cudf::column_view& utcTimestamps,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto offsetSeconds = utcOffset(utcTimestamps, stream, mr);

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

std::unique_ptr<cudf::column> OffsetTable::toUtc(
    const cudf::column_view& localTimestamps,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto indices = activeIntervalIndices(
      inverse_->view().column(0), localTimestamps, stream, mr);
  auto gathered = cudf::gather(
      cudf::table_view{
          {inverse_->view().column(1), inverse_->view().column(2)}},
      indices->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      mr);
  auto gatheredView = gathered->view();

  // utc = local - offset, at the input's resolution so sub-second precision
  // survives.
  const auto durationType =
      durationTypeIdForTimestamp(localTimestamps.type().id());
  auto offset = cudf::cast(
      gatheredView.column(0), cudf::data_type{durationType}, stream, mr);
  auto result = cudf::binary_operation(
      localTimestamps,
      offset->view(),
      cudf::binary_operator::SUB,
      localTimestamps.type(),
      stream,
      mr);

  // A nonexistent local time (spring-forward gap) has no UTC instant; match
  // CPU's toGMT and fail. Null rows are not gaps, so mask them out first.
  cudf::column_view gap = gatheredView.column(1);
  std::unique_ptr<cudf::column> maskedGap;
  if (localTimestamps.nullable() && localTimestamps.null_count() > 0) {
    auto valid = cudf::is_valid(localTimestamps, stream, mr);
    maskedGap = cudf::binary_operation(
        gap,
        valid->view(),
        cudf::binary_operator::LOGICAL_AND,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    gap = maskedGap->view();
  }
  auto anyGap = cudf::reduce(
      gap,
      *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto& anyGapScalar = static_cast<cudf::numeric_scalar<bool>&>(*anyGap);
  if (anyGapScalar.is_valid(stream) && anyGapScalar.value(stream)) {
    VELOX_USER_FAIL(
        "Cannot convert local time to UTC: the time does not exist in the "
        "time zone (daylight savings gap)");
  }
  return result;
}

} // namespace

std::unique_ptr<cudf::column> utcOffsetSeconds(
    const cudf::column_view& utcTimestamps,
    std::string_view timezoneName,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return OffsetTable::get(tz::locateZone(timezoneName))
      ->utcOffset(utcTimestamps, stream, mr);
}

std::unique_ptr<cudf::column> toLocalTimestamp(
    const cudf::column_view& utcTimestamps,
    std::string_view timezoneName,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return OffsetTable::get(tz::locateZone(timezoneName))
      ->toLocal(utcTimestamps, stream, mr);
}

std::unique_ptr<cudf::column> toUtcTimestamp(
    const cudf::column_view& localTimestamps,
    std::string_view timezoneName,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return OffsetTable::get(tz::locateZone(timezoneName))
      ->toUtc(localTimestamps, stream, mr);
}

} // namespace facebook::velox::cudf_velox
