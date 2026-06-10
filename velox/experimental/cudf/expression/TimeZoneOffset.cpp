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
#include "velox/experimental/cudf/expression/TimeZoneOffset.h"

#include "velox/common/base/Exceptions.h"
#include "velox/external/date/date.h"
#include "velox/external/tzdb/time_zone.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime.h>

#include <folly/Synchronized.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

// A single offset interval: at `instant` (UTC seconds) the zone's UTC offset
// becomes `offset` (seconds) and stays constant until the next transition.
struct Transition {
  int64_t instant;
  int64_t offset;
};

// Walks the zone's transitions from 1700 to 2400. That window fully covers the
// representable range of nanosecond timestamps (~1678-2262) with no folding;
// instants beyond it fall back to the last interval's offset.
std::vector<Transition> enumerateTransitions(const tz::TimeZone* timeZone) {
  using std::chrono::seconds;

  // Offset-only zones (e.g. "+05:30") have a single constant offset.
  if (auto fixed = timeZone->offset(); fixed.has_value()) {
    return {Transition{0, std::chrono::duration_cast<seconds>(*fixed).count()}};
  }

  const auto* zone = timeZone->tz();
  VELOX_CHECK_NOT_NULL(
      zone,
      "Time zone has neither a fixed offset nor a database entry: {}",
      timeZone->name());

  const auto yearStart = [](int year) {
    return std::chrono::duration_cast<seconds>(
               date::sys_days{date::year{year} / date::January / 1}
                   .time_since_epoch())
        .count();
  };
  const int64_t horizonSeconds = yearStart(2400);

  std::vector<Transition> transitions;
  int64_t probe = yearStart(1700);
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

// Maps a timestamp resolution to the matching duration resolution so a
// whole-second offset can be scaled exactly (via cudf::cast) and added without
// losing the timestamp's sub-second precision.
cudf::data_type durationTypeFor(cudf::type_id timestampType) {
  switch (timestampType) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      return cudf::data_type{cudf::type_id::DURATION_SECONDS};
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return cudf::data_type{cudf::type_id::DURATION_MILLISECONDS};
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return cudf::data_type{cudf::type_id::DURATION_MICROSECONDS};
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return cudf::data_type{cudf::type_id::DURATION_NANOSECONDS};
    default:
      VELOX_FAIL(
          "Unsupported timestamp resolution for time zone conversion: {}",
          static_cast<int32_t>(timestampType));
  }
}

// Looks up each input timestamp's active offset in `table` and applies it.
// `localToUtc` subtracts the offset (local->UTC) and rejects gap rows;
// otherwise it adds the offset (UTC->local). The input's resolution and null
// mask are preserved.
std::unique_ptr<cudf::column> applyOffset(
    cudf::column_view timestamps,
    const cudf::table& table,
    bool localToUtc,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto tableView = table.view();

  // Truncate to seconds for the lookup key. Offsets only change on whole-second
  // boundaries, so this selects the correct interval at full precision.
  auto key = cudf::cast(
      timestamps,
      cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS},
      stream,
      mr);

  cudf::table_view haystack{{tableView.column(0)}};
  cudf::table_view needles{{key->view()}};
  auto upperBound = cudf::upper_bound(
      haystack,
      needles,
      {cudf::order::ASCENDING},
      {cudf::null_order::AFTER},
      stream,
      mr);

  // The first table key is INT64_MIN, so upper_bound is always >= 1 and the
  // active-interval index (upper_bound - 1) is always in range.
  cudf::numeric_scalar<cudf::size_type> one{1, true, stream, mr};
  auto gatherMap = cudf::binary_operation(
      upperBound->view(),
      one,
      cudf::binary_operator::SUB,
      cudf::data_type{cudf::type_id::INT32},
      stream,
      mr);

  std::vector<cudf::column_view> sourceColumns{tableView.column(1)};
  if (localToUtc) {
    sourceColumns.push_back(tableView.column(2));
  }
  auto gathered = cudf::gather(
      cudf::table_view{sourceColumns},
      gatherMap->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      mr);

  auto offset = cudf::cast(
      gathered->get_column(0).view(),
      durationTypeFor(timestamps.type().id()),
      stream,
      mr);
  auto result = cudf::binary_operation(
      timestamps,
      offset->view(),
      localToUtc ? cudf::binary_operator::SUB : cudf::binary_operator::ADD,
      timestamps.type(),
      stream,
      mr);

  if (localToUtc) {
    auto gap = gathered->get_column(1).view();
    std::unique_ptr<cudf::column> maskedGap;
    if (timestamps.nullable() && timestamps.null_count() > 0) {
      // Null inputs are not gaps; exclude them before checking.
      auto valid = cudf::is_valid(timestamps, stream, mr);
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
          "session time zone (daylight savings gap)");
    }
  }
  return result;
}

std::shared_ptr<const TimeZoneOffsetTable> buildTable(
    const tz::TimeZone* timeZone) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  auto transitions = enumerateTransitions(timeZone);
  VELOX_CHECK(!transitions.empty());

  // Force the first key below every representable input so the active-interval
  // index never goes out of range (see applyOffset).
  constexpr int64_t kFloor = std::numeric_limits<int64_t>::min();

  // Forward table: UTC-keyed instants and offsets.
  std::vector<int64_t> forwardKeys;
  std::vector<int64_t> forwardOffsets;
  forwardKeys.reserve(transitions.size());
  forwardOffsets.reserve(transitions.size());
  for (const auto& transition : transitions) {
    forwardKeys.push_back(transition.instant);
    forwardOffsets.push_back(transition.offset);
  }
  forwardKeys.front() = kFloor;

  // Inverse table: local-keyed breakpoints. A transition from prevOffset to
  // curOffset at UTC instant `inst` shifts the wall clock between
  // inst+prevOffset and inst+curOffset. A forward shift (gap) makes that local
  // range nonexistent; a backward shift (overlap) makes it ambiguous, and
  // keeping the pre-transition offset over the overlap matches toGMT's
  // kEarliest choice (so only one breakpoint, at the later local boundary, is
  // needed).
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

  std::vector<int64_t> inverseKeys;
  std::vector<int64_t> inverseOffsets;
  std::vector<int8_t> inverseGaps;
  for (const auto& breakpoint : breakpoints) {
    // On duplicate local keys keep the last (the stable sort preserves the
    // emission order, which is the intended precedence).
    if (!inverseKeys.empty() && inverseKeys.back() == breakpoint.key) {
      inverseOffsets.back() = breakpoint.offset;
      inverseGaps.back() = breakpoint.gap;
    } else {
      inverseKeys.push_back(breakpoint.key);
      inverseOffsets.push_back(breakpoint.offset);
      inverseGaps.push_back(breakpoint.gap);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> forwardColumns;
  forwardColumns.push_back(makeDeviceColumn(
      forwardKeys, cudf::type_id::TIMESTAMP_SECONDS, stream, mr));
  forwardColumns.push_back(makeDeviceColumn(
      forwardOffsets, cudf::type_id::DURATION_SECONDS, stream, mr));

  std::vector<std::unique_ptr<cudf::column>> inverseColumns;
  inverseColumns.push_back(makeDeviceColumn(
      inverseKeys, cudf::type_id::TIMESTAMP_SECONDS, stream, mr));
  inverseColumns.push_back(makeDeviceColumn(
      inverseOffsets, cudf::type_id::DURATION_SECONDS, stream, mr));
  inverseColumns.push_back(
      makeDeviceColumn(inverseGaps, cudf::type_id::BOOL8, stream, mr));

  auto forward = std::make_unique<cudf::table>(std::move(forwardColumns));
  auto inverse = std::make_unique<cudf::table>(std::move(inverseColumns));
  // The host vectors are freed when this function returns, so the async copies
  // must complete first.
  stream.synchronize();

  return std::make_shared<TimeZoneOffsetTable>(
      std::move(forward), std::move(inverse));
}

} // namespace

TimeZoneOffsetTable::TimeZoneOffsetTable(
    std::unique_ptr<cudf::table> forward,
    std::unique_ptr<cudf::table> inverse)
    : forward_(std::move(forward)), inverse_(std::move(inverse)) {}

// static
std::shared_ptr<const TimeZoneOffsetTable> TimeZoneOffsetTable::get(
    const tz::TimeZone* timeZone) {
  VELOX_CHECK_NOT_NULL(timeZone, "Session time zone must not be null");
  static folly::Synchronized<
      std::unordered_map<int16_t, std::shared_ptr<const TimeZoneOffsetTable>>>
      cache;

  const auto id = timeZone->id();
  {
    auto locked = cache.rlock();
    auto it = locked->find(id);
    if (it != locked->end()) {
      return it->second;
    }
  }
  auto table = buildTable(timeZone);
  auto locked = cache.wlock();
  return locked->emplace(id, std::move(table)).first->second;
}

std::unique_ptr<cudf::column> TimeZoneOffsetTable::toUtc(
    cudf::column_view localTimestamps,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  return applyOffset(
      localTimestamps, *inverse_, /*localToUtc=*/true, stream, mr);
}

std::unique_ptr<cudf::column> TimeZoneOffsetTable::toLocal(
    cudf::column_view utcTimestamps,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  return applyOffset(
      utcTimestamps, *forward_, /*localToUtc=*/false, stream, mr);
}

} // namespace facebook::velox::cudf_velox
