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

#include "velox/experimental/cudf/tests/TimeZoneReference.h"

#include "velox/common/base/VeloxException.h"
#include "velox/external/tzdb/time_zone.h"
#include "velox/type/Timestamp.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <chrono>
#include <string>

namespace facebook::velox::cudf_velox::gpu_sfi {

GpuTimeZone deviceTimeZone(std::string_view timeZone) {
  return gpuTimeZone(*tz::locateZone(timeZone));
}

std::vector<OffsetChange>
offsetChanges(std::string_view timeZone, int32_t fromYear, int32_t toYear) {
  std::vector<OffsetChange> changes;
  const auto* zone = tz::locateZone(timeZone)->tz();
  if (zone == nullptr) {
    return changes;
  }
  const auto to = date::sys_seconds{date::sys_days{date::year{toYear} / 1 / 1}};
  auto info = zone->get_info(
      date::sys_seconds{date::sys_days{date::year{fromYear} / 1 / 1}});
  while (info.end < to) {
    const auto next = zone->get_info(info.end);
    // A change of abbreviation or of saving alone leaves local time where it
    // is; only the total offset matters here.
    if (next.offset != info.offset) {
      changes.push_back(
          OffsetChange{
              info.end.time_since_epoch().count(),
              static_cast<int32_t>(info.offset.count()),
              static_cast<int32_t>(next.offset.count()),
          });
    }
    if (next.end <= info.end) {
      break;
    }
    info = next;
  }
  return changes;
}

std::optional<int64_t> cpuToUtc(
    std::string_view timeZone,
    int64_t localSeconds) {
  Timestamp timestamp(localSeconds, 0);
  try {
    timestamp.toGMT(*tz::locateZone(timeZone));
  } catch (const VeloxUserError& error) {
    // Only the gap counts as "does not exist"; anything else is a test bug.
    if (error.message().find("is in a gap") == std::string::npos) {
      throw;
    }
    return std::nullopt;
  }
  return timestamp.getSeconds();
}

int64_t cpuCorrectedToUtc(std::string_view timeZone, int64_t localSeconds) {
  const auto* zone = tz::locateZone(timeZone);
  Timestamp timestamp(
      zone->correct_nonexistent_time(std::chrono::seconds(localSeconds))
          .count(),
      0);
  timestamp.toGMT(*zone);
  return timestamp.getSeconds();
}

int64_t cpuToLocal(std::string_view timeZone, int64_t utcSeconds) {
  Timestamp timestamp(utcSeconds, 0);
  timestamp.toTimezone(*tz::locateZone(timeZone));
  return timestamp.getSeconds();
}

int64_t yearStartSeconds(int32_t year) {
  return std::chrono::duration_cast<std::chrono::seconds>(
             date::sys_days{date::year{year} / 1 / 1}.time_since_epoch())
      .count();
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
