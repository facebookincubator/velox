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

#include "velox/experimental/cudf/functions/GpuTimeZone.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

/// CPU reference values for the GpuTimeZone device tests. Compiled without the
/// shadow include path, where tz::TimeZone and Timestamp are reachable, and
/// hands the shadow-compiled test only POD and standard types.
namespace facebook::velox::cudf_velox::gpu_sfi {

/// One change of a zone's UTC offset.
struct OffsetChange {
  /// The instant the new offset starts to apply.
  int64_t utcSeconds;
  /// Seconds east of UTC until the instant.
  int32_t offsetBefore;
  /// Seconds east of UTC from the instant on.
  int32_t offsetAfter;
};

/// The device form of the named zone.
GpuTimeZone deviceTimeZone(std::string_view timeZone);

/// The zone's offset changes with instants in [fromYear, toYear). Empty for a
/// fixed-offset zone.
std::vector<OffsetChange>
offsetChanges(std::string_view timeZone, int32_t fromYear, int32_t toYear);

/// What Timestamp::toGMT() answers for a local time in the zone, or nullopt
/// where it raises the user error for a local time that does not exist.
std::optional<int64_t> cpuToUtc(
    std::string_view timeZone,
    int64_t localSeconds);

/// What date_add computes for a local time: correct_nonexistent_time(), then
/// Timestamp::toGMT(). Equals cpuToUtc() wherever that has a value.
int64_t cpuCorrectedToUtc(std::string_view timeZone, int64_t localSeconds);

/// What Timestamp::toTimezone() answers for an instant.
int64_t cpuToLocal(std::string_view timeZone, int64_t utcSeconds);

/// Seconds since the epoch at which the year starts, in UTC.
int64_t yearStartSeconds(int32_t year);

} // namespace facebook::velox::cudf_velox::gpu_sfi
