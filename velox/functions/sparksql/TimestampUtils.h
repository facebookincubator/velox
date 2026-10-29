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

#include "velox/type/Timestamp.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {

/// Lenient conversion of local 'time' to the GMT time.
/// If the 'time' is invalid (DST gap), it will be corrected by
/// shifting forward to the nearest valid time.
void toGMTWithGapCorrection(Timestamp& time, const tz::TimeZone& zone);

/// Spark's (seconds, nanos) timestamp bounds, from Java's Long.MIN/MAX_VALUE.
constexpr int64_t kSparkMaxTimestampSeconds = 9223372036854LL;
constexpr int64_t kSparkMaxTimestampNanos = 775807000LL;
constexpr int64_t kSparkMinTimestampSeconds = -9223372036855LL;
constexpr int64_t kSparkMinTimestampNanos = 224192000LL;

/// Returns false if `timestamp` falls outside Spark's representable range.
inline bool isWithinSparkTimestampRange(const Timestamp& timestamp) {
  if (timestamp.getSeconds() > kSparkMaxTimestampSeconds ||
      (timestamp.getSeconds() == kSparkMaxTimestampSeconds &&
       timestamp.getNanos() > kSparkMaxTimestampNanos)) {
    return false;
  }
  return timestamp.getSeconds() > kSparkMinTimestampSeconds ||
      (timestamp.getSeconds() == kSparkMinTimestampSeconds &&
       timestamp.getNanos() >= kSparkMinTimestampNanos);
}

} // namespace facebook::velox::functions::sparksql
