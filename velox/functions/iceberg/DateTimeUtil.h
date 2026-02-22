/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/functions/lib/TimeUtils.h"

namespace facebook::velox::functions::iceberg {

/// Extract a date year, as years from 1970.
FOLLY_ALWAYS_INLINE int32_t epochYear(int32_t daysSinceEpoch) {
  return functions::getYear(functions::getDateTime(daysSinceEpoch)) - 1970;
}

/// Extract a timestamp year, as years from 1970.
FOLLY_ALWAYS_INLINE int32_t epochYear(Timestamp ts) {
  return functions::getYear(functions::getDateTime(ts, nullptr)) - 1970;
}

/// Extract a date month, as months from 1970-01-01.
FOLLY_ALWAYS_INLINE int32_t epochMonth(int32_t daysSinceEpoch) {
  const std::tm tm = functions::getDateTime(daysSinceEpoch);
  return (functions::getYear(tm) - 1970) * 12 + tm.tm_mon;
}

/// Extract a timestamp month, as months from 1970-01-01.
FOLLY_ALWAYS_INLINE int32_t epochMonth(Timestamp ts) {
  const std::tm tm = functions::getDateTime(ts, nullptr);
  return (functions::getYear(tm) - 1970) * 12 + tm.tm_mon;
}

/// Extract a timestamp day, as days from 1970-01-01.
FOLLY_ALWAYS_INLINE int32_t epochDay(Timestamp ts) {
  const auto seconds = ts.getSeconds();
  return (seconds >= 0) ? seconds / Timestamp::kSecondsInDay
                        : ((seconds + 1) / Timestamp::kSecondsInDay) - 1;
}

/// Extract a timestamp hour, as hours from 1970-01-01 00:00:00.
FOLLY_ALWAYS_INLINE int32_t epochHour(Timestamp ts) {
  const auto seconds = ts.getSeconds();
  return (seconds >= 0) ? seconds / 3'600 : ((seconds + 1) / 3'600) - 1;
}

} // namespace facebook::velox::functions::iceberg
