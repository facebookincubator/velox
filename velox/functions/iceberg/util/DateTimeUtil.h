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

#include "velox/functions/lib/TimeUtils.h"

namespace facebook::velox::functions::iceberg {

FOLLY_ALWAYS_INLINE int32_t epochYear(int32_t daysSinceEpoch) {
  const std::tm tm = functions::getDateTime(daysSinceEpoch);
  // tm_year is the number of years since 1900.
  return tm.tm_year + 1900 - 1970;
}

FOLLY_ALWAYS_INLINE int32_t epochYear(Timestamp ts) {
  return functions::getYear(functions::getDateTime(ts, nullptr)) - 1970;
}

FOLLY_ALWAYS_INLINE int32_t epochMonth(int32_t daysSinceEpoch) {
  const std::tm tm = functions::getDateTime(daysSinceEpoch);
  return (tm.tm_year + 1900 - 1970) * 12 + tm.tm_mon;
}

FOLLY_ALWAYS_INLINE int32_t epochMonth(Timestamp ts) {
  const std::tm tm = functions::getDateTime(ts, nullptr);
  return (tm.tm_year + 1900 - 1970) * 12 + tm.tm_mon;
}

FOLLY_ALWAYS_INLINE int32_t epochDay(int32_t daysSinceEpoch) {
  return daysSinceEpoch;
}

FOLLY_ALWAYS_INLINE int32_t epochDay(Timestamp ts) {
  const auto seconds = ts.getSeconds();
  return (seconds >= 0) ? seconds / Timestamp::kSecondsInDay
                        : ((seconds + 1) / Timestamp::kSecondsInDay) - 1;
}

FOLLY_ALWAYS_INLINE int32_t epochHour(Timestamp ts) {
  const auto seconds = ts.getSeconds();
  return (seconds >= 0) ? seconds / 3600 : ((seconds + 1) / 3600) - 1;
}

} // namespace facebook::velox::functions::iceberg
