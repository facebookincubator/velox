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
#include <type_traits>

#include "velox/external/date/tz.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox {

/// A static class that holds helper functions for TIME type.
class TimeUtil {
 public:
  static constexpr int64_t kMillisecondsInSecond = 1'000;
  // 24 * 60 * 60 * 1000
  static constexpr int64_t kMillisecondPerDay = 86400000LL;

  // Get the number of milliseconds in a day from a given milliseconds
  // millis – the milliseconds from 1970-01-01T00:00:00Z
  // Returns the amount of milliseconds in a day.
  inline static int64_t getMillisOfDay(int64_t millis) {
    if (millis >= 0) {
      return millis % kMillisecondPerDay;
    } else {
      return kMillisecondPerDay - 1 + ((millis + 1) % kMillisecondPerDay);
    }
  }

  // Convert UTC milliseconds to milliseconds in corresponding time zone
  inline static int64_t toTimezone(
      int64_t millis,
      const date::time_zone& zone) {
    auto timePoint = std::chrono::
        time_point<std::chrono::system_clock, std::chrono::milliseconds>(
            std::chrono::milliseconds(millis));

    auto epoch = zone.to_local(timePoint).time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
  }

  // Assuming tzID is in [1, 1680] range.
  // tzID - PrestoDB time zone ID.
  inline static int64_t getPrestoTZOffsetInMillis(int16_t tzID) {
    // PrestoDb time zone ids require some custom code.
    // Mapping is 1-based and covers [-14:00, +14:00] range without 00:00.
    return ((tzID <= 840) ? (tzID - 841) : (tzID - 840)) * 60 *
        kMillisecondsInSecond;
  }

  inline static int64_t toGMT(int64_t millis, int16_t tzID) {
    if (tzID == 0) {
      // No conversion required for time zone id 0, as it is '+00:00'.
      return millis;
    } else if (tzID <= 1680) {
      return millis - getPrestoTZOffsetInMillis(tzID);
    } else {
      // Other ids go this path.
      return toGMT(millis, *date::locate_zone(util::getTimeZoneName(tzID)));
    }
  }

  inline static int64_t toGMT(int64_t millis, const date::time_zone& zone) {
    date::local_time<std::chrono::milliseconds> localTime{
        std::chrono::milliseconds(millis)};
    std::chrono::
        time_point<std::chrono::system_clock, std::chrono::milliseconds>
            sysTime;
    try {
      sysTime = zone.to_sys(localTime);
    } catch (const date::ambiguous_local_time& error) {
      sysTime = zone.to_sys(localTime, date::choose::earliest);
    } catch (const date::nonexistent_local_time& error) {
      // If the time does not exist, fail the conversion.
      VELOX_USER_FAIL(error.what());
    }
    return sysTime.time_since_epoch().count();
  }

}; // TimeUtil
} // namespace facebook::velox
