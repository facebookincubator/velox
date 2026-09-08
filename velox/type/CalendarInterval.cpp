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

#include "velox/type/CalendarInterval.h"

#include <fmt/format.h>
#include <cmath>
#include <sstream>

namespace facebook::velox {

namespace {
constexpr int64_t kMicrosPerSecond = 1'000'000L;
constexpr int64_t kMicrosPerMinute = 60 * kMicrosPerSecond;
constexpr int64_t kMicrosPerHour = 60 * kMicrosPerMinute;
} // namespace

std::string CalendarInterval::toString() const {
  if (months == 0 && days == 0 && microseconds == 0) {
    return "0 seconds";
  }

  std::stringstream ss;
  bool first = true;

  auto append = [&](int64_t value, const char* unit) {
    if (value != 0) {
      if (!first) {
        ss << " ";
      }
      ss << value << " " << unit;
      first = false;
    }
  };

  // Split months into years and remaining months (matching Spark).
  int32_t years = months / 12;
  int32_t remainingMonths = months % 12;
  append(years, "years");
  append(remainingMonths, "months");

  append(days, "days");

  // Split microseconds into hours, minutes, seconds (matching Spark).
  int64_t remaining = microseconds;
  int64_t hours = remaining / kMicrosPerHour;
  remaining %= kMicrosPerHour;
  int64_t minutes = remaining / kMicrosPerMinute;
  remaining %= kMicrosPerMinute;

  append(hours, "hours");
  append(minutes, "minutes");

  // For seconds, preserve sign and fractional part.
  if (remaining != 0) {
    int64_t fracMicros = remaining % kMicrosPerSecond;
    if (fracMicros == 0) {
      int64_t wholeSecs = remaining / kMicrosPerSecond;
      append(wholeSecs, "seconds");
    } else {
      if (!first) {
        ss << " ";
      }
      // Format fractional seconds with up to 6 digits, trimming trailing zeros.
      double secs = static_cast<double>(std::abs(remaining)) / kMicrosPerSecond;
      std::string secStr = fmt::format("{:.6f}", secs);
      // Trim trailing zeros.
      auto pos = secStr.find_last_not_of('0');
      if (pos != std::string::npos && secStr[pos] == '.') {
        secStr.erase(pos);
      } else if (pos != std::string::npos) {
        secStr.erase(pos + 1);
      }
      if (remaining < 0) {
        ss << "-";
      }
      ss << secStr << " seconds";
      first = false;
    }
  }

  return ss.str();
}

} // namespace facebook::velox
