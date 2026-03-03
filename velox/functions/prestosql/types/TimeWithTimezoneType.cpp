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

#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"

#include "velox/type/Time.h"

namespace facebook::velox {

folly::dynamic TimeWithTimezoneType::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "Type";
  obj["type"] = name();
  return obj;
}

StringView TimeWithTimezoneType::valueToString(
    int64_t value,
    char* const startPos) const {
  // TIME WITH TIME ZONE is encoded similarly to TIMESTAMP WITH TIME ZONE
  // with the most significnat 52 bits representing the time component and the
  // least 12 bits representing the timezone minutes. This is different from
  // TIMESTAMP WITH TIMEZONE where the last 12 bits represent the timezone
  // offset. The timezone offset minutes are stored by value, encoded in the
  // type itself. This allows the type to be used in a timezone-agnostic manner.
  //
  // The time component is a 52 bit value representing the number of
  // milliseconds since midnight in UTC.

  int64_t millisUtc = util::unpackMillisUtc(value);

  // Ensure time component is within valid range
  VELOX_CHECK_GE(millisUtc, 0, "Time component is negative");
  VELOX_CHECK_LE(millisUtc, util::kMillisInDay, "Time component is too large");

  // TimeZone's are encoded as a 12 bit value.
  // This represents a range of -14:00 to +14:00, with 0 representing UTC.
  // The range is from -840 to 840 minutes, we thus encode by doing bias
  // encoding and taking 840 as the bias.
  auto timezoneMinutes = util::unpackZoneOffset(value);

  VELOX_CHECK_GE(timezoneMinutes, 0, "Timezone offset is less than -14:00");
  VELOX_CHECK_LE(
      timezoneMinutes, 1680, "Timezone offset is greater than +14:00");

  // Decode timezone offset from bias-encoded value
  int16_t offsetMinutes = util::decodeTimezoneOffset(timezoneMinutes);
  auto decodedMinutes = std::abs(offsetMinutes);

  const auto isBehindUTCString = (offsetMinutes >= 0) ? "+" : "-";

  // Convert UTC time to local time using utility function
  // Example: If UTC time is 06:30:00 and timezone is +05:30,
  // the local time is 12:00:00
  int64_t millisLocal = util::utcToLocalTime(millisUtc, offsetMinutes);

  int64_t hours = millisLocal / util::kMillisInHour;
  int64_t remainingMs = millisLocal % util::kMillisInHour;
  int64_t minutes = remainingMs / util::kMillisInMinute;
  remainingMs = remainingMs % util::kMillisInMinute;
  int64_t seconds = remainingMs / util::kMillisInSecond;
  int64_t millis = remainingMs % util::kMillisInSecond;

  int16_t offsetHours = decodedMinutes / util::kMinutesInHour;
  int16_t remainingOffsetMinutes = decodedMinutes % util::kMinutesInHour;

  fmt::format_to_n(
      startPos,
      kTimeWithTimezoneToVarcharRowSize,
      "{:02d}:{:02d}:{:02d}.{:03d}{}{:02d}:{:02d}",
      hours,
      minutes,
      seconds,
      millis,
      isBehindUTCString,
      offsetHours,
      remainingOffsetMinutes);
  return StringView{startPos, kTimeWithTimezoneToVarcharRowSize};
}

} // namespace facebook::velox
