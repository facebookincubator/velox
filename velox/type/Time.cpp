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

#include "velox/type/Time.h"
#include <charconv>
#include "velox/type/TimestampConversion.h"

namespace facebook::velox::util {

namespace {

// Helper: Parse a number from the given position using std::from_chars
bool parseNumber(const char* data, size_t size, size_t& pos, int32_t& result) {
  if (pos >= size) {
    return false;
  }

  const char* start = data + pos;
  const char* end = data + size;

  auto parseResult = std::from_chars(start, end, result);
  if (parseResult.ec != std::errc{}) {
    return false;
  }

  // Update position
  pos += (parseResult.ptr - start);
  return true;
}

// Helper: Parse fractional seconds (milliseconds)
Expected<int32_t>
parseFractionalSeconds(const char* data, size_t size, size_t& pos) {
  if (pos >= size || data[pos] != '.') {
    return 0; // No fractional part
  }

  pos++; // Skip the '.'

  if (pos >= size) {
    return folly::makeUnexpected(
        Status::UserError(
            "Invalid time format: fractional seconds incomplete"));
  }

  // Parse fractional seconds (milliseconds)
  int32_t fractionalPart = 0;

  const char* start = data + pos;
  const char* end = data + size;

  auto parseResult = std::from_chars(start, end, fractionalPart);
  if (parseResult.ec != std::errc{}) {
    return folly::makeUnexpected(
        Status::UserError(
            "Invalid time format: failed to parse fractional seconds"));
  }

  size_t digitCount = parseResult.ptr - start;
  pos += digitCount;

  // Check that we don't have more than 3 digits (millisecond precision)
  if (digitCount > 3) {
    return folly::makeUnexpected(
        Status::UserError(
            "Invalid time format: Microsecond precision not supported"));
  }

  // Convert to milliseconds by padding with zeros if needed
  // e.g., .1 -> 100ms, .12 -> 120ms, .123 -> 123ms
  for (size_t i = digitCount; i < 3; i++) {
    fractionalPart *= 10;
  }

  return fractionalPart;
}

// Helper: Validate time components
Status validateTimeComponents(const TimeComponents& components) {
  if (components.hour < 0 || components.hour >= kHoursPerDay) {
    return Status::UserError("Invalid hour value: {}", components.hour);
  }
  if (components.minute < 0 || components.minute >= kMinsPerHour) {
    return Status::UserError("Invalid minute value: {}", components.minute);
  }
  if (components.second < 0 || components.second >= kSecsPerMinute) {
    return Status::UserError("Invalid second value: {}", components.second);
  }
  if (components.millis < 0 || components.millis >= kMsecsPerSec) {
    return Status::UserError(
        "Invalid millisecond value: {}", components.millis);
  }
  return Status::OK();
}

// Helper: Convert time components to milliseconds since midnight
Expected<int64_t> timeComponentsToMillis(const TimeComponents& components) {
  int64_t result = static_cast<int64_t>(components.hour) * kMillisInHour +
      static_cast<int64_t>(components.minute) * kMillisInMinute +
      static_cast<int64_t>(components.second) * kMillisInSecond +
      static_cast<int64_t>(components.millis);

  // Validate time range (0 to 86399999 ms in a day)
  if (result < 0 || result >= kMillisInDay) {
    return folly::makeUnexpected(
        Status::UserError(
            "Time value {} is out of range [0, {})", result, kMillisInDay));
  }

  return result;
}

// Helper: Parse time components from string (H:m[:s[.SSS]])
Expected<TimeComponents> parseTimeComponents(const char* buf, size_t len) {
  TimeComponents components;
  size_t pos = 0;

  // Parse hour (required, 1-2 digits)
  if (!parseNumber(buf, len, pos, components.hour)) {
    return folly::makeUnexpected(
        Status::UserError("Invalid time format: failed to parse hour"));
  }

  // Skip first ':'
  if (pos >= len || buf[pos] != ':') {
    return folly::makeUnexpected(
        Status::UserError("Invalid time format: expected ':' after hour"));
  }
  pos++;

  // Parse minute (required, 1-2 digits)
  if (!parseNumber(buf, len, pos, components.minute)) {
    return folly::makeUnexpected(
        Status::UserError("Invalid time format: failed to parse minute"));
  }

  // Check if there's a second ':' for seconds (OPTIONAL)
  if (pos < len && buf[pos] == ':') {
    pos++; // Skip the ':'

    // Parse second (optional, 1-2 digits)
    if (!parseNumber(buf, len, pos, components.second)) {
      return folly::makeUnexpected(
          Status::UserError("Invalid time format: failed to parse second"));
    }

    // Parse optional fractional seconds
    auto millisResult = parseFractionalSeconds(buf, len, pos);
    if (millisResult.hasError()) {
      return folly::makeUnexpected(millisResult.error());
    }
    components.millis = millisResult.value();
  }
  // If no second ':', seconds and millis remain at 0 (default)

  // Check for trailing characters
  if (pos < len) {
    return folly::makeUnexpected(
        Status::UserError(
            "Invalid time format: unexpected trailing characters at position {}",
            pos));
  }

  return components;
}

} // namespace

Expected<int64_t> fromTimeString(const char* buf, size_t len) {
  auto componentsResult = parseTimeComponents(buf, len);
  if (componentsResult.hasError()) {
    return folly::makeUnexpected(componentsResult.error());
  }

  auto components = componentsResult.value();

  // Validate all components
  auto validationStatus = validateTimeComponents(components);
  if (!validationStatus.ok()) {
    return folly::makeUnexpected(validationStatus);
  }

  // Convert to milliseconds since midnight
  return timeComponentsToMillis(components);
}

Expected<int16_t>
parseTimezoneOffset(const char* buf, size_t len, bool allowCompactFormat) {
  if (len < 2) {
    return folly::makeUnexpected(
        Status::UserError("Invalid timezone offset: too short"));
  }

  char sign = buf[0];
  if (sign != '+' && sign != '-') {
    return folly::makeUnexpected(
        Status::UserError("Invalid timezone offset: must start with + or -"));
  }

  int signMultiplier = (sign == '+') ? 1 : -1;

  // Start parsing after the sign
  size_t pos = 1;
  const size_t startPos = pos;

  // Parse offset hours/minutes
  // Supports formats based on allowCompactFormat flag:
  // Always supported:
  //   1. +HH:mm (e.g., +07:09) - colon separates hours and minutes
  //   2. +HH (e.g., +07) - exactly 2 digits, no colon
  // Conditionally supported (allowCompactFormat=true):
  //   3. +HHmm (e.g., +0709) - exactly 4 digits, no colon

  int32_t offsetHours = 0;
  int32_t offsetMinutes = 0;

  // Try to parse as many digits as we can for hours
  if (!parseNumber(buf, len, pos, offsetHours)) {
    return folly::makeUnexpected(
        Status::UserError("Invalid timezone offset: failed to parse hours"));
  }

  size_t digitsRead = pos - startPos;

  // Check what comes next
  if (pos < len && buf[pos] == ':') {
    // Format: +HH:mm
    // Hours should be 2 digits only when followed by colon
    if (digitsRead != 2) {
      return folly::makeUnexpected(
          Status::UserError(
              "Invalid timezone offset format: digits before ':' not equal to 2"));
    }
    pos++; // Skip ':'
    if (!parseNumber(buf, len, pos, offsetMinutes)) {
      return folly::makeUnexpected(
          Status::UserError(
              "Invalid timezone offset: failed to parse minutes after ':'"));
    }
  } else if (digitsRead == 4) {
    // Format: +HHmm (e.g., +0709)
    // Only allowed if allowCompactFormat is true
    if (!allowCompactFormat) {
      return folly::makeUnexpected(
          Status::UserError(
              "Invalid timezone offset format: compact format +HHmm not allowed, use +HH:mm or +HH"));
    }
    // Extract last 2 digits as minutes
    offsetMinutes = offsetHours % 100;
    offsetHours = offsetHours / 100;
  } else if (digitsRead != 2) {
    // Valid : 2
    //    Format: +HH (e.g., +07)
    //    Minutes remain 0
    // Invalid: 1, 3, or 5+ digits without colon
    return folly::makeUnexpected(
        Status::UserError(
            "Invalid timezone offset format: expected +HH:mm or +HH"));
  }

  // Check for trailing characters
  if (pos < len) {
    return folly::makeUnexpected(
        Status::UserError(
            "Invalid timezone offset: unexpected trailing characters at position {}",
            pos));
  }

  // Calculate total offset in minutes
  auto totalOffsetMinutes =
      (signMultiplier * (offsetHours * 60 + offsetMinutes));

  // Validate timezone offset range (-840 to 840 minutes, i.e., -14:00 to
  // +14:00)
  if (totalOffsetMinutes < -kTimeZoneBias ||
      totalOffsetMinutes > kTimeZoneBias) {
    return folly::makeUnexpected(
        Status::UserError(
            "Timezone offset {} minutes is out of range [-840, 840]",
            totalOffsetMinutes));
  }

  return static_cast<int16_t>(totalOffsetMinutes);
}

Expected<int64_t> fromTimeWithTimezoneString(
    const char* buf,
    size_t len,
    bool allowCompactFormat) {
  if (len == 0) {
    return folly::makeUnexpected(
        Status::UserError("Invalid time with timezone: empty string"));
  }

  // Find timezone offset marker (+ or -)
  // Search from right to left for better performance
  std::string_view sv(buf, len);
  auto tzPos = sv.find_last_of("+-");
  int tzStartPos =
      (tzPos != std::string_view::npos) ? static_cast<int>(tzPos) : -1;

  if (tzStartPos == -1) {
    return folly::makeUnexpected(
        Status::UserError(
            "Invalid time with timezone: missing timezone offset"));
  }

  // Extract time component (before timezone, trimming optional space)
  int timeEndPos = tzStartPos;
  if (timeEndPos > 0 && buf[timeEndPos - 1] == ' ') {
    timeEndPos--; // Remove space before timezone
  }

  if (timeEndPos == 0) {
    return folly::makeUnexpected(
        Status::UserError(
            "Invalid time with timezone: missing time component"));
  }

  // Parse time component
  auto timeResult = fromTimeString(buf, timeEndPos);
  if (timeResult.hasError()) {
    return folly::makeUnexpected(timeResult.error());
  }

  int64_t millisLocal = timeResult.value();

  // Parse timezone offset, passing through the allowCompactFormat flag
  auto tzOffsetResult = parseTimezoneOffset(
      buf + tzStartPos, len - tzStartPos, allowCompactFormat);
  if (tzOffsetResult.hasError()) {
    return folly::makeUnexpected(tzOffsetResult.error());
  }

  int16_t offsetMinutes = tzOffsetResult.value();

  // Convert local time to UTC using utility function
  // Example: 12:00:00+05:30 means the local time is 12:00 in UTC+5:30
  // To get UTC time, we subtract 5:30, resulting in 06:30:00 UTC
  int64_t millisUtc = localToUtcTime(millisLocal, offsetMinutes);

  // Encode timezone offset and pack with time
  int16_t encodedTimezone = biasEncode(offsetMinutes);
  return pack(millisUtc, encodedTimezone);
}

} // namespace facebook::velox::util
