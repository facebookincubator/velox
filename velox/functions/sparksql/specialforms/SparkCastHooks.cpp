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

#include "velox/functions/sparksql/specialforms/SparkCastHooks.h"
#include <charconv>
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {

namespace {

// Time constants
constexpr int32_t kHoursPerDay = 24;
constexpr int32_t kMinsPerHour = 60;
constexpr int32_t kSecsPerMinute = 60;

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

// Helper: Parse fractional seconds (microseconds)
Expected<int32_t>
parseFractionalSecondsMicros(const char* data, size_t size, size_t& pos) {
  if (pos >= size || data[pos] != '.') {
    return 0; // No fractional part
  }

  pos++; // Skip the '.'

  if (pos >= size) {
    return folly::makeUnexpected(
        Status::UserError(
            "Invalid time format: fractional seconds incomplete"));
  }

  // Parse fractional seconds (microseconds)
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

  if (digitCount > 6) {
    return folly::makeUnexpected(
        Status::UserError(
            "Invalid time format: precision beyond microseconds not supported"));
  }

  // Convert to microsecond precision by padding with zeros if needed.
  for (size_t i = digitCount; i < 6; i++) {
    fractionalPart *= 10;
  }

  return fractionalPart;
}

/// Represents parsed time components with microsecond precision
struct TimeComponentsMicros {
  int32_t hour = 0;
  int32_t minute = 0;
  int32_t second = 0;
  int32_t micros = 0;
};

// Helper: Validate time components (microseconds)
Status validateTimeComponentsMicros(const TimeComponentsMicros& components) {
  if (components.hour < 0 || components.hour >= kHoursPerDay) {
    return Status::UserError("Invalid hour value: {}", components.hour);
  }
  if (components.minute < 0 || components.minute >= kMinsPerHour) {
    return Status::UserError("Invalid minute value: {}", components.minute);
  }
  if (components.second < 0 || components.second >= kSecsPerMinute) {
    return Status::UserError("Invalid second value: {}", components.second);
  }
  if (components.micros < 0 || components.micros >= 1'000'000) {
    return Status::UserError(
        "Invalid microsecond value: {}", components.micros);
  }
  return Status::OK();
}

// Helper: Convert time components to microseconds since midnight
Expected<int64_t> timeComponentsToMicros(
    const TimeComponentsMicros& components) {
  constexpr int64_t kMicrosInSecond = 1'000'000;
  constexpr int64_t kMicrosInMinute = 60 * kMicrosInSecond;
  constexpr int64_t kMicrosInHour = 60 * kMicrosInMinute;
  constexpr int64_t kMicrosInDay = 24 * kMicrosInHour;

  int64_t result = static_cast<int64_t>(components.hour) * kMicrosInHour +
      static_cast<int64_t>(components.minute) * kMicrosInMinute +
      static_cast<int64_t>(components.second) * kMicrosInSecond +
      static_cast<int64_t>(components.micros);

  // Validate time range (0 to 86399999999 us in a day)
  if (result < 0 || result >= kMicrosInDay) {
    return folly::makeUnexpected(
        Status::UserError(
            "Time value {} is out of range [0, {})", result, kMicrosInDay));
  }

  return result;
}

// Helper: Parse time components from string (H:m:s[.SSSSSS])
Expected<TimeComponentsMicros> parseTimeComponentsMicros(
    const char* buf,
    size_t len) {
  TimeComponentsMicros components;
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

  // Require a second ':' for seconds.
  if (pos >= len || buf[pos] != ':') {
    return folly::makeUnexpected(
        Status::UserError("Invalid time format: expected ':' after minute"));
  }
  pos++; // Skip the ':'

  // Parse second (required, 1-2 digits)
  if (!parseNumber(buf, len, pos, components.second)) {
    return folly::makeUnexpected(
        Status::UserError("Invalid time format: failed to parse second"));
  }

  // Parse optional fractional seconds
  auto microsResult = parseFractionalSecondsMicros(buf, len, pos);
  if (microsResult.hasError()) {
    return folly::makeUnexpected(microsResult.error());
  }
  components.micros = microsResult.value();

  // Check for trailing characters
  if (pos < len) {
    return folly::makeUnexpected(
        Status::UserError(
            "Invalid time format: unexpected trailing characters at position {}",
            pos));
  }

  return components;
}

Expected<int64_t> fromTimeStringMicros(const char* buf, size_t len) {
  auto componentsResult = parseTimeComponentsMicros(buf, len);
  if (componentsResult.hasError()) {
    return folly::makeUnexpected(componentsResult.error());
  }

  auto components = componentsResult.value();

  // Validate all components
  auto validationStatus = validateTimeComponentsMicros(components);
  if (!validationStatus.ok()) {
    return folly::makeUnexpected(validationStatus);
  }

  // Convert to microseconds since midnight
  return timeComponentsToMicros(components);
}

} // namespace

SparkCastHooks::SparkCastHooks(
    const velox::core::QueryConfig& config,
    bool allowOverflow)
    : config_(config), allowOverflow_(allowOverflow) {
  const auto sessionTzName = config.sessionTimezone();
  if (!sessionTzName.empty()) {
    timestampToStringOptions_.timeZone = tz::locateZone(sessionTzName);
  }
}

Expected<Timestamp> SparkCastHooks::castStringToTimestamp(
    const StringView& view) const {
  auto conversionResult = util::fromTimestampWithTimezoneString(
      view.data(), view.size(), util::TimestampParseMode::kSparkCast);
  if (conversionResult.hasError()) {
    return folly::makeUnexpected(conversionResult.error());
  }

  auto sessionTimezone = config_.sessionTimezone().empty()
      ? nullptr
      : tz::locateZone(config_.sessionTimezone());
  return util::fromParsedTimestampWithTimeZone(
      conversionResult.value(), sessionTimezone);
}

template <typename T>
Expected<Timestamp> SparkCastHooks::castNumberToTimestamp(T seconds) const {
  // Spark internally use microsecond precision for timestamp.
  // To avoid overflow, we need to check the range of seconds.
  static constexpr int64_t maxSeconds =
      std::numeric_limits<int64_t>::max() / Timestamp::kMicrosecondsInSecond;
  if (seconds > maxSeconds) {
    return Timestamp::fromMicrosNoError(std::numeric_limits<int64_t>::max());
  }
  if (seconds < -maxSeconds) {
    return Timestamp::fromMicrosNoError(std::numeric_limits<int64_t>::min());
  }

  if constexpr (std::is_floating_point_v<T>) {
    return Timestamp::fromMicrosNoError(
        static_cast<int64_t>(seconds * Timestamp::kMicrosecondsInSecond));
  }

  return Timestamp(seconds, 0);
}

Expected<Timestamp> SparkCastHooks::castIntToTimestamp(int64_t seconds) const {
  return castNumberToTimestamp(seconds);
}

Expected<int64_t> SparkCastHooks::castTimestampToInt(
    Timestamp timestamp) const {
  auto micros = timestamp.toMicros();
  if (micros < 0) {
    return std::floor(
        static_cast<double>(micros) / Timestamp::kMicrosecondsInSecond);
  }
  return micros / Timestamp::kMicrosecondsInSecond;
}

Expected<std::optional<Timestamp>> SparkCastHooks::castDoubleToTimestamp(
    double value) const {
  if (FOLLY_UNLIKELY(std::isnan(value) || std::isinf(value))) {
    return std::nullopt;
  }
  return castNumberToTimestamp(value);
}

Expected<Timestamp> SparkCastHooks::castBooleanToTimestamp(bool val) const {
  return Timestamp::fromMicrosNoError(val ? 1 : 0);
}

Expected<int32_t> SparkCastHooks::castStringToDate(
    const StringView& dateString) const {
  // Allows all patterns supported by Spark:
  // `[+-]yyyy*`
  // `[+-]yyyy*-[m]m`
  // `[+-]yyyy*-[m]m-[d]d`
  // `[+-]yyyy*-[m]m-[d]d *`
  // `[+-]yyyy*-[m]m-[d]dT*`
  // The asterisk `*` in `yyyy*` stands for any numbers.
  // For the last two patterns, the trailing `*` can represent none or any
  // sequence of characters, e.g:
  //   "1970-01-01 123"
  //   "1970-01-01 (BC)"
  return util::fromDateString(
      removeWhiteSpaces(dateString), util::ParseMode::kSparkCast);
}

Expected<int64_t> SparkCastHooks::castStringToTime(
    const StringView& timeString,
    const tz::TimeZone* /* timeZone */,
    int64_t /* sessionStartTimeMs */) const {
  // Spark represents TIME as BIGINT (microseconds since midnight).
  // Parse directly to microseconds - validation and error handling
  // flow through CAST/TRY_CAST semantics automatically.
  return fromTimeStringMicros(timeString.data(), timeString.size());
}

Expected<float> SparkCastHooks::castStringToReal(const StringView& data) const {
  return util::Converter<TypeKind::REAL>::tryCast(data);
}

Expected<double> SparkCastHooks::castStringToDouble(
    const StringView& data) const {
  return util::Converter<TypeKind::DOUBLE>::tryCast(data);
}

StringView SparkCastHooks::removeWhiteSpaces(const StringView& view) const {
  StringView output;
  stringImpl::trimUnicodeWhiteSpace<true, true, StringView, StringView>(
      output, view);
  return output;
}

exec::PolicyType SparkCastHooks::getPolicy() const {
  if (!allowOverflow_) {
    return exec::PolicyType::SparkTryCastPolicy;
  }
  return exec::PolicyType::SparkCastPolicy;
}
} // namespace facebook::velox::functions::sparksql
