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

#include <cmath>
#include <cstdint>
#include <exception>

#include "velox/common/base/VeloxException.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/TimestampUtils.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/Type.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {

namespace {

// Max DATE magnitude before TIMESTAMP_NTZ's microseconds-since-epoch range
// overflows. Spark throws here unconditionally, ANSI or not.
constexpr int64_t kMaxTimestampUtcDays = 106'751'991;

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

TimestampToStringOptions SparkCastHooks::timestampUtcToStringOptions() const {
  auto options = timestampToStringOptions_;
  options.timeZone = nullptr;
  return options;
}

Expected<Timestamp> SparkCastHooks::castStringToTimestamp(
    const StringView& view,
    bool adjustTimezone) const {
  auto conversionResult = util::fromTimestampWithTimezoneString(
      view.data(), view.size(), util::TimestampParseMode::kSparkCast);
  if (conversionResult.hasError()) {
    return folly::makeUnexpected(conversionResult.error());
  }
  if (!adjustTimezone) {
    // For TIMESTAMP UTC, ignore any timezone suffix and store the parsed
    // timestamp fields as-is, not subject to session timezone adjustment.
    return conversionResult.value().timestamp;
  }
  auto sessionTimezone = config_.sessionTimezone().empty()
      ? nullptr
      : tz::locateZone(config_.sessionTimezone());
  return util::fromParsedTimestampWithTimeZone(
      conversionResult.value(), sessionTimezone);
}

template <typename T>
Expected<Timestamp> SparkCastHooks::castNumberToTimestamp(
    T value,
    bool allowOverflow) const {
  static constexpr long double kMaxTimestampMicros =
      static_cast<long double>(std::numeric_limits<int64_t>::max());
  static constexpr long double kMinTimestampMicros =
      static_cast<long double>(std::numeric_limits<int64_t>::min());

  const long double micros =
      static_cast<long double>(value) * Timestamp::kMicrosecondsInSecond;

  if (micros > kMaxTimestampMicros) {
    if (allowOverflow) {
      return Timestamp::fromMicrosNoError(std::numeric_limits<int64_t>::max());
    }
    return folly::makeUnexpected(
        Status::UserError(
            "The value cannot be cast to TIMESTAMP due to an overflow."));
  }

  if (micros < kMinTimestampMicros) {
    if (allowOverflow) {
      return Timestamp::fromMicrosNoError(std::numeric_limits<int64_t>::min());
    }
    return folly::makeUnexpected(
        Status::UserError(
            "The value cannot be cast to TIMESTAMP due to an overflow."));
  }

  return Timestamp::fromMicrosNoError(static_cast<int64_t>(micros));
}

Expected<Timestamp> SparkCastHooks::castIntToTimestamp(int64_t seconds) const {
  return castNumberToTimestamp(seconds, true);
}

Expected<int64_t> SparkCastHooks::castTimestampToBigint(
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
  if (allowOverflow_) {
    if (FOLLY_UNLIKELY(std::isnan(value) || std::isinf(value))) {
      return std::nullopt;
    }

    auto result = castNumberToTimestamp(value, allowOverflow_);
    VELOX_DCHECK(!result.hasError());
    return result.value();
  }

  if (FOLLY_UNLIKELY(std::isnan(value) || std::isinf(value))) {
    return folly::makeUnexpected(
        Status::UserError(
            "The value cannot be cast to TIMESTAMP because it is malformed."));
  }

  auto result = castNumberToTimestamp(value, allowOverflow_);
  if (result.hasError()) {
    return folly::makeUnexpected(result.error());
  }
  return result.value();
}

Expected<Timestamp> SparkCastHooks::castBooleanToTimestamp(bool val) const {
  return Timestamp::fromMicrosNoError(val ? 1 : 0);
}

bool SparkCastHooks::decimalToFloatHighPrecisionCastEnabled() const {
  return SparkQueryConfig{config_}.decimalToFloatHighPrecisionCastEnabled();
}

namespace {

// Spark UTF8String.trimAll trims boundary bytes 0x00..0x20 and 0x7F.
bool isUtf8StringTrimAllByte(char value) {
  const auto byte = static_cast<uint8_t>(value);
  return byte <= 0x20 || byte == 0x7F;
}

// Java String.trim trims boundary characters <= U+0020.
bool isJavaStringTrimByte(char value) {
  return static_cast<uint8_t>(value) <= 0x20;
}

template <typename TShouldTrim>
StringView trimBytes(const StringView& view, TShouldTrim shouldTrim) {
  const auto* data = view.data();
  size_t start{0};
  size_t end{view.size()};

  while (start < end && shouldTrim(data[start])) {
    ++start;
  }

  while (end > start && shouldTrim(data[end - 1])) {
    --end;
  }

  return StringView(data + start, end - start);
}

StringView trimUtf8StringAll(const StringView& view) {
  return trimBytes(view, isUtf8StringTrimAllByte);
}

StringView trimJavaString(const StringView& view) {
  return trimBytes(view, isJavaStringTrimByte);
}

// Type::isDecimal() runs a dynamic_cast for BIGINT and HUGEINT types, which is
// too costly to call for every row. Plain BIGINT and HUGEINT types are
// singletons, so check their identity first. Decimal types are final, so a
// typeid comparison identifies them exactly.
bool isDecimalType(const Type& type) {
  if (&type == BigintType::create().get() ||
      &type == HugeintType::create().get()) {
    return false;
  }
  const auto& typeId = typeid(type);
  return typeId == typeid(ShortDecimalType) ||
      typeId == typeid(LongDecimalType);
}

} // namespace

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
      trimUtf8StringAll(dateString), util::ParseMode::kSparkCast);
}

Expected<int64_t> SparkCastHooks::castStringToTime(
    StringView timeString,
    const tz::TimeZone* /*timeZone*/,
    int64_t /*sessionStartTimeMs*/) const {
  try {
    return TIME_MICRO_UTC()->valueToTime(timeString, /*requireSeconds=*/false);
  } catch (const VeloxException& e) {
    return folly::makeUnexpected(Status::UserError(e.message()));
  } catch (const std::exception& e) {
    return folly::makeUnexpected(Status::UserError(e.what()));
  }
}

Expected<float> SparkCastHooks::castStringToReal(const StringView& data) const {
  return util::Converter<TypeKind::REAL>::tryCast(data);
}

Expected<double> SparkCastHooks::castStringToDouble(
    const StringView& data) const {
  return util::Converter<TypeKind::DOUBLE>::tryCast(data);
}

// Keep these target-type-specific rules in sync with Spark. Spark currently
// uses UTF8String-style trimming for most casts but Java String.trim for
// floating point and decimal casts. See
// https://issues.apache.org/jira/browse/SPARK-59182.
StringView SparkCastHooks::removeWhiteSpaces(
    const StringView& view,
    const Type& toType) const {
  switch (toType.kind()) {
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
      return trimJavaString(view);
    case TypeKind::BIGINT:
    case TypeKind::HUGEINT:
      return isDecimalType(toType) ? trimJavaString(view)
                                   : trimUtf8StringAll(view);
    default:
      return trimUtf8StringAll(view);
  }
}

void SparkCastHooks::castDateTimestampToGMT(
    Timestamp& timestamp,
    const tz::TimeZone& timeZone) const {
  toGMTWithGapCorrection(timestamp, timeZone);
}

bool SparkCastHooks::isDateOverflowForTimestampUtc(int64_t days) const {
  return days > kMaxTimestampUtcDays || days < -kMaxTimestampUtcDays;
}

exec::PolicyType SparkCastHooks::getPolicy() const {
  if (!allowOverflow_) {
    return exec::PolicyType::SparkTryCastPolicy;
  }
  return exec::PolicyType::SparkCastPolicy;
}
} // namespace facebook::velox::functions::sparksql
