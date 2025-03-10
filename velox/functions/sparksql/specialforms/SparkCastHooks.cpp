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
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/functions/sparksql/DecimalUtil.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {

SparkCastHooks::SparkCastHooks(const velox::core::QueryConfig& config)
    : config_(config) {
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

Expected<std::optional<Timestamp>> SparkCastHooks::castDoubleToTimestamp(
    double value) const {
  if (FOLLY_UNLIKELY(std::isnan(value) || std::isinf(value))) {
    return std::nullopt;
  }
  return castNumberToTimestamp(value);
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

namespace {

// Aligns with Spark, which uses BigDecimal in Java.
// Reference:
// https://github.com/openjdk/jdk8u-dev/blob/20e72d16f569e823a9ecdd9951a742b4397ca978/jdk/src/share/classes/java/math/BigDecimal.java#L3294
template <typename T>
struct FloatingTraits {};

template <>
struct FloatingTraits<float> {
  static constexpr int64_t kMaxExact = 1L << 22;

  // Powers of 10 which can be represented exactly in float.
  static constexpr float kPowersOfTen[] = {
      1.0e0f,
      1.0e1f,
      1.0e2f,
      1.0e3f,
      1.0e4f,
      1.0e5f,
      1.0e6f,
      1.0e7f,
      1.0e8f,
      1.0e9f,
      1.0e10f};
  static constexpr size_t kPowersOfTenSize =
      sizeof(kPowersOfTen) / sizeof(kPowersOfTen[0]);
};

template <>
struct FloatingTraits<double> {
  static constexpr int64_t kMaxExact = 1L << 52;

  // Powers of 10 which can be represented exactly in double.
  static constexpr double kPowersOfTen[] = {
      1.0e0,  1.0e1,  1.0e2,  1.0e3,  1.0e4,  1.0e5,  1.0e6,  1.0e7,
      1.0e8,  1.0e9,  1.0e10, 1.0e11, 1.0e12, 1.0e13, 1.0e14, 1.0e15,
      1.0e16, 1.0e17, 1.0e18, 1.0e19, 1.0e20, 1.0e21, 1.0e22};
  static constexpr size_t kPowersOfTenSize =
      sizeof(kPowersOfTen) / sizeof(kPowersOfTen[0]);
};

} // namespace

Expected<float> SparkCastHooks::castShortDecimalToReal(
    int64_t unscaledValue,
    uint8_t precision,
    uint8_t scale) const {
  return doCastDecimalToFloatingType<int64_t, float>(
      unscaledValue, precision, scale);
}

Expected<float> SparkCastHooks::castLongDecimalToReal(
    int128_t unscaledValue,
    uint8_t precision,
    uint8_t scale) const {
  return doCastDecimalToFloatingType<int128_t, float>(
      unscaledValue, precision, scale);
}

Expected<double> SparkCastHooks::castShortDecimalToDouble(
    int64_t unscaledValue,
    uint8_t precision,
    uint8_t scale) const {
  return doCastDecimalToFloatingType<int64_t, double>(
      unscaledValue, precision, scale);
}
Expected<double> SparkCastHooks::castLongDecimalToDouble(
    int128_t unscaledValue,
    uint8_t precision,
    uint8_t scale) const {
  return doCastDecimalToFloatingType<int128_t, double>(
      unscaledValue, precision, scale);
}

template <typename FromNative, typename T>
Expected<T> SparkCastHooks::doCastDecimalToFloatingType(
    FromNative unscaledValue,
    uint8_t precision,
    uint8_t scale) const {
  static_assert(
      std::is_same_v<T, float> || std::is_same_v<T, double>,
      "T must be either float or double");
  if (scale == 0) {
    return static_cast<T>(unscaledValue);
  }

  if (scale < FloatingTraits<T>::kPowersOfTenSize &&
      DecimalUtil::absValue<FromNative>(unscaledValue) <
          FloatingTraits<T>::kMaxExact) {
    return static_cast<T>(unscaledValue) /
        FloatingTraits<T>::kPowersOfTen[scale];
  }

  // Cast decimal to string, then string to floating
  auto rowSize =
      facebook::velox::DecimalUtil::maxStringViewSize(precision, scale);
  char buffer[rowSize];
  memset(buffer, 0, rowSize);
  auto size = facebook::velox::DecimalUtil::castToString<FromNative>(
      unscaledValue, scale, rowSize, buffer);
  if constexpr (std::is_same_v<T, float>) {
    return SparkCastHooks::castStringToReal(StringView(buffer, size));
  } else {
    return SparkCastHooks::castStringToDouble(StringView(buffer, size));
  }
}

exec::PolicyType SparkCastHooks::getPolicy() const {
  return exec::PolicyType::SparkCastPolicy;
}
} // namespace facebook::velox::functions::sparksql
