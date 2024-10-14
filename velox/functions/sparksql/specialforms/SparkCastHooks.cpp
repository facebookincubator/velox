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
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {

SparkCastHooks::SparkCastHooks(const core::QueryConfig& config) : CastHooks() {
  const auto sessionTzName = config.sessionTimezone();
  if (config.adjustTimestampToTimezone() && !sessionTzName.empty()) {
    options_.timeZone = tz::locateZone(sessionTzName);
  }
}

Expected<Timestamp> SparkCastHooks::castStringToTimestamp(
    const StringView& view) const {
  // Allows all patterns supported by Spark:
  // `[+-]yyyy*`
  // `[+-]yyyy*-[m]m`
  // `[+-]yyyy*-[m]m-[d]d`
  // `[+-]yyyy*-[m]m-[d]d `
  // `[+-]yyyy*-[m]m-[d]d [h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
  // `[+-]yyyy*-[m]m-[d]dT[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
  // `[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
  // `T[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
  //
  // where `zone_id` should have one of the forms:
  //   1. Z - Zulu time zone UTC+0
  //   2. +|-[h]h:[m]m
  //   3. A short id, see
  //     https://docs.oracle.com/javase/8/docs/api/java/time/ZoneId.html#SHORT_IDS
  //   4. An id with one of the prefixes UTC+, UTC-, GMT+, GMT-, UT+ or UT-,
  //   and a suffix in the following formats:
  //     a. +|-h[h]
  //     b. +|-hh[:]mm
  //     c. +|-hh:mm:ss
  //     d. +|-hhmmss
  //   5. Region-based zone IDs in the form `area/city`, such as `Europe/Paris`
  const auto conversionResult = util::fromTimestampWithTimezoneString(
      view.data(), view.size(), util::TimestampParseMode::kSparkCast);
  if (conversionResult.hasError()) {
    return folly::makeUnexpected(conversionResult.error());
  }

  auto result = conversionResult.value();

  if (result.second != nullptr) {
    // If the parsed string has timezone information, convert the timestamp at
    // GMT at that time.
    result.first.toGMT(*result.second);
  } else if (options_.timeZone != nullptr) {
    // If the input string contains no timezone information, determine whether
    // it should be interpreted as being in the session timezone and, if so,
    // convert it to GMT.
    result.first.toGMT(*options_.timeZone);
  }
  return result.first;
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

const TimestampToStringOptions& SparkCastHooks::timestampToStringOptions()
    const {
  static constexpr TimestampToStringOptions options = {
      .precision = TimestampToStringOptions::Precision::kMicroseconds,
      .leadingPositiveSign = true,
      .skipTrailingZeros = true,
      .zeroPaddingYear = true,
      .dateTimeSeparator = ' ',
  };
  return options;
}

exec::PolicyType SparkCastHooks::getPolicy() const {
  return exec::PolicyType::SparkCastPolicy;
}
} // namespace facebook::velox::functions::sparksql
