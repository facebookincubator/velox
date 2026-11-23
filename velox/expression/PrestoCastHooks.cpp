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

#include <algorithm>
#include <cmath>
#include <exception>
#include <string_view>

#include <fast_float/fast_float.h>
#include <folly/Expected.h>
#include <folly/Range.h>

#include "velox/common/base/VeloxException.h"
#include "velox/expression/PrestoCastHooks.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/Type.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::exec {

PrestoCastHooks::PrestoCastHooks(const core::QueryConfig& config)
    : CastHooks(),
      legacyCast_(config.isLegacyCast()),
      lenientInfNan_(config.isCastStringToFloatingPointLenientInfNan()) {
  if (!legacyCast_) {
    options_.zeroPaddingYear = true;
    options_.dateTimeSeparator = ' ';
    const auto sessionTzName = config.sessionTimezone();
    if (config.adjustTimestampToTimezone() && !sessionTzName.empty()) {
      options_.timeZone = tz::locateZone(sessionTzName);
    }
  }
}

Expected<Timestamp> PrestoCastHooks::castStringToTimestamp(
    const StringView& view,
    bool /*adjustTimezone*/) const {
  const auto conversionResult = util::fromTimestampWithTimezoneString(
      view.data(),
      view.size(),
      legacyCast_ ? util::TimestampParseMode::kLegacyCast
                  : util::TimestampParseMode::kPrestoCast);
  if (conversionResult.hasError()) {
    return folly::makeUnexpected(conversionResult.error());
  }

  return util::fromParsedTimestampWithTimeZone(
      conversionResult.value(), options_.timeZone);
}

Expected<Timestamp> PrestoCastHooks::castIntToTimestamp(
    int64_t /*seconds*/) const {
  return folly::makeUnexpected(
      Status::UserError("Conversion to Timestamp is not supported"));
}

Expected<int64_t> PrestoCastHooks::castTimestampToBigint(
    Timestamp /*timestamp*/) const {
  return folly::makeUnexpected(
      Status::UserError("Conversion from Timestamp to Int is not supported"));
}

Expected<std::optional<Timestamp>> PrestoCastHooks::castDoubleToTimestamp(
    double /*seconds*/) const {
  return folly::makeUnexpected(
      Status::UserError("Conversion to Timestamp is not supported"));
}

Expected<int32_t> PrestoCastHooks::castStringToDate(
    const StringView& dateString) const {
  // Cast from string to date allows only complete ISO 8601 formatted strings:
  // [+-](YYYY-MM-DD).
  return util::fromDateString(dateString, util::ParseMode::kPrestoCast);
}

Expected<int64_t> PrestoCastHooks::castStringToTime(
    StringView timeString,
    const tz::TimeZone* timeZone,
    int64_t sessionStartTimeMs) const {
  try {
    return TIME()->valueToTime(timeString, timeZone, sessionStartTimeMs);
  } catch (const VeloxException& e) {
    return folly::makeUnexpected(Status::UserError(e.message()));
  } catch (const std::exception& e) {
    return folly::makeUnexpected(Status::UserError(e.what()));
  }
}

Expected<Timestamp> PrestoCastHooks::castBooleanToTimestamp(
    bool /*seconds*/) const {
  return folly::makeUnexpected(
      Status::UserError("Conversion to Timestamp is not supported"));
}

namespace {

// Returns true if 'text' equals 'expected' ignoring ASCII letter case.
bool equalsIgnoreAsciiCase(std::string_view text, std::string_view expected) {
  return text.size() == expected.size() &&
      std::equal(
             text.begin(),
             text.end(),
             expected.begin(),
             folly::AsciiCaseInsensitive{});
}

// Returns true if 'literal', the text fast_float consumed for an infinity or a
// NaN with the sign removed, is an accepted spelling. The nan(n-char-sequence)
// form that fast_float also parses is rejected in both modes.
bool isAcceptedInfNanLiteral(
    std::string_view literal,
    bool isNan,
    bool lenientInfNan) {
  if (isNan) {
    return lenientInfNan ? equalsIgnoreAsciiCase(literal, "nan")
                         : literal == "NaN";
  }
  if (lenientInfNan) {
    return equalsIgnoreAsciiCase(literal, "inf") ||
        equalsIgnoreAsciiCase(literal, "infinity");
  }
  return literal == "Infinity";
}

template <typename T>
Expected<T> doCastToFloatingPoint(const StringView& data, bool lenientInfNan) {
  const char* begin = std::find_if_not(data.begin(), data.end(), [](char c) {
    return functions::stringImpl::isAsciiWhiteSpace(c);
  });
  const char* end = data.end();
  if (begin == end) {
    return folly::makeUnexpected(Status::UserError());
  }
  T result;
  auto [ptr, ec] = fast_float::from_chars(
      begin,
      end,
      result,
      fast_float::chars_format::general |
          fast_float::chars_format::allow_leading_plus);
  // invalid_argument means the string is not a number at all.
  // result_out_of_range means overflow — fast_float sets result to ±infinity,
  // which is the correct Presto behavior for e.g. "1.7E308" cast to REAL.
  if (ec == std::errc::invalid_argument) {
    return folly::makeUnexpected(Status::UserError());
  }
  // Presto allows trailing whitespace after the number but nothing else.
  if (std::find_if_not(ptr, end, [](char c) {
        return functions::stringImpl::isAsciiWhiteSpace(c);
      }) != end) {
    return folly::makeUnexpected(Status::UserError());
  }
  // fast_float parses NaN/Infinity case-insensitively, but Presto accepts only
  // exact case: "NaN" and "Infinity" (with an optional leading +/-), unless
  // 'lenientInfNan' asks for the C-style spellings as well.
  // Overflow paths (ec == result_out_of_range) don't need this check.
  if (ec == std::errc{} && (std::isinf(result) || std::isnan(result))) {
    const char* literalStart = begin;
    if (literalStart < ptr && (*literalStart == '+' || *literalStart == '-')) {
      ++literalStart;
    }
    if (!isAcceptedInfNanLiteral(
            std::string_view(
                literalStart, static_cast<size_t>(ptr - literalStart)),
            std::isnan(result),
            lenientInfNan)) {
      return folly::makeUnexpected(Status::UserError());
    }
  }
  return result;
}

} // namespace

Expected<float> PrestoCastHooks::castStringToReal(
    const StringView& data) const {
  return doCastToFloatingPoint<float>(data, lenientInfNan_);
}

Expected<double> PrestoCastHooks::castStringToDouble(
    const StringView& data) const {
  return doCastToFloatingPoint<double>(data, lenientInfNan_);
}

StringView PrestoCastHooks::removeWhiteSpaces(const StringView& view) const {
  return view;
}

const TimestampToStringOptions& PrestoCastHooks::timestampToStringOptions()
    const {
  return options_;
}

PolicyType PrestoCastHooks::getPolicy() const {
  return legacyCast_ ? PolicyType::LegacyCastPolicy
                     : PolicyType::PrestoCastPolicy;
}
} // namespace facebook::velox::exec
