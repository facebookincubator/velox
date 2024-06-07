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

#include <cmath>

#include <double-conversion/double-conversion.h>
#include <folly/Expected.h>

#include "velox/expression/PrestoCastHooks.h"
#include "velox/external/date/tz.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/type/TimestampConversion.h"

namespace facebook::velox::exec {

PrestoCastHooks::PrestoCastHooks(const core::QueryConfig& config)
    : CastHooks(), legacyCast_(config.isLegacyCast()) {
  if (!legacyCast_) {
    options_.zeroPaddingYear = true;
    options_.dateTimeSeparator = ' ';
    const auto sessionTzName = config.sessionTimezone();
    if (config.adjustTimestampToTimezone() && !sessionTzName.empty()) {
      options_.timeZone = date::locate_zone(sessionTzName);
    }
  }
}

Expected<Timestamp> PrestoCastHooks::castStringToTimestamp(
    const StringView& view) const {
  const auto conversionResult = util::fromTimestampWithTimezoneString(
      view.data(),
      view.size(),
      legacyCast_ ? util::TimestampParseMode::kLegacyCast
                  : util::TimestampParseMode::kPrestoCast);
  if (conversionResult.hasError()) {
    return folly::makeUnexpected(conversionResult.error());
  }

  auto result = conversionResult.value();

  // If the parsed string has timezone information, convert the timestamp at
  // GMT at that time. For example, "1970-01-01 00:00:00 -00:01" is 60 seconds
  // at GMT.
  if (result.second != -1) {
    result.first.toGMT(result.second);

  }
  // If no timezone information is available in the input string, check if we
  // should understand it as being at the session timezone, and if so, convert
  // to GMT.
  else if (options_.timeZone != nullptr) {
    result.first.toGMT(*options_.timeZone);
  }
  return result.first;
}

Expected<int32_t> PrestoCastHooks::castStringToDate(
    const StringView& dateString) const {
  // Cast from string to date allows only complete ISO 8601 formatted strings:
  // [+-](YYYY-MM-DD).
  return util::fromDateString(dateString, util::ParseMode::kPrestoCast);
}

namespace {

using double_conversion::StringToDoubleConverter;

template <typename T>
Expected<T> doCastToFloatingPoint(const StringView& data) {
  static const T kNan = std::numeric_limits<T>::quiet_NaN();
  static StringToDoubleConverter stringToDoubleConverter{
      StringToDoubleConverter::ALLOW_TRAILING_SPACES,
      /*empty_string_value*/ kNan,
      /*junk_string_value*/ kNan,
      "Infinity",
      "NaN"};
  int processedCharactersCount;
  T result;
  auto* begin = std::find_if_not(data.begin(), data.end(), [](char c) {
    return functions::stringImpl::isAsciiWhiteSpace(c);
  });
  auto length = data.end() - begin;
  if (length == 0) {
    // 'data' only contains white spaces.
    return folly::makeUnexpected(Status::UserError());
  }
  if constexpr (std::is_same_v<T, float>) {
    result = stringToDoubleConverter.StringToFloat(
        begin, length, &processedCharactersCount);
  } else if constexpr (std::is_same_v<T, double>) {
    result = stringToDoubleConverter.StringToDouble(
        begin, length, &processedCharactersCount);
  }
  // Since we already removed leading space, if processedCharactersCount == 0,
  // it means the remaining string is either empty or a junk string. So return a
  // user error in this case.
  if UNLIKELY (processedCharactersCount == 0) {
    return folly::makeUnexpected(Status::UserError());
  }
  return result;
}

} // namespace

Expected<float> PrestoCastHooks::castStringToReal(
    const StringView& data) const {
  return doCastToFloatingPoint<float>(data);
}

Expected<double> PrestoCastHooks::castStringToDouble(
    const StringView& data) const {
  return doCastToFloatingPoint<double>(data);
}

StringView PrestoCastHooks::removeWhiteSpaces(const StringView& view) const {
  return view;
}

const TimestampToStringOptions& PrestoCastHooks::timestampToStringOptions()
    const {
  return options_;
}
} // namespace facebook::velox::exec
