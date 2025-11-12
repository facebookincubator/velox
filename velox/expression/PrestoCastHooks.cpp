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

#include "velox/expression/PrestoCastHooks.h"
#include "velox/expression/StringToFloatParser.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::exec {

PrestoCastHooks::PrestoCastHooks(const core::QueryConfig& config)
    : CastHooks(), legacyCast_(config.isLegacyCast()) {
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
    const StringView& view) const {
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

Expected<int64_t> PrestoCastHooks::castTimestampToInt(
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

Expected<Timestamp> PrestoCastHooks::castBooleanToTimestamp(
    bool /*seconds*/) const {
  return folly::makeUnexpected(
      Status::UserError("Conversion to Timestamp is not supported"));
}

namespace {
template <typename T>
Expected<T> doCastToFloatingPoint(const StringView& data) {
  T result;
  const auto status =
      StringToFloatParser::parse<T>(std::string_view(data), result);

  if (status.ok()) {
    return result;
  }

  if (threadSkipErrorDetails()) {
    return folly::makeUnexpected(Status::UserError());
  }
  return folly::makeUnexpected(Status::UserError(status.message()));
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

PolicyType PrestoCastHooks::getPolicy() const {
  return legacyCast_ ? PolicyType::LegacyCastPolicy
                     : PolicyType::PrestoCastPolicy;
}
} // namespace facebook::velox::exec
