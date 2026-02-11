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

Expected<std::optional<Timestamp>> PrestoCastHooks::castDoubleToTimestamp(
    double /*seconds*/) const {
  return folly::makeUnexpected(
      Status::UserError("Conversion to Timestamp is not supported"));
}

Expected<Timestamp> PrestoCastHooks::castBooleanToTimestamp(
    bool /*seconds*/) const {
  return folly::makeUnexpected(
      Status::UserError("Conversion to Timestamp is not supported"));
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
