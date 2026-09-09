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
#pragma once

#include "velox/core/QueryConfig.h"
#include "velox/functions/Macros.h"
#include "velox/functions/sparksql/TimestampUtils.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {

/// Converts a timestamp from one time zone to another: interprets the input
/// as local time in the source zone, resolves it to a UTC instant, then
/// re-expresses that instant as local time in the target zone. The
/// 2-argument overload uses the session timezone as the source.
template <typename T>
struct ConvertTimezoneFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Caches the target timezone when constant.
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* targetTz,
      const arg_type<TimestampUtc>* /*sourceTs*/) {
    auto sessionTzName = config.sessionTimezone();
    if (!sessionTzName.empty()) {
      sessionTimeZone_ = tz::locateZone(sessionTzName);
    }
    if (targetTz) {
      targetTimeZone_ = tz::locateZone(std::string_view(*targetTz), false);
    }
  }

  // Caches the source and target timezones when constant.
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* sourceTz,
      const arg_type<Varchar>* targetTz,
      const arg_type<TimestampUtc>* /*sourceTs*/) {
    if (sourceTz) {
      sourceTimeZone_ = tz::locateZone(std::string_view(*sourceTz), false);
    }
    if (targetTz) {
      targetTimeZone_ = tz::locateZone(std::string_view(*targetTz), false);
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampUtc>& result,
      const arg_type<Varchar>& targetTz,
      const arg_type<TimestampUtc>& sourceTs) {
    const auto* target = targetTimeZone_ != nullptr
        ? targetTimeZone_
        : tz::locateZone(std::string_view(targetTz), false);
    VELOX_USER_CHECK_NOT_NULL(target, "Unknown time zone: '{}'", targetTz);
    convert(result, sessionTimeZone_, target, sourceTs);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampUtc>& result,
      const arg_type<Varchar>& sourceTz,
      const arg_type<Varchar>& targetTz,
      const arg_type<TimestampUtc>& sourceTs) {
    const auto* source = sourceTimeZone_ != nullptr
        ? sourceTimeZone_
        : tz::locateZone(std::string_view(sourceTz), false);
    VELOX_USER_CHECK_NOT_NULL(source, "Unknown time zone: '{}'", sourceTz);
    const auto* target = targetTimeZone_ != nullptr
        ? targetTimeZone_
        : tz::locateZone(std::string_view(targetTz), false);
    VELOX_USER_CHECK_NOT_NULL(target, "Unknown time zone: '{}'", targetTz);
    convert(result, source, target, sourceTs);
  }

 private:
  static void convert(
      out_type<TimestampUtc>& result,
      const tz::TimeZone* sourceTimeZone,
      const tz::TimeZone* targetTimeZone,
      const arg_type<TimestampUtc>& sourceTs) {
    result = sourceTs;
    toGMTWithGapCorrection(result, *sourceTimeZone);
    result.toTimezone(*targetTimeZone);
  }

  // Defaults to GMT when no session timezone is configured. Velox has no
  // OS-local-timezone detection to match Spark's own default (the JVM's
  // local zone), so GMT is used as a deterministic fallback.
  const tz::TimeZone* sessionTimeZone_{tz::locateZone(0)};
  const tz::TimeZone* sourceTimeZone_{nullptr};
  const tz::TimeZone* targetTimeZone_{nullptr};
};

} // namespace facebook::velox::functions::sparksql
