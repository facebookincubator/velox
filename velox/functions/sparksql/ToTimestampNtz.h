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

#include "velox/functions/sparksql/AnsiMode.h"
#include "velox/functions/sparksql/DateTimeFunctions.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/type/TimestampConversion.h"

namespace facebook::velox::functions::sparksql {

/// to_timestamp_ntz(timestamp_str). Ignores timezone suffixes, not
/// session-adjusted.
template <typename T>
struct ToTimestampNtzFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/) {
    ansiEnabled_ = SparkQueryConfig{config}.ansiEnabled();
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<TimestampUtc>& result,
      const arg_type<Varchar>& input) {
    auto parsed = util::fromTimestampWithTimezoneString(
        input, util::TimestampParseMode::kSparkCast);
    if (parsed.hasError()) {
      nullOrUserFail(ansiEnabled_, "{}", parsed.error().message());
      return false;
    }
    result = parsed.value().timestamp;
    return true;
  }

 private:
  bool ansiEnabled_{false};
};

/// Backs get_timestamp (Timestamp) and to_timestamp_ntz's 2-arg form
/// (TimestampUtc).
template <typename T, typename TTimestamp>
struct GetTimestampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* format) {
    legacyFormatter_ = SparkQueryConfig{config}.legacyDateFormatter();
    if constexpr (std::is_same_v<TTimestamp, Timestamp>) {
      auto sessionTimezoneName = config.sessionTimezone();
      if (!sessionTimezoneName.empty()) {
        sessionTimeZone_ = tz::locateZone(sessionTimezoneName);
      }
    } else {
      ansiEnabled_ = SparkQueryConfig{config}.ansiEnabled();
    }
    if (format != nullptr) {
      auto formatter = detail::initializeFormatter(
          std::string_view(*format), legacyFormatter_);
      if (formatter) {
        formatter_ = formatter;
      } else {
        invalidFormat_ = true;
      }
      isConstantTimeFormat_ = true;
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<TTimestamp>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& format) {
    if (invalidFormat_) {
      return false;
    }
    if (!isConstantTimeFormat_) {
      auto formatter = detail::initializeFormatter(
          std::string_view(format), legacyFormatter_);
      if (formatter) {
        formatter_ = formatter;
      } else {
        return false;
      }
    }
    auto dateTimeResult = formatter_->parse(std::string_view(input));
    if (dateTimeResult.hasError()) {
      if constexpr (std::is_same_v<TTimestamp, TimestampUtc>) {
        nullOrUserFail(ansiEnabled_, "{}", dateTimeResult.error().message());
      }
      return false;
    }
    if constexpr (std::is_same_v<TTimestamp, Timestamp>) {
      toGMTWithGapCorrection(
          (*dateTimeResult).timestamp, *getTimeZone(*dateTimeResult));
    }
    result = (*dateTimeResult).timestamp;
    return true;
  }

 private:
  const tz::TimeZone* getTimeZone(const DateTimeResult& result) const {
    return result.timezone ? result.timezone : sessionTimeZone_;
  }

  std::shared_ptr<DateTimeFormatter> formatter_{nullptr};
  bool isConstantTimeFormat_{false};
  const tz::TimeZone* sessionTimeZone_{tz::locateZone(0)}; // default to GMT.
  bool legacyFormatter_{false};
  bool invalidFormat_{false};
  bool ansiEnabled_{false};
};

} // namespace facebook::velox::functions::sparksql
