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

#include <fast_float/fast_float.h>
#include <re2/re2.h>
#include <string_view>
#include "velox/common/base/XxHashInline.h"
#include "velox/functions/lib/DateTimeFormatter.h"
#include "velox/functions/lib/TimeUtils.h"
#include "velox/functions/prestosql/DateTimeImpl.h"
#include "velox/functions/prestosql/detail/DateTimeCalendarFunctions.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneRenderZone.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Time.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/Type.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions {

template <typename T>
struct YearFromIntervalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<IntervalYearMonth>& months) {
    result = months / 12;
  }
};

template <typename T>
struct MonthFromIntervalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<IntervalYearMonth>& months) {
    result = months % 12;
  }
};

template <typename T>
struct DayFromIntervalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<IntervalDayTime>& interval) {
    result = interval / kMillisInDay;
  }
};

namespace {

bool isIntervalWholeDays(int64_t milliseconds) {
  return (milliseconds % kMillisInDay) == 0;
}

int64_t intervalDays(int64_t milliseconds) {
  return milliseconds / kMillisInDay;
}

} // namespace

template <typename T>
struct DateMinusInterval {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Date>& date,
      const arg_type<IntervalDayTime>& interval) {
    VELOX_USER_CHECK(
        isIntervalWholeDays(interval),
        "Cannot subtract hours, minutes, seconds or milliseconds from a date");
    result = addToDate(date, DateTimeUnit::kDay, -intervalDays(interval));
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Date>& date,
      const arg_type<IntervalYearMonth>& interval) {
    result = addToDate(date, DateTimeUnit::kMonth, -interval);
  }
};

template <typename T>
struct DatePlusInterval {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Date>& date,
      const arg_type<IntervalDayTime>& interval) {
    VELOX_USER_CHECK(
        isIntervalWholeDays(interval),
        "Cannot add hours, minutes, seconds or milliseconds to a date");
    result = addToDate(date, DateTimeUnit::kDay, intervalDays(interval));
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Date>& date,
      const arg_type<IntervalYearMonth>& interval) {
    result = addToDate(date, DateTimeUnit::kMonth, interval);
  }
};

template <typename T>
struct TimePlusInterval {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Time>& result,
      const arg_type<Time>& time,
      const arg_type<IntervalDayTime>& interval) {
    result = addToTime(time, interval);
  }
};

template <typename T>
struct TimeMinusInterval {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Time>& result,
      const arg_type<Time>& time,
      const arg_type<IntervalDayTime>& interval) {
    result = addToTime(time, -interval);
  }
};

template <typename T>
struct TimeMinusFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<IntervalDayTime>& result,
      const arg_type<Time>& a,
      const arg_type<Time>& b) {
    // Validate inputs are in valid TIME range [0, 86400000)
    VELOX_USER_CHECK(
        a >= 0 && a < kMillisInDay,
        "TIME value {} is out of range [0, 86400000)",
        a);
    VELOX_USER_CHECK(
        b >= 0 && b < kMillisInDay,
        "TIME value {} is out of range [0, 86400000)",
        b);

    // Simple subtraction returns interval in milliseconds
    result = a - b;
  }
};

template <typename T>
struct IntervalPlusTime {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  FOLLY_ALWAYS_INLINE void call(
      out_type<Time>& result,
      const arg_type<IntervalDayTime>& interval,
      const arg_type<Time>& time) {
    result = addToTime(time, interval);
  }
};

template <typename T>
struct HourFromIntervalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<IntervalDayTime>& millis) {
    result = (millis % kMillisInDay) / kMillisInHour;
  }
};

template <typename T>
struct MinuteFromIntervalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<IntervalDayTime>& millis) {
    result = (millis % kMillisInHour) / kMillisInMinute;
  }
};

template <typename T>
struct SecondFromIntervalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<IntervalDayTime>& millis) {
    result = (millis % kMillisInMinute) / kMillisInSecond;
  }
};

template <typename T>
struct MillisecondFromIntervalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<IntervalDayTime>& millis) {
    result = millis % Timestamp::kMillisecondsInSecond;
  }
};

template <typename T>
struct DateFormatFunction : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Timestamp>* /*timestamp*/,
      const arg_type<Varchar>* formatString) {
    sessionTimeZone_ = getTimeZoneFromConfig(config);
    if (formatString != nullptr) {
      setFormatter(*formatString);
      isConstFormat_ = true;
    }
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<TimestampWithTimezone>* /*timestamp*/,
      const arg_type<Varchar>* formatString) {
    this->initializeTimeZoneSupport(config);
    if (formatString != nullptr) {
      setFormatter(*formatString);
      isConstFormat_ = true;
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Timestamp>& timestamp,
      const arg_type<Varchar>& formatString) {
    if (!isConstFormat_) {
      setFormatter(formatString);
    }

    result.reserve(maxResultSize_);
    const auto resultSize = mysqlDateTime_->format(
        timestamp, sessionTimeZone_, maxResultSize_, result.data());
    result.resize(resultSize);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone,
      const arg_type<Varchar>& formatString) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    call(result, timestamp, formatString);
  }

 private:
  FOLLY_ALWAYS_INLINE void setFormatter(const arg_type<Varchar> formatString) {
    mysqlDateTime_ =
        buildMysqlDateTimeFormatter(
            std::string_view(formatString.data(), formatString.size()))
            .thenOrThrow(folly::identity, [&](const Status& status) {
              VELOX_USER_FAIL("{}", status.message());
            });
    maxResultSize_ = mysqlDateTime_->maxResultSize(sessionTimeZone_);
  }

  const tz::TimeZone* sessionTimeZone_ = nullptr;
  std::shared_ptr<DateTimeFormatter> mysqlDateTime_;
  uint32_t maxResultSize_;
  bool isConstFormat_ = false;
};

template <typename T>
struct FromIso8601Date {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Date>& result, const arg_type<Varchar>& input) {
    const auto castResult = util::fromDateString(
        input.data(), input.size(), util::ParseMode::kIso8601);
    if (castResult.hasError()) {
      return castResult.error();
    }

    result = castResult.value();
    return Status::OK();
  }
};

template <typename T>
struct FromIso8601Timestamp {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/) {
    auto sessionTzName = config.sessionTimezone();
    if (!sessionTzName.empty()) {
      sessionTimeZone_ = tz::locateZone(sessionTzName);
    }
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<Varchar>& input) {
    const auto castResult = util::fromTimestampWithTimezoneString(
        input.data(), input.size(), util::TimestampParseMode::kIso8601);
    if (castResult.hasError()) {
      return castResult.error();
    }

    auto [ts, timeZone, offsetMillis] = castResult.value();
    VELOX_DCHECK(!offsetMillis.has_value());
    // Input string may not contain a timezone - if so, it is interpreted in
    // session timezone.
    if (!timeZone) {
      timeZone = sessionTimeZone_;
    }
    ts.toGMT(*timeZone);
    result = pack(ts, timeZone->id());
    return Status::OK();
  }

 private:
  const tz::TimeZone* sessionTimeZone_{tz::locateZone(0)}; // default to GMT.
};

template <typename T>
struct DateParseFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::shared_ptr<DateTimeFormatter> format_;

  // By default, assume 0 (GMT).
  const tz::TimeZone* sessionTimeZone_{tz::locateZone(0)};
  bool isConstFormat_ = false;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* formatString) {
    if (formatString != nullptr) {
      format_ =
          buildMysqlDateTimeFormatter(
              std::string_view(formatString->data(), formatString->size()))
              .thenOrThrow(folly::identity, [&](const Status& status) {
                VELOX_USER_FAIL("{}", status.message());
              });
      isConstFormat_ = true;
    }

    auto sessionTzName = config.sessionTimezone();
    if (!sessionTzName.empty()) {
      sessionTimeZone_ = tz::locateZone(sessionTzName);
    }
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Timestamp>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& format) {
    if (!isConstFormat_) {
      format_ = buildMysqlDateTimeFormatter(
                    std::string_view(format.data(), format.size()))
                    .thenOrThrow(folly::identity, [&](const Status& status) {
                      VELOX_USER_FAIL("{}", status.message());
                    });
    }

    auto dateTimeResult = format_->parse((std::string_view)(input));
    if (dateTimeResult.hasError()) {
      return dateTimeResult.error();
    }

    dateTimeResult->timestamp.toGMT(*sessionTimeZone_);
    result = dateTimeResult->timestamp;
    return Status::OK();
  }
};

template <typename T>
struct FormatDateTimeFunction : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Timestamp>* /*timestamp*/,
      const arg_type<Varchar>* formatString) {
    sessionTimeZone_ = getTimeZoneFromConfig(config);
    if (formatString != nullptr) {
      setFormatter(*formatString);
      isConstFormat_ = true;
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Timestamp>& timestamp,
      const arg_type<Varchar>& formatString) {
    ensureFormatter(formatString);

    format(timestamp, sessionTimeZone_, maxResultSize_, result);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<TimestampWithTimezone>* /*timestamp*/,
      const arg_type<Varchar>* formatString) {
    this->initializeTimeZoneSupport(config);
    if (formatString != nullptr) {
      setFormatter(*formatString);
      isConstFormat_ = true;
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone,
      const arg_type<Varchar>& formatString) {
    ensureFormatter(formatString);

    const auto timestamp = unpackTimestampUtc(*timestampWithTimezone);
    const auto* timezonePtr = this->renderZone(timestampWithTimezone);

    const auto maxResultSize = jodaDateTime_->maxResultSize(timezonePtr);
    format(timestamp, timezonePtr, maxResultSize, result);
  }

 private:
  FOLLY_ALWAYS_INLINE void ensureFormatter(
      const arg_type<Varchar>& formatString) {
    if (!isConstFormat_) {
      setFormatter(formatString);
    }
  }

  FOLLY_ALWAYS_INLINE void setFormatter(const arg_type<Varchar>& formatString) {
    buildJodaDateTimeFormatter(
        std::string_view(formatString.data(), formatString.size()))
        .thenOrThrow([this](auto formatter) {
          jodaDateTime_ = formatter;
          maxResultSize_ = jodaDateTime_->maxResultSize(sessionTimeZone_);
        });
  }

  void format(
      const Timestamp& timestamp,
      const tz::TimeZone* timeZone,
      uint32_t maxResultSize,
      out_type<Varchar>& result) const {
    result.reserve(maxResultSize);
    const auto resultSize = jodaDateTime_->format(
        timestamp, timeZone, maxResultSize, result.data());
    result.resize(resultSize);
  }

  const tz::TimeZone* sessionTimeZone_ = nullptr;
  std::shared_ptr<DateTimeFormatter> jodaDateTime_;
  uint32_t maxResultSize_;
  bool isConstFormat_ = false;
};

template <typename T>
struct ParseDateTimeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::shared_ptr<DateTimeFormatter> format_;
  const tz::TimeZone* sessionTimeZone_{tz::locateZone(0)}; // GMT
  bool isConstFormat_ = false;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* format) {
    if (format != nullptr) {
      format_ = buildJodaDateTimeFormatter(
                    std::string_view(format->data(), format->size()))
                    .thenOrThrow(folly::identity, [&](const Status& status) {
                      VELOX_USER_FAIL("{}", status.message());
                    });
      isConstFormat_ = true;
    }

    auto sessionTzName = config.sessionTimezone();
    if (!sessionTzName.empty()) {
      sessionTimeZone_ = tz::locateZone(sessionTzName);
    }
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& format) {
    if (!isConstFormat_) {
      format_ = buildJodaDateTimeFormatter(
                    std::string_view(format.data(), format.size()))
                    .thenOrThrow(folly::identity, [&](const Status& status) {
                      VELOX_USER_FAIL("{}", status.message());
                    });
    }
    auto dateTimeResult =
        format_->parse(std::string_view(input.data(), input.size()));
    if (dateTimeResult.hasError()) {
      return dateTimeResult.error();
    }

    // If timezone was not parsed, fallback to the session timezone. If there's
    // no session timezone, fallback to 0 (GMT).
    const auto* timeZone =
        dateTimeResult->timezone ? dateTimeResult->timezone : sessionTimeZone_;
    dateTimeResult->timestamp.toGMT(*timeZone);
    result = pack(dateTimeResult->timestamp, timeZone->id());
    return Status::OK();
  }
};

template <typename T>
struct CurrentDateFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  const tz::TimeZone* timeZone_ = nullptr;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config) {
    timeZone_ = getTimeZoneFromConfig(config);
  }

  FOLLY_ALWAYS_INLINE void call(out_type<Date>& result) {
    auto now = Timestamp::now();
    if (timeZone_ != nullptr) {
      now.toTimezone(*timeZone_);
    }
    const std::chrono::
        time_point<std::chrono::system_clock, std::chrono::milliseconds>
            localTimepoint(std::chrono::milliseconds(now.toMillis()));
    result = std::chrono::floor<date::days>((localTimepoint).time_since_epoch())
                 .count();
  }
};

template <typename T>
struct CurrentTimezoneFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  std::string tzName_;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>&,
      const core::QueryConfig& config) {
    tzName_ = config.sessionTimezone();
  }
  FOLLY_ALWAYS_INLINE void call(out_type<Varchar>& result) {
    result = std::string_view(tzName_);
  }
};

template <typename T>
struct CurrentTimestampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  int64_t result_{0};
  const tz::TimeZone* timeZone_ = nullptr;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /* type */,
      const core::QueryConfig& config) {
    Timestamp ts = Timestamp::fromMillis(config.sessionStartTimeMs());
    timeZone_ = getTimeZoneFromConfig(config);
    if (timeZone_ == nullptr) {
      VELOX_USER_FAIL("Timezone cannot be null");
    }
    result_ = pack(ts, timeZone_->id());
  }

  FOLLY_ALWAYS_INLINE void call(out_type<TimestampWithTimezone>& result) {
    result = result_;
  }
};

template <typename T>
struct ToISO8601Function : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  using TimestampWithTimezoneSupport<T>::initialize;

  ToISO8601Function() {
    auto formatter =
        functions::buildJodaDateTimeFormatter("yyyy-MM-dd'T'HH:mm:ss.SSSZZ");
    VELOX_CHECK(
        !formatter.hasError(),
        "Default format should always be valid, error: {}",
        formatter.error().message());
    formatter_ = formatter.value();
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& config,
      const arg_type<Timestamp>* /*input*/) {
    if (inputTypes[0]->isTimestamp()) {
      VELOX_DCHECK(inputTypes[0]->equivalent(*TIMESTAMP()));
      timeZone_ = getTimeZoneFromConfig(config);
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Date>& date) {
    result = DateType::toIso8601(date);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Timestamp>& timestamp) {
    toIso8601(timestamp, timeZone_, result);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    const auto timestamp = unpackTimestampUtc(*timestampWithTimezone);
    const auto* timeZone = this->renderZone(timestampWithTimezone);

    toIso8601(timestamp, timeZone, result);
  }

 private:
  void toIso8601(
      const Timestamp& timestamp,
      const tz::TimeZone* timeZone,
      out_type<Varchar>& result) const {
    const auto maxResultSize = formatter_->maxResultSize(timeZone);
    result.reserve(maxResultSize);
    const auto resultSize = formatter_->format(
        timestamp, timeZone, maxResultSize, result.data(), false, "Z");
    result.resize(resultSize);
  }

  const tz::TimeZone* timeZone_{nullptr};
  std::shared_ptr<DateTimeFormatter> formatter_;
};

template <typename T>
struct AtTimezoneTimeWithTimezoneFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimeWithTimezone>& result,
      const arg_type<TimeWithTimezone>& timeWithTz,
      const arg_type<Varchar>& targetTimezone) {
    // Extract milliseconds UTC from the input
    auto millisUtc = util::unpackMillisUtc(*timeWithTz);

    // Parse the target timezone offset from the VARCHAR string
    auto offsetResult = util::parseTimezoneOffset(
        targetTimezone.data(),
        targetTimezone.size(),
        /*allowCompactFormat*/ false);
    if (offsetResult.hasError()) {
      if (offsetResult.error().isUserError()) {
        VELOX_USER_FAIL(offsetResult.error().message());
      } else {
        VELOX_FAIL(offsetResult.error().message());
      }
    }

    auto targetOffsetMinutes = offsetResult.value();

    // Encode the timezone offset using bias encoding
    auto encodedOffset = util::biasEncode(targetOffsetMinutes);

    // Pack and return the result with the same UTC time but new timezone
    result = util::pack(millisUtc, encodedOffset);
  }
};

template <typename TExec>
struct ToMillisecondFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void call(
      out_type<int64_t>& result,
      const arg_type<IntervalDayTime>& millis) {
    result = millis;
  }
};

/// xxhash64(Date) → bigint
/// Return a xxhash64 of input Date
template <typename T>
struct XxHash64DateFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE
  void call(out_type<int64_t>& result, const arg_type<Date>& input) {
    // Casted to int64_t to feed into XXH64
    auto date_input = static_cast<int64_t>(input);
    result = XXH64(&date_input, sizeof(date_input), 0);
  }
};

/// xxhash64(Timestamp) → bigint
/// Return a xxhash64 of input Timestamp
template <typename T>
struct XxHash64TimestampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE
  void call(out_type<int64_t>& result, const arg_type<Timestamp>& input) {
    // Use the milliseconds representation of the timestamp
    auto timestamp_millis = input.toMillis();
    result = XXH64(&timestamp_millis, sizeof(timestamp_millis), 0);
  }
};

/// xxhash64(Time) → bigint
/// Return a xxhash64 of input Time
template <typename T>
struct XxHash64TimeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE
  void call(out_type<int64_t>& result, const arg_type<Time>& input) {
    // TIME is represented as milliseconds since midnight
    // Convert to big-endian to match Presto's xxhash64 behavior
    auto bigEndianValue = folly::Endian::big(input);
    result = XXH64(&bigEndianValue, sizeof(bigEndianValue), 0);
  }
};

template <typename T>
struct ParseDurationFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<IntervalDayTime>& result,
      const arg_type<Varchar>& amountUnit) {
    static const RE2 kDurationRegex(R"(^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$)");
    // TODO: Remove re2::StringPiece != std::string_view hacks.
    // It's needed because for some systems in CI,
    // re2 and abseil libraries are old.
    re2::StringPiece valueStr;
    re2::StringPiece unitStr;
    re2::StringPiece amountUnitStr{amountUnit.data(), amountUnit.size()};
    if (!RE2::FullMatch(amountUnitStr, kDurationRegex, &valueStr, &unitStr)) {
      VELOX_USER_FAIL(
          "Input duration is not a valid data duration string: {}",
          std::string_view(amountUnitStr.data(), amountUnitStr.size()));
    }

    double value{};
    auto [_, error] = fast_float::from_chars(
        valueStr.data(), valueStr.data() + valueStr.size(), value);
    if (error == std::errc::result_out_of_range) {
      VELOX_USER_FAIL(
          "Input duration value is out of range for double: {}",
          std::string_view(valueStr.data(), valueStr.size()));
    } else if (error != std::errc{}) {
      VELOX_USER_FAIL(
          "Input duration value is not a valid number: {}",
          std::string_view(valueStr.data(), valueStr.size()));
    }

    result = valueOfTimeUnitToMillis(value, {unitStr.data(), unitStr.size()});
  }
};

template <typename T>
struct LocalTimeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config) {
    auto sessionStartTimeMs = config.sessionStartTimeMs();
    localTimeSinceMidnight_ = sessionStartTimeMs % kMillisInDay;
  }

  // LocalTime just returns the time from midnight in UTC.
  FOLLY_ALWAYS_INLINE void call(out_type<Time>& result) {
    result = localTimeSinceMidnight_;
  }

 private:
  int64_t localTimeSinceMidnight_;
};

template <typename T>
struct LocalTimestampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config) {
    ts_ = Timestamp::fromMillis(config.sessionStartTimeMs());
  }

  FOLLY_ALWAYS_INLINE void call(out_type<Timestamp>& result) {
    result = ts_;
  }

 private:
  Timestamp ts_;
};

template <typename T>
struct CurrentTimeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /* type */,
      const core::QueryConfig& config) {
    const tz::TimeZone* timeZone = getTimeZoneFromConfig(config);
    VELOX_USER_CHECK_NOT_NULL(timeZone, "Timezone cannot be null");

    auto sessionStartTimeMs = config.sessionStartTimeMs();

    auto localMillis =
        timeZone->to_local(std::chrono::milliseconds{sessionStartTimeMs});
    auto localMillisSinceMidnight =
        localMillis - std::chrono::floor<std::chrono::days>(localMillis);

    auto currentOffset = localMillis.count() - sessionStartTimeMs;
    // Safe since timezone offsets are bounded [-840, 840] minutes
    auto currentOffsetMinutes =
        static_cast<int16_t>(currentOffset / (60 * 1000));

    auto encodedOffset = util::biasEncode(currentOffsetMinutes);
    currentTime_ = util::pack(localMillisSinceMidnight.count(), encodedOffset);
  }

  FOLLY_ALWAYS_INLINE void call(out_type<TimeWithTimezone>& result) {
    result = currentTime_;
  }

 private:
  int64_t currentTime_{0};
};

} // namespace facebook::velox::functions
