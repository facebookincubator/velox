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

#include <boost/algorithm/string.hpp>

#include "velox/functions/lib/DateTimeFormatter.h"
#include "velox/functions/lib/TimeUtils.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {

namespace detail {
Expected<std::shared_ptr<DateTimeFormatter>> getDateTimeFormatter(
    const std::string_view& format,
    DateTimeFormatterType type) {
  switch (type) {
    case DateTimeFormatterType::STRICT_SIMPLE:
      return buildSimpleDateTimeFormatter(format, /*lenient=*/false);
    case DateTimeFormatterType::LENIENT_SIMPLE:
      return buildSimpleDateTimeFormatter(format, /*lenient=*/true);
    default:
      return buildJodaDateTimeFormatter(
          std::string_view(format.data(), format.size()));
  }
}
} // namespace detail

namespace {
enum class DateTimeFormat {
  kMicrosecond,
  kMillisecond,
  kSecond,
  kMinute,
  kHour,
  kDay,
  kWeek,
  kMonth,
  kQuarter,
  kYear
};

inline std::optional<DateTimeFormat> fromDateTimeFormatString(
    const StringView& formatString,
    bool throwIfInvalid) {
  static const StringView kMicrosecond("microsecond");
  static const StringView kMillisecond("millisecond");
  static const StringView kSecond("second");
  static const StringView kMinute("minute");
  static const StringView kHour("hour");
  static const StringView kDay("day");
  static const StringView kDd("dd");
  static const StringView kWeek("week");
  static const StringView kMonth("month");
  static const StringView kMon("mon");
  static const StringView kMm("mm");
  static const StringView kQuarter("quarter");
  static const StringView kYear("year");
  static const StringView kYyyy("yyyy");
  static const StringView kYy("yy");

  const auto format = boost::algorithm::to_lower_copy(formatString.str());

  if (format == kMicrosecond) {
    return DateTimeFormat::kMicrosecond;
  }
  if (format == kMillisecond) {
    return DateTimeFormat::kMillisecond;
  }
  if (format == kSecond) {
    return DateTimeFormat::kSecond;
  }
  if (format == kMinute) {
    return DateTimeFormat::kMinute;
  }
  if (format == kHour) {
    return DateTimeFormat::kHour;
  }
  if (format == kDay || format == kDd) {
    return DateTimeFormat::kDay;
  }
  if (format == kWeek) {
    return DateTimeFormat::kWeek;
  }
  if (format == kMonth || format == kMm || format == kMon) {
    return DateTimeFormat::kMonth;
  }
  if (format == kQuarter) {
    return DateTimeFormat::kQuarter;
  }
  if (format == kYear || format == kYyyy || format == kYy) {
    return DateTimeFormat::kYear;
  }
  if (throwIfInvalid) {
    VELOX_UNSUPPORTED("Unsupported datetime format: {}", formatString);
  }
  return std::nullopt;
}
} // namespace

template <typename T>
struct YearFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE int32_t getYear(const std::tm& time) {
    return 1900 + time.tm_year;
  }

  FOLLY_ALWAYS_INLINE void call(
      int32_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getYear(getDateTime(timestamp, this->timeZone_));
  }

  FOLLY_ALWAYS_INLINE void call(int32_t& result, const arg_type<Date>& date) {
    result = getYear(getDateTime(date));
  }
};

template <typename T>
struct WeekFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(int32_t& result, const arg_type<Date>& date) {
    result = getWeek(Timestamp::fromDate(date), nullptr, false);
  }
};

template <typename T>
struct YearOfWeekFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(int32_t& result, const arg_type<Date>& date) {
    const auto dateTime = getDateTime(date);

    int isoWeekDay = dateTime.tm_wday == 0 ? 7 : dateTime.tm_wday;
    // The last few days in December may belong to the next year if they are
    // in the same week as the next January 1 and this January 1 is a Thursday
    // or before.
    if (UNLIKELY(
            dateTime.tm_mon == 11 && dateTime.tm_mday >= 29 &&
            dateTime.tm_mday - isoWeekDay >= 31 - 3)) {
      result = 1900 + dateTime.tm_year + 1;
      return;
    }
    // The first few days in January may belong to the last year if they are
    // in the same week as January 1 and January 1 is a Friday or after.
    if (UNLIKELY(
            dateTime.tm_mon == 0 && dateTime.tm_mday <= 3 &&
            isoWeekDay - (dateTime.tm_mday - 1) >= 5)) {
      result = 1900 + dateTime.tm_year - 1;
      return;
    }
    result = 1900 + dateTime.tm_year;
  }
};

template <typename T>
struct UnixDateFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(int32_t& result, const arg_type<Date>& date) {
    result = date;
  }
};

template <typename T>
struct UnixTimestampFunction {
  // unix_timestamp();
  // If no parameters, return the current unix timestamp without adjusting
  // timezones.
  FOLLY_ALWAYS_INLINE void call(int64_t& result) {
    result = Timestamp::now().getSeconds();
  }
};

template <typename T>
struct UnixTimestampParseFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // unix_timestamp(input);
  // If format is not specified, assume kDefaultFormat.
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/) {
    auto formatter = detail::getDateTimeFormatter(
        kDefaultFormat_,
        config.sparkLegacyDateFormatter() ? DateTimeFormatterType::STRICT_SIMPLE
                                          : DateTimeFormatterType::JODA);
    VELOX_CHECK(!formatter.hasError(), "Default format should always be valid");
    format_ = formatter.value();
    setTimezone(config);
  }

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Varchar>& input) {
    auto dateTimeResult = format_->parse(std::string_view(input));
    // Return null if could not parse.
    if (dateTimeResult.hasError()) {
      return false;
    }
    (*dateTimeResult).timestamp.toGMT(*getTimeZone(*dateTimeResult));
    result = (*dateTimeResult).timestamp.getSeconds();
    return true;
  }

 protected:
  void setTimezone(const core::QueryConfig& config) {
    auto sessionTzName = config.sessionTimezone();
    if (!sessionTzName.empty()) {
      sessionTimeZone_ = tz::locateZone(sessionTzName);
    }
  }

  const tz::TimeZone* getTimeZone(const DateTimeResult& result) {
    // If timezone was not parsed, fallback to the session timezone.
    return result.timezone ? result.timezone : sessionTimeZone_;
  }

  // Default if format is not specified, as per Spark documentation.
  constexpr static std::string_view kDefaultFormat_{"yyyy-MM-dd HH:mm:ss"};
  std::shared_ptr<DateTimeFormatter> format_;
  const tz::TimeZone* sessionTimeZone_{tz::locateZone(0)}; // fallback to GMT.
};

template <typename T>
struct UnixTimestampParseWithFormatFunction
    : public UnixTimestampParseFunction<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // unix_timestamp(input, format):
  // If format is constant, compile it just once per batch.
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* format) {
    legacyFormatter_ = config.sparkLegacyDateFormatter();
    if (format != nullptr) {
      auto formatter = detail::getDateTimeFormatter(
          std::string_view(format->data(), format->size()),
          legacyFormatter_ ? DateTimeFormatterType::STRICT_SIMPLE
                           : DateTimeFormatterType::JODA);
      if (formatter.hasError()) {
        invalidFormat_ = true;
      } else {
        this->format_ = formatter.value();
      }
      isConstFormat_ = true;
    }
    this->setTimezone(config);
  }

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& format) {
    if (invalidFormat_) {
      return false;
    }

    // Format error returns null.
    if (!isConstFormat_) {
      auto formatter = detail::getDateTimeFormatter(
          std::string_view(format.data(), format.size()),
          legacyFormatter_ ? DateTimeFormatterType::STRICT_SIMPLE
                           : DateTimeFormatterType::JODA);
      if (formatter.hasError()) {
        return false;
      }
      this->format_ = formatter.value();
    }
    auto dateTimeResult =
        this->format_->parse(std::string_view(input.data(), input.size()));
    // parsing error returns null
    if (dateTimeResult.hasError()) {
      return false;
    }
    (*dateTimeResult).timestamp.toGMT(*this->getTimeZone(*dateTimeResult));
    result = (*dateTimeResult).timestamp.getSeconds();
    return true;
  }

 private:
  bool isConstFormat_{false};
  bool invalidFormat_{false};
  bool legacyFormatter_{false};
};

// Parses unix time in seconds to a formatted string.
template <typename T>
struct FromUnixtimeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<int64_t>* /*unixtime*/,
      const arg_type<Varchar>* format) {
    legacyFormatter_ = config.sparkLegacyDateFormatter();
    sessionTimeZone_ = getTimeZoneFromConfig(config);
    if (format != nullptr) {
      setFormatter(*format);
      isConstantTimeFormat_ = true;
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<int64_t>& second,
      const arg_type<Varchar>& format) {
    if (!isConstantTimeFormat_) {
      setFormatter(format);
    }
    const Timestamp timestamp{second, 0};
    result.reserve(maxResultSize_);
    int32_t resultSize;
    resultSize = formatter_->format(
        timestamp, sessionTimeZone_, maxResultSize_, result.data(), true);
    result.resize(resultSize);
  }

 private:
  FOLLY_ALWAYS_INLINE void setFormatter(const arg_type<Varchar>& format) {
    formatter_ = detail::getDateTimeFormatter(
                     std::string_view(format.data(), format.size()),
                     legacyFormatter_ ? DateTimeFormatterType::STRICT_SIMPLE
                                      : DateTimeFormatterType::JODA)
                     .thenOrThrow(folly::identity, [&](const Status& status) {
                       VELOX_USER_FAIL("{}", status.message());
                     });
    maxResultSize_ = formatter_->maxResultSize(sessionTimeZone_);
  }

  const tz::TimeZone* sessionTimeZone_{nullptr};
  std::shared_ptr<DateTimeFormatter> formatter_;
  uint32_t maxResultSize_;
  bool isConstantTimeFormat_{false};
  bool legacyFormatter_{false};
};

template <typename T>
struct ToUtcTimestampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* timezone) {
    if (timezone) {
      timeZone_ = tz::locateZone(std::string_view(*timezone), false);
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Timestamp>& result,
      const arg_type<Timestamp>& timestamp,
      const arg_type<Varchar>& timezone) {
    result = timestamp;
    const auto* fromTimezone = timeZone_ != nullptr
        ? timeZone_
        : tz::locateZone(std::string_view(timezone), false);
    VELOX_USER_CHECK_NOT_NULL(
        fromTimezone, "Unknown time zone: '{}'", timezone);
    result.toGMT(*fromTimezone);
  }

 private:
  const tz::TimeZone* timeZone_{nullptr};
};

template <typename T>
struct FromUtcTimestampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* timezone) {
    if (timezone) {
      timeZone_ = tz::locateZone(std::string_view(*timezone), false);
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Timestamp>& result,
      const arg_type<Timestamp>& timestamp,
      const arg_type<Varchar>& timezone) {
    result = timestamp;
    const auto* toTimeZone = timeZone_ != nullptr
        ? timeZone_
        : tz::locateZone(std::string_view(timezone), false);
    VELOX_USER_CHECK_NOT_NULL(toTimeZone, "Unknown time zone: '{}'", timezone);
    result.toTimezone(*toTimeZone);
  }

 private:
  const tz::TimeZone* timeZone_{nullptr};
};

/// Converts date string to Timestmap type.
template <typename T>
struct GetTimestampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* format) {
    legacyFormatter_ = config.sparkLegacyDateFormatter();
    auto sessionTimezoneName = config.sessionTimezone();
    if (!sessionTimezoneName.empty()) {
      sessionTimeZone_ = tz::locateZone(sessionTimezoneName);
    }
    if (format != nullptr) {
      formatter_ = detail::getDateTimeFormatter(
                       std::string_view(*format),
                       legacyFormatter_ ? DateTimeFormatterType::STRICT_SIMPLE
                                        : DateTimeFormatterType::JODA)
                       .thenOrThrow(folly::identity, [&](const Status& status) {
                         VELOX_USER_FAIL("{}", status.message());
                       });
      isConstantTimeFormat_ = true;
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Timestamp>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& format) {
    if (!isConstantTimeFormat_) {
      formatter_ = detail::getDateTimeFormatter(
                       std::string_view(format),
                       legacyFormatter_ ? DateTimeFormatterType::STRICT_SIMPLE
                                        : DateTimeFormatterType::JODA)
                       .thenOrThrow(folly::identity, [&](const Status& status) {
                         VELOX_USER_FAIL("{}", status.message());
                       });
    }
    auto dateTimeResult = formatter_->parse(std::string_view(input));
    // Null as result for parsing error.
    if (dateTimeResult.hasError()) {
      return false;
    }
    (*dateTimeResult).timestamp.toGMT(*getTimeZone(*dateTimeResult));
    result = (*dateTimeResult).timestamp;
    return true;
  }

 private:
  const tz::TimeZone* getTimeZone(const DateTimeResult& result) const {
    // If timezone was not parsed, fallback to the session timezone. If there's
    // no session timezone, fallback to 0 (GMT).
    return result.timezone ? result.timezone : sessionTimeZone_;
  }

  std::shared_ptr<DateTimeFormatter> formatter_{nullptr};
  bool isConstantTimeFormat_{false};
  const tz::TimeZone* sessionTimeZone_{tz::locateZone(0)}; // default to GMT.
  bool legacyFormatter_{false};
};

template <typename T>
struct MakeDateFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const int32_t year,
      const int32_t month,
      const int32_t day) {
    Expected<int64_t> expected = util::daysSinceEpochFromDate(year, month, day);
    if (expected.hasError()) {
      VELOX_DCHECK(expected.error().isUserError());
      VELOX_USER_FAIL(expected.error().message());
    }
    int64_t daysSinceEpoch = expected.value();
    VELOX_USER_CHECK_EQ(
        daysSinceEpoch,
        (int32_t)daysSinceEpoch,
        "Integer overflow in make_date({}, {}, {})",
        year,
        month,
        day);
    result = daysSinceEpoch;
  }
};

template <typename T>
struct LastDayFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE int64_t getYear(const std::tm& time) {
    return 1900 + time.tm_year;
  }

  FOLLY_ALWAYS_INLINE int64_t getMonth(const std::tm& time) {
    return 1 + time.tm_mon;
  }

  FOLLY_ALWAYS_INLINE int64_t getDay(const std::tm& time) {
    return time.tm_mday;
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Date>& date) {
    auto dateTime = getDateTime(date);
    int32_t year = getYear(dateTime);
    int32_t month = getMonth(dateTime);
    auto lastDay = util::getMaxDayOfMonth(year, month);
    Expected<int64_t> expected =
        util::daysSinceEpochFromDate(year, month, lastDay);
    if (expected.hasError()) {
      VELOX_DCHECK(expected.error().isUserError());
      VELOX_USER_FAIL(expected.error().message());
    }
    int64_t daysSinceEpoch = expected.value();
    VELOX_USER_CHECK_EQ(
        daysSinceEpoch,
        (int32_t)daysSinceEpoch,
        "Integer overflow in last_day({})",
        DATE()->toString(date));
    result = daysSinceEpoch;
  }
};

template <typename T>
struct DateFromUnixDateFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(out_type<Date>& result, const int32_t& value) {
    result = value;
  }
};

template <typename T>
struct DateTruncFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  const tz::TimeZone* timeZone_ = nullptr;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* formatString,
      const arg_type<Timestamp>* /*timestamp*/) {
    timeZone_ = getTimeZoneFromConfig(config);
  }

  FOLLY_ALWAYS_INLINE void adjustDateTime(
      std::tm& dateTime,
      const DateTimeFormat& format) {
    switch (format) {
      case DateTimeFormat::kYear:
        dateTime.tm_mon = 0;
        dateTime.tm_yday = 0;
        FMT_FALLTHROUGH;
      case DateTimeFormat::kQuarter:
        dateTime.tm_mon = dateTime.tm_mon / 3 * 3;
        FMT_FALLTHROUGH;
      case DateTimeFormat::kMonth:
        dateTime.tm_mday = 1;
        dateTime.tm_hour = 0;
        dateTime.tm_min = 0;
        dateTime.tm_sec = 0;
        break;
      case DateTimeFormat::kWeek:
        // Subtract the truncation
        dateTime.tm_mday -= dateTime.tm_wday == 0 ? 6 : dateTime.tm_wday - 1;
        // Setting the day of the week to Monday
        dateTime.tm_wday = 1;

        // If the adjusted day of the month falls in the previous month
        // Move to the previous month
        if (dateTime.tm_mday < 1) {
          dateTime.tm_mon -= 1;

          // If the adjusted month falls in the previous year
          // Set to December and Move to the previous year
          if (dateTime.tm_mon < 0) {
            dateTime.tm_mon = 11;
            dateTime.tm_year -= 1;
          }

          // Calculate the correct day of the month based on the number of days
          // in the adjusted month
          static const int daysInMonth[] = {
              31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
          int daysInPrevMonth = daysInMonth[dateTime.tm_mon];

          // Adjust for leap year if February
          if (dateTime.tm_mon == 1 && (dateTime.tm_year + 1900) % 4 == 0 &&
              ((dateTime.tm_year + 1900) % 100 != 0 ||
               (dateTime.tm_year + 1900) % 400 == 0)) {
            daysInPrevMonth = 29;
          }
          // Set to the correct day in the previous month
          dateTime.tm_mday += daysInPrevMonth;
        }
        dateTime.tm_hour = 0;
        dateTime.tm_min = 0;
        dateTime.tm_sec = 0;
        break;
      default:
        VELOX_UNREACHABLE();
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Timestamp>& result,
      const arg_type<Varchar>& formatString,
      const arg_type<Timestamp>& timestamp) {
    std::optional<DateTimeFormat> formatOption =
        fromDateTimeFormatString(formatString, false /*throwIfInvalid*/);
    // Return null if format is illegal.
    if (!formatOption.has_value()) {
      return false;
    }
    DateTimeFormat format = formatOption.value();

    switch (format) {
      // For seconds ,millisecond, microsecond we just truncate the nanoseconds
      // part of the timestamp; no timezone conversion required.
      case DateTimeFormat::kMicrosecond:
        result = Timestamp(
            timestamp.getSeconds(), timestamp.getNanos() / 1000 * 1000);
        return true;

      case DateTimeFormat::kMillisecond:
        result = Timestamp(
            timestamp.getSeconds(), timestamp.getNanos() / 1000000 * 1000000);
        return true;

      case DateTimeFormat::kSecond:
        result = Timestamp(timestamp.getSeconds(), 0);
        return true;

      // Same for minutes; timezones and daylight savings time are at least in
      // the granularity of 30 mins, so we can just truncate the epoch directly.
      case DateTimeFormat::kMinute:
        result = adjustEpoch(timestamp.getSeconds(), 60);
        return true;

      // Hour truncation has to handle the corner case of daylight savings time
      // boundaries. Since conversions from local timezone to UTC may be
      // ambiguous, we need to be carefull about the roundtrip of converting to
      // local time and back. So what we do is to calculate the truncation delta
      // in UTC, then applying it to the input timestamp.
      case DateTimeFormat::kHour: {
        auto epochToAdjust = getSeconds(timestamp, timeZone_);
        auto secondsDelta =
            epochToAdjust - adjustEpoch(epochToAdjust, 60 * 60).getSeconds();
        result = Timestamp(timestamp.getSeconds() - secondsDelta, 0);
        return true;
      }

      // For the truncations below, we may first need to convert to the local
      // timestamp, truncate, then convert back to GMT.
      case DateTimeFormat::kDay:
        result = adjustEpoch(getSeconds(timestamp, timeZone_), 24 * 60 * 60);
        break;

      default:
        auto dateTime = getDateTime(timestamp, timeZone_);
        adjustDateTime(dateTime, format);
        result = Timestamp(Timestamp::calendarUtcToEpoch(dateTime), 0);
        break;
    }

    if (timeZone_ != nullptr) {
      result.toGMT(*timeZone_);
    }
    return true;
  }

 private:
  /// For fixed interval like second, minute, hour, day and week
  /// we can truncate date by a simple arithmetic expression:
  /// floor(seconds / intervalSeconds) * intervalSeconds.
  FOLLY_ALWAYS_INLINE Timestamp
  adjustEpoch(int64_t seconds, int64_t intervalSeconds) {
    int64_t s = seconds / intervalSeconds;
    if (seconds < 0 && seconds % intervalSeconds) {
      s = s - 1;
    }
    int64_t truncedSeconds = s * intervalSeconds;
    return Timestamp(truncedSeconds, 0);
  }
};

template <typename T>
struct DateAddFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Date>& date,
      const TInput& value) {
    __builtin_add_overflow(date, value, &result);
  }
};

template <typename T>
struct DateSubFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Date>& date,
      const TInput& value) {
    __builtin_sub_overflow(date, value, &result);
  }
};

template <typename T>
struct DayOfWeekFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // 1 = Sunday, 2 = Monday, ..., 7 = Saturday
  FOLLY_ALWAYS_INLINE int32_t getDayOfWeek(const std::tm& time) {
    return time.tm_wday + 1;
  }

  FOLLY_ALWAYS_INLINE void call(int32_t& result, const arg_type<Date>& date) {
    result = getDayOfWeek(getDateTime(date));
  }
};

template <typename T>
struct DateDiffFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int32_t& result,
      const arg_type<Date>& endDate,
      const arg_type<Date>& startDate)
#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
      __attribute__((__no_sanitize__("signed-integer-overflow")))
#endif
#endif
  {
    result = endDate - startDate;
  }
};

template <typename T>
struct AddMonthsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void
  call(out_type<Date>& result, const arg_type<Date>& date, int32_t numMonths) {
    const auto dateTime = getDateTime(date);
    const auto year = getYear(dateTime);
    const auto month = getMonth(dateTime);
    const auto day = getDay(dateTime);

    // Similar to handling number in base 12. Here, month - 1 makes it in
    // [0, 11] range.
    int64_t monthAdded = (int64_t)month - 1 + numMonths;
    // Used to adjust month/year when monthAdded is not in [0, 11] range.
    int64_t yearOffset = (monthAdded >= 0 ? monthAdded : monthAdded - 11) / 12;
    // Adjusts monthAdded to natural month number in [1, 12] range.
    auto monthResult = static_cast<int32_t>(monthAdded - yearOffset * 12 + 1);
    // Adjusts year.
    auto yearResult = year + yearOffset;

    auto lastDayOfMonth = util::getMaxDayOfMonth(yearResult, monthResult);
    // Adjusts day to valid one.
    auto dayResult = lastDayOfMonth < day ? lastDayOfMonth : day;

    Expected<int64_t> expected =
        util::daysSinceEpochFromDate(yearResult, monthResult, dayResult);
    if (expected.hasError()) {
      VELOX_DCHECK(expected.error().isUserError());
      VELOX_USER_FAIL(expected.error().message());
    }
    int64_t daysSinceEpoch = expected.value();
    VELOX_USER_CHECK_EQ(
        daysSinceEpoch,
        (int32_t)daysSinceEpoch,
        "Integer overflow in add_months({}, {})",
        DATE()->toString(date),
        numMonths);
    result = daysSinceEpoch;
  }
};

template <typename T>
struct MonthFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(int32_t& result, const arg_type<Date>& date) {
    result = getMonth(getDateTime(date));
  }
};

template <typename T>
struct QuarterFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(int32_t& result, const arg_type<Date>& date) {
    result = getQuarter(getDateTime(date));
  }
};

template <typename T>
struct DayFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(int32_t& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_mday;
  }
};

template <typename T>
struct DayOfYearFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(int32_t& result, const arg_type<Date>& date) {
    result = getDayOfYear(getDateTime(date));
  }
};

template <typename T>
struct WeekdayFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // 0 = Monday, 1 = Tuesday, ..., 6 = Sunday
  FOLLY_ALWAYS_INLINE int32_t getWeekday(const std::tm& time) {
    return (time.tm_wday + 6) % 7;
  }

  FOLLY_ALWAYS_INLINE void call(int32_t& result, const arg_type<Date>& date) {
    result = getWeekday(getDateTime(date));
  }
};

template <typename T>
struct NextDayFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Date>* /*startDate*/,
      const arg_type<Varchar>* dayOfWeek) {
    if (dayOfWeek != nullptr) {
      weekDay_ = getDayOfWeekFromString(*dayOfWeek);
      if (!weekDay_.has_value()) {
        invalidFormat_ = true;
      }
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Date>& result,
      const arg_type<Date>& startDate,
      const arg_type<Varchar>& dayOfWeek) {
    if (invalidFormat_) {
      return false;
    }
    auto weekDay = weekDay_.has_value() ? weekDay_.value()
                                        : getDayOfWeekFromString(dayOfWeek);
    if (!weekDay.has_value()) {
      return false;
    }
    auto nextDay = getNextDate(startDate, weekDay.value());
    if (nextDay != (int32_t)nextDay) {
      return false;
    }
    result = nextDay;
    return true;
  }

 private:
  static FOLLY_ALWAYS_INLINE std::optional<int8_t> getDayOfWeekFromString(
      const StringView& dayOfWeek) {
    std::string lowerDayOfWeek =
        boost::algorithm::to_lower_copy(dayOfWeek.str());
    auto it = kDayOfWeekNames.find(lowerDayOfWeek);
    if (it != kDayOfWeekNames.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  static FOLLY_ALWAYS_INLINE int64_t
  getNextDate(int64_t startDay, int8_t dayOfWeek) {
    return startDay + 1 + ((dayOfWeek - 1 - startDay) % 7 + 7) % 7;
  }

  std::optional<int8_t> weekDay_;
  bool invalidFormat_{false};
};

template <typename T>
struct HourFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int32_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, this->timeZone_).tm_hour;
  }
};

template <typename T>
struct MinuteFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int32_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, this->timeZone_).tm_min;
  }
};

template <typename T>
struct SecondFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int32_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, nullptr).tm_sec;
  }
};

template <typename T>
struct MakeYMIntervalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(out_type<IntervalYearMonth>& result) {
    result = 0;
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<IntervalYearMonth>& result,
      const int32_t year) {
    VELOX_USER_CHECK(
        !__builtin_mul_overflow(year, kMonthInYear, &result),
        "Integer overflow in make_ym_interval({})",
        year);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<IntervalYearMonth>& result,
      const int32_t year,
      const int32_t month) {
    auto totalMonths = (int64_t)year * kMonthInYear + month;
    VELOX_USER_CHECK_EQ(
        totalMonths,
        (int32_t)totalMonths,
        "Integer overflow in make_ym_interval({}, {})",
        year,
        month);
    result = totalMonths;
  }
};

template <typename T>
struct UnixSecondsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = timestamp.getSeconds();
  }
};

template <typename T>
struct TimestampToMicrosFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = timestamp.toMicros();
  }
};

template <typename TExec>
struct MicrosToTimestampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename T>
  FOLLY_ALWAYS_INLINE void call(out_type<Timestamp>& result, const T& micros) {
    result = Timestamp::fromMicrosNoError(micros);
  }
};

template <typename T>
struct TimestampToMillisFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = timestamp.toMillis();
  }
};

template <typename TExec>
struct MillisToTimestampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename T>
  FOLLY_ALWAYS_INLINE void call(out_type<Timestamp>& result, const T& millis) {
    result = Timestamp::fromMillisNoError(millis);
  }
};

} // namespace facebook::velox::functions::sparksql
