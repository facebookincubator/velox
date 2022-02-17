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
#include "velox/core/QueryConfig.h"
#include "velox/external/date/tz.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/JodaDateTime.h"
#include "velox/functions/prestosql/DateTimeImpl.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions {

template <typename T>
struct ToUnixtimeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      double& result,
      const arg_type<Timestamp>& timestamp) {
    result = toUnixtime(timestamp);
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(
      double& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    const auto milliseconds = *timestampWithTimezone.template at<0>();
    Timestamp timestamp{milliseconds / kMillisecondsInSecond, 0UL};
    timestamp.toTimezone(*timestampWithTimezone.template at<1>());
    result = toUnixtime(timestamp);
    return true;
  }
};

template <typename T>
struct FromUnixtimeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      Timestamp& result,
      const arg_type<double>& unixtime) {
    auto resultOptional = fromUnixtime(unixtime);
    if (LIKELY(resultOptional.has_value())) {
      result = resultOptional.value();
      return true;
    }
    return false;
  }
};

namespace {
inline constexpr int64_t kSecondsInDay = 86'400;

FOLLY_ALWAYS_INLINE const date::time_zone* getTimeZoneFromConfig(
    const core::QueryConfig& config) {
  if (config.adjustTimestampToTimezone()) {
    auto sessionTzName = config.sessionTimezone();
    if (!sessionTzName.empty()) {
      return date::locate_zone(sessionTzName);
    }
  }
  return nullptr;
}

FOLLY_ALWAYS_INLINE int64_t
getSeconds(Timestamp timestamp, const date::time_zone* timeZone) {
  if (timeZone != nullptr) {
    timestamp.toTimezoneUTC(*timeZone);
    return timestamp.getSeconds();
  } else {
    return timestamp.getSeconds();
  }
}

FOLLY_ALWAYS_INLINE
std::tm getDateTime(Timestamp timestamp, const date::time_zone* timeZone) {
  int64_t seconds = getSeconds(timestamp, timeZone);
  std::tm dateTime;
  gmtime_r((const time_t*)&seconds, &dateTime);
  return dateTime;
}

FOLLY_ALWAYS_INLINE
std::tm getDateTime(Date date) {
  int64_t seconds = date.days() * kSecondsInDay;
  std::tm dateTime;
  gmtime_r((const time_t*)&seconds, &dateTime);
  return dateTime;
}

template <typename T>
struct InitSessionTimezone {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  const date::time_zone* timeZone_{nullptr};

  FOLLY_ALWAYS_INLINE void initialize(
      const core::QueryConfig& config,
      const arg_type<Timestamp>* /*timestamp*/) {
    timeZone_ = getTimeZoneFromConfig(config);
  }
};

} // namespace

template <typename T>
struct YearFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = 1900 + getDateTime(timestamp, this->timeZone_).tm_year;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Date>& date) {
    result = 1900 + getDateTime(date).tm_year;
    return true;
  }
};

template <typename T>
struct QuarterFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, this->timeZone_).tm_mon / 3 + 1;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_mon / 3 + 1;
    return true;
  }
};

template <typename T>
struct MonthFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = 1 + getDateTime(timestamp, this->timeZone_).tm_mon;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Date>& date) {
    result = 1 + getDateTime(date).tm_mon;
    return true;
  }
};

template <typename T>
struct DayFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, this->timeZone_).tm_mday;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_mday;
    return true;
  }
};

template <typename T>
struct DayOfWeekFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    std::tm dateTime = getDateTime(timestamp, this->timeZone_);
    result = dateTime.tm_wday == 0 ? 7 : dateTime.tm_wday;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Date>& date) {
    std::tm dateTm = getDateTime(date);
    result = dateTm.tm_wday == 0 ? 7 : dateTm.tm_wday;
    return true;
  }
};

template <typename T>
struct DayOfYearFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = 1 + getDateTime(timestamp, this->timeZone_).tm_yday;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Date>& date) {
    result = 1 + getDateTime(date).tm_yday;
    return true;
  }
};

template <typename T>
struct YearOfWeekFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE int64_t computeYearOfWeek(const std::tm& dateTime) {
    int isoWeekDay = dateTime.tm_wday == 0 ? 7 : dateTime.tm_wday;
    // The last few days in December may belong to the next year if they are
    // in the same week as the next January 1 and this January 1 is a Thursday
    // or before.
    if (UNLIKELY(
            dateTime.tm_mon == 11 && dateTime.tm_mday >= 29 &&
            dateTime.tm_mday - isoWeekDay >= 31 - 3)) {
      return 1900 + dateTime.tm_year + 1;
    }
    // The first few days in January may belong to the last year if they are
    // in the same week as January 1 and January 1 is a Friday or after.
    else if (UNLIKELY(
                 dateTime.tm_mon == 0 && dateTime.tm_mday <= 3 &&
                 isoWeekDay - (dateTime.tm_mday - 1) >= 5)) {
      return 1900 + dateTime.tm_year - 1;
    } else {
      return 1900 + dateTime.tm_year;
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    auto dateTime = getDateTime(timestamp, this->timeZone_);
    result = computeYearOfWeek(dateTime);
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Date>& date) {
    auto dateTime = getDateTime(date);
    result = computeYearOfWeek(dateTime);
    return true;
  }
};

template <typename T>
struct HourFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, this->timeZone_).tm_hour;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_hour;
    return true;
  }
};

template <typename T>
struct MinuteFunction : public InitSessionTimezone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, this->timeZone_).tm_min;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_min;
    return true;
  }
};

template <typename T>
struct SecondFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, nullptr).tm_sec;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_sec;
    return true;
  }
};

template <typename T>
struct MillisecondFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = timestamp.getNanos() / kNanosecondsInMillisecond;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Date>& /*date*/) {
    // Dates do not have millisecond granularity.
    result = 0;
    return true;
  }
};

namespace {
inline std::optional<DateTimeUnit> fromDateTimeUnitString(
    const StringView& unitString,
    bool throwIfInvalid) {
  static const StringView kMillisecond("millisecond");
  static const StringView kSecond("second");
  static const StringView kMinute("minute");
  static const StringView kHour("hour");
  static const StringView kDay("day");
  static const StringView kMonth("month");
  static const StringView kQuarter("quarter");
  static const StringView kYear("year");

  if (unitString == kMillisecond) {
    return DateTimeUnit::kMillisecond;
  }
  if (unitString == kSecond) {
    return DateTimeUnit::kSecond;
  }
  if (unitString == kMinute) {
    return DateTimeUnit::kMinute;
  }
  if (unitString == kHour) {
    return DateTimeUnit::kHour;
  }
  if (unitString == kDay) {
    return DateTimeUnit::kDay;
  }
  if (unitString == kMonth) {
    return DateTimeUnit::kMonth;
  }
  if (unitString == kQuarter) {
    return DateTimeUnit::kQuarter;
  }
  if (unitString == kYear) {
    return DateTimeUnit::kYear;
  }
  // TODO Add support for "week".
  if (throwIfInvalid) {
    VELOX_UNSUPPORTED("Unsupported datetime unit: {}", unitString);
  }
  return std::nullopt;
}

inline bool isTimeUnit(const DateTimeUnit unit) {
  return unit == DateTimeUnit::kMillisecond || unit == DateTimeUnit::kSecond ||
      unit == DateTimeUnit::kMinute || unit == DateTimeUnit::kHour;
}

inline bool isDateUnit(const DateTimeUnit unit) {
  return unit == DateTimeUnit::kDay || unit == DateTimeUnit::kMonth ||
      unit == DateTimeUnit::kQuarter || unit == DateTimeUnit::kYear;
}

inline std::optional<DateTimeUnit> getDateUnit(
    const StringView& unitString,
    bool throwIfInvalid) {
  std::optional<DateTimeUnit> unit =
      fromDateTimeUnitString(unitString, throwIfInvalid);
  if (unit.has_value() && !isDateUnit(unit.value())) {
    if (throwIfInvalid) {
      VELOX_USER_FAIL("{} is not a valid DATE field", unitString);
    }
    return std::nullopt;
  }
  return unit;
}

} // namespace

template <typename T>
struct DateTruncFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  const date::time_zone* timeZone_ = nullptr;
  std::optional<DateTimeUnit> unit_;

  FOLLY_ALWAYS_INLINE void initialize(
      const core::QueryConfig& config,
      const arg_type<Varchar>* unitString,
      const arg_type<Timestamp>* /*timestamp*/) {
    timeZone_ = getTimeZoneFromConfig(config);

    if (unitString != nullptr) {
      unit_ = fromDateTimeUnitString(*unitString, false /*throwIfInvalid*/);
      VELOX_USER_CHECK(
          !(unit_.has_value() && unit_.value() == DateTimeUnit::kMillisecond),
          "{} is not a valid TIMESTAMP field",
          *unitString);
    }
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* unitString,
      const arg_type<Date>* /*date*/) {
    if (unitString != nullptr) {
      unit_ = getDateUnit(*unitString, false);
    }
  }

  FOLLY_ALWAYS_INLINE void adjustDateTime(
      std::tm& dateTime,
      const DateTimeUnit& unit) {
    switch (unit) {
      case DateTimeUnit::kYear:
        dateTime.tm_mon = 0;
        dateTime.tm_yday = 0;
        FMT_FALLTHROUGH;
      case DateTimeUnit::kQuarter:
        dateTime.tm_mon = dateTime.tm_mon / 3 * 3;
        FMT_FALLTHROUGH;
      case DateTimeUnit::kMonth:
        dateTime.tm_mday = 1;
        FMT_FALLTHROUGH;
      case DateTimeUnit::kDay:
        dateTime.tm_hour = 0;
        FMT_FALLTHROUGH;
      case DateTimeUnit::kHour:
        dateTime.tm_min = 0;
        FMT_FALLTHROUGH;
      case DateTimeUnit::kMinute:
        dateTime.tm_sec = 0;
        break;
      default:
        VELOX_UNREACHABLE();
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Timestamp>& result,
      const arg_type<Varchar>& unitString,
      const arg_type<Timestamp>& timestamp) {
    DateTimeUnit unit;
    if (unit_.has_value()) {
      unit = unit_.value();
    } else {
      unit =
          fromDateTimeUnitString(unitString, true /*throwIfInvalid*/).value();
      VELOX_USER_CHECK(
          unit != DateTimeUnit::kMillisecond,
          "{} is not a valid TIMESTAMP field",
          unitString);
    }

    if (unit == DateTimeUnit::kSecond) {
      result = Timestamp(timestamp.getSeconds(), 0);
      return true;
    }

    auto dateTime = getDateTime(timestamp, timeZone_);
    adjustDateTime(dateTime, unit);

    result = Timestamp(timegm(&dateTime), 0);
    if (timeZone_ != nullptr) {
      result.toTimezone(*timeZone_);
    }
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Date>& result,
      const arg_type<Varchar>& unitString,
      const arg_type<Date>& date) {
    DateTimeUnit unit = unit_.has_value()
        ? unit_.value()
        : getDateUnit(unitString, true).value();

    if (unit == DateTimeUnit::kDay) {
      result = Date(date.days());
      return true;
    }

    auto dateTime = getDateTime(date);
    adjustDateTime(dateTime, unit);

    result = Date(timegm(&dateTime) / kSecondsInDay);
    return true;
  }
};

template <typename T>
struct DateAddFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  const date::time_zone* sessionTimeZone_ = nullptr;
  std::optional<DateTimeUnit> unit_ = std::nullopt;

  FOLLY_ALWAYS_INLINE void initialize(
      const core::QueryConfig& config,
      const arg_type<Varchar>* unitString,
      const int64_t* /*value*/,
      const arg_type<Timestamp>* /*timestamp*/) {
    sessionTimeZone_ = getTimeZoneFromConfig(config);
    if (unitString != nullptr) {
      unit_ = fromDateTimeUnitString(*unitString, false /*throwIfInvalid*/);
    }
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* unitString,
      const int64_t* /*value*/,
      const arg_type<Date>* /*date*/) {
    if (unitString != nullptr) {
      unit_ = getDateUnit(*unitString, false);
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Timestamp>& result,
      const arg_type<Varchar>& unitString,
      const int64_t value,
      const arg_type<Timestamp>& timestamp) {
    const auto unit = unit_.has_value()
        ? unit_.value()
        : fromDateTimeUnitString(unitString, true /*throwIfInvalid*/).value();

    if (value != (int32_t)value) {
      VELOX_UNSUPPORTED("integer overflow");
    }

    if (LIKELY(sessionTimeZone_ != nullptr)) {
      // sessionTimeZone not null means that the config
      // adjust_timestamp_to_timezone is on.
      Timestamp zonedTimestamp = timestamp;
      zonedTimestamp.toTimezoneUTC(*sessionTimeZone_);

      Timestamp resultTimestamp =
          addToTimestamp(zonedTimestamp, unit, (int32_t)value);

      if (isTimeUnit(unit)) {
        const int64_t offset = static_cast<Timestamp>(timestamp).getSeconds() -
            zonedTimestamp.getSeconds();
        result = Timestamp(
            resultTimestamp.getSeconds() + offset, resultTimestamp.getNanos());
      } else {
        resultTimestamp.toTimezone(*sessionTimeZone_);
        result = resultTimestamp;
      }
    } else {
      result = addToTimestamp(timestamp, unit, (int32_t)value);
    }

    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Date>& result,
      const arg_type<Varchar>& unitString,
      const int64_t value,
      const arg_type<Date>& date) {
    DateTimeUnit unit = unit_.has_value()
        ? unit_.value()
        : getDateUnit(unitString, true).value();

    if (value != (int32_t)value) {
      VELOX_UNSUPPORTED("integer overflow");
    }

    result = addToDate(date, unit, (int32_t)value);
    return true;
  }
};

template <typename T>
struct DateDiffFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  const date::time_zone* sessionTimeZone_ = nullptr;
  std::optional<DateTimeUnit> unit_ = std::nullopt;

  FOLLY_ALWAYS_INLINE void initialize(
      const core::QueryConfig& config,
      const arg_type<Varchar>* unitString,
      const arg_type<Timestamp>* /*timestamp1*/,
      const arg_type<Timestamp>* /*timestamp2*/) {
    if (unitString != nullptr) {
      unit_ = fromDateTimeUnitString(*unitString, false /*throwIfInvalid*/);
    }

    sessionTimeZone_ = getTimeZoneFromConfig(config);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* unitString,
      const arg_type<Date>* /*date1*/,
      const arg_type<Date>* /*date2*/) {
    if (unitString != nullptr) {
      unit_ = getDateUnit(*unitString, false);
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Varchar>& unitString,
      const arg_type<Timestamp>& timestamp1,
      const arg_type<Timestamp>& timestamp2) {
    const auto unit = unit_.has_value()
        ? unit_.value()
        : fromDateTimeUnitString(unitString, true /*throwIfInvalid*/).value();

    if (LIKELY(sessionTimeZone_ != nullptr)) {
      // sessionTimeZone not null means that the config
      // adjust_timestamp_to_timezone is on.
      Timestamp fromZonedTimestamp = timestamp1;
      fromZonedTimestamp.toTimezoneUTC(*sessionTimeZone_);

      Timestamp toZonedTimestamp = timestamp2;
      if (isTimeUnit(unit)) {
        const int64_t offset = static_cast<Timestamp>(timestamp1).getSeconds() -
            fromZonedTimestamp.getSeconds();
        toZonedTimestamp = Timestamp(
            toZonedTimestamp.getSeconds() - offset,
            toZonedTimestamp.getNanos());
      } else {
        toZonedTimestamp.toTimezoneUTC(*sessionTimeZone_);
      }
      result = diffTimestamp(unit, fromZonedTimestamp, toZonedTimestamp);
    } else {
      result = diffTimestamp(unit, timestamp1, timestamp2);
    }
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Varchar>& unitString,
      const arg_type<Date>& date1,
      const arg_type<Date>& date2) {
    DateTimeUnit unit = unit_.has_value()
        ? unit_.value()
        : getDateUnit(unitString, true).value();

    result = diffDate(unit, date1, date2);
    return true;
  }
};

template <typename T>
struct ParseDateTimeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::optional<JodaFormatter> format_;
  std::optional<int64_t> sessionTzID_;

  FOLLY_ALWAYS_INLINE void initialize(
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* format) {
    if (format != nullptr) {
      format_.emplace(*format);
    }

    auto sessionTzName = config.sessionTimezone();
    if (!sessionTzName.empty()) {
      sessionTzID_ = util::getTimeZoneID(sessionTzName);
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& format) {
    auto jodaResult = format_.has_value() ? format_->parse(input)
                                          : JodaFormatter(format).parse(input);

    // If timezone was not parsed, fallback to the session timezone. If there's
    // no session timezone, fallback to 0 (GMT).
    int16_t timezoneId = jodaResult.timezoneId != -1 ? jodaResult.timezoneId
                                                     : sessionTzID_.value_or(0);
    result = std::make_tuple(jodaResult.timestamp.toMillis(), timezoneId);
    return true;
  }
};

} // namespace facebook::velox::functions
