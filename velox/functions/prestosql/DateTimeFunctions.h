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
#include "velox/external/date/tz.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/JodaDateTime.h"
#include "velox/functions/prestosql/DateTimeImpl.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

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
    std::tm dateTime = getDateTime(date);
    result = dateTime.tm_wday == 0 ? 7 : dateTime.tm_wday;
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
    result = timestamp.getNanos() / kNanosecondsInMilliseconds;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Date>& date) {
    // Dates do not have millisecond granularity.
    result = 0;
    return true;
  }
};

namespace {
enum class DateTimeUnit { kSecond, kMinute, kHour, kDay, kMonth, kYear };

inline std::optional<DateTimeUnit> fromDateTimeUnitString(
    const StringView& unitString,
    bool throwIfInvalid) {
  static const StringView kSecond("second");
  static const StringView kMinute("minute");
  static const StringView kHour("hour");
  static const StringView kDay("day");
  static const StringView kMonth("month");
  static const StringView kYear("year");

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
  if (unitString == kYear) {
    return DateTimeUnit::kYear;
  }
  // TODO Add support for "quarter" and "week".
  if (throwIfInvalid) {
    VELOX_UNSUPPORTED("Unsupported datetime unit: {}", unitString);
  }
  return std::nullopt;
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
    }
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const core::QueryConfig& config,
      const arg_type<Varchar>* unitString,
      const arg_type<Date>* /*date*/) {
    if (unitString != nullptr) {
      unit_ = fromDateTimeUnitString(*unitString, false /*throwIfInvalid*/);
    }
  }

  FOLLY_ALWAYS_INLINE void fixTmForTruncation(
      std::tm& tmValue,
      const DateTimeUnit& unit) {
    switch (unit) {
      case DateTimeUnit::kYear:
        tmValue.tm_mon = 0;
        tmValue.tm_yday = 0;
        FMT_FALLTHROUGH;
      case DateTimeUnit::kMonth:
        tmValue.tm_mday = 1;
        FMT_FALLTHROUGH;
      case DateTimeUnit::kDay:
        tmValue.tm_hour = 0;
        FMT_FALLTHROUGH;
      case DateTimeUnit::kHour:
        tmValue.tm_min = 0;
        FMT_FALLTHROUGH;
      case DateTimeUnit::kMinute:
        tmValue.tm_sec = 0;
        break;
      default:
        VELOX_UNREACHABLE();
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Timestamp>& result,
      const arg_type<Varchar>& unitString,
      const arg_type<Timestamp>& timestamp) {
    const auto unit = unit_.has_value()
        ? unit_.value()
        : fromDateTimeUnitString(unitString, true /*throwIfInvalid*/).value();
    if (unit == DateTimeUnit::kSecond) {
      result = Timestamp(timestamp.getSeconds(), 0);
      return true;
    }

    auto dateTime = getDateTime(timestamp, timeZone_);
    fixTmForTruncation(dateTime, unit);

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
    const auto unit = unit_.has_value()
        ? unit_.value()
        : fromDateTimeUnitString(unitString, true /*throwIfInvalid*/).value();
    if (unit == DateTimeUnit::kSecond || unit == DateTimeUnit::kMinute ||
        unit == DateTimeUnit::kHour) {
      VELOX_USER_FAIL("{} is not a valid DATE field", unitString);
    }

    auto dateTm = getDateTime(date);
    fixTmForTruncation(dateTm, unit);

    result = Date(timegm(&dateTm) / kSecondsInDay);
    return true;
  }
};

template <typename T>
struct ParseDateTimeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::optional<JodaFormatter> format_;

  FOLLY_ALWAYS_INLINE void initialize(
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* format) {
    if (format != nullptr) {
      format_.emplace(*format);
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& format) {
    auto ts = format_.has_value() ? format_->parse(input)
                                  : JodaFormatter(format).parse(input);
    // TODO: Need to extend JodaFormatter to parse and add the timezone
    // information as the second argument.
    result = std::make_tuple(ts.toMillis(), (int16_t)0);
    return true;
  }
};

} // namespace facebook::velox::functions
