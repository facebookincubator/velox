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
#include <velox/type/Timestamp.h>
#include <string_view>
#include "velox/core/QueryConfig.h"
#include "velox/external/date/tz.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/DateTimeFormatter.h"
#include "velox/functions/lib/JodaDateTime.h"
#include "velox/functions/prestosql/DateTimeImpl.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Type.h"
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
    result = (double)milliseconds / kMillisecondsInSecond;
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
    timestamp.toTimezone(*timeZone);
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

template <typename T>
struct TimestampWithTimezoneSupport {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Convert timestampWithTimezone to a timestamp representing the moment at the
  // zone in timestampWithTimezone.
  FOLLY_ALWAYS_INLINE
  Timestamp toTimestamp(
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    const auto milliseconds = *timestampWithTimezone.template at<0>();
    Timestamp timestamp = Timestamp::fromMillis(milliseconds);
    timestamp.toTimezone(*timestampWithTimezone.template at<1>());

    return timestamp;
  }
};

} // namespace

template <typename T>
struct YearFunction : public InitSessionTimezone<T>,
                      public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE int64_t getYear(const std::tm& time) {
    return 1900 + time.tm_year;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE
  void call(
      TInput& result,
      const arg_type<Timestamp>& timestamp) {
    result = getYear(getDateTime(timestamp, this->timeZone_));
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE
  void call(TInput& result, const arg_type<Date>& date) {
    result = getYear(getDateTime(date));
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE
  void call(
      TInput& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getYear(getDateTime(timestamp, nullptr));
  }
};

template <typename T>
struct QuarterFunction : public InitSessionTimezone<T>,
                         public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE int64_t getQuarter(const std::tm& time) {
    return time.tm_mon / 3 + 1;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<Timestamp>& timestamp) {
    result = getQuarter(getDateTime(timestamp, this->timeZone_));
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, const arg_type<Date>& date) {
    result = getQuarter(getDateTime(date));
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getQuarter(getDateTime(timestamp, nullptr));
  }
};

template <typename T>
struct MonthFunction : public InitSessionTimezone<T>,
                       public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE int64_t getMonth(const std::tm& time) {
    return 1 + time.tm_mon;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<Timestamp>& timestamp) {
    result = getMonth(getDateTime(timestamp, this->timeZone_));
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, const arg_type<Date>& date) {
    result = getMonth(getDateTime(date));
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getMonth(getDateTime(timestamp, nullptr));
  }
};

template <typename T>
struct DayFunction : public InitSessionTimezone<T>,
                     public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, this->timeZone_).tm_mday;
  }


  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_mday;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getDateTime(timestamp, nullptr).tm_mday;
  }
};

template <typename T>
struct DayOfWeekFunction : public InitSessionTimezone<T>,
                           public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE int64_t getDayOfWeek(const std::tm& time) {
    return time.tm_wday + 1 == 0 ? 7 : time.tm_wday + 1;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDayOfWeek(getDateTime(timestamp, this->timeZone_));
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, const arg_type<Date>& date) {
    result = getDayOfWeek(getDateTime(date));
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getDayOfWeek(getDateTime(timestamp, nullptr));
  }
};

template <typename T>
struct DayOfYearFunction : public InitSessionTimezone<T>,
                           public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE int64_t getDayOfYear(const std::tm& time) {
    return time.tm_yday + 1;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDayOfYear(getDateTime(timestamp, this->timeZone_));
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, const arg_type<Date>& date) {
    result = getDayOfYear(getDateTime(date));
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getDayOfYear(getDateTime(timestamp, nullptr));
  }

};


template <typename T>
struct YearOfWeekFunction : public InitSessionTimezone<T>,
                            public TimestampWithTimezoneSupport<T> {
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

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<Timestamp>& timestamp) {
    result = computeYearOfWeek(getDateTime(timestamp, this->timeZone_));
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, const arg_type<Date>& date) {
    result = computeYearOfWeek(getDateTime(date));
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = computeYearOfWeek(getDateTime(timestamp, nullptr));
  }
};


template <typename T>
struct HourFunction : public InitSessionTimezone<T>,
                      public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, this->timeZone_).tm_hour;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_hour;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getDateTime(timestamp, nullptr).tm_hour;
  }
};

template <typename T>
struct MinuteFunction : public InitSessionTimezone<T>,
                        public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, this->timeZone_).tm_min;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_min;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getDateTime(timestamp, nullptr).tm_min;
  }
};

template <typename T>
struct SecondFunction : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, nullptr).tm_sec;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_sec;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getDateTime(timestamp, nullptr).tm_sec;
  }
};

template <typename T>
struct MillisecondFunction : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<Timestamp>& timestamp) {
    result = timestamp.getNanos() / kNanosecondsInMillisecond;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<Date>& /*date*/) {
    // Dates do not have millisecond granularity.
    result = 0;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(
      TInput& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = timestamp.getNanos() / kNanosecondsInMillisecond;
  }
};

} // namespace facebook::velox::functions
