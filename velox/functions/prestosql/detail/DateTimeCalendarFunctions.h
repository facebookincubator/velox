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

#include <chrono>
#include <optional>
#include <string_view>
#include <vector>

#include "velox/functions/Macros.h"
#include "velox/functions/lib/DateTimeUtil.h"
#include "velox/functions/lib/TimeUtils.h"
#include "velox/functions/prestosql/DateTimeImpl.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneRenderZone.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Timestamp.h"
#include "velox/type/tz/TimeZoneMap.h"

// Calendar arithmetic and field extraction over DATE, TIMESTAMP and TIMESTAMP
// WITH TIME ZONE. These structs depend only on the time zone database and
// the date arithmetic in velox/functions/lib, so a translation unit can
// compile them without the formatters, parsers and vector machinery that the
// rest of DateTimeFunctions.h needs.
namespace facebook::velox::functions {

template <typename T>
struct ToUnixtimeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      double& result,
      const arg_type<Timestamp>& timestamp) {
    result = toUnixtime(timestamp);
  }

  FOLLY_ALWAYS_INLINE void call(
      double& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    const auto milliseconds = unpackMillisUtc(*timestampWithTimezone);
    result = (double)milliseconds / Timestamp::kMillisecondsInSecond;
  }
};

template <typename T>
struct FromUnixtimeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // (double) -> timestamp
  FOLLY_ALWAYS_INLINE void call(
      Timestamp& result,
      const arg_type<double>& unixtime) {
    result = fromUnixtime(unixtime);
  }

  // (double, varchar) -> timestamp with time zone
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<double>* /*unixtime*/,
      const arg_type<Varchar>* timezone) {
    if (timezone != nullptr) {
      tzID_ = tz::getTimeZoneID((std::string_view)(*timezone));
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<double>& unixtime,
      const arg_type<Varchar>& timeZone) {
    int16_t timeZoneId =
        tzID_.value_or(tz::getTimeZoneID((std::string_view)timeZone));
    result = fromUnixtime(unixtime, timeZoneId);
  }

  // (double, bigint, bigint) -> timestamp with time zone
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<double>* /*unixtime*/,
      const arg_type<int64_t>* hours,
      const arg_type<int64_t>* minutes) {
    if (hours != nullptr && minutes != nullptr) {
      tzID_ = tz::getTimeZoneID(
          checkedPlus(checkedMultiply<int64_t>(*hours, 60), *minutes));
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<double>& unixtime,
      const arg_type<int64_t>& hours,
      const arg_type<int64_t>& minutes) {
    int16_t timezoneId = tzID_.value_or(
        tz::getTimeZoneID(
            checkedPlus(checkedMultiply<int64_t>(hours, 60), minutes)));
    result = pack(fromUnixtime(unixtime).toMillis(), timezoneId);
  }

 private:
  std::optional<int64_t> tzID_;
};

namespace {

// Returns the embedded zone's UTC offset at the represented instant.
FOLLY_ALWAYS_INLINE int64_t
getTimeZoneOffsetSeconds(int64_t timestampWithTimezone) {
  const auto* embeddedZone =
      tz::locateZone(unpackZoneKeyId(timestampWithTimezone));
  auto inputTimestamp = unpackTimestampUtc(timestampWithTimezone);
  inputTimestamp.toTimezone(*embeddedZone);
  auto gmtTimestamp = inputTimestamp;
  gmtTimestamp.toGMT(*embeddedZone);
  return inputTimestamp.getSeconds() - gmtTimestamp.getSeconds();
}

template <typename T>
struct TimestampWithTimezoneSupport {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<TimestampWithTimezone>* /*timestampWithTimezone*/) {
    initializeTimeZoneSupport(config);
  }

  // Converts timestampWithTimezone to a timestamp representing the same
  // instant in the render zone. If `asGMT` is true, returns the GMT time at
  // that instant.
  FOLLY_ALWAYS_INLINE
  Timestamp toTimestamp(
      const arg_type<TimestampWithTimezone>& timestampWithTimezone,
      bool asGMT = false) {
    auto timestamp = unpackTimestampUtc(*timestampWithTimezone);
    if (!asGMT) {
      timestamp.toTimezone(*renderZone(timestampWithTimezone));
    }

    return timestamp;
  }

 protected:
  FOLLY_ALWAYS_INLINE void initializeTimeZoneSupport(
      const core::QueryConfig& config) {
    renderZone_.emplace(config);
  }

  FOLLY_ALWAYS_INLINE void initializeTimeZoneSupport(
      const core::QueryConfig& config,
      TimestampWithTimeZoneRenderZone::SessionZoneResolution resolution) {
    renderZone_.emplace(config, resolution);
  }

  // Returns the embedded zone under legacy behavior, the session zone
  // otherwise.
  FOLLY_ALWAYS_INLINE const tz::TimeZone* renderZone(
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) const {
    VELOX_CHECK(
        renderZone_.has_value(),
        "TimestampWithTimezoneSupport must be initialized before use");
    return renderZone_->get(*timestampWithTimezone);
  }

 private:
  // Holds the query-scoped rendering policy after initialization.
  std::optional<TimestampWithTimeZoneRenderZone> renderZone_;
};

} // namespace

template <typename T>
struct DateFunction : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* date) {
    timeZone_ = getTimeZoneFromConfig(config);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Timestamp>* timestamp) {
    timeZone_ = getTimeZoneFromConfig(config);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<TimestampWithTimezone>* /*timestampWithTimezone*/) {
    this->initializeTimeZoneSupport(config);
  }

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Date>& result, const arg_type<Varchar>& date) {
    auto days = util::fromDateString(date, util::ParseMode::kPrestoCast);
    if (days.hasError()) {
      return days.error();
    }

    result = days.value();
    return Status::OK();
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Timestamp>& timestamp) {
    result = util::toDate(timestamp, timeZone_);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    result = util::toDate(this->toTimestamp(timestampWithTimezone), nullptr);
  }

 private:
  const tz::TimeZone* timeZone_ = nullptr;
};

template <typename T>
struct WeekFunction : public InitSessionTimezone<T>,
                      public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  using InitSessionTimezone<T>::initialize;
  using TimestampWithTimezoneSupport<T>::initialize;

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getWeek(timestamp, this->timeZone_, false);
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Date>& date) {
    result = getWeek(Timestamp::fromDate(date), nullptr, false);
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getWeek(timestamp, nullptr, false);
  }
};

template <typename T>
struct YearFunction : public InitSessionTimezone<T>,
                      public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  using InitSessionTimezone<T>::initialize;
  using TimestampWithTimezoneSupport<T>::initialize;

  FOLLY_ALWAYS_INLINE int64_t getYear(const std::tm& time) {
    return 1900 + time.tm_year;
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getYear(getDateTime(timestamp, this->timeZone_));
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Date>& date) {
    result = getYear(getDateTime(date));
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getYear(getDateTime(timestamp, nullptr));
  }
};

template <typename T>
struct QuarterFunction : public InitSessionTimezone<T>,
                         public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  using InitSessionTimezone<T>::initialize;
  using TimestampWithTimezoneSupport<T>::initialize;

  FOLLY_ALWAYS_INLINE int64_t getQuarter(const std::tm& time) {
    return time.tm_mon / 3 + 1;
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getQuarter(getDateTime(timestamp, this->timeZone_));
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Date>& date) {
    result = getQuarter(getDateTime(date));
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getQuarter(getDateTime(timestamp, nullptr));
  }
};

template <typename T>
struct MonthFunction : public InitSessionTimezone<T>,
                       public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  using InitSessionTimezone<T>::initialize;
  using TimestampWithTimezoneSupport<T>::initialize;

  FOLLY_ALWAYS_INLINE int64_t getMonth(const std::tm& time) {
    return 1 + time.tm_mon;
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getMonth(getDateTime(timestamp, this->timeZone_));
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Date>& date) {
    result = getMonth(getDateTime(date));
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getMonth(getDateTime(timestamp, nullptr));
  }
};

template <typename T>
struct DayFunction : public InitSessionTimezone<T>,
                     public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  using InitSessionTimezone<T>::initialize;
  using TimestampWithTimezoneSupport<T>::initialize;

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, this->timeZone_).tm_mday;
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_mday;
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getDateTime(timestamp, nullptr).tm_mday;
  }
};

template <typename T>
struct LastDayOfMonthFunction : public InitSessionTimezone<T>,
                                public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  using InitSessionTimezone<T>::initialize;
  using TimestampWithTimezoneSupport<T>::initialize;

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Timestamp>& timestamp) {
    auto dt = getDateTime(timestamp, this->timeZone_);
    Expected<int64_t> daysSinceEpochFromDate =
        util::lastDayOfMonthSinceEpochFromDate(dt);
    if (daysSinceEpochFromDate.hasError()) {
      VELOX_DCHECK(daysSinceEpochFromDate.error().isUserError());
      VELOX_USER_FAIL(daysSinceEpochFromDate.error().message());
    }
    result = daysSinceEpochFromDate.value();
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Date>& date) {
    auto dt = getDateTime(date);
    Expected<int64_t> lastDayOfMonthSinceEpoch =
        util::lastDayOfMonthSinceEpochFromDate(dt);
    if (lastDayOfMonthSinceEpoch.hasError()) {
      VELOX_DCHECK(lastDayOfMonthSinceEpoch.error().isUserError());
      VELOX_USER_FAIL(lastDayOfMonthSinceEpoch.error().message());
    }
    result = lastDayOfMonthSinceEpoch.value();
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    auto dt = getDateTime(timestamp, nullptr);
    Expected<int64_t> lastDayOfMonthSinceEpoch =
        util::lastDayOfMonthSinceEpochFromDate(dt);
    if (lastDayOfMonthSinceEpoch.hasError()) {
      VELOX_DCHECK(lastDayOfMonthSinceEpoch.error().isUserError());
      VELOX_USER_FAIL(lastDayOfMonthSinceEpoch.error().message());
    }
    result = lastDayOfMonthSinceEpoch.value();
  }
};

template <typename T>
struct TimestampMinusFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<IntervalDayTime>& result,
      const arg_type<Timestamp>& a,
      const arg_type<Timestamp>& b) {
    result = a.toMillis() - b.toMillis();
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<IntervalDayTime>& result,
      const arg_type<TimestampWithTimezone>& a,
      const arg_type<TimestampWithTimezone>& b) {
    result = unpackMillisUtc(*a) - unpackMillisUtc(*b);
  }
};

template <typename T>
struct TimestampPlusInterval : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Timestamp>& result,
      const arg_type<Timestamp>& a,
      const arg_type<IntervalDayTime>& b)
#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
      __attribute__((__no_sanitize__("signed-integer-overflow")))
#endif
#endif
  {
    result = Timestamp::fromMillisNoError(a.toMillis() + b);
  }

  // We only need to capture the time zone session config if we are operating on
  // a timestamp and a IntervalYearMonth.
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Timestamp>*,
      const arg_type<IntervalYearMonth>*) {
    sessionTimeZone_ = getTimeZoneFromConfig(config);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Timestamp>& result,
      const arg_type<Timestamp>& timestamp,
      const arg_type<IntervalYearMonth>& interval) {
    result = addToTimestamp(
        timestamp, DateTimeUnit::kMonth, interval, sessionTimeZone_);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone,
      const arg_type<IntervalDayTime>& interval) {
    result = addMillisToTimestampWithTimezone(*timestampWithTimezone, interval);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<TimestampWithTimezone>*,
      const arg_type<IntervalYearMonth>*) {
    this->initializeTimeZoneSupport(config);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone,
      const arg_type<IntervalYearMonth>& interval) {
    result = addToTimestampWithTimezone(
        *timestampWithTimezone,
        DateTimeUnit::kMonth,
        interval,
        *this->renderZone(timestampWithTimezone));
  }

 private:
  // Only set if the parameters are timestamp and IntervalYearMonth.
  const tz::TimeZone* sessionTimeZone_ = nullptr;
};

template <typename T>
struct IntervalPlusTimestamp : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Timestamp>& result,
      const arg_type<IntervalDayTime>& a,
      const arg_type<Timestamp>& b)
#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
      __attribute__((__no_sanitize__("signed-integer-overflow")))
#endif
#endif
  {
    result = Timestamp::fromMillisNoError(a + b.toMillis());
  }

  // We only need to capture the time zone session config if we are operating on
  // a timestamp and a IntervalYearMonth.
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<IntervalYearMonth>*,
      const arg_type<Timestamp>*) {
    sessionTimeZone_ = getTimeZoneFromConfig(config);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Timestamp>& result,
      const arg_type<IntervalYearMonth>& interval,
      const arg_type<Timestamp>& timestamp) {
    result = addToTimestamp(
        timestamp, DateTimeUnit::kMonth, interval, sessionTimeZone_);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<IntervalDayTime>& interval,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    result = addMillisToTimestampWithTimezone(*timestampWithTimezone, interval);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<IntervalYearMonth>*,
      const arg_type<TimestampWithTimezone>*) {
    this->initializeTimeZoneSupport(config);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<IntervalYearMonth>& interval,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    result = addToTimestampWithTimezone(
        *timestampWithTimezone,
        DateTimeUnit::kMonth,
        interval,
        *this->renderZone(timestampWithTimezone));
  }

 private:
  // Only set if the parameters are timestamp and IntervalYearMonth.
  const tz::TimeZone* sessionTimeZone_ = nullptr;
};

template <typename T>
struct TimestampMinusInterval : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Timestamp>& result,
      const arg_type<Timestamp>& a,
      const arg_type<IntervalDayTime>& b)
#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
      __attribute__((__no_sanitize__("signed-integer-overflow")))
#endif
#endif
  {
    result = Timestamp::fromMillisNoError(a.toMillis() - b);
  }

  // We only need to capture the time zone session config if we are operating on
  // a timestamp and a IntervalYearMonth.
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Timestamp>*,
      const arg_type<IntervalYearMonth>*) {
    sessionTimeZone_ = getTimeZoneFromConfig(config);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Timestamp>& result,
      const arg_type<Timestamp>& timestamp,
      const arg_type<IntervalYearMonth>& interval) {
    result = addToTimestamp(
        timestamp, DateTimeUnit::kMonth, -interval, sessionTimeZone_);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone,
      const arg_type<IntervalDayTime>& interval) {
    result =
        addMillisToTimestampWithTimezone(*timestampWithTimezone, -interval);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<TimestampWithTimezone>*,
      const arg_type<IntervalYearMonth>*) {
    this->initializeTimeZoneSupport(config);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone,
      const arg_type<IntervalYearMonth>& interval) {
    result = addToTimestampWithTimezone(
        *timestampWithTimezone,
        DateTimeUnit::kMonth,
        -interval,
        *this->renderZone(timestampWithTimezone));
  }

 private:
  // Only set if the parameters are timestamp and IntervalYearMonth.
  const tz::TimeZone* sessionTimeZone_ = nullptr;
};

template <typename T>
struct DayOfWeekFunction : public InitSessionTimezone<T>,
                           public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  using InitSessionTimezone<T>::initialize;
  using TimestampWithTimezoneSupport<T>::initialize;

  FOLLY_ALWAYS_INLINE int64_t getDayOfWeek(const std::tm& time) {
    return time.tm_wday == 0 ? 7 : time.tm_wday;
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDayOfWeek(getDateTime(timestamp, this->timeZone_));
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Date>& date) {
    result = getDayOfWeek(getDateTime(date));
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getDayOfWeek(getDateTime(timestamp, nullptr));
  }
};

template <typename T>
struct DayOfYearFunction : public InitSessionTimezone<T>,
                           public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  using InitSessionTimezone<T>::initialize;
  using TimestampWithTimezoneSupport<T>::initialize;

  FOLLY_ALWAYS_INLINE int64_t getDayOfYear(const std::tm& time) {
    return time.tm_yday + 1;
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDayOfYear(getDateTime(timestamp, this->timeZone_));
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Date>& date) {
    result = getDayOfYear(getDateTime(date));
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getDayOfYear(getDateTime(timestamp, nullptr));
  }
};

template <typename T>
struct YearOfWeekFunction : public InitSessionTimezone<T>,
                            public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  using InitSessionTimezone<T>::initialize;
  using TimestampWithTimezoneSupport<T>::initialize;

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

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = computeYearOfWeek(getDateTime(timestamp, this->timeZone_));
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Date>& date) {
    result = computeYearOfWeek(getDateTime(date));
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = computeYearOfWeek(getDateTime(timestamp, nullptr));
  }
};

template <typename T>
struct HourFunction : public InitSessionTimezone<T>,
                      public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  using InitSessionTimezone<T>::initialize;
  using TimestampWithTimezoneSupport<T>::initialize;

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, this->timeZone_).tm_hour;
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_hour;
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getDateTime(timestamp, nullptr).tm_hour;
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Time>& time) {
    VELOX_USER_CHECK(
        time >= 0 && time < kMillisInDay,
        "TIME value {} is out of range [0, 86400000)",
        time);
    result = std::chrono::duration_cast<std::chrono::hours>(
                 std::chrono::milliseconds(time))
                 .count();
  }
};

template <typename T>
struct MinuteFunction : public InitSessionTimezone<T>,
                        public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  using InitSessionTimezone<T>::initialize;
  using TimestampWithTimezoneSupport<T>::initialize;

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, this->timeZone_).tm_min;
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_min;
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getDateTime(timestamp, nullptr).tm_min;
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Time>& time) {
    VELOX_USER_CHECK(
        time >= 0 && time < kMillisInDay,
        "TIME value {} is out of valid range [0, 86399999]",
        time);
    auto duration = std::chrono::milliseconds(time);
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    result = minutes.count() % kMinutesInHour;
  }
};

template <typename T>
struct SecondFunction : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = getDateTime(timestamp, nullptr).tm_sec;
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Date>& date) {
    result = getDateTime(date).tm_sec;
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    auto timestamp = this->toTimestamp(timestampWithTimezone);
    result = getDateTime(timestamp, nullptr).tm_sec;
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Time>& time) {
    VELOX_USER_CHECK(
        time >= 0 && time < kMillisInDay,
        "TIME value {} is out of range [0, 86400000)",
        time);
    auto duration = std::chrono::milliseconds(time);
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    result = seconds.count() % kSecondsInMinute;
  }
};

template <typename T>
struct MillisecondFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = timestamp.getNanos() / Timestamp::kNanosecondsInMillisecond;
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Date>& /*date*/) {
    // Dates do not have millisecond granularity.
    result = 0;
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    const auto timestamp = unpackTimestampUtc(*timestampWithTimezone);
    result = timestamp.getNanos() / Timestamp::kNanosecondsInMillisecond;
  }

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const arg_type<Time>& time) {
    VELOX_USER_CHECK(
        time >= 0 && time < kMillisInDay,
        "TIME value {} is out of range [0, 86400000)",
        time);
    auto time_duration = std::chrono::milliseconds(time);
    result = (time_duration % std::chrono::seconds(1)).count();
  }
};

namespace {
inline bool isDateUnit(const DateTimeUnit unit) {
  return unit == DateTimeUnit::kDay || unit == DateTimeUnit::kMonth ||
      unit == DateTimeUnit::kQuarter || unit == DateTimeUnit::kYear ||
      unit == DateTimeUnit::kWeek;
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

inline std::optional<DateTimeUnit> getTimestampUnit(
    const StringView& unitString) {
  std::optional<DateTimeUnit> unit =
      fromDateTimeUnitString(unitString, /*throwIfInvalid=*/false);
  VELOX_USER_CHECK(
      !(unit.has_value() && unit.value() == DateTimeUnit::kMillisecond),
      "{} is not a valid TIMESTAMP field",
      unitString);

  return unit;
}

inline std::optional<DateTimeUnit> getTimeUnit(
    const StringView& unitString,
    bool throwIfInvalid = true) {
  std::optional<DateTimeUnit> unit =
      fromDateTimeUnitString(unitString, /*throwIfInvalid=*/false);

  // Presto does not support microseconds for TIME type operations.
  // Only millisecond, second, minute, and hour are valid TIME fields.
  // See: presto-main-base/.../DateTimeFunctions.java:getTimeField()
  if (unit.has_value() && isTimeUnit(unit.value()) &&
      unit.value() != DateTimeUnit::kMicrosecond) {
    return unit;
  }

  if (throwIfInvalid) {
    VELOX_USER_FAIL("{} is not a valid TIME field", unitString);
  }
  return std::nullopt;
}

inline void checkValueInInt32Range(int64_t value) {
  if (value != static_cast<int32_t>(value)) {
    VELOX_UNSUPPORTED(
        "Value should be in range [{}, {}]",
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max());
  }
}

} // namespace

template <typename T>
struct DateTruncFunction : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  const tz::TimeZone* timeZone_ = nullptr;
  std::optional<DateTimeUnit> unit_;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* unitString,
      const arg_type<Timestamp>* /*timestamp*/) {
    timeZone_ = getTimeZoneFromConfig(config);

    if (unitString != nullptr) {
      unit_ = getTimestampUnit(*unitString);
    }
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* unitString,
      const arg_type<Date>* /*date*/) {
    if (unitString != nullptr) {
      unit_ = getDateUnit(*unitString, false);
    }
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* unitString,
      const arg_type<TimestampWithTimezone>* /*timestamp*/) {
    if (unitString != nullptr) {
      unit_ = getTimestampUnit(*unitString);
      if (unit_.has_value() && unit_.value() != DateTimeUnit::kSecond) {
        this->initializeTimeZoneSupport(config);
      }
    } else {
      this->initializeTimeZoneSupport(
          config,
          TimestampWithTimeZoneRenderZone::SessionZoneResolution::kOnDemand);
    }
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* unitString,
      const arg_type<Time>* /*time*/) {
    if (unitString != nullptr) {
      unit_ = getTimeUnit(*unitString);
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Timestamp>& result,
      const arg_type<Varchar>& unitString,
      const arg_type<Timestamp>& timestamp) {
    DateTimeUnit unit;
    if (unit_.has_value()) {
      unit = unit_.value();
    } else {
      unit = getTimestampUnit(unitString).value();
    }
    result = truncateTimestamp(timestamp, unit, timeZone_);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Varchar>& unitString,
      const arg_type<Date>& date) {
    DateTimeUnit unit = unit_.has_value()
        ? unit_.value()
        : getDateUnit(unitString, true).value();

    if (unit == DateTimeUnit::kDay) {
      result = date;
      return;
    }

    auto dateTime = getDateTime(date);
    adjustDateTime(dateTime, unit);

    result = Timestamp::calendarUtcToEpoch(dateTime) / kSecondsInDay;
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<Varchar>& unitString,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    DateTimeUnit unit;
    if (unit_.has_value()) {
      unit = unit_.value();
    } else {
      unit = getTimestampUnit(unitString).value();
    }

    if (unit == DateTimeUnit::kSecond) {
      const auto utcTimestamp = unpackTimestampUtc(*timestampWithTimezone);
      result = pack(
          utcTimestamp.getSeconds() * 1000,
          unpackZoneKeyId(*timestampWithTimezone));
      return;
    }

    const auto timestamp = this->toTimestamp(timestampWithTimezone);
    auto dateTime = getDateTime(timestamp, nullptr);
    adjustDateTime(dateTime, unit);

    uint64_t resultMillis;

    if (unit < DateTimeUnit::kDay) {
      // If the unit is less than a day, we compute the difference in
      // milliseconds between the local timestamp and the truncated local
      // timestamp. We then subtract this difference from the UTC timestamp,
      // this handles things like ambiguous timestamps in the local time zone.
      const auto millisDifference =
          timestamp.toMillis() - Timestamp::calendarUtcToEpoch(dateTime) * 1000;

      resultMillis = unpackMillisUtc(*timestampWithTimezone) - millisDifference;
    } else {
      // If the unit is at least a day, we do the truncation on the local
      // timestamp and then convert it to a system time directly. This handles
      // cases like when a time zone has daylight savings time, a "day" can be
      // 25 or 23 hours at the transition points.
      auto updatedTimestamp =
          Timestamp::fromMillis(Timestamp::calendarUtcToEpoch(dateTime) * 1000);
      updatedTimestamp.toGMT(*this->renderZone(timestampWithTimezone));

      resultMillis = updatedTimestamp.toMillis();
    }

    result = pack(resultMillis, unpackZoneKeyId(*timestampWithTimezone));
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Time>& result,
      const arg_type<Varchar>& unitString,
      const arg_type<Time>& time) {
    DateTimeUnit unit;
    if (unit_.has_value()) {
      unit = unit_.value();
    } else {
      unit = getTimeUnit(unitString).value();
    }
    result = truncateTime(time, unit);
  }
};

template <typename T>
struct DateAddFunction : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  const tz::TimeZone* sessionTimeZone_ = nullptr;
  std::optional<DateTimeUnit> unit_ = std::nullopt;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* unitString,
      const int64_t* /*value*/,
      const arg_type<Timestamp>* /*timestamp*/) {
    sessionTimeZone_ = getTimeZoneFromConfig(config);
    if (unitString != nullptr) {
      unit_ = fromDateTimeUnitString(*unitString, /*throwIfInvalid=*/true);
    }
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* unitString,
      const int64_t* /*value*/,
      const arg_type<Date>* /*date*/) {
    if (unitString != nullptr) {
      unit_ = getDateUnit(*unitString, false);
    }
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* unitString,
      const int64_t* /*value*/,
      const arg_type<Time>* /*time*/) {
    if (unitString != nullptr) {
      unit_ = getTimeUnit(*unitString, /*throwIfInvalid=*/true);
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Timestamp>& result,
      const arg_type<Varchar>& unitString,
      const int64_t value,
      const arg_type<Timestamp>& timestamp) {
    const auto unit = unit_.has_value()
        ? unit_.value()
        : fromDateTimeUnitString(unitString, /*throwIfInvalid=*/true).value();

    checkValueInInt32Range(value);
    result = addToTimestamp(
        unit, static_cast<int32_t>(value), timestamp, sessionTimeZone_);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* unitString,
      const int64_t* /*value*/,
      const arg_type<TimestampWithTimezone>* /*timestamp*/) {
    if (unitString != nullptr) {
      unit_ = fromDateTimeUnitString(*unitString, /*throwIfInvalid=*/true);
      if (unit_.value() >= DateTimeUnit::kDay) {
        this->initializeTimeZoneSupport(config);
      }
    } else {
      this->initializeTimeZoneSupport(
          config,
          TimestampWithTimeZoneRenderZone::SessionZoneResolution::kOnDemand);
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<Varchar>& unitString,
      const int64_t value,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone) {
    const auto unit = unit_.has_value()
        ? unit_.value()
        : fromDateTimeUnitString(unitString, /*throwIfInvalid=*/true).value();

    checkValueInInt32Range(value);

    if (unit < DateTimeUnit::kDay) {
      result = addToTimestampWithTimezone(
          *timestampWithTimezone, unit, static_cast<int32_t>(value));
    } else {
      result = addToTimestampWithTimezone(
          *timestampWithTimezone,
          unit,
          static_cast<int32_t>(value),
          *this->renderZone(timestampWithTimezone));
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Varchar>& unitString,
      const int64_t value,
      const arg_type<Date>& date) {
    DateTimeUnit unit = unit_.has_value()
        ? unit_.value()
        : getDateUnit(unitString, true).value();

    checkValueInInt32Range(value);

    result = addToDate(date, unit, static_cast<int32_t>(value));
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Time>& result,
      const arg_type<Varchar>& unitString,
      const int64_t value,
      const arg_type<Time>& time) {
    DateTimeUnit unit;
    if (unit_.has_value()) {
      unit = unit_.value();
    } else {
      unit = getTimeUnit(unitString, /*throwIfInvalid=*/true).value();
    }

    checkValueInInt32Range(value);

    result = addToTime(unit, static_cast<int32_t>(value), time);
  }
};

template <typename T>
struct DateDiffFunction : public TimestampWithTimezoneSupport<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  const tz::TimeZone* sessionTimeZone_ = nullptr;
  std::optional<DateTimeUnit> unit_ = std::nullopt;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* unitString,
      const arg_type<Timestamp>* /*timestamp1*/,
      const arg_type<Timestamp>* /*timestamp2*/) {
    if (unitString != nullptr) {
      unit_ = fromDateTimeUnitString(*unitString, /*throwIfInvalid=*/true);
    }

    sessionTimeZone_ = getTimeZoneFromConfig(config);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* unitString,
      const arg_type<Date>* /*date1*/,
      const arg_type<Date>* /*date2*/) {
    if (unitString != nullptr) {
      unit_ = getDateUnit(*unitString, false);
    }
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* unitString,
      const arg_type<TimestampWithTimezone>* /*timestampWithTimezone1*/,
      const arg_type<TimestampWithTimezone>* /*timestampWithTimezone2*/) {
    if (unitString != nullptr) {
      unit_ = fromDateTimeUnitString(*unitString, /*throwIfInvalid=*/true);
      if (unit_.value() >= DateTimeUnit::kDay) {
        this->initializeTimeZoneSupport(config);
      }
    } else {
      this->initializeTimeZoneSupport(
          config,
          TimestampWithTimeZoneRenderZone::SessionZoneResolution::kOnDemand);
    }
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* unitString,
      const arg_type<Time>* /*time1*/,
      const arg_type<Time>* /*time2*/) {
    if (unitString != nullptr) {
      unit_ = getTimeUnit(*unitString, /*throwIfInvalid=*/true);
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Varchar>& unitString,
      const arg_type<Timestamp>& timestamp1,
      const arg_type<Timestamp>& timestamp2) {
    const auto unit = unit_.has_value()
        ? unit_.value()
        : fromDateTimeUnitString(unitString, /*throwIfInvalid=*/true).value();
    result = diffTimestamp(unit, timestamp1, timestamp2, sessionTimeZone_);
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Varchar>& unitString,
      const arg_type<Date>& date1,
      const arg_type<Date>& date2) {
    DateTimeUnit unit = unit_.has_value()
        ? unit_.value()
        : getDateUnit(unitString, true).value();

    result = diffDate(unit, date1, date2);
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Varchar>& unitString,
      const arg_type<TimestampWithTimezone>& timestampWithTz1,
      const arg_type<TimestampWithTimezone>& timestampWithTz2) {
    const auto unit = unit_.has_value()
        ? unit_.value()
        : fromDateTimeUnitString(unitString, /*throwIfInvalid=*/true).value();

    if (unit < DateTimeUnit::kDay) {
      result = diffTimestamp(
          unit,
          unpackTimestampUtc(*timestampWithTz1),
          unpackTimestampUtc(*timestampWithTz2));
    } else {
      // Legacy uses the first argument's zone; the session zone otherwise.
      // Normalizing to UTC is incorrect across daylight saving boundaries.
      result = diffTimestampWithTimeZone(
          unit,
          *timestampWithTz1,
          *timestampWithTz2,
          *this->renderZone(timestampWithTz1));
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<Varchar>& unitString,
      const arg_type<Time>& time1,
      const arg_type<Time>& time2) {
    DateTimeUnit unit;
    if (unit_.has_value()) {
      unit = unit_.value();
    } else {
      unit = getTimeUnit(unitString, /*throwIfInvalid=*/true).value();
    }

    result = diffTime(unit, time1, time2);
  }
};

template <typename T>
struct TimeZoneHourFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& input) {
    auto offset = getTimeZoneOffsetSeconds(*input);
    result = offset / 3600;
  }
};

template <typename T>
struct TimeZoneMinuteFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& input) {
    auto offset = getTimeZoneOffsetSeconds(*input);
    result = (offset / 60) % 60;
  }
};

template <typename T>
struct AtTimezoneFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::optional<int64_t> targetTimezoneID_;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<TimestampWithTimezone>* /*tsWithTz*/,
      const arg_type<Varchar>* timezone) {
    if (timezone) {
      targetTimezoneID_ = tz::getTimeZoneID(
          std::string_view(timezone->data(), timezone->size()));
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<TimestampWithTimezone>& result,
      const arg_type<TimestampWithTimezone>& tsWithTz,
      const arg_type<Varchar>& timezone) {
    const auto inputMs = unpackMillisUtc(*tsWithTz);
    const auto targetTimezoneID = targetTimezoneID_.has_value()
        ? targetTimezoneID_.value()
        : tz::getTimeZoneID(std::string_view(timezone.data(), timezone.size()));

    // Input and output TimestampWithTimezones should not contain
    // different timestamp values - solely timezone ID should differ between the
    // two, as timestamp is stored as a UTC offset. The timestamp is then
    // resolved to the respective timezone at the time of display.
    result = pack(inputMs, targetTimezoneID);
  }
};

/// Converts a TIMESTAMP WITH TIME ZONE to the wall clock read in the target
/// zone, dropping the zone. The zone the input carries is ignored; only its
/// instant is used.
template <typename T>
struct AtTimezoneConvertToTimestampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Target zone when the timezone argument is constant; null otherwise.
  const tz::TimeZone* targetTimeZone_{nullptr};

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<TimestampWithTimezone>* /*timestampWithTimezone*/,
      const arg_type<Varchar>* timezone) {
    if (timezone) {
      targetTimeZone_ =
          tz::locateZone(std::string_view(timezone->data(), timezone->size()));
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Timestamp>& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimezone,
      const arg_type<Varchar>& timezone) {
    const auto* targetTimeZone = targetTimeZone_ != nullptr
        ? targetTimeZone_
        : tz::locateZone(std::string_view(timezone.data(), timezone.size()));

    Timestamp timestamp = unpackTimestampUtc(*timestampWithTimezone);
    timestamp.toTimezone(*targetTimeZone);
    result = timestamp;
  }
};

} // namespace facebook::velox::functions
