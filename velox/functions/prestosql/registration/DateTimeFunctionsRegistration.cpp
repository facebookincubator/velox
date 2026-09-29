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

#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/DateTimeFunctions.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneRegistration.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneRegistration.h"

namespace facebook::velox::functions {
namespace {

// Register timestamp + interval and interval + timestamp
// functions for specified TTimestamp type and 2 supported interval types
// (IntervalDayTime and IntervalYearMonth).
// @tparam TTimestamp Timestamp or TimestampWithTimezone.
template <typename TTimestamp>
void registerTimestampPlusInterval(
    const std::string& name,
    std::string_view defaultOwner) {
  registerFunction<
      TimestampPlusInterval,
      TTimestamp,
      TTimestamp,
      IntervalDayTime>({name}, {}, true, defaultOwner);
  registerFunction<
      TimestampPlusInterval,
      TTimestamp,
      TTimestamp,
      IntervalYearMonth>({name}, {}, true, defaultOwner);
  registerFunction<
      IntervalPlusTimestamp,
      TTimestamp,
      IntervalDayTime,
      TTimestamp>({name}, {}, true, defaultOwner);
  registerFunction<
      IntervalPlusTimestamp,
      TTimestamp,
      IntervalYearMonth,
      TTimestamp>({name}, {}, true, defaultOwner);
}

// Register timestamp - IntervalYearMonth and timestamp - IntervalDayTime
// functions for specified TTimestamp type.
// @tparam TTimestamp Timestamp or TimestampWithTimezone.
template <typename TTimestamp>
void registerTimestampMinusInterval(
    const std::string& name,
    std::string_view defaultOwner) {
  registerFunction<
      TimestampMinusInterval,
      TTimestamp,
      TTimestamp,
      IntervalDayTime>({name}, {}, true, defaultOwner);
  registerFunction<
      TimestampMinusInterval,
      TTimestamp,
      TTimestamp,
      IntervalYearMonth>({name}, {}, true, defaultOwner);
}

void registerFromUnixtime(
    const std::string& name,
    std::string_view defaultOwner) {
  registerFunction<FromUnixtimeFunction, Timestamp, double>(
      {name}, {}, true, defaultOwner);
  registerFunction<
      FromUnixtimeFunction,
      TimestampWithTimezone,
      double,
      Varchar>({name}, {}, true, defaultOwner);
  registerFunction<
      FromUnixtimeFunction,
      TimestampWithTimezone,
      double,
      int64_t,
      int64_t>({name}, {}, true, defaultOwner);
}

void registerSimpleFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  // Date time functions.
  registerFunction<ToUnixtimeFunction, double, Timestamp>(
      {prefix + "to_unixtime"}, {}, true, defaultOwner);
  registerFunction<ToUnixtimeFunction, double, TimestampWithTimezone>(
      {prefix + "to_unixtime"}, {}, true, defaultOwner);

  registerFromUnixtime(prefix + "from_unixtime", defaultOwner);
  registerFunction<CurrentTimeFunction, TimeWithTimezone>(
      {prefix + "current_time"}, {}, true, defaultOwner);
  registerFunction<CurrentTimezoneFunction, Varchar>(
      {prefix + "current_timezone"}, {}, true, defaultOwner);
  registerFunction<CurrentTimestampFunction, TimestampWithTimezone>(
      {prefix + "current_timestamp", prefix + "now"}, {}, true, defaultOwner);

  registerFunction<DateFunction, Date, Varchar>(
      {prefix + "date"}, {}, true, defaultOwner);
  registerFunction<DateFunction, Date, Timestamp>(
      {prefix + "date"}, {}, true, defaultOwner);
  registerFunction<DateFunction, Date, TimestampWithTimezone>(
      {prefix + "date"}, {}, true, defaultOwner);
  registerFunction<TimeZoneHourFunction, int64_t, TimestampWithTimezone>(
      {prefix + "timezone_hour"}, {}, true, defaultOwner);

  registerFunction<TimeZoneMinuteFunction, int64_t, TimestampWithTimezone>(
      {prefix + "timezone_minute"}, {}, true, defaultOwner);

  registerFunction<YearFunction, int64_t, Timestamp>(
      {prefix + "year"}, {}, true, defaultOwner);
  registerFunction<YearFunction, int64_t, Date>(
      {prefix + "year"}, {}, true, defaultOwner);
  registerFunction<YearFunction, int64_t, TimestampWithTimezone>(
      {prefix + "year"}, {}, true, defaultOwner);
  registerFunction<YearFromIntervalFunction, int64_t, IntervalYearMonth>(
      {prefix + "year"}, {}, true, defaultOwner);

  registerFunction<WeekFunction, int64_t, Timestamp>(
      {prefix + "week", prefix + "week_of_year"}, {}, true, defaultOwner);
  registerFunction<WeekFunction, int64_t, Date>(
      {prefix + "week", prefix + "week_of_year"}, {}, true, defaultOwner);
  registerFunction<WeekFunction, int64_t, TimestampWithTimezone>(
      {prefix + "week", prefix + "week_of_year"}, {}, true, defaultOwner);
  registerFunction<QuarterFunction, int64_t, Timestamp>(
      {prefix + "quarter"}, {}, true, defaultOwner);
  registerFunction<QuarterFunction, int64_t, Date>(
      {prefix + "quarter"}, {}, true, defaultOwner);
  registerFunction<QuarterFunction, int64_t, TimestampWithTimezone>(
      {prefix + "quarter"}, {}, true, defaultOwner);

  registerFunction<MonthFunction, int64_t, Timestamp>(
      {prefix + "month"}, {}, true, defaultOwner);
  registerFunction<MonthFunction, int64_t, Date>(
      {prefix + "month"}, {}, true, defaultOwner);
  registerFunction<MonthFunction, int64_t, TimestampWithTimezone>(
      {prefix + "month"}, {}, true, defaultOwner);
  registerFunction<MonthFromIntervalFunction, int64_t, IntervalYearMonth>(
      {prefix + "month"}, {}, true, defaultOwner);

  registerFunction<DayFunction, int64_t, Timestamp>(
      {prefix + "day", prefix + "day_of_month"}, {}, true, defaultOwner);
  registerFunction<DayFunction, int64_t, Date>(
      {prefix + "day", prefix + "day_of_month"}, {}, true, defaultOwner);
  registerFunction<DayFromIntervalFunction, int64_t, IntervalDayTime>(
      {prefix + "day", prefix + "day_of_month"}, {}, true, defaultOwner);

  registerFunction<DateMinusInterval, Date, Date, IntervalDayTime>(
      {prefix + "minus"}, {}, true, defaultOwner);
  registerFunction<DateMinusInterval, Date, Date, IntervalYearMonth>(
      {prefix + "minus"}, {}, true, defaultOwner);
  registerFunction<DatePlusInterval, Date, Date, IntervalDayTime>(
      {prefix + "plus"}, {}, true, defaultOwner);
  registerFunction<DatePlusInterval, Date, Date, IntervalYearMonth>(
      {prefix + "plus"}, {}, true, defaultOwner);

  registerTimestampPlusInterval<Timestamp>({prefix + "plus"}, defaultOwner);
  registerTimestampMinusInterval<Timestamp>({prefix + "minus"}, defaultOwner);
  registerTimestampPlusInterval<TimestampWithTimezone>(
      {prefix + "plus"}, defaultOwner);
  registerTimestampMinusInterval<TimestampWithTimezone>(
      {prefix + "minus"}, defaultOwner);

  // Register Time + Interval and Interval + Time functions
  registerFunction<TimePlusInterval, Time, Time, IntervalDayTime>(
      {prefix + "plus"}, {}, true, defaultOwner);

  registerFunction<IntervalPlusTime, Time, IntervalDayTime, Time>(
      {prefix + "plus"}, {}, true, defaultOwner);

  // Register Time - Interval function
  registerFunction<TimeMinusInterval, Time, Time, IntervalDayTime>(
      {prefix + "minus"}, {}, true, defaultOwner);

  // Register Time - Time function (returns IntervalDayTime)
  registerFunction<TimeMinusFunction, IntervalDayTime, Time, Time>(
      {prefix + "minus"}, {}, true, defaultOwner);

  // Use optimized vector function for Time + IntervalYearMonth (identity
  // function)
  exec::registerVectorFunction(
      prefix + "plus",
      TimeIntervalYearMonthVectorFunction::signaturesPlus(),
      std::make_unique<TimeIntervalYearMonthVectorFunction>(),
      {},
      /*overwrite=*/true,
      defaultOwner);

  // Use optimized vector function for Time - IntervalYearMonth (identity
  // function). Only supports (time, interval), not (interval, time).
  exec::registerVectorFunction(
      prefix + "minus",
      TimeIntervalYearMonthVectorFunction::signaturesMinus(),
      std::make_unique<TimeIntervalYearMonthVectorFunction>(),
      {},
      /*overwrite=*/true,
      defaultOwner);

  registerFunction<
      TimestampMinusFunction,
      IntervalDayTime,
      Timestamp,
      Timestamp>({prefix + "minus"}, {}, true, defaultOwner);

  registerFunction<
      TimestampMinusFunction,
      IntervalDayTime,
      TimestampWithTimezone,
      TimestampWithTimezone>({prefix + "minus"}, {}, true, defaultOwner);

  registerFunction<DayFunction, int64_t, TimestampWithTimezone>(
      {prefix + "day", prefix + "day_of_month"}, {}, true, defaultOwner);
  registerFunction<DayOfWeekFunction, int64_t, Timestamp>(
      {prefix + "dow", prefix + "day_of_week"}, {}, true, defaultOwner);
  registerFunction<DayOfWeekFunction, int64_t, Date>(
      {prefix + "dow", prefix + "day_of_week"}, {}, true, defaultOwner);
  registerFunction<DayOfWeekFunction, int64_t, TimestampWithTimezone>(
      {prefix + "dow", prefix + "day_of_week"}, {}, true, defaultOwner);
  registerFunction<DayOfYearFunction, int64_t, Timestamp>(
      {prefix + "doy", prefix + "day_of_year"}, {}, true, defaultOwner);
  registerFunction<DayOfYearFunction, int64_t, Date>(
      {prefix + "doy", prefix + "day_of_year"}, {}, true, defaultOwner);
  registerFunction<DayOfYearFunction, int64_t, TimestampWithTimezone>(
      {prefix + "doy", prefix + "day_of_year"}, {}, true, defaultOwner);
  registerFunction<YearOfWeekFunction, int64_t, Timestamp>(
      {prefix + "yow", prefix + "year_of_week"}, {}, true, defaultOwner);
  registerFunction<YearOfWeekFunction, int64_t, Date>(
      {prefix + "yow", prefix + "year_of_week"}, {}, true, defaultOwner);
  registerFunction<YearOfWeekFunction, int64_t, TimestampWithTimezone>(
      {prefix + "yow", prefix + "year_of_week"}, {}, true, defaultOwner);

  registerFunction<HourFunction, int64_t, Timestamp>(
      {prefix + "hour"}, {}, true, defaultOwner);
  registerFunction<HourFunction, int64_t, Date>(
      {prefix + "hour"}, {}, true, defaultOwner);
  registerFunction<HourFunction, int64_t, TimestampWithTimezone>(
      {prefix + "hour"}, {}, true, defaultOwner);
  registerFunction<HourFunction, int64_t, Time>(
      {prefix + "hour"}, {}, true, defaultOwner);
  registerFunction<HourFromIntervalFunction, int64_t, IntervalDayTime>(
      {prefix + "hour"}, {}, true, defaultOwner);

  registerFunction<LastDayOfMonthFunction, Date, Timestamp>(
      {prefix + "last_day_of_month"}, {}, true, defaultOwner);
  registerFunction<LastDayOfMonthFunction, Date, Date>(
      {prefix + "last_day_of_month"}, {}, true, defaultOwner);
  registerFunction<LastDayOfMonthFunction, Date, TimestampWithTimezone>(
      {prefix + "last_day_of_month"}, {}, true, defaultOwner);

  registerFunction<MinuteFunction, int64_t, Timestamp>(
      {prefix + "minute"}, {}, true, defaultOwner);
  registerFunction<MinuteFunction, int64_t, Date>(
      {prefix + "minute"}, {}, true, defaultOwner);
  registerFunction<MinuteFunction, int64_t, TimestampWithTimezone>(
      {prefix + "minute"}, {}, true, defaultOwner);
  registerFunction<MinuteFunction, int64_t, Time>(
      {prefix + "minute"}, {}, true, defaultOwner);
  registerFunction<MinuteFromIntervalFunction, int64_t, IntervalDayTime>(
      {prefix + "minute"}, {}, true, defaultOwner);

  registerFunction<SecondFunction, int64_t, Timestamp>(
      {prefix + "second"}, {}, true, defaultOwner);
  registerFunction<SecondFunction, int64_t, Date>(
      {prefix + "second"}, {}, true, defaultOwner);
  registerFunction<SecondFunction, int64_t, TimestampWithTimezone>(
      {prefix + "second"}, {}, true, defaultOwner);
  registerFunction<SecondFunction, int64_t, Time>(
      {prefix + "second"}, {}, true, defaultOwner);
  registerFunction<SecondFromIntervalFunction, int64_t, IntervalDayTime>(
      {prefix + "second"}, {}, true, defaultOwner);

  registerFunction<MillisecondFunction, int64_t, Timestamp>(
      {prefix + "millisecond"}, {}, true, defaultOwner);
  registerFunction<MillisecondFunction, int64_t, Date>(
      {prefix + "millisecond"}, {}, true, defaultOwner);
  registerFunction<MillisecondFunction, int64_t, TimestampWithTimezone>(
      {prefix + "millisecond"}, {}, true, defaultOwner);
  registerFunction<MillisecondFunction, int64_t, Time>(
      {prefix + "millisecond"}, {}, true, defaultOwner);
  registerFunction<MillisecondFromIntervalFunction, int64_t, IntervalDayTime>(
      {prefix + "millisecond"}, {}, true, defaultOwner);

  registerFunction<DateTruncFunction, Timestamp, Varchar, Timestamp>(
      {prefix + "date_trunc"}, {}, true, defaultOwner);
  registerFunction<DateTruncFunction, Date, Varchar, Date>(
      {prefix + "date_trunc"}, {}, true, defaultOwner);
  registerFunction<
      DateTruncFunction,
      TimestampWithTimezone,
      Varchar,
      TimestampWithTimezone>({prefix + "date_trunc"}, {}, true, defaultOwner);
  registerFunction<DateTruncFunction, Time, Varchar, Time>(
      {prefix + "date_trunc"}, {}, true, defaultOwner);
  registerFunction<DateAddFunction, Date, Varchar, int64_t, Date>(
      {prefix + "date_add"}, {}, true, defaultOwner);
  registerFunction<DateAddFunction, Timestamp, Varchar, int64_t, Timestamp>(
      {prefix + "date_add"}, {}, true, defaultOwner);
  registerFunction<
      DateAddFunction,
      TimestampWithTimezone,
      Varchar,
      int64_t,
      TimestampWithTimezone>({prefix + "date_add"}, {}, true, defaultOwner);
  registerFunction<DateAddFunction, Time, Varchar, int64_t, Time>(
      {prefix + "date_add"}, {}, true, defaultOwner);
  registerFunction<DateDiffFunction, int64_t, Varchar, Date, Date>(
      {prefix + "date_diff"}, {}, true, defaultOwner);
  registerFunction<DateDiffFunction, int64_t, Varchar, Timestamp, Timestamp>(
      {prefix + "date_diff"}, {}, true, defaultOwner);
  registerFunction<
      DateDiffFunction,
      int64_t,
      Varchar,
      TimestampWithTimezone,
      TimestampWithTimezone>({prefix + "date_diff"}, {}, true, defaultOwner);
  registerFunction<DateDiffFunction, int64_t, Varchar, Time, Time>(
      {prefix + "date_diff"}, {}, true, defaultOwner);
  registerFunction<DateFormatFunction, Varchar, Timestamp, Varchar>(
      {prefix + "date_format"}, {}, true, defaultOwner);
  registerFunction<DateFormatFunction, Varchar, TimestampWithTimezone, Varchar>(
      {prefix + "date_format"}, {}, true, defaultOwner);
  registerFunction<FormatDateTimeFunction, Varchar, Timestamp, Varchar>(
      {prefix + "format_datetime"}, {}, true, defaultOwner);
  registerFunction<
      FormatDateTimeFunction,
      Varchar,
      TimestampWithTimezone,
      Varchar>({prefix + "format_datetime"}, {}, true, defaultOwner);
  registerFunction<
      ParseDateTimeFunction,
      TimestampWithTimezone,
      Varchar,
      Varchar>({prefix + "parse_datetime"}, {}, true, defaultOwner);
  registerFunction<DateParseFunction, Timestamp, Varchar, Varchar>(
      {prefix + "date_parse"}, {}, true, defaultOwner);
  registerFunction<FromIso8601Date, Date, Varchar>(
      {prefix + "from_iso8601_date"}, {}, true, defaultOwner);
  registerFunction<FromIso8601Timestamp, TimestampWithTimezone, Varchar>(
      {prefix + "from_iso8601_timestamp"}, {}, true, defaultOwner);
  registerFunction<CurrentDateFunction, Date>(
      {prefix + "current_date"}, {}, true, defaultOwner);

  registerFunction<ToISO8601Function, Varchar, Date>(
      {prefix + "to_iso8601"}, {}, true, defaultOwner);
  registerFunction<ToISO8601Function, Varchar, Timestamp>(
      {prefix + "to_iso8601"}, {}, true, defaultOwner);
  registerFunction<ToISO8601Function, Varchar, TimestampWithTimezone>(
      {prefix + "to_iso8601"}, {}, true, defaultOwner);

  registerFunction<
      AtTimezoneFunction,
      TimestampWithTimezone,
      TimestampWithTimezone,
      Varchar>({prefix + "at_timezone"}, {}, true, defaultOwner);

  registerFunction<
      AtTimezoneTimeWithTimezoneFunction,
      TimeWithTimezone,
      TimeWithTimezone,
      Varchar>({prefix + "at_timezone"}, {}, true, defaultOwner);

  registerFunction<
      AtTimezoneConvertToTimestampFunction,
      Timestamp,
      TimestampWithTimezone,
      Varchar>({prefix + "at_timezone_convert"});

  registerFunction<
      AtTimezoneTimeWithTimezoneFunction,
      TimeWithTimezone,
      TimeWithTimezone,
      Varchar>({prefix + "at_timezone_convert"});

  registerFunction<ToMillisecondFunction, int64_t, IntervalDayTime>(
      {prefix + "to_milliseconds"}, {}, true, defaultOwner);

  registerFunction<XxHash64DateFunction, int64_t, Date>(
      {prefix + "xxhash64_internal"}, {}, true, defaultOwner);
  registerFunction<XxHash64TimestampFunction, int64_t, Timestamp>(
      {prefix + "xxhash64_internal"}, {}, true, defaultOwner);
  registerFunction<XxHash64TimeFunction, int64_t, Time>(
      {prefix + "xxhash64_internal"}, {}, true, defaultOwner);

  registerFunction<ParseDurationFunction, IntervalDayTime, Varchar>(
      {prefix + "parse_duration"}, {}, true, defaultOwner);

  registerFunction<LocalTimeFunction, Time>(
      {prefix + "localtime"}, {}, true, defaultOwner);
  registerFunction<LocalTimestampFunction, Timestamp>(
      {prefix + "localtimestamp"}, {}, true, defaultOwner);
}
} // namespace

void registerDateTimeFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerTimestampWithTimeZoneType();
  registerTimeWithTimezoneType();
  registerSimpleFunctions(prefix, defaultOwner);
}
} // namespace facebook::velox::functions
