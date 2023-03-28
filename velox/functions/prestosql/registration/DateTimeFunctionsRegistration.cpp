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

#include "velox/expression/VectorFunction.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/DateTimeFunctions.h"

namespace facebook::velox::functions {
namespace {
void registerSimpleFunctions(const std::string& prefix) {
  // Date time functions.
  registerFunction<ToUnixtimeFunction, double, Timestamp>(
      {prefix + "to_unixtime"});
  registerFunction<ToUnixtimeFunction, double, TimestampWithTimezone>(
      {prefix + "to_unixtime"});
  registerFunction<FromUnixtimeFunction, Timestamp, double>(
      {prefix + "from_unixtime"});

  registerFunction<YearFunction, int64_t, Timestamp>({prefix + "year"});
  registerFunction<YearFunction, int64_t, Date>({prefix + "year"});
  registerFunction<YearFunction, int64_t, TimestampWithTimezone>(
      {prefix + "year"});
  registerFunction<WeekFunction, int64_t, Timestamp>(
      {prefix + "week", prefix + "week_of_year"});
  registerFunction<WeekFunction, int64_t, Date>(
      {prefix + "week", prefix + "week_of_year"});
  registerFunction<WeekFunction, int64_t, TimestampWithTimezone>(
      {prefix + "week", prefix + "week_of_year"});
  registerFunction<QuarterFunction, int64_t, Timestamp>({prefix + "quarter"});
  registerFunction<QuarterFunction, int64_t, Date>({prefix + "quarter"});
  registerFunction<QuarterFunction, int64_t, TimestampWithTimezone>(
      {prefix + "quarter"});
  registerFunction<MonthFunction, int64_t, Timestamp>({prefix + "month"});
  registerFunction<MonthFunction, int64_t, Date>({prefix + "month"});
  registerFunction<MonthFunction, int64_t, TimestampWithTimezone>(
      {prefix + "month"});
  registerFunction<DayFunction, int64_t, Timestamp>(
      {prefix + "day", prefix + "day_of_month"});
  registerFunction<DayFunction, int64_t, Date>(
      {prefix + "day", prefix + "day_of_month"});
  registerFunction<DateMinusIntervalDayTime, Date, Date, IntervalDayTime>(
      {prefix + "minus"});
  registerFunction<DatePlusIntervalDayTime, Date, Date, IntervalDayTime>(
      {prefix + "plus"});
  registerFunction<DayFunction, int64_t, TimestampWithTimezone>(
      {prefix + "day", prefix + "day_of_month"});
  registerFunction<DayOfWeekFunction, int64_t, Timestamp>(
      {prefix + "dow", prefix + "day_of_week"});
  registerFunction<DayOfWeekFunction, int64_t, Date>(
      {prefix + "dow", prefix + "day_of_week"});
  registerFunction<DayOfWeekFunction, int64_t, TimestampWithTimezone>(
      {prefix + "dow", prefix + "day_of_week"});
  registerFunction<DayOfYearFunction, int64_t, Timestamp>(
      {prefix + "doy", prefix + "day_of_year"});
  registerFunction<DayOfYearFunction, int64_t, Date>(
      {prefix + "doy", prefix + "day_of_year"});
  registerFunction<DayOfYearFunction, int64_t, TimestampWithTimezone>(
      {prefix + "doy", prefix + "day_of_year"});
  registerFunction<YearOfWeekFunction, int64_t, Timestamp>(
      {prefix + "yow", prefix + "year_of_week"});
  registerFunction<YearOfWeekFunction, int64_t, Date>(
      {prefix + "yow", prefix + "year_of_week"});
  registerFunction<YearOfWeekFunction, int64_t, TimestampWithTimezone>(
      {prefix + "yow", prefix + "year_of_week"});
  registerFunction<HourFunction, int64_t, Timestamp>({prefix + "hour"});
  registerFunction<HourFunction, int64_t, Date>({prefix + "hour"});
  registerFunction<HourFunction, int64_t, TimestampWithTimezone>(
      {prefix + "hour"});
  registerFunction<MinuteFunction, int64_t, Timestamp>({prefix + "minute"});
  registerFunction<MinuteFunction, int64_t, Date>({prefix + "minute"});
  registerFunction<MinuteFunction, int64_t, TimestampWithTimezone>(
      {prefix + "minute"});
  registerFunction<SecondFunction, int64_t, Timestamp>({prefix + "second"});
  registerFunction<SecondFunction, int64_t, Date>({prefix + "second"});
  registerFunction<SecondFunction, int64_t, TimestampWithTimezone>(
      {prefix + "second"});
  registerFunction<MillisecondFunction, int64_t, Timestamp>(
      {prefix + "millisecond"});
  registerFunction<MillisecondFunction, int64_t, Date>(
      {prefix + "millisecond"});
  registerFunction<MillisecondFunction, int64_t, TimestampWithTimezone>(
      {prefix + "millisecond"});
  registerFunction<DateTruncFunction, Timestamp, Varchar, Timestamp>(
      {prefix + "date_trunc"});
  registerFunction<DateTruncFunction, Date, Varchar, Date>(
      {prefix + "date_trunc"});
  registerFunction<
      DateTruncFunction,
      TimestampWithTimezone,
      Varchar,
      TimestampWithTimezone>({prefix + "date_trunc"});
  registerFunction<DateAddFunction, Date, Varchar, int64_t, Date>(
      {prefix + "date_add"});
  registerFunction<DateAddFunction, Timestamp, Varchar, int64_t, Timestamp>(
      {prefix + "date_add"});
  registerFunction<DateDiffFunction, int64_t, Varchar, Date, Date>(
      {prefix + "date_diff"});
  registerFunction<DateDiffFunction, int64_t, Varchar, Timestamp, Timestamp>(
      {prefix + "date_diff"});
  registerFunction<DateFormatFunction, Varchar, Timestamp, Varchar>(
      {prefix + "date_format"});
  registerFunction<DateFormatFunction, Varchar, TimestampWithTimezone, Varchar>(
      {prefix + "date_format"});
  registerFunction<FormatDateTimeFunction, Varchar, Timestamp, Varchar>(
      {prefix + "format_datetime"});
  registerFunction<
      ParseDateTimeFunction,
      TimestampWithTimezone,
      Varchar,
      Varchar>({prefix + "parse_datetime"});
  registerFunction<DateParseFunction, Timestamp, Varchar, Varchar>(
      {prefix + "date_parse"});
}
} // namespace

void registerDateTimeFunctions(const std::string& prefix) {
  registerTimestampWithTimeZoneType();

  registerSimpleFunctions(prefix);
  VELOX_REGISTER_VECTOR_FUNCTION(udf_from_unixtime, prefix + "from_unixtime");
}
} // namespace facebook::velox::functions
