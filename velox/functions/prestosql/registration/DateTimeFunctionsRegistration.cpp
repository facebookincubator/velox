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
void registerSimpleFunctions() {
  // Date time functions.
  registerFunction<ToUnixtimeFunction, double, Timestamp>({"to_unixtime"});
  registerFunction<ToUnixtimeFunction, double, TimestampWithTimezone>(
      {"to_unixtime"});
  registerFunction<FromUnixtimeFunction, Timestamp, double>({"from_unixtime"});

  registerFunction<YearFunction, int64_t, Timestamp>({"year"});
  registerFunction<YearFunction, int64_t, Date>({"year"});
  registerFunction<YearFunction, int64_t, TimestampWithTimezone>({"year"});
  registerFunction<WeekFunction, int64_t, Timestamp>({"week", "week_of_year"});
  registerFunction<WeekFunction, int64_t, Date>({"week", "week_of_year"});
  registerFunction<WeekFunction, int64_t, TimestampWithTimezone>(
      {"week", "week_of_year"});
  registerFunction<QuarterFunction, int64_t, Timestamp>({"quarter"});
  registerFunction<QuarterFunction, int64_t, Date>({"quarter"});
  registerFunction<QuarterFunction, int64_t, TimestampWithTimezone>(
      {"quarter"});
  registerFunction<MonthFunction, int64_t, Timestamp>({"month"});
  registerFunction<MonthFunction, int64_t, Date>({"month"});
  registerFunction<MonthFunction, int64_t, TimestampWithTimezone>({"month"});
  registerFunction<DayFunction, int64_t, Timestamp>({"day", "day_of_month"});
  registerFunction<DayFunction, int64_t, Date>({"day", "day_of_month"});
  registerFunction<DateMinusIntervalDayTime, Date, Date, IntervalDayTime>(
      {"minus"});
  registerFunction<DatePlusIntervalDayTime, Date, Date, IntervalDayTime>(
      {"plus"});
  registerFunction<DayFunction, int64_t, TimestampWithTimezone>(
      {"day", "day_of_month"});
  registerFunction<DayOfWeekFunction, int64_t, Timestamp>(
      {"dow", "day_of_week"});
  registerFunction<DayOfWeekFunction, int64_t, Date>({"dow", "day_of_week"});
  registerFunction<DayOfWeekFunction, int64_t, TimestampWithTimezone>(
      {"dow", "day_of_week"});
  registerFunction<DayOfYearFunction, int64_t, Timestamp>(
      {"doy", "day_of_year"});
  registerFunction<DayOfYearFunction, int64_t, Date>({"doy", "day_of_year"});
  registerFunction<DayOfYearFunction, int64_t, TimestampWithTimezone>(
      {"doy", "day_of_year"});
  registerFunction<YearOfWeekFunction, int64_t, Timestamp>(
      {"yow", "year_of_week"});
  registerFunction<YearOfWeekFunction, int64_t, Date>({"yow", "year_of_week"});
  registerFunction<YearOfWeekFunction, int64_t, TimestampWithTimezone>(
      {"yow", "year_of_week"});
  registerFunction<HourFunction, int64_t, Timestamp>({"hour"});
  registerFunction<HourFunction, int64_t, Date>({"hour"});
  registerFunction<HourFunction, int64_t, TimestampWithTimezone>({"hour"});
  registerFunction<MinuteFunction, int64_t, Timestamp>({"minute"});
  registerFunction<MinuteFunction, int64_t, Date>({"minute"});
  registerFunction<MinuteFunction, int64_t, TimestampWithTimezone>({"minute"});
  registerFunction<SecondFunction, int64_t, Timestamp>({"second"});
  registerFunction<SecondFunction, int64_t, Date>({"second"});
  registerFunction<SecondFunction, int64_t, TimestampWithTimezone>({"second"});
  registerFunction<MillisecondFunction, int64_t, Timestamp>({"millisecond"});
  registerFunction<MillisecondFunction, int64_t, Date>({"millisecond"});
  registerFunction<MillisecondFunction, int64_t, TimestampWithTimezone>(
      {"millisecond"});
  registerFunction<DateTruncFunction, Timestamp, Varchar, Timestamp>(
      {"date_trunc"});
  registerFunction<DateTruncFunction, Date, Varchar, Date>({"date_trunc"});
  registerFunction<
      DateTruncFunction,
      TimestampWithTimezone,
      Varchar,
      TimestampWithTimezone>({"date_trunc"});
  registerFunction<DateAddFunction, Date, Varchar, int64_t, Date>({"date_add"});
  registerFunction<DateAddFunction, Timestamp, Varchar, int64_t, Timestamp>(
      {"date_add"});
  registerFunction<DateDiffFunction, int64_t, Varchar, Date, Date>(
      {"date_diff"});
  registerFunction<DateDiffFunction, int64_t, Varchar, Timestamp, Timestamp>(
      {"date_diff"});
  registerFunction<DateFormatFunction, Varchar, Timestamp, Varchar>(
      {"date_format"});
  registerFunction<DateFormatFunction, Varchar, TimestampWithTimezone, Varchar>(
      {"date_format"});
  registerFunction<FormatDateTimeFunction, Varchar, Timestamp, Varchar>(
      {"format_datetime"});
  registerFunction<
      ParseDateTimeFunction,
      TimestampWithTimezone,
      Varchar,
      Varchar>({"parse_datetime"});
  registerFunction<DateParseFunction, Timestamp, Varchar, Varchar>(
      {"date_parse"});
}
} // namespace

void registerDateTimeFunctions() {
  registerTimestampWithTimeZoneType();

  registerSimpleFunctions();
  VELOX_REGISTER_VECTOR_FUNCTION(udf_from_unixtime, "from_unixtime");
}
} // namespace facebook::velox::functions
