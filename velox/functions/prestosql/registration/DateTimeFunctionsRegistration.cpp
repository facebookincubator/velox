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
  registerFunction<ToUnixtimeFunction, double, Timestamp>(
      {"to_unixtime", "to_unix_timestamp"});
  registerFunction<FromUnixtimeFunction, Timestamp, double>({"from_unixtime"});

  registerFunction<YearFunction, int64_t, Timestamp>({"year"});
  registerFunction<YearFunction, int64_t, Date>({"year"});
  registerFunction<QuarterFunction, int64_t, Timestamp>({"quarter"});
  registerFunction<QuarterFunction, int64_t, Date>({"quarter"});
  registerFunction<MonthFunction, int64_t, Timestamp>({"month"});
  registerFunction<MonthFunction, int64_t, Date>({"month"});
  registerFunction<DayFunction, int64_t, Timestamp>({"day", "day_of_month"});
  registerFunction<DayFunction, int64_t, Date>({"day", "day_of_month"});
  registerFunction<DayOfWeekFunction, int64_t, Timestamp>(
      {"dow", "day_of_week"});
  registerFunction<DayOfWeekFunction, int64_t, Date>({"dow", "day_of_week"});
  registerFunction<DayOfYearFunction, int64_t, Timestamp>(
      {"doy", "day_of_year"});
  registerFunction<YearOfWeekFunction, int64_t, Timestamp>(
      {"yow", "year_of_week"});
  registerFunction<YearOfWeekFunction, int64_t, Date>({"yow", "year_of_week"});
  registerFunction<DayOfYearFunction, int64_t, Date>({"doy", "day_of_year"});
  registerFunction<HourFunction, int64_t, Timestamp>({"hour"});
  registerFunction<HourFunction, int64_t, Date>({"hour"});
  registerFunction<MinuteFunction, int64_t, Timestamp>({"minute"});
  registerFunction<MinuteFunction, int64_t, Date>({"minute"});
  registerFunction<SecondFunction, int64_t, Timestamp>({"second"});
  registerFunction<SecondFunction, int64_t, Date>({"second"});
  registerFunction<MillisecondFunction, int64_t, Timestamp>({"millisecond"});
  registerFunction<MillisecondFunction, int64_t, Date>({"millisecond"});
  registerFunction<DateTruncFunction, Timestamp, Varchar, Timestamp>(
      {"date_trunc"});
  registerFunction<DateTruncFunction, Date, Varchar, Date>({"date_trunc"});
  registerFunction<
      ParseDateTimeFunction,
      TimestampWithTimezone,
      Varchar,
      Varchar>({"parse_datetime"});
}
} // namespace

void registerDateTimeFunctions() {
  registerSimpleFunctions();

  registerType("timestamp with time zone", [](auto /*childTypes*/) {
    return TIMESTAMP_WITH_TIME_ZONE();
  });
  VELOX_REGISTER_VECTOR_FUNCTION(udf_from_unixtime, "from_unixtime");
}
} // namespace facebook::velox::functions
