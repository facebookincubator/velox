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

#include "velox/functions/iceberg/DateTimeFunctions.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/iceberg/DateTimeUtil.h"

namespace facebook::velox::functions::iceberg {

// years(input) -> years from 1970
// Input is date or timestamp.
template <typename TExec>
struct YearsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void call(
      int32_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = epochYear(timestamp);
  }

  FOLLY_ALWAYS_INLINE void call(int32_t& result, const arg_type<Date>& date) {
    result = epochYear(date);
  }
};

// months(input) -> months from 1970-01-01
// Input is date or timestamp.
template <typename TExec>
struct MonthsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void call(
      int32_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = epochMonth(timestamp);
  }

  FOLLY_ALWAYS_INLINE void call(int32_t& result, const arg_type<Date>& date) {
    result = epochMonth(date);
  }
};

// days(input) -> days from 1970-01-01
// Input is date or timestamp.
template <typename TExec>
struct DaysFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Timestamp>& timestamp) {
    result = epochDay(timestamp);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Date>& result,
      const arg_type<Date>& date) {
    result = date;
  }
};

// hours(input) -> hours from 1970-01-01 00:00:00
// Input is timestamp.
template <typename TExec>
struct HoursFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void call(
      int32_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = epochHour(timestamp);
  }
};

void registerDateTimeFunctions(const std::string& prefix) {
  registerFunction<YearsFunction, int32_t, Timestamp>({prefix + "years"});
  registerFunction<YearsFunction, int32_t, Date>({prefix + "years"});
  registerFunction<MonthsFunction, int32_t, Timestamp>({prefix + "months"});
  registerFunction<MonthsFunction, int32_t, Date>({prefix + "months"});
  registerFunction<DaysFunction, Date, Timestamp>({prefix + "days"});
  registerFunction<DaysFunction, Date, Date>({prefix + "days"});
  registerFunction<HoursFunction, int32_t, Timestamp>({prefix + "hours"});
}

} // namespace facebook::velox::functions::iceberg
