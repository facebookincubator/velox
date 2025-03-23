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

#include "velox/functions/sparksql/DateTimeFunctions.h"
#include <iostream>

namespace facebook::velox::functions::sparksql {


inline int64_t safeMul(int64_t a, int64_t b, const char* context) {
  int64_t result;
  VELOX_USER_CHECK(
      !__builtin_mul_overflow(a, b, &result),
      "Overflow during multiplication in {}: {} * {}",
      context,
      a,
      b);
  return result;
}

inline int64_t safeAdd(int64_t a, int64_t b, const char* context) {
  int64_t result;
  VELOX_USER_CHECK(
      !__builtin_add_overflow(a, b, &result),
      "Overflow during addition in {}: {} + {}",
      context,
      a,
      b);
  return result;
}


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
struct MakeDTIntervalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<IntervalDayTime>& result,
      int32_t days = 0,
      int32_t hours = 0,
      int32_t minutes = 0,
      double secondsMicros = 0) {
    
    int64_t daysMicros = safeMul(
        static_cast<int64_t>(days) * Timestamp::kSecondsInDay,
        Timestamp::kMicrosecondsInSecond,
        "days");

    int64_t hoursMicros = safeMul(
        static_cast<int64_t>(hours) * 3600LL,
        Timestamp::kMicrosecondsInSecond,
        "hours");

    int64_t minutesMicros = safeMul(
        static_cast<int64_t>(minutes) * 60LL,
        Timestamp::kMicrosecondsInSecond,
        "minutes");

    int64_t secondsComponent = static_cast<int64_t>(secondsMicros);

    result = safeAdd(
        safeAdd(
            safeAdd(daysMicros, hoursMicros, "days + hours"),
            minutesMicros,
            "previous + minutes"),
        secondsComponent,
        "previous + seconds");
  }
};

} // namespace facebook::velox::functions::sparksql 