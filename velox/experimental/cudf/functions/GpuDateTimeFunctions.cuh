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

#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/experimental/cudf/functions/GpuTimeZone.h"

#include "velox/common/base/Macros.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/TimeUtilsCore.h"
// For SimpleTypeTrait<Date>, which registration reads to name the signature.
#include "velox/type/SimpleFunctionApi.h"

#include <ctime>
#include <vector>

/// DATE and TIMESTAMP field extractors for GPU SFI.
///
/// Presto's YearFunction and its siblings cannot be instantiated here:
/// DateTimeFunctions.h reaches re2, the time zone database and the vector
/// layer, none of which parse under nvcc, and their tz::TimeZone* is host
/// memory. These wrappers compute the same thing with the same device-callable
/// calendar accessors in TimeUtilsCore.h, converting UTC to local time through
/// GpuTimeZone, which the host derives from the same tz::TimeZone the CPU uses.
/// GpuInitSessionTimeZone stands in for InitSessionTimezone. TIMESTAMP WITH
/// TIME ZONE, which carries a time zone per row, is not supported.
namespace facebook::velox::cudf_velox::gpu_sfi {

namespace detail {

VELOX_GPU_COMPATIBLE inline int64_t floorDivide(
    int64_t value,
    int64_t divisor) {
  const int64_t quotient = value / divisor;
  return (value % divisor < 0) ? quotient - 1 : quotient;
}

/// ISO 8601 week-numbering year and week for a day number, both read from the
/// calendar date of that week's Thursday.
struct IsoWeek {
  int64_t year;
  int64_t week;
};

VELOX_GPU_COMPATIBLE inline IsoWeek isoWeek(int64_t days) {
  // 1970-01-01 was a Thursday, ISO weekday 4.
  const int64_t isoWeekday = days - floorDivide(days + 3, 7) * 7 + 4;
  const int64_t thursday = days - isoWeekday + 4;
  const std::tm time =
      functions::getDateTimeUtc(thursday * functions::kSecondsInDay);
  return IsoWeek{1900 + time.tm_year, time.tm_yday / 7 + 1};
}

VELOX_GPU_COMPATIBLE inline int64_t dayOfWeek(const std::tm& time) {
  // tm_wday counts from Sunday; Presto counts Monday 1 through Sunday 7.
  return time.tm_wday == 0 ? 7 : time.tm_wday;
}

} // namespace detail

/// Stands in for InitSessionTimezone: initialize() resolves the session time
/// zone once per call site for the TIMESTAMP overloads. DATE registrations have
/// no initialize(), which is matched on the argument types.
template <typename T>
struct GpuInitSessionTimeZone {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Timestamp>* /*timestamp*/) {
    timeZone_ = gpuSessionTimeZone(config);
  }

  /// The calendar fields of a timestamp in the session time zone, as
  /// getDateTime(timestamp, timeZone) computes them on the CPU.
  VELOX_GPU_COMPATIBLE std::tm localDateTime(
      const arg_type<Timestamp>& timestamp) const {
    return functions::getDateTimeUtc(timeZone_.toLocal(timestamp.seconds));
  }

  /// Days since the epoch in the session time zone.
  VELOX_GPU_COMPATIBLE int64_t
  localDays(const arg_type<Timestamp>& timestamp) const {
    return detail::floorDivide(
        timeZone_.toLocal(timestamp.seconds), functions::kSecondsInDay);
  }

  GpuTimeZone timeZone_;
};

template <typename T>
struct GpuYearFunction : public GpuInitSessionTimeZone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = functions::getYear(this->localDateTime(timestamp));
  }

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = functions::getYear(functions::getDateTime(date));
  }
};

template <typename T>
struct GpuMonthFunction : public GpuInitSessionTimeZone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = functions::getMonth(this->localDateTime(timestamp));
  }

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = functions::getMonth(functions::getDateTime(date));
  }
};

template <typename T>
struct GpuDayFunction : public GpuInitSessionTimeZone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = functions::getDay(this->localDateTime(timestamp));
  }

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = functions::getDay(functions::getDateTime(date));
  }
};

template <typename T>
struct GpuQuarterFunction : public GpuInitSessionTimeZone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = functions::getQuarter(this->localDateTime(timestamp));
  }

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = functions::getQuarter(functions::getDateTime(date));
  }
};

template <typename T>
struct GpuDayOfYearFunction : public GpuInitSessionTimeZone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = functions::getDayOfYear(this->localDateTime(timestamp));
  }

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = functions::getDayOfYear(functions::getDateTime(date));
  }
};

template <typename T>
struct GpuDayOfWeekFunction : public GpuInitSessionTimeZone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = detail::dayOfWeek(this->localDateTime(timestamp));
  }

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = detail::dayOfWeek(functions::getDateTime(date));
  }
};

template <typename T>
struct GpuWeekFunction : public GpuInitSessionTimeZone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = detail::isoWeek(this->localDays(timestamp)).week;
  }

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = detail::isoWeek(date).week;
  }
};

template <typename T>
struct GpuYearOfWeekFunction : public GpuInitSessionTimeZone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = detail::isoWeek(this->localDays(timestamp)).year;
  }

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = detail::isoWeek(date).year;
  }
};

template <typename T>
struct GpuHourFunction : public GpuInitSessionTimeZone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = this->localDateTime(timestamp).tm_hour;
  }
};

template <typename T>
struct GpuMinuteFunction : public GpuInitSessionTimeZone<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = this->localDateTime(timestamp).tm_min;
  }
};

/// No time zone: Velox's SecondFunction reads the field in UTC.
template <typename T>
struct GpuSecondFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = functions::getDateTimeUtc(timestamp.seconds).tm_sec;
  }
};

template <typename T>
struct GpuMillisecondFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<Timestamp>& timestamp) {
    result = static_cast<int64_t>(timestamp.nanos / 1'000'000);
  }
};

} // namespace facebook::velox::cudf_velox::gpu_sfi
