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

#include "velox/common/base/Macros.h"
#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/TimeUtilsCore.h"
// For SimpleTypeTrait<Date>, which registration reads to name the signature.
// Without it the empty primary template is chosen and Date registers as no
// type at all.
#include "velox/type/SimpleFunctionApi.h"

/// DATE field extractors for GPU SFI.
///
/// Everywhere else GPU SFI instantiates the Velox function struct itself. The
/// datetime family is the exception, and it is worth being precise about why,
/// because the exception looks like a shortcut and is not one.
///
/// Presto's YearFunction and its siblings inherit InitSessionTimezone and
/// TimestampWithTimezoneSupport, which hold a tz::TimeZone* filled in by an
/// initialize() that takes a QueryConfig. A single struct serves both the
/// timezone-aware Timestamp overload and the timezone-free Date overload, so
/// there is no way to instantiate one without the other. Reaching the struct
/// at all means parsing DateTimeFunctions.h, which brings in re2, the timezone
/// database, the datetime formatter and the vector layer -- roughly a hundred
/// errors in a device translation unit, none of them related to reading a field
/// out of a std::tm.
///
/// What that struct computes for a Date argument is a call into getDateTime()
/// followed by one field read. Both already live in TimeUtilsCore.h and are
/// device-callable, so the algorithm -- the calendar conversion, which is the
/// part with any real behaviour in it -- is genuinely shared with the CPU
/// implementation. These wrappers restate only the three lines of glue, and
/// they call the same accessors, so a field extracted on GPU agrees with the
/// CPU result by construction rather than by testing.
///
/// The struct shape deliberately mirrors the upstream one. If the timezone-free
/// overloads are ever split into base structs, these become redundant and can
/// be deleted in favour of the real thing.
namespace facebook::velox::cudf_velox::gpu_sfi {

template <typename T>
struct GpuYearFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = functions::getYear(functions::getDateTime(date));
  }
};

template <typename T>
struct GpuMonthFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = functions::getMonth(functions::getDateTime(date));
  }
};

template <typename T>
struct GpuDayFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = functions::getDay(functions::getDateTime(date));
  }
};

template <typename T>
struct GpuQuarterFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = functions::getQuarter(functions::getDateTime(date));
  }
};

template <typename T>
struct GpuDayOfYearFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    result = functions::getDayOfYear(functions::getDateTime(date));
  }
};

/// tm_wday is days since Sunday; Presto counts Monday as 1 through Sunday as 7.
template <typename T>
struct GpuDayOfWeekFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(int64_t& result, const arg_type<Date>& date) {
    const auto time = functions::getDateTime(date);
    result = time.tm_wday == 0 ? 7 : time.tm_wday;
  }
};

} // namespace facebook::velox::cudf_velox::gpu_sfi
