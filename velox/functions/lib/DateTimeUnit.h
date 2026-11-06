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

/// The unit argument of date_add, date_diff, date_trunc and the interval
/// operators. Kept in its own header, with no other includes, so that CUDA
/// translation units can use it without the date/time formatter.
namespace facebook::velox::functions {

/// Ordered from the shortest unit to the longest, so that comparisons such as
/// unit < DateTimeUnit::kDay separate fixed-length units from calendar units.
enum class DateTimeUnit {
  kMicrosecond,
  kMillisecond,
  kSecond,
  kMinute,
  kHour,
  kDay,
  kWeek,
  kMonth,
  kQuarter,
  kYear,
};

/// Returns true for the units shorter than a day. Their length is fixed, so
/// arithmetic in them needs no calendar and no time zone.
VELOX_GPU_COMPATIBLE inline bool isTimeUnit(DateTimeUnit unit) {
  return unit < DateTimeUnit::kDay;
}

} // namespace facebook::velox::functions
