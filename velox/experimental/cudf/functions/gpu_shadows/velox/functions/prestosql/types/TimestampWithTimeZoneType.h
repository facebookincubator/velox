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

// GPU shadow for velox/functions/prestosql/types/TimestampWithTimeZoneType.h.
// Keeps the simple-function tag and the packing helpers, which parse under
// nvcc, and drops TimestampWithTimeZoneType itself, which derives from Type.
// The tag and the helpers are copied from the real header: registration reads
// the tag's typeName for the signature string the host matches against, and a
// kernel must pack a value bit for bit as the CPU does, so they must stay
// identical.
#pragma once

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Macros.h"
#include "velox/type/SimpleFunctionApi.h"
#include "velox/type/Timestamp.h"

#include <cstdint>

namespace facebook::velox {

using TimeZoneKey = int16_t;

constexpr int32_t kMillisShift = 12;
constexpr int32_t kTimezoneMask = (1 << kMillisShift) - 1;
// The maximum and minimum millis UTC we can represent in a
// TimestampWithTimeZone value given the bits we have to store it.
// We have 64 bits minus the bits for the time zone minus 1 for the sign bit.
constexpr int64_t kMaxMillisUtc =
    (1LL << (64 - (int64_t)kMillisShift - 1)) - 1LL;
constexpr int64_t kMinMillisUtc = (kMaxMillisUtc + 1) * -1LL;

VELOX_GPU_COMPATIBLE inline int64_t unpackMillisUtc(
    int64_t dateTimeWithTimeZone) {
  return dateTimeWithTimeZone >> kMillisShift;
}

VELOX_GPU_COMPATIBLE inline TimeZoneKey unpackZoneKeyId(
    int64_t dateTimeWithTimeZone) {
  return dateTimeWithTimeZone & kTimezoneMask;
}

VELOX_GPU_COMPATIBLE inline int64_t pack(
    int64_t millisUtc,
    TimeZoneKey timeZoneKey) {
  VELOX_USER_CHECK(
      millisUtc <= kMaxMillisUtc && millisUtc >= kMinMillisUtc,
      "TimestampWithTimeZone overflow: {} ms",
      millisUtc);
  return (millisUtc << kMillisShift) | (timeZoneKey & kTimezoneMask);
}

VELOX_GPU_COMPATIBLE inline int64_t pack(
    const Timestamp& timestamp,
    TimeZoneKey timeZoneKey) {
  return pack(timestamp.toMillis(), timeZoneKey);
}

VELOX_GPU_COMPATIBLE inline Timestamp unpackTimestampUtc(
    int64_t dateTimeWithTimeZone) {
  return Timestamp::fromMillis(unpackMillisUtc(dateTimeWithTimeZone));
}

// Type used for function registration.
struct TimestampWithTimezoneT {
  using type = int64_t;
  static constexpr const char* typeName = "timestamp with time zone";
};

using TimestampWithTimezone = CustomType<TimestampWithTimezoneT, true>;

} // namespace facebook::velox
