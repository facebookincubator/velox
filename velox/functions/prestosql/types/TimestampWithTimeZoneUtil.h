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

#include "velox/type/Timestamp.h"

namespace facebook::velox {
using TimeZoneKey = int16_t;

constexpr int32_t kTimezoneMask = 0xFFF;
constexpr int32_t kMillisShift = 12;

inline int64_t unpackMillisUtc(int64_t dateTimeWithTimeZone) {
  return dateTimeWithTimeZone >> kMillisShift;
}

inline TimeZoneKey unpackZoneKeyId(int64_t dateTimeWithTimeZone) {
  return dateTimeWithTimeZone & kTimezoneMask;
}

inline int64_t pack(int64_t millisUtc, int16_t timeZoneKey) {
  return (millisUtc << kMillisShift) | (timeZoneKey & kTimezoneMask);
}

inline Timestamp unpackTimestampUtc(int64_t dateTimeWithTimeZone) {
  return Timestamp::fromMillis(unpackMillisUtc(dateTimeWithTimeZone));
}
} // namespace facebook::velox
