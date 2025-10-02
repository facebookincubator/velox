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

#include "velox/type/Type.h"

namespace facebook::velox {

/// Represents TIME WITH TIME ZONE as a bigint.
/// This type stores time with timezone information, typically encoded as:
/// - Most significant bits: milliseconds since midnight UTC (similar to TIME)
/// - Least significant bits : timezone information
class TimeWithTimezoneType final : public BigintType {
  TimeWithTimezoneType() = default;

 public:
  static constexpr int16_t kTimeZoneBias = 840;
  static constexpr int16_t kMinutesInHour = 60;

  static std::shared_ptr<const TimeWithTimezoneType> get() {
    VELOX_CONSTEXPR_SINGLETON TimeWithTimezoneType kInstance;
    return {std::shared_ptr<const TimeWithTimezoneType>{}, &kInstance};
  }

  bool equivalent(const Type& other) const override {
    // Pointer comparison works since this type is a singleton.
    return this == &other;
  }

  const char* name() const override {
    return "TIME WITH TIME ZONE";
  }

  std::string toString() const override {
    return name();
  }

  /// Returns the time with timezone 'value' formatted as HH:MM:SS.mmmZZ
  /// where the timezone offset is included in the representation.
  std::string valueToString(int64_t value) const;

  folly::dynamic serialize() const override;

  static TypePtr deserialize(const folly::dynamic& /*obj*/) {
    return TimeWithTimezoneType::get();
  }

  bool isOrderable() const override {
    return true;
  }

  bool isComparable() const override {
    return true;
  }

  /// Encodes the timezone offset in the upper bits of the bigint.
  /// The timezone offset is encoded as an integer in the range [-840, 840]
  /// representing the number of minutes from UTC. The bias is added to the
  /// timezone offset to ensure that the timezone offset is positive.
  /// Typically called before a call to pack which will encode the timezone
  /// offset along with the time value.
  static inline int16_t biasEncode(int16_t timeZoneOffsetMinutes) {
    VELOX_CHECK(
        -kTimeZoneBias <= timeZoneOffsetMinutes &&
            timeZoneOffsetMinutes <= kTimeZoneBias,
        "Timezone offset must be between -840 and 840 minutes. Got: ",
        timeZoneOffsetMinutes);
    return timeZoneOffsetMinutes + kTimeZoneBias;
  }
};

inline bool isTimeWithTimeZone(const TypePtr& other) {
  return TimeWithTimezoneType::get() == other;
}

using TimeWithTimezoneTypePtr = std::shared_ptr<const TimeWithTimezoneType>;

FOLLY_ALWAYS_INLINE TimeWithTimezoneTypePtr TIME_WITH_TIME_ZONE() {
  return TimeWithTimezoneType::get();
}

} // namespace facebook::velox
