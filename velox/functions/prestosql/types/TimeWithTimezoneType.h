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

#include "velox/type/SimpleFunctionApi.h"
#include "velox/type/Type.h"

namespace facebook::velox {

/// Represents TIME WITH TIME ZONE as a bigint.
/// This type stores time with timezone information, typically encoded as:
/// - Most significant bits: milliseconds since midnight UTC (similar to TIME)
/// - Least significant bits : timezone information
class TimeWithTimezoneType final : public BigintType {
  TimeWithTimezoneType() = default;

 public:
  static constexpr int16_t kTimeWithTimezoneToVarcharRowSize = 18;

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
  StringView valueToString(int64_t value, char* const startPos) const;

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
};

inline bool isTimeWithTimeZone(const TypePtr& other) {
  return TimeWithTimezoneType::get() == other;
}

using TimeWithTimezoneTypePtr = std::shared_ptr<const TimeWithTimezoneType>;

FOLLY_ALWAYS_INLINE TimeWithTimezoneTypePtr TIME_WITH_TIME_ZONE() {
  return TimeWithTimezoneType::get();
}

struct TimeWithTimezoneT {
  using type = int64_t;
  static constexpr const char* typeName = "time with time zone";
};

using TimeWithTimezone = CustomType<TimeWithTimezoneT>;

} // namespace facebook::velox
