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
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox {

/// Represents timestamp with time zone as a number of milliseconds since epoch
/// and time zone ID.
class TimestampWithTimeZoneType : public RowType {
  TimestampWithTimeZoneType()
      : RowType({"timestamp", "timezone"}, {BIGINT(), SMALLINT()}) {}

 public:
  static const std::shared_ptr<const TimestampWithTimeZoneType>& get() {
    static const std::shared_ptr<const TimestampWithTimeZoneType> instance =
        std::shared_ptr<TimestampWithTimeZoneType>(
            new TimestampWithTimeZoneType());

    return instance;
  }

  bool equivalent(const Type& other) const override {
    // Pointer comparison works since this type is a singleton.
    return this == &other;
  }

  const char* name() const override {
    return "TIMESTAMP WITH TIME ZONE";
  }

  const std::vector<TypeParameter>& parameters() const override {
    static const std::vector<TypeParameter> kEmpty = {};
    return kEmpty;
  }

  std::string toString() const override {
    return name();
  }

  folly::dynamic serialize() const override {
    folly::dynamic obj = folly::dynamic::object;
    obj["name"] = "Type";
    obj["type"] = name();
    return obj;
  }
};

inline bool isTimestampWithTimeZoneType(const TypePtr& type) {
  // Pointer comparison works since this type is a singleton.
  return TimestampWithTimeZoneType::get() == type;
}

inline std::shared_ptr<const TimestampWithTimeZoneType>
TIMESTAMP_WITH_TIME_ZONE() {
  return TimestampWithTimeZoneType::get();
}

// Type used for function registration.
struct TimestampWithTimezoneT {
  using type = Row<int64_t, int16_t>;
  static constexpr const char* typeName = "timestamp with time zone";
};

using TimestampWithTimezone = CustomType<TimestampWithTimezoneT>;

class TimestampWithTimeZoneTypeFactories : public CustomTypeFactories {
 public:
  TypePtr getType() const override {
    return TIMESTAMP_WITH_TIME_ZONE();
  }

  // Type casting from and to TimestampWithTimezone is not supported yet.
  exec::CastOperatorPtr getCastOperator() const override {
    VELOX_NYI(
        "Casting of {} is not implemented yet.",
        TIMESTAMP_WITH_TIME_ZONE()->toString());
  }
};

void registerTimestampWithTimeZoneType();

} // namespace facebook::velox
