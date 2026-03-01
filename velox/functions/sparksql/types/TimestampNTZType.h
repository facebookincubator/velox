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

namespace facebook::velox::functions::sparksql {

/// The timestamp without time zone type represents a local time in microsecond
/// precision, which is independent of time zone.
class TimestampNTZType final : public BigintType {
  constexpr TimestampNTZType() : BigintType{} {}

 public:
  static std::shared_ptr<const TimestampNTZType> get() {
    VELOX_CONSTEXPR_SINGLETON TimestampNTZType kInstance;
    return {std::shared_ptr<const TimestampNTZType>{}, &kInstance};
  }

  bool equivalent(const Type& other) const override {
    // Pointer comparison works since this type is a singleton.
    return this == &other;
  }

  const char* name() const override {
    return "TIMESTAMP_NTZ";
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

inline bool isTimestampNTZType(const TypePtr& type) {
  // Pointer comparison works since this type is a singleton.
  return TimestampNTZType::get() == type;
}

inline std::shared_ptr<const TimestampNTZType> TIMESTAMP_NTZ() {
  return TimestampNTZType::get();
}

// Type used for function registration.
struct TimestampNTZT {
  using type = int64_t;
  static constexpr const char* typeName = "timestamp_ntz";
};

using TimestampNTZ = CustomType<TimestampNTZT>;

} // namespace facebook::velox::functions::sparksql
