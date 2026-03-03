/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

class KHyperLogLogType final : public VarbinaryType {
  KHyperLogLogType() = default;

 public:
  static std::shared_ptr<const KHyperLogLogType> get() {
    VELOX_CONSTEXPR_SINGLETON KHyperLogLogType kInstance;
    return {std::shared_ptr<const KHyperLogLogType>{}, &kInstance};
  }

  bool equivalent(const Type& other) const override {
    // Pointer comparison works since this type is a singleton.
    return this == &other;
  }

  const char* name() const override {
    return "KHYPERLOGLOG";
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

  bool isOrderable() const override {
    return false;
  }
};

inline bool isKHyperLogLogType(const TypePtr& type) {
  // Pointer comparison works since this type is a singleton.
  return KHyperLogLogType::get() == type;
}

inline std::shared_ptr<const KHyperLogLogType> KHYPERLOGLOG() {
  return KHyperLogLogType::get();
}

// Type to use for inputs and outputs of simple functions, e.g.
// arg_type<KHyperLogLog> and out_type<KHyperLogLog>.
struct KHyperLogLogT {
  using type = Varbinary;
  static constexpr const char* typeName = "khyperloglog";
};

using KHyperLogLog = CustomType<KHyperLogLogT>;

} // namespace facebook::velox
