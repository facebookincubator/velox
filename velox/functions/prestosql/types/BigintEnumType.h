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

#include <boost/algorithm/string.hpp>

#include "folly/CPortability.h"
#include "velox/type/Type.h"

namespace facebook::velox {

class BigintEnumType : public BigintType {
 public:
  BigintEnumType(
      const std::string& name,
      const std::unordered_map<std::string, int64_t>& values)
      : BigintType(), enumName_(name), enumMap_(values) {}

  /// Creates a BigintEnumType with the given name and enum map and inserts into
  /// the static map of instances.
  static std::shared_ptr<const BigintEnumType> create(
      const std::string& enumName,
      const std::unordered_map<std::string, int64_t>& enumMap);

  /// Returns the enum type if an enum type has been created with the given
  /// name; otherwise returns nullopt.
  static std::optional<std::shared_ptr<const BigintEnumType>> get(
      const std::string& enumName);

  // TODO: Move the parsing logic to Prestissimo as it is tightly coupled with
  // how Presto coordinator serializes the enum type
  static std::pair<std::string, std::unordered_map<std::string, int64_t>>
  parseTypeInfo(const std::string& enumTypeString);

  bool equivalent(const Type& other) const override {
    return this->name() == other.name();
  }

  const char* name() const override {
    return enumName_.c_str();
  }

  std::string toString() const override {
    return name();
  }

  const std::unordered_map<std::string, int64_t>& enumMap() const {
    return enumMap_;
  }

  bool containsValue(const int64_t& value) const {
    return std::any_of(
        enumMap_.begin(), enumMap_.end(), [value](const auto& pair) {
          return pair.second == value;
        });
  }

  folly::dynamic serialize() const override {
    folly::dynamic obj = folly::dynamic::object;
    obj["name"] = "Type";
    obj["type"] = name();
    return obj;
  }

 private:
  // Returns a static map of enum name to the enum type, to avoid the same enum
  // type being created multiple times.
  static std::unordered_map<std::string, std::shared_ptr<const BigintEnumType>>&
  getInstances();

  std::string enumName_;
  std::unordered_map<std::string, int64_t> enumMap_;
};

FOLLY_ALWAYS_INLINE std::shared_ptr<const BigintEnumType> BIGINT_ENUM(
    const std::string& enumName) {
  auto enumType = BigintEnumType::get(enumName);
  if (enumType) {
    return enumType.value();
  }
  VELOX_FAIL("Unregistered type: {}", enumName);
}
} // namespace facebook::velox
