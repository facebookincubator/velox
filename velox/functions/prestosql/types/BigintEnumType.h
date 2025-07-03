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

#include <map>
#include "folly/CPortability.h"
#include "velox/type/Type.h"

namespace facebook::velox {

class BigintEnumType : public BigintType {
 public:
  explicit BigintEnumType(const std::string& typeInfoString)
      : BigintType(/*providesCustomComparison*/ true),
        typeInfoString_(typeInfoString) {
    try {
      parseTypeInfo(typeInfoString);
    } catch (std::invalid_argument& e) {
      VELOX_USER_FAIL("Failed to parse type {}, {}", typeInfoString, e.what());
    }
  }

  FOLLY_EXPORT static const std::shared_ptr<const BigintEnumType>& get(
      const std::string& typeInfoString);

  int32_t compare(const int64_t& left, const int64_t& right) const override {
    return left < right ? -1 : left == right ? 0 : 1;
  }

  uint64_t hash(const int64_t& value) const override {
    return folly::hasher<int64_t>()(value);
  }

  bool equivalent(const Type& other) const override {
    return this->name() == other.name();
  }

  const char* name() const override {
    return enumName_.c_str();
  }

  std::string toString() const override {
    return name();
  }

  std::map<std::string, int64_t> getEnumMap() const {
    return enumMap_;
  }

  std::string getTypeInfoString() const {
    return typeInfoString_;
  }

  folly::dynamic serialize() const override {
    folly::dynamic obj = folly::dynamic::object;
    obj["name"] = "Type";
    obj["type"] = name();
    return obj;
  }

 private:
  void parseTypeInfo(const std::string& typeInfoString);

  const std::string typeInfoString_;

  std::string enumName_;
  std::map<std::string, int64_t> enumMap_;
};

FOLLY_ALWAYS_INLINE std::shared_ptr<const BigintEnumType> BIGINT_ENUM(
    const std::string& typeInfoString) {
  return BigintEnumType::get(typeInfoString);
}
} // namespace facebook::velox
