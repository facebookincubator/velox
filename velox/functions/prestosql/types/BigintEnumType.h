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

#include <folly/container/EvictingCacheMap.h>

#include "velox/type/Type.h"

namespace facebook::velox {

class BigintEnumType : public BigintType {
 public:
  static const std::shared_ptr<const BigintEnumType>& get(
      const std::vector<TypeParameter>& typeParameters);

  bool equivalent(const Type& other) const override {
    return this == &other;
  }

  const char* name() const override {
    return "BIGINT_ENUM";
  }

  const std::vector<TypeParameter>& parameters() const override {
    return parameters_;
  }

  std::string toString() const override {
    return fmt::format(
        "{}:BigintEnum({}{{{}}})",
        parameters_[0].stringLiteral.value(),
        parameters_[0].stringLiteral.value(),
        parameters_[1].stringLiteral.value());
  }

  folly::dynamic serialize() const override {
    folly::dynamic obj = folly::dynamic::object;
    obj["name"] = "Type";
    obj["type"] = name();
    std::vector<std::string> stringParams;
    stringParams.reserve(parameters_.size());
    for (auto& param : parameters_) {
      stringParams.push_back(param.stringLiteral.value());
    }
    obj["stringParams"] = velox::ISerializable::serialize(stringParams);

    return obj;
  }

  bool containsValue(const int64_t& value) const {
    return std::any_of(
        valuesObj_.items().begin(),
        valuesObj_.items().end(),
        [value](const auto& pair) { return pair.second == value; });
  }

  const std::string& enumName() const {
    return parameters_[0].stringLiteral.value();
  }

 private:
  BigintEnumType(const std::vector<TypeParameter>& typeParameters)
      : parameters_{typeParameters} {
    VELOX_CHECK_EQ(typeParameters.size(), 2);
    VELOX_CHECK(typeParameters[0].stringLiteral.has_value());
    VELOX_CHECK(typeParameters[1].stringLiteral.has_value());
    try {
      valuesObj_ =
          folly::parseJson("{" + typeParameters[1].stringLiteral.value() + "}");
    } catch (const std::runtime_error& e) {
      VELOX_FAIL(
          "Failed to parse enum values {}, {}",
          typeParameters[1].stringLiteral.value(),
          e.what());
    }
  }

  const std::vector<TypeParameter> parameters_;
  folly::dynamic valuesObj_;
};

inline std::shared_ptr<const BigintEnumType> BIGINT_ENUM(
    const std::vector<TypeParameter>& typeParameters) {
  return BigintEnumType::get(typeParameters);
}

FOLLY_ALWAYS_INLINE bool isBigintEnumType(const TypePtr& type) {
  if (type->parameters().size() != 2) {
    return false;
  }
  return BigintEnumType::get(type->parameters()) == type;
}

} // namespace facebook::velox
