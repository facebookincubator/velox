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

class BigintEnumType;
using BigintEnumTypePtr = std::shared_ptr<const BigintEnumType>;

/// BigintEnumType represents an enumerated value where the physical type is a
/// bigint. Each enum type has a name and a set of string keys which map to
/// bigint values, passed in as a LongEnumParameter TypeParameterKind.
class BigintEnumType : public BigintType {
 public:
  explicit BigintEnumType(const std::vector<TypeParameter>& typeParameters);

  static BigintEnumTypePtr get(
      const std::vector<TypeParameter>& typeParameters);

  static std::string mapToString(
      std::unordered_map<std::string, int64_t> longEnumMap);

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
    return fmt::format("{}:BigintEnum({}{})", name_, name_, mapToString(map_));
  }

  folly::dynamic serialize() const override;

  folly::dynamic serializeEnumParameter(
      const std::string& name,
      const std::unordered_map<std::string, int64_t>& map) const;

  bool containsValue(int64_t value) const {
    return flippedMap_.find(value) != flippedMap_.end();
  }

  const std::string& enumName() const {
    return name_;
  }

 private:
  const std::vector<TypeParameter> parameters_;

  std::string name_;
  std::unordered_map<std::string, int64_t> map_;
  std::unordered_map<int64_t, std::string> flippedMap_;
};

inline BigintEnumTypePtr BIGINT_ENUM(
    const std::vector<TypeParameter>& typeParameters) {
  return BigintEnumType::get(typeParameters);
}

FOLLY_ALWAYS_INLINE bool isBigintEnumType(const TypePtr& type) {
  return dynamic_cast<const BigintEnumType*>(type.get()) != nullptr;
}

} // namespace facebook::velox
