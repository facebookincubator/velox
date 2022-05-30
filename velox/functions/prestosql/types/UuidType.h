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
class UuidType : public VarcharType {
 public:
  UuidType() = default;

  static const std::shared_ptr<const UuidType>& get() {
    static const std::shared_ptr<const UuidType> kInstance{
        std::make_shared<const UuidType>()};
    return kInstance;
  }

  std::string toString() const override {
    static const auto typeName = "UUID";
    return typeName;
  }
};

FOLLY_ALWAYS_INLINE bool isUuidType(const TypePtr& type) {
  // Pointer comparison works since this type is a singleton.
  return UuidType::get() == type;
}

FOLLY_ALWAYS_INLINE std::shared_ptr<const UuidType> UUID() {
  return UuidType::get();
}

struct UuidT {
  using type = StringView;
  static constexpr const char* typeName = "uuid";
};

using Uuid = CustomType<UuidT>;

class UuidTypeFactories : public CustomTypeFactories {
 public:
  TypePtr getType(std::vector<TypePtr> /*childTypes*/) const override {
    return UUID();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    return nullptr;
  }
};

} // namespace facebook::velox
