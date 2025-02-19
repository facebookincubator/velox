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

#include "velox/expression/CastExpr.h"
#include "velox/type/SimpleFunctionApi.h"
#include "velox/type/Type.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox {

// Currently only supports TDigest of Doubles
class TDigestType : public VarbinaryType {
  TypePtr elementType_;

 public:
  TDigestType(TypePtr elementType) : elementType_(std::move(elementType)) {
    VELOX_USER_CHECK(
        elementType_->kind() == TypeKind::DOUBLE,
        "TDigestType only supports DOUBLE");
  }

  static const std::shared_ptr<const TDigestType>& get() {
    static const std::shared_ptr<const TDigestType> instance =
        std::shared_ptr<TDigestType>(new TDigestType(DOUBLE()));
    return instance;
  }

  static std::shared_ptr<const TDigestType> create(TypePtr elementType) {
    VELOX_USER_CHECK(
        elementType->kind() == TypeKind::DOUBLE,
        "TDigestType only supports DOUBLE");
    return std::shared_ptr<TDigestType>(
        new TDigestType(std::move(elementType)));
  }

  bool equivalent(const Type& other) const override {
    return dynamic_cast<const TDigestType*>(&other) != nullptr;
  }

  const char* name() const override {
    return "TDIGEST";
  }

  std::string toString() const override {
    return fmt::format("TDIGEST({})", elementType_->toString());
  }

  folly::dynamic serialize() const override {
    folly::dynamic obj = folly::dynamic::object;
    obj["name"] = "Type";
    obj["type"] = name();
    obj["elementType"] = elementType_->serialize();
    return obj;
  }

  const TypePtr& elementType() const {
    return elementType_;
  }
};

inline bool isTDigestType(const TypePtr& type) {
  return std::dynamic_pointer_cast<const TDigestType>(type) != nullptr;
}

inline std::shared_ptr<const TDigestType> TDIGEST(TypePtr typePtr) {
  VELOX_USER_CHECK(
      typePtr->kind() == TypeKind::DOUBLE, "TDigestType only supports DOUBLE");
  return TDigestType::get();
}

struct TDigestT {
  using type = Varbinary;
  static constexpr const char* typeName = "tdigest";
};

using TDigest = CustomType<TDigestT>;

class TDigestTypeFactories : public CustomTypeFactories {
 public:
  TypePtr getType() const override {
    return TDIGEST(DOUBLE());
  }

  // TDigest should be treated as Varbinary during type castings.
  exec::CastOperatorPtr getCastOperator() const override {
    return nullptr;
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& /*config*/) const override {
    return nullptr;
  }
};

void registerTDigestType();

} // namespace facebook::velox
