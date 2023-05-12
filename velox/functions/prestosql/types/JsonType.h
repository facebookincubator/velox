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
#include "velox/type/Type.h"

namespace facebook::velox {

/// Custom operator for casts from and to Json type.
class JsonCastOperator : public exec::CastOperator {
 public:
  static const std::shared_ptr<const CastOperator>& get() {
    static const std::shared_ptr<const CastOperator> instance{
        new JsonCastOperator()};

    return instance;
  }

  bool isSupportedFromType(const TypePtr& other) const override;

  bool isSupportedToType(const TypePtr& other) const override;

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override;

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override;

 private:
  JsonCastOperator() = default;
};

/// Represents JSON as a string.
class JsonType : public VarcharType {
  JsonType() = default;

 public:
  static const std::shared_ptr<const JsonType>& get() {
    static const std::shared_ptr<const JsonType> instance{new JsonType()};

    return instance;
  }

  static const std::shared_ptr<const exec::CastOperator>& getCastOperator() {
    return JsonCastOperator::get();
  }

  bool equivalent(const Type& other) const override {
    // Pointer comparison works since this type is a singleton.
    return this == &other;
  }

  const char* name() const override {
    return "JSON";
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

FOLLY_ALWAYS_INLINE bool isJsonType(const TypePtr& type) {
  // Pointer comparison works since this type is a singleton.
  return JsonType::get() == type;
}

FOLLY_ALWAYS_INLINE std::shared_ptr<const JsonType> JSON() {
  return JsonType::get();
}

// Type used for function registration.
struct JsonT {
  using type = Varchar;
  static constexpr const char* typeName = "json";
};

using Json = CustomType<JsonT>;

class JsonTypeFactories : public CustomTypeFactories {
 public:
  JsonTypeFactories() = default;

  TypePtr getType() const override {
    return JSON();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    return JsonCastOperator::get();
  }
};

void registerJsonType();

} // namespace facebook::velox
