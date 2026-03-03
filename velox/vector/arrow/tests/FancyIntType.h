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

namespace facebook::velox::test {

class FancyIntType final : public BigintType {
  constexpr FancyIntType() : BigintType{} {}

 public:
  static std::shared_ptr<const FancyIntType> get() {
    VELOX_CONSTEXPR_SINGLETON FancyIntType kInstance;
    return {std::shared_ptr<const FancyIntType>{}, &kInstance};
  }

  bool equivalent(const Type& other) const override {
    // Pointer comparison works since this type is a singleton.
    return this == &other;
  }

  const char* name() const override {
    return "FANCY_INT";
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

inline std::shared_ptr<const FancyIntType> FANCY_INT() {
  return FancyIntType::get();
}

class FancyIntTypeFactory : public CustomTypeFactory {
 public:
  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return FANCY_INT();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    VELOX_UNSUPPORTED();
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& config) const override {
    VELOX_UNSUPPORTED();
  }

  const char* getArrowFormatString() const override {
    return "fi";
  }
};

inline void registerFancyIntType() {
  registerCustomType(
      FANCY_INT()->name(), std::make_unique<const FancyIntTypeFactory>());
}

} // namespace facebook::velox::test
