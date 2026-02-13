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

#include "velox/common/base/Exceptions.h"
#include "velox/type/SimpleFunctionApi.h"
#include "velox/type/Type.h"

namespace facebook::velox {

namespace exec {
class CastOperator;
}

// Given a class T, create a custom type that can be used to refer to it, with
// an underlying opaque vector type.
// Note that name must be stored in a variable and passed and cant be inlined.
// OpaqueCustomTypeRegister<T, "type"> wont compile.
// but static constexpr char* type = "type", OpaqueCustomTypeRegister<T, type>
// works.
template <typename T, const char* customTypeName>
class OpaqueCustomTypeRegister {
 public:
  using SerializeValueFunc = OpaqueType::SerializeFunc<T>;
  using DeserializeValueFunc = OpaqueType::DeserializeFunc<T>;

  // @param serializeValueFunc Optional serialization function for values of
  // type T.
  // @param deserializeValueFunc Optional deserialization function for values of
  // type T. Both serializeValueFunc and deserializeValueFunc must be specified
  // or not together.
  static bool registerType(
      SerializeValueFunc serializeValueFunc = nullptr,
      DeserializeValueFunc deserializeValueFunc = nullptr);

  static bool unregisterType();

  // Type used in the simple function interface as CustomType<TypeT>.
  struct TypeT {
    using type = std::shared_ptr<T>;
    static constexpr const char* typeName = customTypeName;
  };

  using SimpleType = CustomType<TypeT>;

  class VeloxType;
  using VeloxTypePtr = std::shared_ptr<const VeloxType>;

  class VeloxType : public OpaqueType {
   public:
    VeloxType() : OpaqueType(std::type_index(typeid(T))) {}

    static const VeloxTypePtr& get() {
      static const VeloxType kInstance;
      static const VeloxTypePtr kInstancePtr{TypePtr{}, &kInstance};
      return kInstancePtr;
    }

    static const std::shared_ptr<const exec::CastOperator>& getCastOperator() {
      VELOX_UNSUPPORTED();
    }

    bool equivalent(const velox::Type& other) const override {
      // Pointer comparison works since this type is a singleton.
      return this == &other;
    }

    const char* name() const override {
      return customTypeName;
    }

    std::string toString() const override {
      return customTypeName;
    }

    folly::dynamic serialize() const override {
      folly::dynamic obj = folly::dynamic::object;
      obj["name"] = "Type";
      obj["type"] = customTypeName;
      return obj;
    }
  };

  static const VeloxTypePtr& singletonTypePtr() {
    return VeloxType::get();
  }

  static const VeloxTypePtr& get() {
    return VeloxType::get();
  }

 private:
  class TypeFactory : public CustomTypeFactory {
   public:
    TypeFactory() = default;

    TypePtr getType(
        const std::vector<TypeParameter>& parameters) const override {
      VELOX_CHECK(parameters.empty());
      return singletonTypePtr();
    }

    exec::CastOperatorPtr getCastOperator() const override {
      VELOX_UNSUPPORTED();
    }

    AbstractInputGeneratorPtr getInputGenerator(
        const InputGeneratorConfig& /*config*/) const override {
      return nullptr;
    }
  };
};

template <typename T, const char* customTypeName>
bool OpaqueCustomTypeRegister<T, customTypeName>::registerType(
    SerializeValueFunc serializeValueFunc,
    DeserializeValueFunc deserializeValueFunc) {
  VELOX_USER_CHECK(
      ((serializeValueFunc && deserializeValueFunc) ||
       (!serializeValueFunc && !deserializeValueFunc)),
      "Both serialization and deserialization functions need to be registered for custom type {}",
      customTypeName);
  if (registerCustomType(
          customTypeName, std::make_unique<const TypeFactory>())) {
    if (serializeValueFunc && deserializeValueFunc) {
      OpaqueType::registerSerialization<T>(
          customTypeName, serializeValueFunc, deserializeValueFunc);
    }
    return true;
  }
  return false;
}

template <typename T, const char* customTypeName>
bool OpaqueCustomTypeRegister<T, customTypeName>::unregisterType() {
  if (unregisterCustomType(customTypeName)) {
    OpaqueType::unregisterSerialization(singletonTypePtr(), customTypeName);
    return true;
  }
  return false;
}

} // namespace facebook::velox
