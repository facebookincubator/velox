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

#include "velox/common/base/Exceptions.h"
#include "velox/substrait/SubstraitType.h"

namespace facebook::velox::substrait {

struct SubstraitFunctionArgument {
  /// whether the argument is required or not.
  virtual const bool isRequired() const = 0;
  /// convert argument type to short type string based on
  /// https://substrait.io/extensions/#function-signature-compound-names
  virtual const std::string toTypeString() const = 0;
};

using SubstraitFunctionArgumentPtr =
    std::shared_ptr<const SubstraitFunctionArgument>;

struct SubstraitEnumArgument : public SubstraitFunctionArgument {
  bool required;
  bool const isRequired() const override {
    return required;
  }

  const std::string toTypeString() const override {
    return required ? "req" : "opt";
  }
};

struct SubstraitTypeArgument : public SubstraitFunctionArgument {
  const std::string toTypeString() const override {
    return "type";
  }
  const bool isRequired() const override {
    return true;
  }
};

class SubstraitValueArgument : public SubstraitFunctionArgument {
 public:
  SubstraitTypePtr type;
  const std::string toTypeString() const override {
    return type->signature();
  }

  const bool isRequired() const override {
    return true;
  }

  const bool isWildcard() const {
    return type->isWildcard();
  }
};

struct SubstraitFunctionAnchor {
  /// uri of function anchor corresponding the file
  std::string uri;

  /// function signature which is combination of function name and type of
  /// arguments.
  std::string key;

  bool operator==(const SubstraitFunctionAnchor& other) const {
    return (uri == other.uri && key == other.key);
  }
};

struct SubstraitFunctionVariant {
  /// scalar function name.
  std::string name;
  /// scalar function uri.
  std::string uri;
  std::vector<SubstraitFunctionArgumentPtr> arguments;
  /// return type of scalar function.
  std::string returnType;

  static std::string constructKey(
      const std::string& name,
      const std::vector<SubstraitFunctionArgumentPtr>& arguments);

  std::string key() const {
    return SubstraitFunctionVariant::constructKey(name, arguments);
  }

  SubstraitFunctionAnchor anchor() const {
    return {uri, key()};
  }

  std::vector<SubstraitFunctionArgumentPtr> requireArguments() const;
};

using SubstraitFunctionVariantPtr = std::shared_ptr<SubstraitFunctionVariant>;

struct SubstraitScalarFunctionVariant : public SubstraitFunctionVariant {};

struct SubstraitAggregateFunctionVariant : public SubstraitFunctionVariant {};

struct SubstraitScalarFunction {
  /// scalar function name.
  std::string name;
  /// A collection of scalar function variants.
  std::vector<std::shared_ptr<SubstraitScalarFunctionVariant>> impls;
};

struct SubstraitAggregateFunction {
  /// aggregate function name.
  std::string name;
  /// A collection of aggregate function variants.
  std::vector<std::shared_ptr<SubstraitAggregateFunctionVariant>> impls;
};

/// class used to deserialize substrait YAML extension files.
struct SubstraitExtension {
  /// deserialize default substrait extension.
  static std::shared_ptr<SubstraitExtension> loadExtension();

  /// deserialize substrait extension by given basePath and extensionFiles.
  static std::shared_ptr<SubstraitExtension> loadExtension(
      const std::string& basePath,
      const std::vector<std::string>& extensionFiles);

  /// a collection of scalar function variants loaded from Substrait extension
  /// yaml.
  std::vector<SubstraitFunctionVariantPtr> scalarFunctionVariants;
  /// a collection of aggregate function variants loaded from Substrait
  /// extension yaml.
  std::vector<SubstraitFunctionVariantPtr> aggregateFunctionVariants;

  std::vector<SubstraitTypeAnchorPtr> types;
};

using SubstraitExtensionPtr = std::shared_ptr<const SubstraitExtension>;

} // namespace facebook::velox::substrait

namespace std {

/// hash function of facebook::velox::substrait::SubstraitFunctionAnchor
template <>
struct hash<facebook::velox::substrait::SubstraitFunctionAnchor> {
  size_t operator()(
      const facebook::velox::substrait::SubstraitFunctionAnchor& k) const {
    return hash<std::string>()(k.key) ^ hash<std::string>()(k.uri);
  }
};

}; // namespace std
