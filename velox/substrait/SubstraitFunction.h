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

#include <algorithm>
#include "folly/FBString.h"
#include "functional"
#include "sstream"
#include "velox/common/base/Exceptions.h"
#include "velox/substrait/SubstraitParser.h"
#include "velox/substrait/SubstraitType.h"

namespace facebook::velox::substrait {

struct SubstraitFunctionArgument {
  virtual const bool isRequired() const = 0;
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

struct SubstraitValueArgument : public SubstraitFunctionArgument {
  std::string type;

  const std::string toTypeString() const override {
    return SubstraitType::argumentToSignature(type);
  }

  const bool isRequired() const override {
    return true;
  }

  const bool isWildcard() const {
    return SubstraitType::isWildcard(type);
  }
};

struct SubstraitFunctionAnchor {
  std::string uri;
  std::string key;

  bool operator==(const SubstraitFunctionAnchor& other) const {
    return (uri == other.uri && key == other.key);
  }
};

struct SubstraitFunctionVariant {
  std::string name;
  std::string uri;
  std::vector<SubstraitFunctionArgumentPtr> arguments;
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

struct SubstraitAggregateFunctionVariant : public SubstraitFunctionVariant {
  std::string intermediate;
};

struct SubstraitScalarFunction {
  std::string name;
  std::vector<std::shared_ptr<SubstraitScalarFunctionVariant>> impls;
};

struct SubstraitAggregateFunction {
  std::string name;
  std::vector<std::shared_ptr<SubstraitAggregateFunctionVariant>> impls;
};

} // namespace facebook::velox::substrait

/// hash function for type facebook::velox::substrait::SubstraitFunctionAnchor
namespace std {

template <>
struct hash<facebook::velox::substrait::SubstraitFunctionAnchor> {
  size_t operator()(
      const facebook::velox::substrait::SubstraitFunctionAnchor& k) const {
    return hash<std::string>()(k.key) ^ hash<std::string>()(k.uri);
  }
};

}; // namespace std
