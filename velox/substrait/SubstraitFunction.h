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
#include "velox/substrait/SubstraitParser.h"

namespace facebook::velox::substrait {

struct SubstraitFunctionArgument {
  virtual bool isRequired() const = 0;
  virtual std::shared_ptr<std::string> toTypeString() const = 0;
};

struct SubstraitEnumFunctionArgument : public SubstraitFunctionArgument {
  std::vector<std::string> options;
  std::string name;
  bool required;
  bool isRequired() const override {
    return required;
  }

  std::shared_ptr<std::string> toTypeString() const override {
    std::string res = required ? "req" : "opt";
    return std::make_shared<std::string>(res);
  }
};

struct SubstraitTypeFunctionArgument : public SubstraitFunctionArgument {
  std::string type;
  std::string name;
  std::shared_ptr<std::string> toTypeString() const override {
    return std::make_shared<std::string>("type");
  }

  bool isRequired() const override {
    return true;
  }
};

struct SubstraitValueFunctionArgument : public SubstraitFunctionArgument {
  std::string type;
  std::string name;
  std::shared_ptr<std::string> toTypeString() const override {
    return std::make_shared<std::string>(type);
  }

  bool isRequired() const override {
    return true;
  }
};

enum class Nullability { MIRROR, DECLARED_OUTPUT, DISCRETE };

enum class Decomposability { NONE, ONE, MANY };

struct Variadic {
  int min;
  int max;
};

struct SubstraitFunction {
  std::string name;
  std::string uri;
  Variadic variadic;
  std::string description;
  std::vector<std::shared_ptr<SubstraitFunctionArgument>> args;
  Nullability nullability;
  std::string returnType;
};

using SubstraitFunctionPtr = std::shared_ptr<const SubstraitFunction>;

struct SubstraitScalarFunction : public SubstraitFunction {};

using SubstraitScalarFunctionPtr = std::shared_ptr<const SubstraitScalarFunction>;

struct SubstraitAggregateFunction : public SubstraitFunction {
  Decomposability decomposability;
  std::string intermediate;
};

using SubstraitAggregateFunctionPtr = std::shared_ptr<const SubstraitAggregateFunction>;

enum class WindowType { PARTITION, STREAMING };

struct SubstraitWindowFunction : public SubstraitFunction {
  WindowType windowType;
};

using SubstraitWindowFunctionPtr = std::shared_ptr<const SubstraitWindowFunction>;

} // namespace facebook::velox::substrait
