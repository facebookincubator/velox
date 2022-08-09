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

#include "SubstraitExtension.h"

namespace YAML {

using EnumArgument = facebook::velox::substrait::SubstraitEnumArgument;

using ValueArgument = facebook::velox::substrait::SubstraitValueArgument;

using TypeArgument = facebook::velox::substrait::SubstraitTypeArgument;

using ScalarFunctionVariant =
    facebook::velox::substrait::SubstraitScalarFunctionVariant;

using AggregateFunctionVariant =
    facebook::velox::substrait::SubstraitAggregateFunctionVariant;

using ScalarFunction = facebook::velox::substrait::SubstraitScalarFunction;

using AggregateFunction =
    facebook::velox::substrait::SubstraitAggregateFunction;

template <>
struct convert<EnumArgument> {
  static bool decode(const Node& node, EnumArgument& argument) {
    // 'options' is required  property
    auto& options = node["options"];
    if (options && options.IsSequence()) {
      auto& required = node["required"];
      argument.required = required && required.as<bool>();
      return true;
    } else {
      return false;
    }
  }
};

template <>
struct convert<ValueArgument> {
  static bool decode(const Node& node, ValueArgument& argument) {
    auto& value = node["value"];
    if (value && value.IsScalar()) {
      argument.type = value.as<std::string>();
      return true;
    }
    return false;
  }
};

template <>
struct convert<TypeArgument> {
  static bool decode(const Node& node, TypeArgument& argument) {
    if (node["type"]) {
      return true;
    } else {
      return false;
    }
  }
};

template <>
struct convert<ScalarFunctionVariant> {
  static bool decode(const Node& node, ScalarFunctionVariant& function) {
    auto& returnType = node["return"];
    if (returnType && returnType.IsScalar()) {
      function.returnType = returnType.as<std::string>();
      auto& args = node["args"];
      if (args && args.IsSequence()) {
        for (auto& arg : args) {
          if (arg["options"]) { // enum argument
            auto enumArgument =
                std::make_shared<EnumArgument>(arg.as<EnumArgument>());
            function.arguments.emplace_back(enumArgument);
          } else if (arg["value"]) {
            auto valueArgument =
                std::make_shared<ValueArgument>(arg.as<ValueArgument>());
            function.arguments.emplace_back(valueArgument);
          } else {
            auto typeArgument =
                std::make_shared<TypeArgument>(arg.as<TypeArgument>());
            function.arguments.emplace_back(typeArgument);
          }
        }
      }
      return true;
    }
    return false;
  }
};

template <>
struct convert<AggregateFunctionVariant> {
  static bool decode(const Node& node, AggregateFunctionVariant& function) {
    auto& returnType = node["return"];
    if (returnType && returnType.IsScalar()) {
      function.returnType = returnType.as<std::string>();
      auto& args = node["args"];
      if (args && args.IsSequence()) {
        for (auto& arg : args) {
          if (arg["options"]) { // enum argument
            auto enumArgument =
                std::make_shared<EnumArgument>(arg.as<EnumArgument>());
            function.arguments.emplace_back(enumArgument);
          } else if (arg["value"]) { // value argument
            auto valueArgument =
                std::make_shared<ValueArgument>(arg.as<ValueArgument>());
            function.arguments.emplace_back(valueArgument);
          } else { // type argument
            auto typeArgument =
                std::make_shared<TypeArgument>(arg.as<TypeArgument>());
            function.arguments.emplace_back(typeArgument);
          }
        }
      }
      return true;
    }
    return false;
  }
};

template <>
struct convert<ScalarFunction> {
  static bool decode(const Node& node, ScalarFunction& function) {
    auto& name = node["name"];
    if (name && name.IsScalar()) {
      function.name = name.as<std::string>();
      auto& impls = node["impls"];
      if (impls && impls.IsSequence() && impls.size() > 0) {
        for (auto& impl : impls) {
          auto scalarFunctionVariant = impl.as<ScalarFunctionVariant>();
          scalarFunctionVariant.name = function.name;
          function.impls.emplace_back(
              std::make_shared<ScalarFunctionVariant>(scalarFunctionVariant));
        }
      }
      return true;
    }
    return false;
  }
};

template <>
struct convert<AggregateFunction> {
  static bool decode(const Node& node, AggregateFunction& function) {
    auto& name = node["name"];
    if (name && name.IsScalar()) {
      function.name = name.as<std::string>();
      auto& impls = node["impls"];
      if (impls && impls.IsSequence() && impls.size() > 0) {
        for (auto& impl : impls) {
          auto aggregateFunctionVariant = impl.as<AggregateFunctionVariant>();
          aggregateFunctionVariant.name = function.name;
          function.impls.emplace_back(
              std::make_shared<AggregateFunctionVariant>(
                  aggregateFunctionVariant));
        }
      }
      return true;
    }
    return false;
  }
};

template <>
struct convert<facebook::velox::substrait::SubstraitExtension> {
  static bool decode(
      const Node& node,
      facebook::velox::substrait::SubstraitExtension& extension) {
    auto& scalarFunctions = node["scalar_functions"];
    auto& aggregateFunctions = node["aggregate_functions"];
    const bool scalarFunctionsExists =
        scalarFunctions && scalarFunctions.IsSequence();
    const bool aggregateFunctionsExists =
        aggregateFunctions && aggregateFunctions.IsSequence();
    if (!scalarFunctionsExists && !aggregateFunctionsExists) {
      return false;
    }

    if (scalarFunctionsExists) {
      for (auto& scalarFunctionNode : scalarFunctions) {
        const auto& scalarFunction = scalarFunctionNode.as<ScalarFunction>();
        for (auto& scalaFunctionVariant : scalarFunction.impls) {
          extension.scalarFunctionVariants.emplace_back(scalaFunctionVariant);
        }
      }
    }

    if (aggregateFunctionsExists) {
      for (auto& aggregateFunctionNode : aggregateFunctions) {
        const auto& aggregateFunction =
            aggregateFunctionNode.as<AggregateFunction>();
        for (auto& aggregateFunctionVariant : aggregateFunction.impls) {
          extension.aggregateFunctionVariants.emplace_back(
              aggregateFunctionVariant);
        }
      }
    }

    return true;
  }
};

} // namespace YAML

namespace facebook::velox::substrait {

std::string getSubstraitExtensionAbsolutePath() {
  const std::string absolute_path = __FILE__;
  auto const pos = absolute_path.find_last_of('/');
  return absolute_path.substr(0, pos) + "/extensions/";
}

std::shared_ptr<SubstraitExtension> SubstraitExtension::load() {
  SubstraitExtension mergedExtension;
  std::vector<std::string> extensionFiles = {
      "functions_aggregate_approx.yaml",
      "functions_aggregate_generic.yaml",
      "functions_arithmetic.yaml",
      "functions_arithmetic_decimal.yaml",
      "functions_boolean.yaml",
      "functions_comparison.yaml",
      "functions_datetime.yaml",
      "functions_logarithmic.yaml",
      "functions_rounding.yaml",
      "functions_string.yaml"};

  for (const auto& extensionFile : extensionFiles) {
    const auto& extensionUri =
        getSubstraitExtensionAbsolutePath() + extensionFile;
    const auto& substraitExtension =
        YAML::Load(extensionUri).as<SubstraitExtension>();

    for (auto& scalarFunctionVariant :
         substraitExtension.scalarFunctionVariants) {
      scalarFunctionVariant->uri = extensionUri;
      mergedExtension.scalarFunctionVariants.emplace_back(
          scalarFunctionVariant);
    }

    for (auto& aggregateFunctionVariant :
         substraitExtension.aggregateFunctionVariants) {
      aggregateFunctionVariant->uri = extensionUri;
      mergedExtension.aggregateFunctionVariants.emplace_back(
          aggregateFunctionVariant);
    }
  }

  return std::make_shared<SubstraitExtension>(mergedExtension);
}

} // namespace facebook::velox::substrait
