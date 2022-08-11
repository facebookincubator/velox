/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "SubstraitFunctionLookup.h"

namespace facebook::velox::substrait {

FunctionMappingMap& SubstraitFunctionMappings::scalarMappings() {
  static FunctionMappingMap scalarMap{
      {"plus", "add"},
      {"minus", "subtract"},
      {"mod", "modulus"},
      {"eq", "equal"},
      {"neq", "not_equal"},
  };
  return scalarMap;
}

FunctionMappingMap& SubstraitFunctionMappings::aggregateMappings() {
  static FunctionMappingMap aggregateMap;
  return aggregateMap;
}

FunctionMappingMap& SubstraitFunctionMappings::windowMappings() {
  static FunctionMappingMap windowMap;
  return windowMap;
}

SubstraitFunctionLookup::SubstraitFunctionLookup(
    const std::vector<SubstraitFunctionVariantPtr>& functions) {
  std::unordered_map<std::string, std::vector<SubstraitFunctionVariantPtr>>
      signatures;

  for (const auto& function : functions) {
    if (signatures.find(function->name) == signatures.end()) {
      std::vector<SubstraitFunctionVariantPtr> nameFunctions;
      nameFunctions.emplace_back(function);
      signatures.insert({function->name, nameFunctions});
    } else {
      auto& nameFunctions = signatures.at(function->name);
      nameFunctions.emplace_back(function);
    }
  }

  for (const auto& [name, signature] : signatures) {
    auto functionFinder =
        std::make_shared<SubstraitFunctionFinder>(name, signature);
    functionSignatures_.insert({name, functionFinder});
  }
}

std::optional<SubstraitFunctionVariantPtr>
SubstraitFunctionLookup::lookupFunction(
    google::protobuf::Arena& arena,
    const core::CallTypedExprPtr& callTypeExpr) const {
  const auto& veloxFunctionName = callTypeExpr->name();
  const auto& functionMappings = this->getFunctionMappings();
  const auto& substraitFunctionName =
      functionMappings.find(veloxFunctionName) != functionMappings.end()
      ? functionMappings.at(veloxFunctionName)
      : veloxFunctionName;

  if (functionSignatures_.find(substraitFunctionName) ==
      functionSignatures_.end()) {
    return std::nullopt;
  }

  const auto& functionFinder = functionSignatures_.at(substraitFunctionName);
  return functionFinder->lookupFunction(
      substraitFunctionName, arena, callTypeExpr);
}

SubstraitFunctionLookup::SubstraitFunctionFinder::SubstraitFunctionFinder(
    const std::string& name,
    const std::vector<SubstraitFunctionVariantPtr>& functions)
    : name_(name) {
  for (const auto& function : functions) {
    directMap_.insert({function->key(), function});

    if (function->requireArguments().size() != function->arguments.size()) {
      const std::string& functionKey = SubstraitFunctionVariant::constructKey(
          function->name, function->requireArguments());
      directMap_.insert({functionKey, function});
    }
  }
  anyTypeOption_ = std::nullopt;
  for (const auto& function : functions) {
    for (const auto& arg : function->arguments) {
      if (const auto& valueArgument =
              std::dynamic_pointer_cast<const SubstraitValueArgument>(arg)) {
        if (valueArgument->isWildcard()) {
          anyTypeOption_ = std::make_optional(function);
          break;
        }
      }
    }
  }
}

std::optional<SubstraitFunctionVariantPtr>
SubstraitFunctionLookup::SubstraitFunctionFinder::lookupFunction(
    const std::string& substraitFuncName,
    google::protobuf::Arena& arena,
    const core::CallTypedExprPtr& expr) const {
  std::vector<::substrait::Type> types;

  for (auto& input : expr->inputs()) {
    auto& substraitType = typeConvertor_->toSubstraitType(arena, input->type());
    types.emplace_back(substraitType);
  }

  const auto& signature =
      SubstraitTypeUtil::signature(substraitFuncName, types);
  /// try to do a direct match
  if (directMap_.find(signature) != directMap_.end()) {
    return std::make_optional(directMap_.at(signature));
  } else {
    return anyTypeOption_;
  }
}

} // namespace facebook::velox::substrait
