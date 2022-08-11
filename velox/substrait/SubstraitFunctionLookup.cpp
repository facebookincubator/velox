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

SubstraitFunctionLookup::SubstraitFunctionLookup(
    const std::vector<SubstraitFunctionVariantPtr>& functions,
    const SubstraitFunctionMappingsPtr& functionMappings)
    : functionMappings_(functionMappings) {
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

const std::optional<SubstraitFunctionVariantPtr>
SubstraitFunctionLookup::lookupFunction(
    const std::string& functionName,
    const std::vector<::substrait::Type>& types) const {
  const auto& functionMappings = getFunctionMappings();
  const auto& substraitFunctionName =
      functionMappings.find(functionName) != functionMappings.end()
      ? functionMappings.at(functionName)
      : functionName;

  if (functionSignatures_.find(substraitFunctionName) ==
      functionSignatures_.end()) {
    return std::nullopt;
  }

  const auto& functionFinder = functionSignatures_.at(substraitFunctionName);
  return functionFinder->lookupFunction(substraitFunctionName, types);
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
    const std::vector<::substrait::Type>& types) const {
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
