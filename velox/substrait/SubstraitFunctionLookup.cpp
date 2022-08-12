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
#include "limits.h"

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
    const std::vector<SubstraitTypePtr>& substraitTypes) const {
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
  return functionFinder->lookupFunction(substraitFunctionName, substraitTypes);
}

SubstraitFunctionLookup::SubstraitFunctionFinder::SubstraitFunctionFinder(
    const std::string& name,
    const std::vector<SubstraitFunctionVariantPtr>& functions)
    : name_(name) {
  size_t minArgs =
      functions.empty() ? 0 : functions[0]->requireArguments().size();
  size_t maxArgs = functions.empty() ? 0 : functions[0]->arguments.size();
  for (const auto& function : functions) {
    minArgs = std::min(minArgs, function->requireArguments().size());
    maxArgs = std::max(maxArgs, function->arguments.size());
    directMap_.insert({function->key(), function});
    if (function->requireArguments().size() != function->arguments.size()) {
      const std::string& functionKey = SubstraitFunctionVariant::constructKey(
          function->name, function->requireArguments());
      directMap_.insert({functionKey, function});
    }
  }
  for (const auto& function : functions) {
    if (function->hasAny()) {
      anyPositionMap_.insert({function->key(), function->anyPosition()});
      if (function->requireArguments().size() != function->arguments.size()) {
        const std::string& functionKey = SubstraitFunctionVariant::constructKey(
            function->name, function->requireArguments());
        auto pos = function->anyPosition(function->requireArguments());
        anyPositionMap_.insert({functionKey, pos});
      }
    }
  }
  argRange_ = {minArgs, maxArgs};
}

const std::optional<SubstraitFunctionVariantPtr>
SubstraitFunctionLookup::SubstraitFunctionFinder::lookupFunction(
    const std::string& substraitFuncName,
    const std::vector<SubstraitTypePtr>& types) const {
  // Check number of types within argRange
  if (types.size() < argRange_.first || types.size() > argRange_.second) {
    return std::nullopt;
  }

  const auto& signature =
      SubstraitTypeUtil::signature(substraitFuncName, types);
  /// try to do a direct match
  if (directMap_.find(signature) != directMap_.end()) {
    return std::make_optional(directMap_.at(signature));
  } else {
    for (const auto& [anySignature, anyPosition] : anyPositionMap_) {
      std::vector<SubstraitTypePtr> anyTypes;
      anyTypes.reserve(types.size());
      for (int i = 0; i < types.size(); i++) {
        if (anyPosition.find(i) != anyPosition.end() && anyPosition.at(i)) {
          anyTypes.emplace_back(std::make_shared<SubstraitAnyType>("any"));
        } else {
          anyTypes.emplace_back(types.at(i));
        }
      }
      const auto& newSignature =
          SubstraitTypeUtil::signature(substraitFuncName, anyTypes);
      if (directMap_.find(newSignature) != directMap_.end()) {
        return std::make_optional(directMap_.at(newSignature));
      }
    }
    return std::nullopt;
  }
}

} // namespace facebook::velox::substrait
