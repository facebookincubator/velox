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
#include "velox/substrait/TypeUtils.h"

namespace facebook::velox::substrait {

SubstraitFunctionLookup::SubstraitFunctionFinder::SubstraitFunctionFinder(
    const std::string name,
    const std::vector<SubstraitFunctionPtr>& functions)
    : name_(name) {
  for (const auto& function : functions) {
    const std::string& functionKey = SubstraitFunction::constructKey(
        function->name, function->requireArguments());

    directMap_.insert({functionKey,function});
  }
}

SubstraitFunctionLookup::SubstraitFunctionLookup(
    const std::vector<SubstraitFunctionPtr>& functions) {
  // TODO creaate signatures_ based on functions
  std::unordered_map<std::string, std::vector<SubstraitFunctionPtr>&>
      signatures;

  for (const auto& function : functions) {
    if (signatures.find(function->name) != signatures.end()) {
      std::vector<SubstraitFunctionPtr> nameFunctions;
      nameFunctions.emplace_back(function);
      signatures.insert({function->name, nameFunctions});
    } else {
      auto nameFunctions = signatures.at(function->name);
      nameFunctions.emplace_back(function);
    }
  }
}

const std::optional<SubstraitFunctionPtr>
SubstraitFunctionLookup::lookupFunction(
    const core::CallTypedExprPtr& callTypeExpr) const {
  auto& veloxFunctionName = callTypeExpr->name();
  auto& functionMappings = getFunctionMappings();
  auto& substraitFunctionName =
      functionMappings.find(veloxFunctionName) != functionMappings.end()
      ? functionMappings.at(veloxFunctionName)
      : veloxFunctionName;

  if (functionSignatures_.find(substraitFunctionName) ==
      functionSignatures_.end()) {
    return std::nullopt;
  }

  const auto& functionFinder = functionSignatures_.at(substraitFunctionName);
  return functionFinder->lookupFunction(callTypeExpr);
}

const std::optional<SubstraitFunctionPtr>
SubstraitFunctionLookup::SubstraitFunctionFinder::lookupFunction(
    const core::TypedExprPtr& exprPtr) const {


  return std::nullopt;
}

} // namespace facebook::velox::substrait
