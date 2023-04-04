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

#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::exec {

void addVariablesInTypeToList(
    const TypeSignature& type,
    const std::unordered_map<std::string, SignatureVariable>& allVariables,
    std::unordered_map<std::string, SignatureVariable>& usedVariables) {
  auto iter = allVariables.find(type.baseName());
  if (iter != allVariables.end()) {
    usedVariables.emplace(iter->first, iter->second);
  }
  for (const auto& parameter : type.parameters()) {
    addVariablesInTypeToList(parameter, allVariables, usedVariables);
  }
}

std::unordered_map<std::string, SignatureVariable> getUsedTypeVariables(
    const std::vector<TypeSignature>& types,
    const std::unordered_map<std::string, SignatureVariable>& allVariables) {
  std::unordered_map<std::string, SignatureVariable> usedVariables;
  for (const auto& type : types) {
    addVariablesInTypeToList(type, allVariables, usedVariables);
  }
  return usedVariables;
}

} // namespace facebook::velox::exec
