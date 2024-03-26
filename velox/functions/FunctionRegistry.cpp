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
#include "velox/functions/FunctionRegistry.h"

#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <iterator>
#include <optional>
#include <sstream>
#include "velox/common/base/Exceptions.h"
#include "velox/core/SimpleFunctionMetadata.h"
#include "velox/exec/WindowFunction.h"
#include "velox/expression/FunctionCallToSpecialForm.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/expression/VectorFunction.h"
#include "velox/type/Type.h"

namespace facebook::velox {
namespace {

void populateSimpleFunctionSignatures(FunctionSignatureMap& map) {
  const auto& simpleFunctions = exec::simpleFunctions();
  for (const auto& functionName : simpleFunctions.getFunctionNames()) {
    map[functionName] = simpleFunctions.getFunctionSignatures(functionName);
  }
}

void populateVectorFunctionSignatures(FunctionSignatureMap& map) {
  auto vectorFunctions = exec::vectorFunctionFactories();
  vectorFunctions.withRLock([&map](const auto& locked) {
    for (const auto& it : locked) {
      const auto& allSignatures = it.second.signatures;
      auto& curSignatures = map[it.first];
      std::transform(
          allSignatures.begin(),
          allSignatures.end(),
          std::back_inserter(curSignatures),
          [](std::shared_ptr<exec::FunctionSignature> signature)
              -> exec::FunctionSignature* { return signature.get(); });
    }
  });
}

} // namespace

FunctionSignatureMap getFunctionSignatures() {
  FunctionSignatureMap result;
  populateSimpleFunctionSignatures(result);
  populateVectorFunctionSignatures(result);
  return result;
}

FunctionSignatureMap getVectorFunctionSignatures() {
  FunctionSignatureMap result;
  populateVectorFunctionSignatures(result);
  return result;
}

void clearFunctionRegistry() {
  exec::mutableSimpleFunctions().clearRegistry();
  exec::vectorFunctionFactories().withWLock(
      [](auto& functionMap) { functionMap.clear(); });
}

std::optional<bool> isDeterministic(const std::string& functionName) {
  const auto simpleFunctions =
      exec::simpleFunctions().getFunctionSignaturesAndMetadata(functionName);
  const auto metadata = exec::getVectorFunctionMetadata(functionName);
  if (simpleFunctions.empty() && !metadata.has_value()) {
    return std::nullopt;
  }

  for (const auto& [metadata, _] : simpleFunctions) {
    if (!metadata.deterministic) {
      return false;
    }
  }
  if (metadata.has_value() && !metadata.value().deterministic) {
    return false;
  }
  return true;
}

TypePtr resolveFunction(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes) {
  // Check if this is a simple function.
  if (auto returnType = resolveSimpleFunction(functionName, argTypes)) {
    return returnType;
  }

  // Check if VectorFunctions has this function name + signature.
  return resolveVectorFunction(functionName, argTypes);
}

std::optional<std::pair<TypePtr, exec::VectorFunctionMetadata>>
resolveFunctionWithMetadata(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes) {
  if (auto resolvedFunction =
          exec::simpleFunctions().resolveFunction(functionName, argTypes)) {
    return std::pair<TypePtr, exec::VectorFunctionMetadata>{
        resolvedFunction->type(), resolvedFunction->metadata()};
  }

  return exec::resolveVectorFunctionWithMetadata(functionName, argTypes);
}

TypePtr resolveFunctionOrCallableSpecialForm(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes) {
  if (auto returnType = resolveCallableSpecialForm(functionName, argTypes)) {
    return returnType;
  }

  return resolveFunction(functionName, argTypes);
}

TypePtr resolveCallableSpecialForm(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes) {
  return exec::resolveTypeForSpecialForm(functionName, argTypes);
}

TypePtr resolveSimpleFunction(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes) {
  if (auto resolvedFunction =
          exec::simpleFunctions().resolveFunction(functionName, argTypes)) {
    return resolvedFunction->type();
  }

  return nullptr;
}

TypePtr resolveVectorFunction(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes) {
  return exec::resolveVectorFunction(functionName, argTypes);
}

std::optional<std::pair<TypePtr, exec::VectorFunctionMetadata>>
resolveVectorFunctionWithMetadata(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes) {
  return exec::resolveVectorFunctionWithMetadata(functionName, argTypes);
}

bool isCompanionFunctionName(
    const std::string& name,
    const std::unordered_map<std::string, exec::AggregateFunctionEntry>&
        aggregateFunctions) {
  static const std::vector<std::string> kCompanionFunctionSuffixList = {
      "_partial", "_merge_extract", "_merge", "_extract"};
  for (const auto& companionFunctionSuffix : kCompanionFunctionSuffixList) {
    auto suffixOffset = name.rfind(companionFunctionSuffix);
    if (suffixOffset != std::string::npos) {
      return aggregateFunctions.count(name.substr(0, suffixOffset)) > 0;
    }
  }
  return false;
}

const std::vector<std::string> getScalarNames(bool sortResult) {
  // Do not print "internal" functions.
  static const std::unordered_set<std::string> kBlockList = {
      "row_constructor", "in", "is_null"};

  auto functions = getFunctionSignatures();
  std::vector<std::string> names;
  names.reserve(functions.size());
  exec::aggregateFunctions().withRLock([&](const auto& aggregateFunctions) {
    for (const auto& func : functions) {
      const auto& name = func.first;
      if (!isCompanionFunctionName(name, aggregateFunctions) &&
          kBlockList.count(name) == 0) {
        names.emplace_back(name);
      }
    }
  });
  if (sortResult) {
    std::sort(names.begin(), names.end());
  }
  return names;
}

const std::vector<std::string> getAggregateNames(bool sortResult) {
  std::vector<std::string> names;
  exec::aggregateFunctions().withRLock([&](const auto& functions) {
    names.reserve(functions.size());
    for (const auto& entry : functions) {
      if (!isCompanionFunctionName(entry.first, functions)) {
        names.emplace_back(entry.first);
      }
    }
  });
  if (sortResult) {
    std::sort(names.begin(), names.end());
  }
  return names;
}

const std::vector<std::string> getWindowNames(bool sortResult) {
  const auto& functions = exec::windowFunctions();
  std::vector<std::string> names;
  names.reserve(functions.size());
  // Removing all aggregates from the window functions list.
  exec::aggregateFunctions().withRLock([&](const auto& aggregateFunctions) {
    for (const auto& entry : functions) {
      if (aggregateFunctions.count(entry.first) == 0) {
        names.emplace_back(entry.first);
      }
    }
  });

  if (sortResult) {
    std::sort(names.begin(), names.end());
  }
  return names;
}

} // namespace facebook::velox
