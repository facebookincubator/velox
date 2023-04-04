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

#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::exec {

std::unordered_map<std::string, SignatureVariable> getUsedTypeVariables(
    const std::vector<TypeSignature>& types,
    const std::unordered_map<std::string, SignatureVariable>& allVariables);

class CompanionSignatures {
 public:
  static std::vector<AggregateFunctionSignaturePtr> partialFunctionSignatures(
      const std::vector<AggregateFunctionSignaturePtr>& aggregateSignatures) {
    std::vector<AggregateFunctionSignaturePtr> signatures;
    for (const auto& signature : aggregateSignatures) {
      std::vector<TypeSignature> usedTypes = signature->argumentTypes();
      usedTypes.push_back(signature->intermediateType());
      auto variables = getUsedTypeVariables(usedTypes, signature->variables());

      signatures.push_back(std::make_shared<AggregateFunctionSignature>(
          variables,
          signature->intermediateType(),
          signature->intermediateType(),
          signature->argumentTypes(),
          signature->constantArguments(),
          signature->variableArity()));
    }
    return signatures;
  }

  static std::string partialFunctionName(const std::string& name) {
    return name + "_partial";
  }

  static std::vector<AggregateFunctionSignaturePtr> mergeFunctionSignatures(
      const std::vector<AggregateFunctionSignaturePtr>& aggregateSignatures) {
    std::unordered_set<TypeSignature> distinctIntermediateTypes;
    std::vector<AggregateFunctionSignaturePtr> signatures;
    for (const auto& signature : aggregateSignatures) {
      if (distinctIntermediateTypes.count(signature->intermediateType()) > 0) {
        continue;
      }
      distinctIntermediateTypes.insert(signature->intermediateType());

      std::vector<TypeSignature> usedTypes = {signature->intermediateType()};
      auto variables = getUsedTypeVariables(usedTypes, signature->variables());

      signatures.push_back(std::make_shared<AggregateFunctionSignature>(
          variables,
          signature->intermediateType(),
          signature->intermediateType(),
          std::vector<TypeSignature>{signature->intermediateType()},
          signature->constantArguments(),
          signature->variableArity()));
    }
    return signatures;
  }

  static std::string mergeFunctionName(const std::string& name) {
    return name + "_merge";
  }

  static bool hasSameIntermediateTypesAcrossSignatures(
      const std::vector<AggregateFunctionSignaturePtr>& signatures) {
    std::unordered_set<TypeSignature> seenTypes;
    for (const auto& signature : signatures) {
      if (seenTypes.count(signature->intermediateType()) > 0) {
        return true;
      }
      seenTypes.insert(signature->intermediateType());
    }
    return false;
  }

  static FunctionSignaturePtr extractFunctionSignature(
      const AggregateFunctionSignaturePtr& signature) {
    std::vector<TypeSignature> usedTypes = {
        signature->intermediateType(), signature->returnType()};
    auto variables = getUsedTypeVariables(usedTypes, signature->variables());
    return std::make_shared<FunctionSignature>(
        variables,
        signature->returnType(),
        std::vector<TypeSignature>{signature->intermediateType()},
        std::vector<bool>{false},
        false);
  }

  static std::string extractFunctionNameWithSuffix(
      const std::string& name,
      const TypeSignature& resultType) {
    return name + "_extract_" + resultType.toString();
  }

  static std::vector<FunctionSignaturePtr> extractFunctionSignatures(
      const std::vector<AggregateFunctionSignaturePtr>& signatures) {
    std::vector<FunctionSignaturePtr> extractSignatures;
    for (const auto& signature : signatures) {
      extractSignatures.push_back(extractFunctionSignature(signature));
    }
    return extractSignatures;
  }

  static std::string extractFunctionName(const std::string& name) {
    return name + "_extract";
  }

  static std::vector<FunctionSignaturePtr> retractFunctionSignatures(
      const std::vector<AggregateFunctionSignaturePtr>& signatures) {
    std::vector<FunctionSignaturePtr> retractSignatures;
    for (const auto& signature : signatures) {
      std::vector<TypeSignature> usedTypes = {signature->intermediateType()};
      auto variables = getUsedTypeVariables(usedTypes, signature->variables());
      retractSignatures.push_back(std::make_shared<FunctionSignature>(
          variables,
          signature->intermediateType(),
          std::vector<TypeSignature>{
              signature->intermediateType(), signature->intermediateType()},
          std::vector<bool>{false, false},
          false));
    }
    return retractSignatures;
  }

  static std::string retractFunctionName(const std::string& name) {
    return name + "_retract";
  }
};

} // namespace facebook::velox::exec
