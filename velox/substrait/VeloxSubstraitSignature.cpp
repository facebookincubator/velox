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

#include "velox/substrait/VeloxSubstraitSignature.h"
#include <boost/algorithm/string.hpp>
#include <iostream>
#include "velox/exec/Aggregate.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/functions/FunctionRegistry.h"

namespace facebook::velox::substrait {

std::string VeloxSubstraitSignature::toVeloxSignature(
    const std::string& functionName,
    const std::vector<facebook::velox::TypePtr>& inputs) {
  std::ostringstream signature;
  signature << functionName << "(";
  for (auto i = 0; i < inputs.size(); i++) {
    if (i > 0) {
      signature << ", ";
    }
    signature << inputs[i]->toString();
  }
  signature << ")";
  return signature.str();
}

std::string VeloxSubstraitSignature::toVeloxSignature(
    const std::vector<const facebook::velox::exec::FunctionSignature*>&
        signatures) {
  std::stringstream out;
  for (auto i = 0; i < signatures.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << signatures[i]->toString();
  }
  return out.str();
}

const exec::FunctionSignature& VeloxSubstraitSignature::resolveFunction(
    const std::string& functionName,
    const std::vector<TypePtr>& arguments) {
  const auto veloxScalarFunctions = velox::getFunctionSignatures();
  const auto scalarFunctionsIt = veloxScalarFunctions.find(functionName);
  if (scalarFunctionsIt != veloxScalarFunctions.end()) {
    for (const auto& candidateSignature : scalarFunctionsIt->second) {
      if (exec::SignatureBinder(*candidateSignature, arguments).tryBind()) {
        return *candidateSignature;
      }
    }
    VELOX_USER_FAIL(
        "Scalar Function signature is not supported: {}. Supported signatures: {}.",
        VeloxSubstraitSignature::toVeloxSignature(functionName, arguments),
        VeloxSubstraitSignature::toVeloxSignature(scalarFunctionsIt->second));
  }

  auto aggregateFunctionsOption =
      exec::getAggregateFunctionSignatures(functionName);
  if (aggregateFunctionsOption.has_value()) {
    std::vector<const exec::FunctionSignature*> supportedAggFunSigs;
    const auto& aggregateFunctionSignatures = aggregateFunctionsOption.value();
    supportedAggFunSigs.reserve(aggregateFunctionSignatures.size());
    for (const auto& candidateSignature : aggregateFunctionSignatures) {
      supportedAggFunSigs.emplace_back(candidateSignature.get());
      auto binder = exec::SignatureBinder(*candidateSignature, arguments);
      if (exec::SignatureBinder(*candidateSignature, arguments).tryBind()) {
        return *candidateSignature;
      }
    }
    if (arguments.size() == 1) {
      for (const auto& candidateSignature : aggregateFunctionSignatures) {
        auto binder = exec::SignatureBinder(*candidateSignature, arguments);
        const auto& resolveType =
            binder.tryResolveType(candidateSignature->intermediateType());
        if (resolveType && resolveType->equivalent(*arguments.at(0))) {
          return *candidateSignature;
        }
      }
    }
    VELOX_USER_FAIL(
        "Aggregate Function signature is not supported: {}. Supported signatures {}.",
        VeloxSubstraitSignature::toVeloxSignature(functionName, arguments),
        VeloxSubstraitSignature::toVeloxSignature(supportedAggFunSigs));
  }

  if (auto vectorFunctionSignatures =
          exec::getVectorFunctionSignatures(functionName)) {
    for (const auto& signature : vectorFunctionSignatures.value()) {
      if (exec::SignatureBinder(*signature, arguments).tryBind()) {
        return *signature;
      }
    }
  }

  VELOX_USER_FAIL(
      "Function signature is not supported: {}.",
      VeloxSubstraitSignature::toVeloxSignature(functionName, arguments));
}

std::string VeloxSubstraitSignature::toSubstraitSignature(
    const exec::TypeSignature& typeSignature) {
  if ("T" == typeSignature.baseType() ||
      boost::algorithm::starts_with(typeSignature.baseType(), "__user_T")) {
    return "any";
  }
  auto typeKind = mapNameToTypeKind(
      boost::algorithm::to_upper_copy(typeSignature.baseType()));
  switch (typeKind) {
    case TypeKind::BOOLEAN:
      return "bool";
    case TypeKind::TINYINT:
      return "i8";
    case TypeKind::SMALLINT:
      return "i16";
    case TypeKind::INTEGER:
      return "i32";
    case TypeKind::BIGINT:
      return "i64";
    case TypeKind::REAL:
      return "fp32";
    case TypeKind::DOUBLE:
      return "fp64";
    case TypeKind::VARCHAR:
      return "str";
    case TypeKind::VARBINARY:
      return "vbin";
    case TypeKind::TIMESTAMP:
      return "ts";
    case TypeKind::DATE:
      return "date";
    case TypeKind::SHORT_DECIMAL:
      return "dec";
    case TypeKind::LONG_DECIMAL:
      return "dec";
    case TypeKind::ARRAY:
      return "list";
    case TypeKind::MAP:
      return "map";
    case TypeKind::ROW:
      return "struct";
    case TypeKind::UNKNOWN:
      return "u!name";
    default:
      VELOX_UNSUPPORTED(
          "Substrait type signature conversion not supported for type {}.",
          mapTypeKindToName(typeKind));
  }
}

std::string VeloxSubstraitSignature::toSubstraitSignature(
    const std::string& functionName,
    const std::vector<TypePtr>& arguments) {
  if (functionName == "and" || functionName == "or" || functionName == "xor") {
    return functionName + ":bool_bool";
  }
  if (functionName == "not") {
    return functionName + ":bool";
  }

  const auto& veloxFunctionSignature =
      VeloxSubstraitSignature::resolveFunction(functionName, arguments);
  if (veloxFunctionSignature.argumentTypes().empty()) {
    return functionName;
  }

  std::vector<std::string> substraitTypeSignatures;
  substraitTypeSignatures.reserve(
      veloxFunctionSignature.argumentTypes().size());
  for (const auto& typeSignature : veloxFunctionSignature.argumentTypes()) {
    substraitTypeSignatures.emplace_back(toSubstraitSignature(typeSignature));
  }
  return functionName + ":" + folly::join("_", substraitTypeSignatures);
}

} // namespace facebook::velox::substrait
