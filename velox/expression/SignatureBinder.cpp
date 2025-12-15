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
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <algorithm>
#include <optional>
#include <unordered_map>

#include "velox/common/base/Exceptions.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/expression/TypeSignature.h"
#include "velox/expression/type_calculation/TypeCalculation.h"
#include "velox/type/Type.h"
#include "velox/type/TypeCoercer.h"
#include "velox/type/TypeUtil.h"

namespace facebook::velox::exec {
namespace {

bool isAny(const TypeSignature& typeSignature) {
  return typeSignature.baseName() == "any";
}

/// Returns true only if 'str' contains digits.
bool isPositiveInteger(const std::string& str) {
  return !str.empty() &&
      std::find_if(str.begin(), str.end(), [](unsigned char c) {
        return !std::isdigit(c);
      }) == str.end();
}

std::optional<int> tryResolveLongLiteral(
    const TypeSignature& parameter,
    const std::unordered_map<std::string, SignatureVariable>& variables,
    std::unordered_map<std::string, int>& integerVariablesBindings) {
  const auto& variable = parameter.baseName();

  if (isPositiveInteger(variable)) {
    // Handle constant.
    return atoi(variable.c_str());
  };

  if (integerVariablesBindings.count(variable)) {
    return integerVariablesBindings.at(variable);
  }

  auto it = variables.find(variable);
  if (it == variables.end()) {
    return std::nullopt;
  }

  const auto& constraints = it->second.constraint();

  if (constraints.empty()) {
    return std::nullopt;
  }

  // Try to assign value based on constraints.
  // Check constraints and evaluate.
  const auto calculation = fmt::format("{}={}", variable, constraints);
  expression::calculation::evaluate(calculation, integerVariablesBindings);
  VELOX_CHECK(
      integerVariablesBindings.count(variable),
      "Variable {} calculation failed.",
      variable);
  return integerVariablesBindings.at(variable);
}

std::optional<LongEnumParameter> tryResolveLongEnumLiteral(
    const TypeSignature& parameter,
    const std::unordered_map<std::string, LongEnumParameter>&
        longEnumParameterVariableBindings) {
  auto it = longEnumParameterVariableBindings.find(parameter.baseName());
  if (it != longEnumParameterVariableBindings.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<VarcharEnumParameter> tryResolveVarcharEnumLiteral(
    const TypeSignature& parameter,
    const std::unordered_map<std::string, VarcharEnumParameter>&
        varcharEnumParameterVariableBindings) {
  auto it = varcharEnumParameterVariableBindings.find(parameter.baseName());
  if (it != varcharEnumParameterVariableBindings.end()) {
    return it->second;
  }
  return std::nullopt;
}

// If the parameter is a named field from a row, ensure the names are
// compatible. For example:
//
// > row(bigint) - binds any row with bigint as field.
// > row(foo bigint) - only binds rows where bigint field is named foo.
bool checkNamedRowField(
    const TypeSignature& signature,
    const TypePtr& actualType,
    size_t idx) {
  if (!signature.rowFieldName()) {
    return true;
  }
  return *signature.rowFieldName() == asRowType(actualType)->nameOf(idx);
}

bool checkSignatureProperties(
    const SignatureVariable& signature,
    const TypePtr& actualType) {
  if (signature.knownTypesOnly() && actualType->isUnKnown()) {
    return false;
  }
  if (signature.orderableTypesOnly() && !actualType->isOrderable()) {
    return false;
  }
  if (signature.comparableTypesOnly() && !actualType->isComparable()) {
    return false;
  }
  return true;
}

} // namespace

bool SignatureBinder::tryBind(std::vector<Coercion>* coercions) {
  const auto& formalArgs = signature_.argumentTypes();
  const auto formalArgsCnt = formalArgs.size();

  if (signature_.variableArity()) {
    if (actualTypes_.size() < formalArgsCnt - 1) {
      return false;
    }
    if (!isAny(signature_.argumentTypes().back()) &&
        actualTypes_.size() > formalArgsCnt) {
      const auto& variableType = actualTypes_[formalArgsCnt - 1];
      if (!variableType) {
        return false;
      }
      for (size_t i = formalArgsCnt; i < actualTypes_.size(); ++i) {
        if (!actualTypes_[i]) {
          return false;
        }
        if (actualTypes_[i] == UNKNOWN()) {
          continue;
        }
        if (!actualTypes_[i]->equivalent(*variableType)) {
          return false;
        }
      }
    }
  } else if (formalArgsCnt != actualTypes_.size()) {
    return false;
  }

  size_t bindArgsCnt = actualTypes_.size();
  if (coercions) {
    coercions->clear();
    coercions->resize(bindArgsCnt);
  } else if (formalArgsCnt < bindArgsCnt) {
    bindArgsCnt = formalArgsCnt;
  }

  for (size_t i = 0; i < bindArgsCnt; ++i) {
    const auto& formalArgSignature =
        i < formalArgsCnt ? formalArgs[i] : formalArgs.back();
    if (!SignatureBinderBase::tryBind(
            formalArgSignature,
            actualTypes_[i],
            coercions ? &(*coercions)[i] : nullptr)) {
      return false;
    }
  }

  return true;
}

bool SignatureBinderBase::checkOrSetLongEnumParameter(
    const std::string& parameterName,
    const LongEnumParameter& params) {
  auto it = longEnumVariablesBindings_.find(parameterName);
  if (it != longEnumVariablesBindings_.end()) {
    if (longEnumVariablesBindings_[parameterName] != params) {
      return false;
    }
  }
  longEnumVariablesBindings_[parameterName] = params;
  return true;
}

bool SignatureBinderBase::checkOrSetVarcharEnumParameter(
    const std::string& parameterName,
    const VarcharEnumParameter& params) {
  auto it = varcharEnumVariablesBindings_.find(parameterName);
  if (it != varcharEnumVariablesBindings_.end()) {
    if (varcharEnumVariablesBindings_[parameterName] != params) {
      return false;
    }
  }
  varcharEnumVariablesBindings_[parameterName] = params;
  return true;
}

bool SignatureBinderBase::checkOrSetIntegerParameter(
    const std::string& parameterName,
    int value) {
  if (isPositiveInteger(parameterName)) {
    return atoi(parameterName.c_str()) == value;
  }
  if (!variables().count(parameterName)) {
    // Return false if the parameter is not found in the signature.
    return false;
  }

  const auto& constraint = variables().at(parameterName).constraint();
  if (isPositiveInteger(constraint) && atoi(constraint.c_str()) != value) {
    // Return false if the actual value does not match the constraint.
    return false;
  }

  if (integerVariablesBindings_.count(parameterName)) {
    // Return false if the parameter is found with a different value.
    if (integerVariablesBindings_[parameterName] != value) {
      return false;
    }
  }
  // Bind the variable.
  integerVariablesBindings_[parameterName] = value;
  return true;
}

bool SignatureBinderBase::tryBind(
    const exec::TypeSignature& typeSignature,
    const TypePtr& actualType,
    Coercion* coercion) {
  if (!actualType) {
    return false;
  }

  if (isAny(typeSignature)) {
    return true;
  }

  const auto& baseName = typeSignature.baseName();

  if (auto varIt = variables().find(baseName); varIt != variables().end()) {
    VELOX_CHECK(
        typeSignature.parameters().empty(),
        "Variables with parameters are not supported");
    VELOX_CHECK(varIt->second.isTypeParameter());

    auto& varType = typeVariablesBindings_[varIt->second.name()];
    if (varType) {
      return varType->equivalent(*actualType);
    }
    if (!checkSignatureProperties(varIt->second, actualType)) {
      return false;
    }
    varType = actualType;
    return true;
  }

  const auto& params = typeSignature.parameters();

  // Type is not a variable.
  auto typeName = boost::algorithm::to_upper_copy(baseName);
  if (!boost::algorithm::iequals(typeName, actualType->name())) {
    if (!coercion || !params.empty()) {
      return false;
    }
    auto availableCoercion = TypeCoercer::coerceTypeBase(actualType, typeName);
    if (!availableCoercion) {
      return false;
    }
    *coercion = *availableCoercion;
    return true;
  }

  // Handle homogeneous row case: row(T, ...)
  if (typeSignature.isHomogeneousRow()) {
    VELOX_CHECK_EQ(
        params.size(), 1, "Homogeneous row must have exactly one parameter");

    if (actualType->kind() != TypeKind::ROW) {
      return false;
    }

    if (actualType->size() == 0) {
      // Empty row is always compatible with homogeneous row.
      return true;
    }

    // All children must unify to the same type variable T
    const auto& typeParam = params[0];
    const auto& paramBaseName = typeParam.baseName();

    // First, check and extract the common child type if homogeneous.
    const auto actualChildType =
        velox::type::tryGetHomogeneousRowChild(actualType);
    if (!actualChildType) {
      return false;
    }

    if (variables().count(paramBaseName)) {
      auto it = typeVariablesBindings_.find(paramBaseName);
      if (it != typeVariablesBindings_.end()) {
        return it->second->equivalent(*actualChildType);
      } else {
        typeVariablesBindings_[paramBaseName] = actualChildType;
        return true;
      }
    } else {
      return tryBind(typeParam, actualChildType);
    }
  }

  // Type Parameters can recurse.
  if (params.size() != actualType->parameters().size()) {
    return false;
  }

  for (size_t i = 0; i < params.size(); i++) {
    const auto& actualParameter = actualType->parameters()[i];
    switch (actualParameter.kind) {
      case TypeParameterKind::kLongLiteral:
        if (!checkOrSetIntegerParameter(
                params[i].baseName(), actualParameter.longLiteral.value())) {
          return false;
        }
        break;
      case TypeParameterKind::kLongEnumLiteral:
        if (!checkOrSetLongEnumParameter(
                params[i].baseName(),
                actualParameter.longEnumLiteral.value())) {
          return false;
        }
        break;
      case TypeParameterKind::kVarcharEnumLiteral:
        if (!checkOrSetVarcharEnumParameter(
                params[i].baseName(),
                actualParameter.varcharEnumLiteral.value())) {
          return false;
        }
        break;
      case TypeParameterKind::kType:
        if (!checkNamedRowField(params[i], actualType, i)) {
          return false;
        }
        if (!tryBind(params[i], actualParameter.type)) {
          // TODO Allow coercions for complex types.
          return false;
        }
        break;
    }
  }
  return true;
}

TypePtr SignatureBinder::tryResolveType(
    const exec::TypeSignature& typeSignature,
    const std::unordered_map<std::string, SignatureVariable>& variables,
    const std::unordered_map<std::string, TypePtr>& typeVariablesBindings,
    std::unordered_map<std::string, int>& integerVariablesBindings,
    const std::unordered_map<std::string, LongEnumParameter>&
        longEnumParameterVariableBindings,
    const std::unordered_map<std::string, VarcharEnumParameter>&
        varcharEnumParameterVariableBindings) {
  const auto& baseName = typeSignature.baseName();

  if (variables.count(baseName)) {
    auto it = typeVariablesBindings.find(baseName);
    if (it == typeVariablesBindings.end()) {
      return nullptr;
    }
    return it->second;
  }

  // Type is not a variable.
  auto typeName = boost::algorithm::to_upper_copy(baseName);

  const auto& params = typeSignature.parameters();
  std::vector<TypeParameter> typeParameters;

  for (auto& param : params) {
    auto literal =
        tryResolveLongLiteral(param, variables, integerVariablesBindings);
    if (literal.has_value()) {
      typeParameters.emplace_back(literal.value());
      continue;
    }
    auto longEnumParameterliteral =
        tryResolveLongEnumLiteral(param, longEnumParameterVariableBindings);
    if (longEnumParameterliteral.has_value()) {
      typeParameters.emplace_back(longEnumParameterliteral.value());
      continue;
    }
    auto varcharEnumParameterliteral = tryResolveVarcharEnumLiteral(
        param, varcharEnumParameterVariableBindings);
    if (varcharEnumParameterliteral.has_value()) {
      typeParameters.emplace_back(varcharEnumParameterliteral.value());
      continue;
    }

    auto type = tryResolveType(
        param,
        variables,
        typeVariablesBindings,
        integerVariablesBindings,
        longEnumParameterVariableBindings,
        varcharEnumParameterVariableBindings);
    if (!type) {
      return nullptr;
    }
    typeParameters.emplace_back(type, param.rowFieldName());
  }

  try {
    if (auto type = getType(typeName, typeParameters)) {
      return type;
    }
  } catch (const std::exception&) {
    // TODO Perhaps, modify getType to add suppress-errors flag.
    return nullptr;
  }

  auto typeKind = TypeKindName::tryToTypeKind(typeName);
  if (!typeKind.has_value()) {
    return nullptr;
  }

  // getType(parameters) function doesn't support OPAQUE type.
  switch (*typeKind) {
    case TypeKind::OPAQUE:
      return OpaqueType::create<void>();
    default:
      return nullptr;
  }
}
} // namespace facebook::velox::exec
