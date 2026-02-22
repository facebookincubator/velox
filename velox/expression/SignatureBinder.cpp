/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <optional>

#include "velox/expression/SignatureBinder.h"
#include "velox/expression/type_calculation/TypeCalculation.h"
#include "velox/type/Type.h"
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

  {
    auto integerIt = integerVariablesBindings.find(variable);
    if (integerIt != integerVariablesBindings.end()) {
      return integerIt->second;
    }
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

  auto integerIt = integerVariablesBindings.find(variable);
  VELOX_CHECK(
      integerIt != integerVariablesBindings.end(),
      "Variable calculation failed: {}",
      variable);
  return integerIt->second;
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
  if (signature.rowFieldName().has_value() &&
      (*signature.rowFieldName() != asRowType(actualType)->nameOf(idx))) {
    return false;
  }
  return true;
}

} // namespace

bool SignatureBinder::tryBindWithCoercions(std::vector<Coercion>& coercions) {
  return tryBind(true, coercions);
}

bool SignatureBinder::tryBind() {
  std::vector<Coercion> coercions;
  return tryBind(false, coercions);
}

bool SignatureBinder::tryBind(
    bool allowCoercions,
    std::vector<Coercion>& coercions) {
  const auto numActualTypes = actualTypes_.size();

  if (allowCoercions) {
    coercions.clear();
    coercions.resize(numActualTypes);
  }

  const auto& formalArgs = signature_.argumentTypes();
  const auto numFormalArgs = formalArgs.size();

  if (signature_.variableArity()) {
    if (numActualTypes < numFormalArgs - 1) {
      return false;
    }
  } else {
    if (numFormalArgs != numActualTypes) {
      return false;
    }
  }

  if (allowCoercions && !variables().empty()) {
    for (auto i = 0; i < numActualTypes; i++) {
      if (actualTypes_[i]) {
        const auto& typeSignature =
            i < numFormalArgs ? formalArgs[i] : formalArgs[numFormalArgs - 1];

        if (!tryBindVariablesWithCoercion(typeSignature, actualTypes_[i])) {
          return false;
        }
      }
    }
  }

  for (auto i = 0; i < numFormalArgs && i < numActualTypes; i++) {
    if (actualTypes_[i]) {
      if (allowCoercions) {
        if (!SignatureBinderBase::tryBindWithCoercion(
                formalArgs[i], actualTypes_[i], coercions[i])) {
          return false;
        }
      } else {
        if (!SignatureBinderBase::tryBind(formalArgs[i], actualTypes_[i])) {
          return false;
        }
      }
    } else {
      return false;
    }
  }

  if (signature_.variableArity()) {
    if (!isAny(signature_.argumentTypes().back())) {
      if (numActualTypes > numFormalArgs) {
        if (allowCoercions) {
          auto firstType = actualTypes_[numFormalArgs - 1];
          if (coercions[numFormalArgs - 1].type != nullptr) {
            firstType = coercions[numFormalArgs - 1].type;
          }

          for (auto i = numFormalArgs; i < numActualTypes; i++) {
            if (auto cost =
                    TypeCoercer::coercible(actualTypes_[i], firstType)) {
              if (cost.value() > 0) {
                coercions[i] = Coercion{firstType, cost.value()};
              }
            } else {
              return false;
            }
          }

        } else {
          const auto& firstType = actualTypes_[numFormalArgs - 1];
          for (auto i = numFormalArgs; i < numActualTypes; i++) {
            if (!firstType->equivalent(*actualTypes_[i])) {
              return false;
            }
          }
        }
      }
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
  if (!variables().contains(parameterName)) {
    // Return false if the parameter is not found in the signature.
    return false;
  }

  const auto& constraint = variables().at(parameterName).constraint();
  if (isPositiveInteger(constraint) && atoi(constraint.c_str()) != value) {
    // Return false if the actual value does not match the constraint.
    return false;
  }

  auto integerIt = integerVariablesBindings_.find(parameterName);
  if (integerIt != integerVariablesBindings_.end()) {
    // Return false if the parameter is found with a different value.
    if (integerIt->second != value) {
      return false;
    }
  }

  // Bind the variable.
  integerVariablesBindings_[parameterName] = value;
  return true;
}

std::optional<bool> SignatureBinderBase::checkSetTypeVariable(
    const exec::TypeSignature& typeSignature,
    const TypePtr& actualType,
    bool allowCoercion,
    Coercion& coercion) {
  const auto& baseName = typeSignature.baseName();

  auto variableIt = variables().find(baseName);
  if (variableIt == variables().end()) {
    return std::nullopt;
  }

  if (allowCoercion) {
    // Variables must be already set.
    auto bindingIt = typeVariablesBindings_.find(baseName);
    VELOX_CHECK(bindingIt != typeVariablesBindings_.end());

    const auto& boundType = bindingIt->second;
    const auto cost = TypeCoercer::coercible(actualType, boundType);
    VELOX_CHECK(cost.has_value());

    if (cost.value() > 0) {
      coercion.type = boundType;
      coercion.cost = cost.value();
    }
    return true;
  }

  // Variables cannot have further parameters.
  VELOX_CHECK(
      typeSignature.parameters().empty(),
      "Variables with parameters are not supported");
  const auto& variable = variableIt->second;
  VELOX_CHECK(variable.isTypeParameter(), "Not expecting integer variable");

  auto bindingIt = typeVariablesBindings_.find(baseName);
  if (bindingIt != typeVariablesBindings_.end()) {
    return bindingIt->second->equivalent(*actualType);
  }

  if (!variable.isEligibleType(*actualType)) {
    return false;
  }

  typeVariablesBindings_[baseName] = actualType;
  return true;
}

bool SignatureBinderBase::tryBindWithCoercion(
    const exec::TypeSignature& typeSignature,
    const TypePtr& actualType,
    Coercion& coercion) {
  return tryBind(typeSignature, actualType, true, coercion);
}

bool SignatureBinderBase::tryBind(
    const exec::TypeSignature& typeSignature,
    const TypePtr& actualType) {
  Coercion coercion;
  return tryBind(typeSignature, actualType, false, coercion);
}

bool SignatureBinder::tryBindVariablesWithCoercion(
    const exec::TypeSignature& typeSignature,
    const TypePtr& actualType) {
  const auto& baseName = typeSignature.baseName();

  auto variableIt = variables().find(baseName);
  if (variableIt != variables().end()) {
    // Variables cannot have further parameters.
    VELOX_CHECK(
        typeSignature.parameters().empty(),
        "Variables with parameters are not supported");
    const auto& variable = variableIt->second;
    VELOX_CHECK(variable.isTypeParameter(), "Not expecting integer variable");

    if (!variable.isEligibleType(*actualType)) {
      return false;
    }

    auto bindingIt = typeVariablesBindings_.find(baseName);
    if (bindingIt == typeVariablesBindings_.end()) {
      typeVariablesBindings_[baseName] = actualType;
      return true;
    }

    if (auto superType =
            TypeCoercer::leastCommonSuperType(actualType, bindingIt->second)) {
      typeVariablesBindings_[baseName] = superType;
      return true;
    }

    return false;
  }

  if (typeSignature.isHomogeneousRow()) {
    // TODO Add coercion support.
    return true;
  }

  const auto& params = typeSignature.parameters();
  if (params.size() != actualType->parameters().size()) {
    return false;
  }

  for (auto i = 0; i < params.size(); i++) {
    const auto& actualParameter = actualType->parameters()[i];
    if (actualParameter.kind == TypeParameterKind::kType) {
      if (!tryBindVariablesWithCoercion(params[i], actualParameter.type)) {
        return false;
      }
    }
  }

  return true;
}

bool SignatureBinderBase::tryBind(
    const exec::TypeSignature& typeSignature,
    const TypePtr& actualType,
    bool allowCoercion,
    Coercion& coercion) {
  coercion.reset();
  if (isAny(typeSignature)) {
    return true;
  }

  if (auto result = checkSetTypeVariable(
          typeSignature, actualType, allowCoercion, coercion)) {
    return result.value();
  }

  // Type is not a variable.
  const auto& baseName = typeSignature.baseName();
  auto typeName = boost::algorithm::to_upper_copy(baseName);
  if (!boost::algorithm::iequals(typeName, actualType->name())) {
    if (allowCoercion) {
      if (auto availableCoercion =
              TypeCoercer::coerceTypeBase(actualType, typeName)) {
        coercion = availableCoercion.value();
        return true;
      }
    }
    return false;
  }

  const auto& params = typeSignature.parameters();

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

    // First, check and extract the common child type if homogeneous.
    const auto actualChildType =
        velox::type::tryGetHomogeneousRowChild(actualType);
    if (!actualChildType) {
      return false;
    }

    // TODO Add coercion support.
    if (auto result = checkSetTypeVariable(
            typeParam, actualChildType, /*allowCoercion=*/false, coercion)) {
      return result.value();
    }

    return tryBind(typeParam, actualChildType);
  }

  // Type Parameters can recurse.
  if (params.size() != actualType->parameters().size()) {
    return false;
  }

  std::vector<Coercion> paramCoercions;
  if (allowCoercion) {
    paramCoercions.resize(params.size());
  }

  for (auto i = 0; i < params.size(); i++) {
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

        if (allowCoercion) {
          if (!tryBindWithCoercion(
                  params[i], actualParameter.type, paramCoercions[i])) {
            return false;
          }

        } else if (!tryBind(params[i], actualParameter.type)) {
          return false;
        }

        break;
    }
  }

  if (allowCoercion) {
    const bool hasCoercion = std::ranges::any_of(
        paramCoercions,
        [](const auto& coercion) { return coercion.type != nullptr; });

    if (hasCoercion) {
      std::vector<TypeParameter> newParams;
      newParams.reserve(params.size());
      for (auto i = 0; i < params.size(); i++) {
        if (paramCoercions[i].type != nullptr) {
          newParams.push_back(
              TypeParameter(paramCoercions[i].type, params[i].rowFieldName()));
          coercion.cost += paramCoercions[i].cost;
        } else {
          newParams.push_back(actualType->parameters()[i]);
        }
      }

      coercion.type = getType(typeName, newParams);
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

  if (variables.contains(baseName)) {
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
TypePtr tryResolveReturnTypeWithCoercions(
    const std::vector<FunctionSignaturePtr>& signatures,
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions) {
  std::vector<std::pair<std::vector<Coercion>, TypePtr>> candidates;
  for (const auto& signature : signatures) {
    SignatureBinder binder(*signature, argTypes);
    std::vector<Coercion> requiredCoercions;
    if (binder.tryBindWithCoercions(requiredCoercions)) {
      auto type = binder.tryResolveReturnType();
      bool needsCoercion = false;
      for (const auto& c : requiredCoercions) {
        if (c.type != nullptr) {
          needsCoercion = true;
          break;
        }
      }
      if (!needsCoercion) {
        // Exact match. No coercions needed.
        coercions.resize(argTypes.size(), nullptr);
        return type;
      }
      candidates.emplace_back(std::move(requiredCoercions), type);
    }
  }

  if (auto index = Coercion::pickLowestCost(candidates)) {
    const auto& requiredCoercions = candidates[index.value()].first;
    coercions.reserve(requiredCoercions.size());
    for (const auto& coercion : requiredCoercions) {
      coercions.push_back(coercion.type);
    }
    return candidates[index.value()].second;
  }

  return nullptr;
}

} // namespace facebook::velox::exec
