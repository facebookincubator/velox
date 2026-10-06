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
#include <charconv>
#include <limits>
#include <optional>
#include <string_view>

#include "velox/expression/SignatureBinder.h"
#include "velox/expression/type_calculation/TypeCalculation.h"
#include "velox/type/Type.h"
#include "velox/type/TypeUtil.h"

namespace facebook::velox::exec {
namespace {

bool isAny(const TypeSignature& typeSignature) {
  return typeSignature.baseName() == "any";
}

bool containsOnlyDigits(std::string_view value) {
  return !value.empty() &&
      std::find_if(value.begin(), value.end(), [](unsigned char c) {
        return !std::isdigit(c);
      }) == value.end();
}

// Parses a digit-only literal that fits the binder's integer range.
std::optional<int> tryParseInteger(std::string_view value) {
  if (!containsOnlyDigits(value)) {
    return std::nullopt;
  }

  int result{0};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size()) {
    return std::nullopt;
  }
  return result;
}

std::optional<int> tryResolveLongLiteral(
    const TypeSignature& parameter,
    const std::unordered_map<std::string, SignatureVariable>& variables,
    std::unordered_map<std::string, int>& integerVariablesBindings) {
  const auto& variable = parameter.baseName();

  if (containsOnlyDigits(variable)) {
    return tryParseInteger(variable);
  }

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

// Extracts only long literals that fit the binder's integer range.
std::optional<int> tryGetIntegerLiteral(const TypeParameter& parameter) {
  if (parameter.kind != TypeParameterKind::kLongLiteral) {
    return std::nullopt;
  }

  const auto value = parameter.longLiteral.value();
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    return std::nullopt;
  }
  return static_cast<int>(value);
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

// The coercion that takes 'actualType' to 'boundType', the type a variable
// resolved to across every argument sharing it.
//
// Two DECIMALs of different precision or scale are different types that share
// a name, which the coercer reports as interchangeable at no cost. Reaching
// the bound type still means rescaling the value, so say so here.
std::optional<Coercion> coerceToBoundType(
    const TypeCoercer& coercer,
    const TypePtr& actualType,
    const TypePtr& boundType) {
  if (actualType->isDecimal() && boundType->isDecimal() &&
      !actualType->equivalent(*boundType)) {
    const auto commonType = coercer.leastCommonSuperType(actualType, boundType);
    if (!commonType || !commonType->equivalent(*boundType)) {
      return std::nullopt;
    }
    return Coercion{.type = boundType, .cost = 1};
  }
  return coercer.coerce(actualType, boundType);
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
  const bool hasUnresolvedArgument = std::ranges::any_of(
      actualTypes_,
      [](const TypePtr& actualType) { return actualType == nullptr; });
  if (!allowCoercions || hasUnresolvedArgument) {
    // Partial bindings from resolved arguments are used to type lambda inputs.
    return tryBindImpl(allowCoercions, coercions);
  }

  const auto originalTypeBindings = typeVariablesBindings_;
  const auto originalIntegerBindings = integerVariablesBindings_;
  const auto originalLongEnumBindings = longEnumVariablesBindings_;
  const auto originalVarcharEnumBindings = varcharEnumVariablesBindings_;

  const auto restoreBindings = [&]() {
    typeVariablesBindings_ = originalTypeBindings;
    integerVariablesBindings_ = originalIntegerBindings;
    longEnumVariablesBindings_ = originalLongEnumBindings;
    varcharEnumVariablesBindings_ = originalVarcharEnumBindings;
    coercions.clear();
  };

  try {
    if (tryBindImpl(allowCoercions, coercions)) {
      return true;
    }
  } catch (...) {
    restoreBindings();
    throw;
  }

  restoreBindings();
  return false;
}

bool SignatureBinder::tryBindImpl(
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

    if (!validateLiteralParameterConstraints()) {
      return false;
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
            if (auto coercion =
                    coerceToBoundType(coercer_, actualTypes_[i], firstType)) {
              if (coercion->cost > 0) {
                coercions[i] = Coercion{firstType, coercion->cost};
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
  if (containsOnlyDigits(parameterName)) {
    const auto literal = tryParseInteger(parameterName);
    return literal && literal.value() == value;
  }
  if (!variables().contains(parameterName)) {
    // Return false if the parameter is not found in the signature.
    return false;
  }

  const auto& constraint = variables().at(parameterName).constraint();
  if (containsOnlyDigits(constraint)) {
    const auto literal = tryParseInteger(constraint);
    // Return false if the constraint is out of range or does not match.
    return literal && literal.value() == value;
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

namespace {

// Preserves existing bindings for shared variables while seeding unbound ones
// from the actual type.
std::optional<std::vector<int>> tryResolveProvisionalLiteralParameters(
    const std::vector<exec::TypeSignature>& parameters,
    std::span<const TypeParameter> actualParameters,
    const std::unordered_map<std::string, SignatureVariable>& variables,
    std::unordered_map<std::string, int>& integerVariablesBindings) {
  if (parameters.size() != actualParameters.size()) {
    return std::nullopt;
  }

  std::vector<int> provisionalValues;
  provisionalValues.reserve(parameters.size());
  for (auto i = 0; i < parameters.size(); ++i) {
    const auto actualValue = tryGetIntegerLiteral(actualParameters[i]);
    if (!actualValue) {
      return std::nullopt;
    }

    const auto& parameterName = parameters[i].baseName();
    if (containsOnlyDigits(parameterName)) {
      const auto literal = tryParseInteger(parameterName);
      if (!literal) {
        return std::nullopt;
      }
      provisionalValues.push_back(literal.value());
      continue;
    }

    const auto variableIt = variables.find(parameterName);
    if (variableIt == variables.end() ||
        !variableIt->second.isIntegerParameter()) {
      return std::nullopt;
    }

    const auto binding =
        integerVariablesBindings.emplace(parameterName, actualValue.value())
            .first;
    provisionalValues.push_back(binding->second);
  }
  return provisionalValues;
}

// Widens a variable DECIMAL precision to preserve the source's integral
// digits, then validates the resulting precision and scale.
bool tryWidenDecimalParameters(
    const TypePtr& actualType,
    const std::vector<exec::TypeSignature>& parameters,
    const std::unordered_map<std::string, SignatureVariable>& variables,
    std::unordered_map<std::string, int>& integerVariablesBindings,
    std::vector<int>& provisionalValues) {
  if (!actualType->isDecimal() || parameters.size() != 2) {
    return true;
  }

  const auto& precisionName = parameters[0].baseName();
  const auto precisionVariable = variables.find(precisionName);
  if (precisionVariable != variables.end() &&
      precisionVariable->second.isIntegerParameter()) {
    const auto [sourcePrecision, sourceScale] =
        getDecimalPrecisionScale(*actualType);
    const auto targetScale = provisionalValues.at(1);
    const auto minimumPrecision =
        static_cast<int64_t>(sourcePrecision) - sourceScale + targetScale;
    if (minimumPrecision < 1 || minimumPrecision < targetScale ||
        minimumPrecision > LongDecimalType::kMaxPrecision) {
      return false;
    }
    if (provisionalValues.at(0) < minimumPrecision) {
      const auto widenedPrecision = static_cast<int>(minimumPrecision);
      provisionalValues.at(0) = widenedPrecision;
      integerVariablesBindings[precisionName] = widenedPrecision;
    }
  }

  return provisionalValues.at(0) >= 1 &&
      provisionalValues.at(0) <= LongDecimalType::kMaxPrecision &&
      provisionalValues.at(1) >= 0 &&
      provisionalValues.at(1) <= provisionalValues.at(0);
}

// Requires the common type to retain the formal type's base and arity so its
// literal parameters still describe this signature.
TypePtr tryResolveLiteralParameterTarget(
    const TypeCoercer& coercer,
    const exec::TypeSignature& typeSignature,
    const TypePtr& actualType,
    const std::vector<int>& provisionalValues) {
  std::vector<TypeParameter> provisionalParameters;
  provisionalParameters.reserve(provisionalValues.size());
  for (const auto value : provisionalValues) {
    provisionalParameters.emplace_back(value);
  }

  TypePtr provisionalType;
  try {
    provisionalType = getType(
        boost::algorithm::to_upper_copy(typeSignature.baseName()),
        provisionalParameters);
  } catch (const std::exception&) {
    return nullptr;
  }

  auto targetType = provisionalType
      ? coercer.leastCommonSuperType(actualType, provisionalType)
      : nullptr;
  if (!targetType ||
      !boost::algorithm::iequals(
          targetType->name(), typeSignature.baseName()) ||
      targetType->parameters().size() != typeSignature.parameters().size()) {
    return nullptr;
  }
  return targetType;
}

// Rejects fixed-literal mismatches before replacing provisional bindings with
// the target's parameters.
bool tryBindTargetLiteralParameters(
    const std::vector<exec::TypeSignature>& parameters,
    const TypePtr& targetType,
    std::unordered_map<std::string, int>& integerVariablesBindings) {
  for (auto i = 0; i < parameters.size(); ++i) {
    const auto targetValue = tryGetIntegerLiteral(targetType->parameters()[i]);
    if (!targetValue) {
      return false;
    }

    const auto& parameterName = parameters[i].baseName();
    if (containsOnlyDigits(parameterName)) {
      const auto literal = tryParseInteger(parameterName);
      if (!literal || literal.value() != targetValue.value()) {
        return false;
      }
      continue;
    }
    integerVariablesBindings[parameterName] = targetValue.value();
  }
  return true;
}

} // namespace

bool SignatureBinder::tryBindLiteralParametersWithCoercion(
    const exec::TypeSignature& typeSignature,
    const TypePtr& actualType) {
  const auto& parameters = typeSignature.parameters();
  auto provisionalValues = tryResolveProvisionalLiteralParameters(
      parameters,
      actualType->parameters(),
      variables(),
      integerVariablesBindings_);
  if (!provisionalValues) {
    return false;
  }

  if (!tryWidenDecimalParameters(
          actualType,
          parameters,
          variables(),
          integerVariablesBindings_,
          provisionalValues.value())) {
    return false;
  }

  const auto targetType = tryResolveLiteralParameterTarget(
      coercer_, typeSignature, actualType, provisionalValues.value());
  if (!targetType) {
    return false;
  }
  return tryBindTargetLiteralParameters(
      parameters, targetType, integerVariablesBindings_);
}

bool SignatureBinder::validateLiteralParameterConstraints() const {
  for (const auto& [name, variable] : variables()) {
    if (!variable.isIntegerParameter() || variable.constraint().empty()) {
      continue;
    }

    const auto binding = integerVariablesBindings_.find(name);
    if (binding == integerVariablesBindings_.end()) {
      continue;
    }

    auto calculatedBindings = integerVariablesBindings_;
    expression::calculation::evaluate(
        fmt::format("{}={}", name, variable.constraint()), calculatedBindings);

    const auto calculated = calculatedBindings.find(name);
    if (calculated == calculatedBindings.end() ||
        calculated->second != binding->second) {
      return false;
    }
  }
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
    const auto availableCoercion =
        coerceToBoundType(coercer_, actualType, boundType);
    VELOX_CHECK(availableCoercion.has_value());

    if (availableCoercion->cost > 0) {
      coercion.type = boundType;
      coercion.cost = availableCoercion->cost;
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
  if (isAny(typeSignature)) {
    return true;
  }

  const auto& baseName = typeSignature.baseName();

  auto variableIt = variables().find(baseName);
  if (variableIt != variables().end()) {
    // Variables cannot have further parameters.
    VELOX_CHECK(
        typeSignature.parameters().empty(),
        "Variables with parameters are not supported");
    const auto& variable = variableIt->second;
    if (!variable.isTypeParameter()) {
      // Integer variables (e.g. decimal precision and scale) are bound
      // later by tryBind. Skip them here.
      return true;
    }

    if (!variable.isEligibleType(*actualType)) {
      return false;
    }

    auto bindingIt = typeVariablesBindings_.find(baseName);
    if (bindingIt == typeVariablesBindings_.end()) {
      typeVariablesBindings_[baseName] = actualType;
      return true;
    }

    if (auto superType =
            coercer_.leastCommonSuperType(actualType, bindingIt->second)) {
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

  auto typeToBind = actualType;
  if (!boost::algorithm::iequals(baseName, actualType->name())) {
    const auto typeName = boost::algorithm::to_upper_copy(baseName);
    const auto baseCoercion =
        coercer_.tryCoerceToTypeBase(*actualType, typeName);
    if (!baseCoercion) {
      if (actualType->isUnknown()) {
        // Bind type variables to UNKNOWN so the parameterized formal resolves.
        for (const auto& param : params) {
          if (!tryBindVariablesWithCoercion(param, UNKNOWN())) {
            return false;
          }
        }
        return true;
      }
      return false;
    }
    typeToBind = baseCoercion->type;
  }

  if (params.size() != typeToBind->parameters().size()) {
    return false;
  }

  if (!params.empty() &&
      std::ranges::all_of(typeToBind->parameters(), [](const auto& parameter) {
        return parameter.kind == TypeParameterKind::kLongLiteral;
      })) {
    return tryBindLiteralParametersWithCoercion(typeSignature, typeToBind);
  }

  for (auto i = 0; i < params.size(); i++) {
    const auto& actualParameter = typeToBind->parameters()[i];
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

  if (allowCoercion && !typeSignature.parameters().empty()) {
    const auto resolvedType = SignatureBinder::tryResolveType(
        typeSignature,
        variables(),
        typeVariablesBindings_,
        integerVariablesBindings_,
        longEnumVariablesBindings_,
        varcharEnumVariablesBindings_);
    if (resolvedType &&
        std::ranges::all_of(
            resolvedType->parameters(), [](const auto& parameter) {
              return parameter.kind == TypeParameterKind::kLongLiteral;
            })) {
      const auto availableCoercion =
          coerceToBoundType(coercer_, actualType, resolvedType);
      if (!availableCoercion) {
        return false;
      }
      if (availableCoercion->cost > 0) {
        coercion = availableCoercion.value();
      }
      return true;
    }
  }

  // Type is not a variable.
  const auto& baseName = typeSignature.baseName();
  auto typeName = boost::algorithm::to_upper_copy(baseName);
  if (!boost::algorithm::iequals(typeName, actualType->name())) {
    if (allowCoercion &&
        (typeSignature.parameters().empty() || actualType->isUnknown())) {
      auto resolvedType = SignatureBinder::tryResolveType(
          typeSignature,
          variables(),
          typeVariablesBindings_,
          integerVariablesBindings_,
          longEnumVariablesBindings_,
          varcharEnumVariablesBindings_);
      if (resolvedType) {
        if (auto availableCoercion =
                coercer_.coerce(actualType, resolvedType)) {
          coercion = availableCoercion.value();
          return true;
        }
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
    std::vector<TypePtr>& coercions,
    const TypeCoercer& coercer) {
  std::vector<std::pair<std::vector<Coercion>, TypePtr>> candidates;
  for (const auto& signature : signatures) {
    SignatureBinder binder(*signature, argTypes, coercer);
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

  // Aggregate/window signatures don't model null-on-null, so UNKNOWN ties stay
  // ambiguous here (no tie-break).
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
