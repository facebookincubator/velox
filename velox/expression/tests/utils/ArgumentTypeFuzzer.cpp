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

#include "velox/expression/tests/utils/ArgumentTypeFuzzer.h"

#include <boost/algorithm/string.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include "velox/expression/ReverseSignatureBinder.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/expression/type_calculation/TypeCalculation.h"
#include "velox/type/SimpleFunctionApi.h"
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::test {

std::string typeToBaseName(const TypePtr& type) {
  return type->isDecimal()
      ? "decimal"
      : boost::algorithm::to_lower_copy(std::string{type->kindName()});
}

std::optional<TypeKind> baseNameToTypeKind(const std::string& typeName) {
  auto kindName = boost::algorithm::to_upper_copy(typeName);
  return tryMapNameToTypeKind(kindName);
}

void ArgumentTypeFuzzer::determineUnboundedIntegerVariables() {
  // Assign a random value for all integer values.
  for (const auto& [variableName, variableInfo] : variables()) {
    if (!variableInfo.isIntegerParameter() ||
        integerVariablesBindings_.count(variableName)) {
      continue;
    }

    // When decimal function is registered as vector function, the variable name
    // contains 'precision' like 'a_precision'.
    if (auto pos = variableName.find("precision"); pos != std::string::npos) {
      // Generate a random precision, and corresponding scale should not exceed
      // the precision.
      const auto precision =
          boost::random::uniform_int_distribution<uint32_t>(1, 38)(rng_);
      integerVariablesBindings_[variableName] = precision;
      const auto colName = variableName.substr(0, pos);
      integerVariablesBindings_[colName + "scale"] =
          boost::random::uniform_int_distribution<uint32_t>(0, precision)(rng_);
      continue;
    }

    // When decimal function is registered as simple function, the variable name
    // contains 'i' like 'i1'.
    if (auto pos = variableName.find("i"); pos != std::string::npos) {
      VELOX_USER_CHECK_GE(variableName.size(), 2);
      const auto index =
          std::stoi(variableName.substr(pos + 1, variableName.size()));
      if (index <= kIntegerPairSize) {
        // Generate a random precision, and corresponding scale should not
        // exceed the precision.
        const auto precision =
            boost::random::uniform_int_distribution<uint32_t>(1, 38)(rng_);
        integerVariablesBindings_[variableName] = precision;
        const auto scaleIndex = index + kIntegerPairSize;
        const auto scaleName = "i" + std::to_string(scaleIndex);
        integerVariablesBindings_[scaleName] =
            boost::random::uniform_int_distribution<uint32_t>(
                0, precision)(rng_);
      }
      continue;
    }

    integerVariablesBindings_[variableName] =
        boost::random::uniform_int_distribution<int32_t>()(rng_);
  }

  // Handle constraints.
  for (const auto& [variableName, variableInfo] : variables()) {
    const auto constraint = variableInfo.constraint();
    if (constraint == "") {
      continue;
    }
    const auto calculation = fmt::format("{}={}", variableName, constraint);
    expression::calculation::evaluate(calculation, integerVariablesBindings_);
  }
}

void ArgumentTypeFuzzer::determineUnboundedTypeVariables() {
  for (auto& [variableName, variableInfo] : variables()) {
    if (!variableInfo.isTypeParameter()) {
      continue;
    }

    if (bindings_[variableName] != nullptr) {
      continue;
    }

    // Random randomType() never generates unknown here.
    // TODO: we should extend randomType types and exclude unknown based
    // on variableInfo.
    if (variableInfo.orderableTypesOnly()) {
      bindings_[variableName] = randOrderableType();
    } else {
      bindings_[variableName] = randType();
    }
  }
}

TypePtr ArgumentTypeFuzzer::randType() {
  return velox::randType(rng_, 2);
}

TypePtr ArgumentTypeFuzzer::randOrderableType() {
  return velox::randOrderableType(rng_, 2);
}

bool ArgumentTypeFuzzer::fuzzArgumentTypes(uint32_t maxVariadicArgs) {
  const auto& formalArgs = signature_.argumentTypes();
  auto formalArgsCnt = formalArgs.size();

  if (returnType_) {
    exec::ReverseSignatureBinder binder{signature_, returnType_};
    if (!binder.tryBind()) {
      return false;
    }
    bindings_ = binder.bindings();
  } else {
    for (const auto& [name, _] : signature_.variables()) {
      bindings_.insert({name, nullptr});
    }
  }

  determineUnboundedIntegerVariables();
  determineUnboundedTypeVariables();
  for (auto i = 0; i < formalArgsCnt; i++) {
    TypePtr actualArg;
    if (formalArgs[i].baseName() == "any") {
      actualArg = randType();
    } else {
      actualArg = exec::SignatureBinder::tryResolveType(
          formalArgs[i], variables(), bindings_, integerVariablesBindings_);
      VELOX_CHECK(actualArg != nullptr);
    }
    argumentTypes_.push_back(actualArg);
  }

  // Generate random repeats of the last argument type if the signature is
  // variadic.
  if (signature_.variableArity()) {
    auto repeat = boost::random::uniform_int_distribution<uint32_t>(
        0, maxVariadicArgs)(rng_);
    auto last = argumentTypes_[formalArgsCnt - 1];
    for (int i = 0; i < repeat; ++i) {
      argumentTypes_.push_back(last);
    }
  }

  return true;
}

TypePtr ArgumentTypeFuzzer::fuzzReturnType() {
  VELOX_CHECK_EQ(
      returnType_,
      nullptr,
      "Only fuzzing uninitialized return type is allowed.");

  determineUnboundedIntegerVariables();
  determineUnboundedTypeVariables();
  if (signature_.returnType().baseName() == "any") {
    returnType_ = randType();
    return returnType_;
  } else {
    returnType_ = exec::SignatureBinder::tryResolveType(
        signature_.returnType(),
        variables(),
        bindings_,
        integerVariablesBindings_);
    VELOX_CHECK_NE(returnType_, nullptr);
    return returnType_;
  }
}

} // namespace facebook::velox::test
