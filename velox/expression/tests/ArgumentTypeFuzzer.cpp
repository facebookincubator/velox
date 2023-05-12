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

#include "velox/expression/tests/ArgumentTypeFuzzer.h"

#include <boost/algorithm/string.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include "velox/expression/ReverseSignatureBinder.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/type/Type.h"

namespace facebook::velox::test {

namespace {

// Return a random type among those in kSupportedTypes determined by rng.
// TODO: Extend this function to return arbitrary random types including nested
// complex types.
TypePtr randomType(std::mt19937& rng) {
  // Decimal types are not supported because VectorFuzzer doesn't support them.
  static std::vector<TypePtr> kSupportedTypes{
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      TIMESTAMP(),
      DATE(),
      INTERVAL_DAY_TIME()};
  auto index = boost::random::uniform_int_distribution<uint32_t>(
      0, kSupportedTypes.size() - 1)(rng);
  return kSupportedTypes[index];
}

} // namespace

std::string typeToBaseName(const TypePtr& type) {
  return boost::algorithm::to_lower_copy(std::string{type->kindName()});
}

std::optional<TypeKind> baseNameToTypeKind(const std::string& typeName) {
  auto kindName = boost::algorithm::to_upper_copy(typeName);
  return tryMapNameToTypeKind(kindName);
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
    bindings_[variableName] = randomType(rng_);
  }
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

  determineUnboundedTypeVariables();
  for (auto i = 0; i < formalArgsCnt; i++) {
    TypePtr actualArg;
    if (formalArgs[i].baseName() == "any") {
      actualArg = randomType(rng_);
    } else {
      actualArg = exec::SignatureBinder::tryResolveType(
          formalArgs[i], variables(), bindings_);
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

  determineUnboundedTypeVariables();
  if (signature_.returnType().baseName() == "any") {
    returnType_ = randomType(rng_);
    return returnType_;
  } else {
    returnType_ = exec::SignatureBinder::tryResolveType(
        signature_.returnType(), variables(), bindings_);
    VELOX_CHECK_NE(returnType_, nullptr);
    return returnType_;
  }
}

} // namespace facebook::velox::test
