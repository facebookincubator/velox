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
#include "velox/exec/AggregateFunctionRegistry.h"

#include "velox/exec/Aggregate.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/type/Type.h"

namespace facebook::velox::exec {

namespace {
constexpr char kSignatureNotSupportedError[] =
    "Aggregate function signature is not supported: {}. Supported signatures: {}.";
} // namespace

TypePtr resolveResultType(
    const std::string& name,
    const std::vector<TypePtr>& argTypes) {
  if (auto signatures = getAggregateFunctionSignatures(name)) {
    for (const auto& signature : signatures.value()) {
      SignatureBinder binder(*signature, argTypes, TypeCoercer::defaults());
      if (binder.tryBind()) {
        return binder.tryResolveReturnType();
      }
    }

    VELOX_USER_FAIL(
        kSignatureNotSupportedError,
        toString(name, argTypes),
        toString(signatures.value()));
  }

  VELOX_USER_FAIL("Aggregate function not registered: {}", name);
}

TypePtr resolveResultTypeWithCoercions(
    const std::string& name,
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions,
    const TypeCoercer& coercer) {
  coercions.clear();

  if (auto signatures = getAggregateFunctionSignatures(name)) {
    std::vector<FunctionSignaturePtr> baseSignatures(
        signatures.value().begin(), signatures.value().end());
    if (auto type = tryResolveReturnTypeWithCoercions(
            baseSignatures, argTypes, coercions, coercer)) {
      return type;
    }

    VELOX_USER_FAIL(
        kSignatureNotSupportedError,
        toString(name, argTypes),
        toString(signatures.value()));
  }

  VELOX_USER_FAIL("Aggregate function not registered: {}", name);
}

TypePtr resolveIntermediateType(
    const std::string& name,
    const std::vector<TypePtr>& argTypes) {
  if (auto signatures = getAggregateFunctionSignatures(name)) {
    for (const auto& signature : signatures.value()) {
      SignatureBinder binder(*signature, argTypes, TypeCoercer::defaults());
      if (binder.tryBind()) {
        return binder.tryResolveType(signature->intermediateType());
      }
    }

    VELOX_USER_FAIL(
        kSignatureNotSupportedError,
        toString(name, argTypes),
        toString(signatures.value()));
  } else {
    VELOX_USER_FAIL("Aggregate function not registered: {}", name);
  }
}

std::vector<std::string> getAggregateFunctionNames() {
  std::vector<std::string> names;
  exec::aggregateFunctions().withRLock([&](const auto& map) {
    names.reserve(map.size());
    for (const auto& function : map) {
      names.push_back(function.first);
    }
  });

  return names;
}

} // namespace facebook::velox::exec
