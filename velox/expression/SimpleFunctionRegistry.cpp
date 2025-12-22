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

#include "velox/expression/SimpleFunctionRegistry.h"

namespace facebook::velox::exec {
namespace {

SimpleFunctionRegistry& simpleFunctionsInternal() {
  static SimpleFunctionRegistry instance;
  return instance;
}
} // namespace

const SimpleFunctionRegistry& simpleFunctions() {
  return simpleFunctionsInternal();
}

SimpleFunctionRegistry& mutableSimpleFunctions() {
  return simpleFunctionsInternal();
}

bool SimpleFunctionRegistry::registerFunctionInternal(
    const std::string& name,
    const std::shared_ptr<const Metadata>& metadata,
    const FunctionFactory& factory,
    bool overwrite) {
  const auto sanitizedName = sanitizeName(name);
  return registeredFunctions_.withWLock([&](auto& map) {
    SignatureMap& signatureMap = map[sanitizedName];
    auto& functions = signatureMap[*metadata->signature()];

    for (auto it = functions.begin(); it != functions.end(); ++it) {
      const auto& otherMetadata = (*it)->getMetadata();

      if (metadata->physicalSignatureEquals(otherMetadata)) {
        if (!overwrite) {
          return false;
        }
        functions.erase(it);
        break;
      }

      // Make sure default-null-behavior and deterministic properties are the
      // same for all implementations of a given logical signature.
      VELOX_USER_CHECK_EQ(
          metadata->defaultNullBehavior(), otherMetadata.defaultNullBehavior());
      VELOX_USER_CHECK_EQ(
          metadata->isDeterministic(), otherMetadata.isDeterministic());
    }

    functions.emplace_back(
        std::make_unique<const FunctionEntry>(metadata, factory));
    return true;
  });
}

void SimpleFunctionRegistry::removeFunction(const std::string& name) {
  const auto sanitizedName = sanitizeName(name);
  registeredFunctions_.withWLock([&](auto& map) { map.erase(sanitizedName); });
}

namespace {
// This function is not thread safe. It should be called only from within a
// synchronized read region of registeredFunctions_.
const SignatureMap* getSignatureMap(
    const std::string& name,
    const FunctionMap& registeredFunctions) {
  const auto sanitizedName = sanitizeName(name);
  const auto it = registeredFunctions.find(sanitizedName);
  return it != registeredFunctions.end() ? &it->second : nullptr;
}
} // namespace

std::unordered_map<std::string, std::vector<const FunctionSignature*>>
SimpleFunctionRegistry::getFunctionSignatureMap() const {
  std::unordered_map<std::string, std::vector<const FunctionSignature*>> result;
  registeredFunctions_.withRLock([&](const auto& map) {
    result.reserve(map.size());

    for (const auto& entry : map) {
      std::vector<const FunctionSignature*> signatures;
      const auto signatureMap = &entry.second;
      signatures.reserve(signatureMap->size());
      for (const auto& pair : *signatureMap) {
        signatures.emplace_back(&pair.first);
      }
      result[entry.first] = signatures;
    }
  });
  return result;
}

std::vector<const FunctionSignature*>
SimpleFunctionRegistry::getFunctionSignatures(const std::string& name) const {
  std::vector<const FunctionSignature*> signatures;
  registeredFunctions_.withRLock([&](const auto& map) {
    if (const auto* signatureMap = getSignatureMap(name, map)) {
      signatures.reserve(signatureMap->size());
      for (const auto& pair : *signatureMap) {
        signatures.emplace_back(&pair.first);
      }
    }
  });

  return signatures;
}

std::vector<std::pair<VectorFunctionMetadata, const FunctionSignature*>>
SimpleFunctionRegistry::getFunctionSignaturesAndMetadata(
    const std::string& name) const {
  std::vector<std::pair<VectorFunctionMetadata, const FunctionSignature*>>
      result;
  registeredFunctions_.withRLock([&](const auto& map) {
    if (const auto* signatureMap = getSignatureMap(name, map)) {
      result.reserve(signatureMap->size());
      for (const auto& [signature, functions] : *signatureMap) {
        VectorFunctionMetadata metadata{
            false,
            functions[0]->getMetadata().isDeterministic(),
            functions[0]->getMetadata().defaultNullBehavior()};
        result.emplace_back(
            std::pair<VectorFunctionMetadata, const FunctionSignature*>{
                metadata, &signature});
      }
    }
  });
  return result;
}

namespace {

// Same as Type::kindEquals, but matches UNKNOWN in 'physicalType' to any type.
bool physicalTypeMatches(const TypePtr& type, const TypePtr& physicalType) {
  if (physicalType->isUnKnown()) {
    return true;
  }

  if (type->kind() != physicalType->kind()) {
    return false;
  }

  if (type->size() != physicalType->size()) {
    return false;
  }

  for (auto i = 0; i < type->size(); ++i) {
    if (!physicalTypeMatches(type->childAt(i), physicalType->childAt(i))) {
      return false;
    }
  }

  return true;
}

} // namespace

std::optional<SimpleFunctionRegistry::ResolvedSimpleFunction>
SimpleFunctionRegistry::resolveFunction(
    const std::string& name,
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>* coercions) const {
  using Candidate = std::pair<const FunctionEntry*, TypePtr>;
  const FunctionEntry* selectedFunction{};
  TypePtr selectedType;
  auto selectedPriority = kImpossibleCoercionCost;
  std::vector<Coercion> selectedCoercions;
  size_t selectedCount = 0;
  registeredFunctions_.withRLock([&](const auto& map) {
    const auto* signatureMap = getSignatureMap(name, map);
    if (!signatureMap) {
      return;
    }
    std::vector<Coercion> requiredCoercions;

    for (const auto& [candidateSignature, functionEntry] : *signatureMap) {
      SignatureBinder binder(candidateSignature, argTypes);

      if (!binder.tryBind(coercions ? &requiredCoercions : nullptr)) {
        continue;
      }
      for (const auto& currentCandidate : functionEntry) {
        const auto& m = currentCandidate->getMetadata();

        // For variadic signatures, number of arguments in function call
        // may be one less than number of arguments in the signature.
        const auto numArgsToMatch =
            std::min(argTypes.size(), m.argPhysicalTypes().size());

        bool match = true;
        for (auto i = 0; i < numArgsToMatch; ++i) {
          auto argType = argTypes[i];
          if (coercions && requiredCoercions[i].type != nullptr) {
            argType = requiredCoercions[i].type;
          }
          if (!physicalTypeMatches(argType, m.argPhysicalTypes()[i])) {
            match = false;
            break;
          }
        }

        if (!match) {
          continue;
        }

        auto resultType = binder.tryResolveReturnType();
        if (!resultType) {
          continue;
        }

        if (physicalTypeMatches(resultType, m.resultPhysicalType())) {
          // Coercion cost is required to match more suitable signature.
          // For example, let's assume function has two signatures:
          // 1. BIGINT -> BIGINT
          // 2. Generic<T> -> BIGINT
          // Then binding function to INTEGER should pick the second signature.
          // So we need to ensure that any non-zero coercion cost is higher than
          // the maximum possible cost of function signatures without coercion.
          const auto coercionCost = Coercion::overallCost(requiredCoercions);
          const auto currentPriority = m.priority() +
              core::kMaxFunctionRank * core::kMaxFunctionArgs * coercionCost;

          if (currentPriority < selectedPriority) {
            selectedFunction = currentCandidate.get();
            selectedType = resultType;
            selectedPriority = currentPriority;
            selectedCoercions = requiredCoercions;
            selectedCount = 1;
          } else if (currentPriority == selectedPriority) {
            ++selectedCount;
          }
        }
      }
    }
  });
  if (selectedCount != 1) {
    if (coercions) {
      coercions->clear();
    }
    return std::nullopt;
  }
  Coercion::convert(selectedCoercions, coercions);
  return ResolvedSimpleFunction{*selectedFunction, selectedType};
}

} // namespace facebook::velox::exec
