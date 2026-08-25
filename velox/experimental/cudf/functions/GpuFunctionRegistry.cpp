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

#include "velox/experimental/cudf/functions/GpuFunctionLookup.h"

#include <algorithm>
#include <cctype>
#include <mutex>
#include <unordered_set>

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

using Registry = std::unordered_map<std::string, std::vector<GpuFunctionEntry>>;

Registry& registry() {
  static Registry instance;
  return instance;
}

std::mutex& registryMutex() {
  static std::mutex instance;
  return instance;
}

/// Velox lowercases function names before keying the registry; matching that
/// keeps GPU and CPU lookups agreeing on what counts as the same function.
std::string sanitizeName(const std::string& name) {
  std::string result(name.size(), '\0');
  std::transform(name.begin(), name.end(), result.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return result;
}

/// Turns the strings that crossed the device boundary into the signature Velox
/// would have built for the same function.
///
/// This is the one place the two worlds are stitched together, and it is a
/// direct translation because the device side derives its strings from
/// SimpleTypeTrait<T>::name -- the same source Velox's TypeAnalysis reads when
/// it builds a signature for the CPU registration of that function.
exec::FunctionSignaturePtr toVeloxSignature(
    const GpuFunctionSignature& signature) {
  exec::FunctionSignatureBuilder builder;
  // Declared before use, and deduplicated because the builder treats a repeat
  // as an error while a type naming the same variable twice is normal.
  std::unordered_set<std::string> declared;
  for (const auto& variable : signature.integerVariables) {
    if (declared.insert(variable).second) {
      builder.integerVariable(variable);
    }
  }
  builder.returnType(signature.returnType);
  for (const auto& argumentType : signature.argumentTypes) {
    builder.argumentType(argumentType);
  }
  if (signature.variadicTail) {
    builder.variableArity();
  }
  return builder.build();
}

} // namespace

bool registerGpuKernel(
    const std::vector<std::string>& aliases,
    GpuFunctionSignature signature,
    GpuLaunchFn launch,
    bool overwrite) {
  auto veloxSignature = toVeloxSignature(signature);

  std::lock_guard<std::mutex> guard(registryMutex());

  bool registeredAll = true;
  for (const auto& alias : aliases) {
    auto& entries = registry()[sanitizeName(alias)];

    // FunctionSignature::operator== compares argument types, return type and
    // variable arity together, so an overload differing only in arity is
    // correctly a different entry rather than a replacement.
    auto existing = std::find_if(
        entries.begin(), entries.end(), [&](const GpuFunctionEntry& entry) {
          return *entry.signature == *veloxSignature;
        });

    if (existing != entries.end()) {
      if (!overwrite) {
        registeredAll = false;
        continue;
      }
      existing->launch = launch;
      continue;
    }

    entries.push_back(GpuFunctionEntry{veloxSignature, launch});
  }
  return registeredAll;
}

const std::unordered_map<std::string, std::vector<GpuFunctionEntry>>&
gpuFunctionRegistry() {
  return registry();
}

void clearGpuFunctionRegistry() {
  std::lock_guard<std::mutex> guard(registryMutex());
  registry().clear();
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
