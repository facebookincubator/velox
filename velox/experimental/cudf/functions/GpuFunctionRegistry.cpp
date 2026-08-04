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

#include "velox/experimental/cudf/functions/GpuFunctionRegistry.h"

#include <algorithm>
#include <cctype>
#include <mutex>

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

bool sameSignature(
    const GpuFunctionSignature& lhs,
    const GpuFunctionSignature& rhs) {
  return lhs.returnType == rhs.returnType &&
      lhs.argumentTypes == rhs.argumentTypes;
}

} // namespace

bool registerGpuKernel(
    const std::vector<std::string>& aliases,
    GpuFunctionSignature signature,
    GpuLaunchFn launch,
    bool overwrite) {
  std::lock_guard<std::mutex> guard(registryMutex());

  bool registeredAll = true;
  for (const auto& alias : aliases) {
    auto& entries = registry()[sanitizeName(alias)];

    auto existing = std::find_if(
        entries.begin(), entries.end(), [&](const GpuFunctionEntry& entry) {
          return sameSignature(entry.signature, signature);
        });

    if (existing != entries.end()) {
      if (!overwrite) {
        registeredAll = false;
        continue;
      }
      existing->launch = launch;
      continue;
    }

    entries.push_back(GpuFunctionEntry{signature, launch});
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
