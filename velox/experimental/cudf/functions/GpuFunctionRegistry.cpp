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
#include "velox/experimental/cudf/functions/GpuVectorFunction.h"

namespace facebook::velox::gpu {

GpuFunctionRegistry& GpuFunctionRegistry::instance() {
  static GpuFunctionRegistry reg;
  return reg;
}

void GpuFunctionRegistry::registerFunction(
    GpuFunctionKey key,
    GpuFunctionFactory factory) {
  std::lock_guard<std::mutex> lock(mu_);
  factories_[std::move(key)] = std::move(factory);
}

GpuVectorFunction* GpuFunctionRegistry::resolveFunction(
    const GpuFunctionKey& key) const {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = resolved_.find(key);
  if (it != resolved_.end()) {
    return it->second.get();
  }

  auto factIt = factories_.find(key);
  if (factIt == factories_.end()) {
    return nullptr;
  }

  auto fn = factIt->second();
  auto* ptr = fn.get();
  const_cast<GpuFunctionRegistry*>(this)->resolved_[key] = std::move(fn);
  return ptr;
}

bool GpuFunctionRegistry::hasFunction(const GpuFunctionKey& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  return factories_.count(key) > 0;
}

size_t GpuFunctionRegistry::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return factories_.size();
}

void GpuFunctionRegistry::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  factories_.clear();
  resolved_.clear();
}

} // namespace facebook::velox::gpu
