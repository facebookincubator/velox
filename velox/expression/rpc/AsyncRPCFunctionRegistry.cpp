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

#include "velox/expression/rpc/AsyncRPCFunctionRegistry.h"

namespace facebook::velox::exec::rpc {

std::mutex& AsyncRPCFunctionRegistry::mutex() {
  static std::mutex instance;
  return instance;
}

std::unordered_map<std::string, AsyncRPCFunctionRegistry::Factory>&
AsyncRPCFunctionRegistry::factories() {
  static std::unordered_map<std::string, Factory> instance;
  return instance;
}

bool AsyncRPCFunctionRegistry::registerFunction(
    const std::string& name,
    Factory factory) {
  std::lock_guard<std::mutex> lock(mutex());
  auto& registry = factories();
  if (registry.count(name) > 0) {
    return false; // Already registered
  }
  // Note: Do NOT use LOG() here as this function may be called during static
  // initialization, before glog is initialized.
  registry[name] = std::move(factory);
  return true;
}

std::shared_ptr<AsyncRPCFunction> AsyncRPCFunctionRegistry::create(
    const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex());
  auto& registry = factories();
  auto it = registry.find(name);
  if (it == registry.end()) {
    return nullptr;
  }
  return it->second();
}

bool AsyncRPCFunctionRegistry::isRegistered(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex());
  return factories().count(name) > 0;
}

std::unordered_set<std::string>
AsyncRPCFunctionRegistry::registeredFunctions() {
  std::lock_guard<std::mutex> lock(mutex());
  std::unordered_set<std::string> result;
  for (const auto& [name, _] : factories()) {
    result.insert(name);
  }
  return result;
}

void AsyncRPCFunctionRegistry::clear() {
  std::lock_guard<std::mutex> lock(mutex());
  factories().clear();
}

} // namespace facebook::velox::exec::rpc
