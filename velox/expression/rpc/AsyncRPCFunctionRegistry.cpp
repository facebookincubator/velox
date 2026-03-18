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

#include <glog/logging.h>

#include "velox/expression/rpc/RPCFunctionStubs.h"

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

std::unordered_map<std::string, AsyncRPCFunctionRegistry::Signatures>&
AsyncRPCFunctionRegistry::signatureStore() {
  static std::unordered_map<std::string, Signatures> instance;
  return instance;
}

bool AsyncRPCFunctionRegistry::registerFunction(
    const std::string& name,
    Factory factory) {
  std::lock_guard<std::mutex> lock(mutex());
  return registerEntryLocked(name, std::move(factory), {});
}

bool AsyncRPCFunctionRegistry::registerFunction(
    const std::string& name,
    Factory factory,
    Signatures signatures) {
  std::lock_guard<std::mutex> lock(mutex());
  return registerEntryLocked(name, std::move(factory), std::move(signatures));
}

bool AsyncRPCFunctionRegistry::registerEntryLocked(
    const std::string& name,
    Factory factory,
    Signatures signatures) {
  auto& registry = factories();
  if (registry.count(name) > 0) {
    return false; // Already registered
  }
  registry[name] = std::move(factory);
  if (!signatures.empty()) {
    signatureStore()[name] = std::move(signatures);
  }
  // Note: Do NOT use LOG() here as this function is called during static
  // initialization, before glog is initialized. Using LOG() would cause
  // a SIGSEGV crash (Static Initialization Order Fiasco).
  return true;
}

void AsyncRPCFunctionRegistry::registerStubs(
    const std::string& namespacePrefix) {
  std::unordered_map<std::string, Signatures> sigsCopy;
  {
    std::lock_guard<std::mutex> lock(mutex());
    for (const auto& [name, sigs] : signatureStore()) {
      if (!sigs.empty()) {
        sigsCopy[name] = sigs;
      }
    }
  }
  LOG(INFO) << "[RPC] registerStubs: namespacePrefix='" << namespacePrefix
            << "', found " << sigsCopy.size() << " function(s) with signatures";
  for (auto& [name, sigs] : sigsCopy) {
    std::string stubName = namespacePrefix + name;
    LOG(INFO) << "[RPC] registerStubs: registering stub '" << stubName
              << "' with " << sigs.size() << " signature(s)";
    registerRPCFunctionStub(stubName, std::move(sigs));
  }
  LOG(INFO) << "[RPC] registerStubs: completed, registered " << sigsCopy.size()
            << " stub(s)";
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

void AsyncRPCFunctionRegistry::testingClear() {
  std::lock_guard<std::mutex> lock(mutex());
  factories().clear();
  signatureStore().clear();
}

} // namespace facebook::velox::exec::rpc
