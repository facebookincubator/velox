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

#include "velox/common/base/Exceptions.h"
#include "velox/expression/rpc/RPCFunctionStubs.h"

namespace facebook::velox::exec::rpc {

std::mutex& AsyncRPCFunctionRegistry::mutex() {
  static std::mutex instance;
  return instance;
}

std::unordered_map<std::string, AsyncRPCFunctionRegistry::Registration>&
AsyncRPCFunctionRegistry::registrations() {
  static std::unordered_map<std::string, Registration> instance;
  return instance;
}

bool AsyncRPCFunctionRegistry::registerFunction(
    const std::string& name,
    Factory factory,
    Signatures signatures) {
  return registerFunction(name, std::move(factory), std::move(signatures), {});
}

bool AsyncRPCFunctionRegistry::registerFunction(
    const std::string& name,
    Factory factory,
    Signatures signatures,
    Metadata metadata) {
  VELOX_CHECK(
      !signatures.empty(),
      "RPC function must be registered with at least one signature: {}",
      name);
  std::lock_guard<std::mutex> lock(mutex());
  // Note: Do NOT use LOG() here as this function is called during static
  // initialization, before glog is initialized. Using LOG() would cause
  // a SIGSEGV crash (Static Initialization Order Fiasco).
  return registrations()
      .emplace(
          name,
          Registration{
              std::move(factory), std::move(signatures), std::move(metadata)})
      .second;
}

void AsyncRPCFunctionRegistry::registerStubs(
    const std::string& namespacePrefix) {
  auto entries = functions();
  LOG(INFO) << "[RPC] registerStubs: namespacePrefix='" << namespacePrefix
            << "', found " << entries.size() << " function(s)";
  for (auto& entry : entries) {
    const std::string stubName = namespacePrefix + entry.name;
    LOG(INFO) << "[RPC] registerStubs: registering stub '" << stubName
              << "' with " << entry.signatures.size() << " signature(s)";
    registerRPCFunctionStub(
        stubName, std::move(entry.signatures), std::move(entry.metadata));
  }
  LOG(INFO) << "[RPC] registerStubs: completed, registered " << entries.size()
            << " stub(s)";
}

std::optional<AsyncRPCFunctionRegistry::FunctionEntry>
AsyncRPCFunctionRegistry::find(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex());
  const auto& all = registrations();
  const auto it = all.find(name);
  if (it == all.end()) {
    return std::nullopt;
  }
  return FunctionEntry{name, it->second.signatures, it->second.metadata};
}

std::vector<AsyncRPCFunctionRegistry::FunctionEntry>
AsyncRPCFunctionRegistry::functions() {
  std::lock_guard<std::mutex> lock(mutex());
  const auto& all = registrations();
  std::vector<FunctionEntry> entries;
  entries.reserve(all.size());
  for (const auto& [name, registration] : all) {
    entries.push_back(
        FunctionEntry{name, registration.signatures, registration.metadata});
  }
  return entries;
}

std::shared_ptr<AsyncRPCFunction> AsyncRPCFunctionRegistry::create(
    const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex());
  const auto& all = registrations();
  const auto it = all.find(name);
  if (it == all.end()) {
    return nullptr;
  }
  return it->second.factory();
}

bool AsyncRPCFunctionRegistry::isRegistered(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex());
  return registrations().contains(name);
}

std::unordered_set<std::string>
AsyncRPCFunctionRegistry::registeredFunctions() {
  std::lock_guard<std::mutex> lock(mutex());
  std::unordered_set<std::string> result;
  for (const auto& [name, _] : registrations()) {
    result.insert(name);
  }
  return result;
}

void AsyncRPCFunctionRegistry::testingClear() {
  std::lock_guard<std::mutex> lock(mutex());
  registrations().clear();
}

} // namespace facebook::velox::exec::rpc
