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

#include "velox/exec/OutputTransportRegistry.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/DefaultOutputBufferManager.h"

namespace facebook::velox::exec {

namespace {

// Keep kInMemory available without explicit registration for compatibility.
// TODO: Register it at engine initialization and let unregisterAll() clear the
// registry completely.
OutputTransportRegistry::Registry::Map builtinEntries() {
  OutputTransportRegistry::Registry::Map entries;
  entries.emplace(
      std::string{core::TransportKind::kInMemory},
      DefaultOutputBufferManager::makeDefaultTransportEntry());
  return entries;
}

// Initialize the process-wide registry before a child can reference it.
ScopedRegistry<std::string, OutputTransportEntry>& outputTransports() {
  static ScopedRegistry<std::string, OutputTransportEntry> instance;
  [[maybe_unused]] static const bool seeded = [] {
    instance.replaceAll(builtinEntries());
    return true;
  }();
  return instance;
}

OutputTransportRegistry::Registry& registryFor(const core::QueryCtx& queryCtx) {
  auto registry = queryCtx.registry<OutputTransportRegistry::Registry>(
      OutputTransportRegistry::kRegistryKey);
  return registry ? *registry : OutputTransportRegistry::global();
}

} // namespace

// static
OutputTransportRegistry::Registry& OutputTransportRegistry::global() {
  return outputTransports();
}

// static
std::shared_ptr<OutputTransportRegistry::Registry>
OutputTransportRegistry::create(const Registry* parent) {
  return std::make_shared<Registry>(parent);
}

// static
std::shared_ptr<OutputTransportEntry> OutputTransportRegistry::tryGet(
    const core::QueryCtx& queryCtx,
    const std::string& id) {
  return registryFor(queryCtx).find(id);
}

// static
std::shared_ptr<OutputTransportEntry> OutputTransportRegistry::tryGet(
    const std::string& id) {
  return global().find(id);
}

// static
std::vector<std::pair<std::string, std::shared_ptr<OutputTransportEntry>>>
OutputTransportRegistry::getAll(const core::QueryCtx& queryCtx) {
  return snapshot(queryCtx);
}

// static
std::vector<std::pair<std::string, std::shared_ptr<OutputTransportEntry>>>
OutputTransportRegistry::getAll() {
  std::vector<std::pair<std::string, std::shared_ptr<OutputTransportEntry>>>
      result;
  for (auto& [id, entry] : global().snapshot()) {
    if (entry != nullptr) {
      result.emplace_back(id, entry);
    }
  }
  return result;
}

// static
void OutputTransportRegistry::unregisterAll(const core::QueryCtx& queryCtx) {
  auto registry = queryCtx.registry<OutputTransportRegistry::Registry>(
      OutputTransportRegistry::kRegistryKey);
  if (registry) {
    registry->clear();
  }
}

// static
void OutputTransportRegistry::unregisterAll() {
  // Restore only the built-in entry atomically.
  global().replaceAll(builtinEntries());
}

// static
std::vector<std::pair<std::string, std::shared_ptr<OutputTransportEntry>>>
OutputTransportRegistry::snapshot(const core::QueryCtx& queryCtx) {
  std::vector<std::pair<std::string, std::shared_ptr<OutputTransportEntry>>>
      result;
  for (auto& [id, entry] : registryFor(queryCtx).snapshot()) {
    if (entry != nullptr) {
      result.emplace_back(id, entry);
    }
  }
  return result;
}

} // namespace facebook::velox::exec
