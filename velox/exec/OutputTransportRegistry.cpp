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

// Seeds 'registry' with the built-in in-memory transport, keeping it a
// first-class entry so lookups and enumeration stay plain reads.
//
// Backward-compat shim: kInMemory must resolve with zero registration to
// preserve the pre-registry guarantee that the default output buffer manager is
// always present. The target end state is to register kInMemory explicitly at
// engine init like any other transport, which retires this seeding (and the
// unregisterAll() re-seed) and leaves a plain map: unregisterAll() fully clears
// and isolation is uniform.
void registerBuiltinDefault(OutputTransportRegistry::Registry& registry) {
  registry.insert(
      std::string{core::TransportKind::kInMemory},
      DefaultOutputBufferManager::makeDefaultTransportEntry(),
      /*overwrite=*/true);
}

// The process-wide root registry, seeded with the built-in in-memory transport
// on first access -- before any child scope can exist, since children are
// created via create(&global()) -- so scoped lookups never mutate a parent, per
// ScopedRegistry's contract.
ScopedRegistry<std::string, OutputTransportEntry>& outputTransports() {
  static ScopedRegistry<std::string, OutputTransportEntry> instance;
  [[maybe_unused]] static const bool seeded = [] {
    registerBuiltinDefault(instance);
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
  // Reset to baseline: drop user registrations and restore the built-in
  // in-memory default. The re-seed is part of the backward-compat shim (see
  // registerBuiltinDefault); with init-time registration this reduces to a
  // plain clear().
  global().clear();
  registerBuiltinDefault(global());
}

// static
std::vector<std::pair<std::string, std::shared_ptr<OutputTransportEntry>>>
OutputTransportRegistry::snapshot(const core::QueryCtx& queryCtx) {
  // Merges the per-query override with the global registry, consistent with
  // tryGet(queryCtx, ...).
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
