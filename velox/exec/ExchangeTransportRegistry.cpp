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

#include "velox/exec/ExchangeTransportRegistry.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/InMemoryExchangeClient.h"

namespace facebook::velox::exec {

namespace {

// Keep kInMemory available without explicit registration for compatibility.
//
// TODO: Register kInMemory explicitly at engine init like any other transport.
// Then this seeding goes away and unregisterAll() becomes a plain clear().
ExchangeTransportRegistry::Registry::Map builtinEntries() {
  ExchangeTransportRegistry::Registry::Map entries;
  entries.emplace(
      std::string{core::TransportKind::kInMemory},
      InMemoryExchangeClient::makeDefaultTransportEntry());
  return entries;
}

// Initialize the process-wide registry before a child can reference it.
ScopedRegistry<std::string, ExchangeTransportEntry>& exchangeTransports() {
  static ScopedRegistry<std::string, ExchangeTransportEntry> instance;
  [[maybe_unused]] static const bool kSeeded = [] {
    instance.replaceAll(builtinEntries());
    return true;
  }();
  return instance;
}

ExchangeTransportRegistry::Registry& registryFor(
    const core::QueryCtx& queryCtx) {
  auto registry = queryCtx.registry<ExchangeTransportRegistry::Registry>(
      ExchangeTransportRegistry::kRegistryKey);
  return registry ? *registry : ExchangeTransportRegistry::global();
}

} // namespace

// static
ExchangeTransportRegistry::Registry& ExchangeTransportRegistry::global() {
  return exchangeTransports();
}

// static
std::shared_ptr<ExchangeTransportRegistry::Registry>
ExchangeTransportRegistry::create(const Registry* parent) {
  return std::make_shared<Registry>(parent);
}

// static
std::shared_ptr<ExchangeTransportEntry> ExchangeTransportRegistry::tryGet(
    const core::QueryCtx& queryCtx,
    const std::string& id) {
  return registryFor(queryCtx).find(id);
}

// static
std::shared_ptr<ExchangeTransportEntry> ExchangeTransportRegistry::tryGet(
    const std::string& id) {
  return global().find(id);
}

// static
std::vector<std::pair<std::string, std::shared_ptr<ExchangeTransportEntry>>>
ExchangeTransportRegistry::getAll(const core::QueryCtx& queryCtx) {
  return snapshot(queryCtx);
}

// static
std::vector<std::pair<std::string, std::shared_ptr<ExchangeTransportEntry>>>
ExchangeTransportRegistry::getAll() {
  std::vector<std::pair<std::string, std::shared_ptr<ExchangeTransportEntry>>>
      result;
  for (auto& [id, entry] : global().snapshot()) {
    if (entry != nullptr) {
      result.emplace_back(id, entry);
    }
  }
  return result;
}

// static
void ExchangeTransportRegistry::unregisterAll(const core::QueryCtx& queryCtx) {
  auto registry = queryCtx.registry<ExchangeTransportRegistry::Registry>(
      ExchangeTransportRegistry::kRegistryKey);
  if (registry) {
    registry->clear();
  }
}

// static
void ExchangeTransportRegistry::unregisterAll() {
  // Reset to the built-in entries under one lock so readers never observe the
  // default transport as temporarily unregistered.
  global().replaceAll(builtinEntries());
}

// static
std::vector<std::pair<std::string, std::shared_ptr<ExchangeTransportEntry>>>
ExchangeTransportRegistry::snapshot(const core::QueryCtx& queryCtx) {
  std::vector<std::pair<std::string, std::shared_ptr<ExchangeTransportEntry>>>
      result;
  for (auto& [id, entry] : registryFor(queryCtx).snapshot()) {
    if (entry != nullptr) {
      result.emplace_back(id, entry);
    }
  }
  return result;
}

} // namespace facebook::velox::exec
