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

#include "velox/connectors/ConnectorRegistry.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "velox/connectors/Connector.h"
#include "velox/connectors/ConnectorRegistryInternal.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::connector {

namespace {

ConnectorRegistry::Registry& registryFor(const core::QueryCtx& queryCtx) {
  auto registry = queryCtx.registry<ConnectorRegistry::Registry>(
      ConnectorRegistry::kRegistryKey);
  return registry ? *registry : ConnectorRegistry::global();
}

} // namespace

// static
ConnectorRegistry::Registry& ConnectorRegistry::global() {
  return connectors();
}

// static
std::shared_ptr<ConnectorRegistry::Registry> ConnectorRegistry::create(
    const Registry* parent) {
  return std::make_shared<Registry>(parent);
}

// static
std::shared_ptr<Connector> ConnectorRegistry::tryGet(
    const core::QueryCtx& queryCtx,
    const std::string& connectorId) {
  return registryFor(queryCtx).find(connectorId);
}

// static
std::shared_ptr<Connector> ConnectorRegistry::tryGet(
    const std::string& connectorId) {
  return global().find(connectorId);
}

// static
void ConnectorRegistry::unregisterAll(const core::QueryCtx& queryCtx) {
  auto registry = queryCtx.registry<ConnectorRegistry::Registry>(
      ConnectorRegistry::kRegistryKey);
  if (registry) {
    registry->clear();
  }
}

// static
void ConnectorRegistry::unregisterAll() {
  global().clear();
}

// static
std::vector<std::pair<std::string, std::shared_ptr<Connector>>>
ConnectorRegistry::snapshot(const core::QueryCtx& queryCtx) {
  return registryFor(queryCtx).snapshot();
}

} // namespace facebook::velox::connector
