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
#include "velox/connectors/ConnectorRegistryInternal.h"

namespace facebook::velox::connector {

// static
std::shared_ptr<Connector> ConnectorRegistry::tryGet(
    const std::string& connectorId) {
  return connectors().withRLock(
      [&](const auto& registry) -> std::shared_ptr<Connector> {
        auto it = registry.find(connectorId);
        if (it != registry.end()) {
          return it->second;
        }
        return nullptr;
      });
}

// static
void ConnectorRegistry::unregisterAll() {
  folly::F14FastMap<std::string, std::shared_ptr<Connector>> entries;
  connectors().withWLock([&](auto& registry) { entries.swap(registry); });
}

// static
void ConnectorRegistry::forEach(
    std::function<void(const std::shared_ptr<Connector>&)> func) {
  connectors().withRLock([&](const auto& registry) {
    for (const auto& [_, connector] : registry) {
      func(connector);
    }
  });
}

} // namespace facebook::velox::connector
