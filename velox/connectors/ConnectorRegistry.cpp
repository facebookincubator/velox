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
  auto it = connectors().find(connectorId);
  if (it != connectors().end()) {
    return it->second;
  }
  return nullptr;
}

// static
void ConnectorRegistry::unregisterAll() {
  connectors().clear();
}

// static
const std::unordered_map<std::string, std::shared_ptr<Connector>>&
ConnectorRegistry::all() {
  return connectors();
}

} // namespace facebook::velox::connector
