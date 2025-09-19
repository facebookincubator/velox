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

#include "velox/connectors/ConnectorObjectFactory.h"

namespace facebook::velox::connector {

ConnectorObjectFactory::~ConnectorObjectFactory() = default;

std::unordered_map<std::string, std::shared_ptr<ConnectorObjectFactory>>&
connectorObjectFactories() {
  static std::
      unordered_map<std::string, std::shared_ptr<ConnectorObjectFactory>>
          factories;
  return factories;
}

bool registerConnectorObjectFactory(
    std::shared_ptr<ConnectorObjectFactory> factory) {
  bool ok =
      connectorObjectFactories().insert({factory->name(), factory}).second;
  VELOX_CHECK(
      ok,
      "ConnectorObjectFactory with name '{}' is already registered",
      factory->name());
  return true;
}

bool hasConnectorObjectFactory(const std::string& connectorName) {
  return connectorObjectFactories().count(connectorName) == 1;
}

bool unregisterConnectorObjectFactory(const std::string& connectorName) {
  auto count = connectorObjectFactories().erase(connectorName);
  return count == 1;
}

std::shared_ptr<ConnectorObjectFactory> getConnectorObjectFactory(
    const std::string& connectorName) {
  auto it = connectorObjectFactories().find(connectorName);
  VELOX_CHECK(
      it != connectorObjectFactories().end(),
      "ConnectorObjectFactory with name '{}' not registered",
      connectorName);
  return it->second;
}

} // namespace facebook::velox::connector
