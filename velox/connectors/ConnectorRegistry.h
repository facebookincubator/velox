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

#pragma once

#include "velox/connectors/Connector.h"

namespace facebook::velox::connector {

/// Provides static methods for connector registry operations.
class ConnectorRegistry {
 public:
  /// Return the connector with the specified ID, or nullptr if not registered.
  static std::shared_ptr<Connector> tryGet(const std::string& connectorId);

  /// Return all connectors whose implementation is of type T.
  template <typename T>
  static std::vector<std::shared_ptr<T>> findAll() {
    std::vector<std::shared_ptr<T>> result;
    for (const auto& [_, connector] : all()) {
      if (auto casted = std::dynamic_pointer_cast<T>(connector)) {
        result.push_back(std::move(casted));
      }
    }
    return result;
  }

  /// Unregister all connectors.
  static void unregisterAll();

 private:
  // Return a reference to the internal connector map.
  static const std::unordered_map<std::string, std::shared_ptr<Connector>>&
  all();
};

} // namespace facebook::velox::connector
