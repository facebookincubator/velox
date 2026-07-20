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

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/common/ScopedRegistry.h"
#include "velox/connectors/Connector.h"

namespace facebook::velox::core {
class QueryCtx;
} // namespace facebook::velox::core

namespace facebook::velox::connector {

/// Manages connector registration and lookup. All methods are thread-safe.
///
/// Two groups of APIs:
///
/// - Query-scoped APIs take a QueryCtx& and check for per-query registry
///   overrides before falling back to the global registry. Use these in
///   operator and expression evaluation code where a QueryCtx is available.
///
/// - Global APIs operate directly on the global registry. Use these for
///   process-level operations: startup registration, shutdown cleanup, and
///   process-wide lookups (e.g., periodic stats reporting).
/// All methods are thread-safe.
class ConnectorRegistry {
 public:
  using Registry = ScopedRegistry<std::string, Connector>;

  /// Registry key for per-query connector overrides on QueryCtx.
  static constexpr std::string_view kRegistryKey = "connectors";

  /// Return the global registry (root scope).
  static Registry& global();

  /// Create a per-query registry. If 'parent' is provided, lookups fall back
  /// to it. Pass nullptr for isolation mode (no fallback).
  static std::shared_ptr<Registry> create(const Registry* parent = nullptr);

  /// Return the connector with the specified ID, or nullptr if not registered.
  /// Checks per-query override on QueryCtx first, falls back to the global
  /// registry if no override is set.
  static std::shared_ptr<Connector> tryGet(
      const core::QueryCtx& queryCtx,
      const std::string& connectorId);

  /// Return the connector with the specified ID from the global registry, or
  /// nullptr if not registered.
  static std::shared_ptr<Connector> tryGet(const std::string& connectorId);

  /// Return all connectors whose implementation is of type T. Checks per-query
  /// override on QueryCtx first, falls back to the global registry if no
  /// override is set.
  template <typename T>
  static std::vector<std::shared_ptr<T>> findAll(
      const core::QueryCtx& queryCtx) {
    std::vector<std::shared_ptr<T>> result;
    for (auto& [_, connector] : snapshot(queryCtx)) {
      if (auto casted = std::dynamic_pointer_cast<T>(connector)) {
        result.push_back(std::move(casted));
      }
    }
    return result;
  }

  /// Return all connectors from the global registry whose implementation is
  /// of type T.
  template <typename T>
  static std::vector<std::shared_ptr<T>> findAll() {
    std::vector<std::shared_ptr<T>> result;
    for (auto& [_, connector] : global().snapshot()) {
      if (auto casted = std::dynamic_pointer_cast<T>(connector)) {
        result.push_back(std::move(casted));
      }
    }
    return result;
  }

  /// Unregister all connectors from the registry visible to the given query.
  static void unregisterAll(const core::QueryCtx& queryCtx);

  /// Unregister all connectors from the global registry.
  static void unregisterAll();

 private:
  // Return a snapshot of all connectors visible to the given query. Uses
  // per-query registry if set, otherwise the global registry.
  static std::vector<std::pair<std::string, std::shared_ptr<Connector>>>
  snapshot(const core::QueryCtx& queryCtx);
};

} // namespace facebook::velox::connector
