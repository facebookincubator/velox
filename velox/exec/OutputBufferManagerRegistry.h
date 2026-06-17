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

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/common/ScopedRegistry.h"
#include "velox/exec/IOutputBufferManager.h"

namespace facebook::velox::core {
class QueryCtx;
} // namespace facebook::velox::core

namespace facebook::velox::exec {

/// A registered output buffer manager together with an optional per-query
/// availability predicate.
struct OutputBufferManagerEntry {
  /// The output buffer manager for a transport.
  std::shared_ptr<IOutputBufferManager> manager;
  /// Per-query availability predicate. When empty, the manager is always
  /// available. When set, the manager is visible to tryGet() only for queries
  /// where this returns true; it controls visibility, not instance count.
  std::function<bool(const core::QueryCtx&)> isAvailable;
};

/// Manages output buffer manager registration and lookup. All methods are
/// thread-safe.
///
/// Two groups of APIs:
///
/// - Query-scoped APIs take a QueryCtx& and check for per-query registry
///   overrides before falling back to the global registry. Use these in
///   operator and task code where a QueryCtx is available.
///
/// - Global APIs operate directly on the global registry. Use these for
///   process-level operations: startup registration, shutdown cleanup, and
///   process-wide lookups.
class OutputBufferManagerRegistry {
 public:
  using Registry = ScopedRegistry<std::string, OutputBufferManagerEntry>;

  /// Registry key for per-query output buffer manager overrides on QueryCtx.
  static constexpr std::string_view kRegistryKey = "outputBufferManagers";

  /// Return the global registry (root scope).
  static Registry& global();

  /// Create a per-query registry. If 'parent' is provided, lookups fall back
  /// to it. Pass nullptr for isolation mode (no fallback).
  static std::shared_ptr<Registry> create(const Registry* parent = nullptr);

  /// Return the output buffer manager with the specified ID, or nullptr if not
  /// registered. Checks per-query override on QueryCtx first, falls back to the
  /// global registry if no override is set.
  static std::shared_ptr<IOutputBufferManager> tryGet(
      const core::QueryCtx& queryCtx,
      const std::string& id);

  /// Return the output buffer manager with the specified ID from the global
  /// registry, or nullptr if not registered.
  static std::shared_ptr<IOutputBufferManager> tryGet(const std::string& id);

  /// Return true if a manager with the specified ID is registered for the
  /// given query (checking the per-query override then the global registry),
  /// regardless of its availability predicate. Use this to distinguish a
  /// capability gap (not registered at all) from a policy opt-out (registered
  /// but unavailable for this query).
  static bool contains(const core::QueryCtx& queryCtx, const std::string& id);

  /// Return all registered output buffer managers visible to the given query.
  /// Checks per-query override on QueryCtx first, falls back to the global
  /// registry if no override is set.
  static std::vector<
      std::pair<std::string, std::shared_ptr<IOutputBufferManager>>>
  getAll(const core::QueryCtx& queryCtx);

  /// Return all registered output buffer managers from the global registry.
  static std::vector<
      std::pair<std::string, std::shared_ptr<IOutputBufferManager>>>
  getAll();

  /// Unregister all output buffer managers from the registry visible to the
  /// given query.
  static void unregisterAll(const core::QueryCtx& queryCtx);

  /// Unregister all output buffer managers from the global registry.
  static void unregisterAll();

 private:
  static std::vector<
      std::pair<std::string, std::shared_ptr<IOutputBufferManager>>>
  snapshot(const core::QueryCtx& queryCtx);
};

} // namespace facebook::velox::exec
