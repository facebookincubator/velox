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

/// Registry value pairing an output buffer manager with an optional
/// availability predicate.
///
/// The predicate decouples registration from selection: a manager is registered
/// once, typically based on process-level capability, while 'isAvailable'
/// determines per query whether it may be used. A registered manager is
/// therefore visible only to the queries whose configuration selects it. The
/// predicate gates visibility, not lifetime, so a shared (e.g. process-wide)
/// manager instance can be wrapped without additional cost.
///
/// A null 'isAvailable' marks the manager as unconditionally available, as used
/// by the default HTTP manager.
struct OutputBufferManagerEntry {
  /// The registered output buffer manager.
  std::shared_ptr<IOutputBufferManager> manager;

  /// Predicate deciding whether 'manager' may be used by a given query.
  /// A null predicate means the manager is unconditionally available.
  std::function<bool(const core::QueryCtx&)> isAvailable;

  explicit OutputBufferManagerEntry(
      std::shared_ptr<IOutputBufferManager> manager,
      std::function<bool(const core::QueryCtx&)> isAvailable = nullptr)
      : manager{std::move(manager)}, isAvailable{std::move(isAvailable)} {}
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

  /// Return the manager registered under 'id' that is available for 'queryCtx',
  /// or nullptr. Resolution checks the per-query override on 'queryCtx' before
  /// falling back to the global registry, and a resolved entry is returned only
  /// if its availability predicate holds for 'queryCtx'. Callers should treat
  /// this as the authoritative per-query availability check.
  static std::shared_ptr<IOutputBufferManager> tryGet(
      const core::QueryCtx& queryCtx,
      const std::string& id);

  /// Return the manager registered under 'id' in the global registry, or
  /// nullptr. This overload does not evaluate any availability predicate; use
  /// the QueryCtx overload for per-query availability.
  static std::shared_ptr<IOutputBufferManager> tryGet(const std::string& id);

  /// Return whether a manager is registered under 'id' in the global registry,
  /// independent of any availability predicate. Intended as a process-level
  /// capability check.
  static bool contains(const std::string& id);

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
