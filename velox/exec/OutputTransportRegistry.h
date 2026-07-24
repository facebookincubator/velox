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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/common/ScopedRegistry.h"
#include "velox/common/base/Exceptions.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/PartitionedOutputFactory.h"

namespace facebook::velox::core {
class QueryCtx;
} // namespace facebook::velox::core

namespace facebook::velox::exec {

/// Registry value pairing an output buffer manager with the factory that builds
/// its matching output operator, keyed by transport id. Registering the two
/// together ensures a transport's operator and manager cannot diverge. Build
/// entries with make(), which binds the operator to this manager and rejects
/// null halves; direct aggregate initialization is for tests passing real
/// values.
struct OutputTransportEntry {
  std::shared_ptr<OutputBufferManager> manager;

  /// Builds this transport's output operator, binding 'manager'.
  PartitionedOutputFactory makeOutputOperator;

  /// Preferred way to build an entry: pairs 'manager' with an operator builder
  /// that receives that same manager, so the operator can't be wired to a
  /// different one than the entry stores. The manager is captured weakly (the
  /// entry owns it) and locked when building, honoring the
  /// PartitionedOutputFactory ownership contract.
  template <typename TManager>
  static std::shared_ptr<OutputTransportEntry> make(
      std::shared_ptr<TManager> manager,
      std::function<std::unique_ptr<Operator>(
          int32_t operatorId,
          DriverCtx* ctx,
          const std::shared_ptr<const core::PartitionedOutputNode>& node,
          bool eagerFlush,
          const std::shared_ptr<TManager>& manager)> build) {
    VELOX_CHECK_NOT_NULL(manager, "Output transport manager is null");
    VELOX_CHECK(build != nullptr, "Output transport operator builder is null");
    std::weak_ptr<TManager> weakManager = manager;
    return std::make_shared<OutputTransportEntry>(
        std::move(manager),
        [weakManager = std::move(weakManager), build = std::move(build)](
            int32_t operatorId,
            DriverCtx* ctx,
            const std::shared_ptr<const core::PartitionedOutputNode>& node,
            bool eagerFlush) -> std::unique_ptr<Operator> {
          auto manager = weakManager.lock();
          VELOX_CHECK_NOT_NULL(manager, "Output buffer manager has expired");
          return build(operatorId, ctx, node, eagerFlush, manager);
        });
  }
};

/// Manages output transport registration and lookup, keyed by transport id.
/// Each entry pairs an output buffer manager with the factory that builds its
/// matching output operator. All methods are thread-safe.
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
class OutputTransportRegistry {
 public:
  using Registry = ScopedRegistry<std::string, OutputTransportEntry>;

  /// Registry key for per-query output transport overrides on QueryCtx.
  static constexpr std::string_view kRegistryKey = "outputTransports";

  /// Returns the global registry (root scope).
  static Registry& global();

  /// Creates a per-query registry. If 'parent' is provided, lookups fall back
  /// to it. Pass nullptr for isolation mode (no fallback).
  static std::shared_ptr<Registry> create(const Registry* parent = nullptr);

  /// Returns the transport entry registered under 'id' for 'queryCtx'
  /// (per-query override, then global registry), or nullptr.
  static std::shared_ptr<OutputTransportEntry> tryGet(
      const core::QueryCtx& queryCtx,
      const std::string& id);

  /// Returns the transport entry registered under 'id' in the global registry,
  /// or nullptr. Ignores per-query overrides; use the QueryCtx overload to
  /// honor them.
  static std::shared_ptr<OutputTransportEntry> tryGet(const std::string& id);

  /// Returns all transports visible to 'queryCtx' as (id, entry) pairs
  /// (per-query override merged over the global registry).
  static std::vector<
      std::pair<std::string, std::shared_ptr<OutputTransportEntry>>>
  getAll(const core::QueryCtx& queryCtx);

  /// Returns all registered transports from the global registry, as
  /// (id, entry) pairs.
  static std::vector<
      std::pair<std::string, std::shared_ptr<OutputTransportEntry>>>
  getAll();

  /// Clears the per-query transport overrides; global registrations remain.
  static void unregisterAll(const core::QueryCtx& queryCtx);

  /// Clears all registered transports, keeping the built-in in-memory default.
  static void unregisterAll();

 private:
  /// Returns the (id, entry) pairs visible to 'queryCtx' -- the per-query
  /// override merged with the global registry. Backs the QueryCtx-scoped
  /// getAll().
  static std::vector<
      std::pair<std::string, std::shared_ptr<OutputTransportEntry>>>
  snapshot(const core::QueryCtx& queryCtx);
};

} // namespace facebook::velox::exec
