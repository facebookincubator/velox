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
#include "velox/exec/Operator.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/PartitionedOutputFactory.h"

namespace facebook::velox::core {
class QueryCtx;
} // namespace facebook::velox::core

namespace facebook::velox::exec {

/// Pairs an output buffer manager with its operator factory. Use make() to bind
/// the factory to the manager's concrete type.
struct OutputTransportEntry {
  /// Owns the output buffers for this transport.
  const std::shared_ptr<OutputBufferManager> manager;

  /// Builds output operators bound to 'manager'.
  const PartitionedOutputFactory makeOutputOperator;

  /// Pairs 'manager' with an operator builder that receives the same concrete
  /// manager. The entry owns the manager; the factory captures it weakly.
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
    return std::shared_ptr<OutputTransportEntry>(new OutputTransportEntry(
        std::move(manager),
        [weakManager = std::move(weakManager), build = std::move(build)](
            int32_t operatorId,
            DriverCtx* ctx,
            const std::shared_ptr<const core::PartitionedOutputNode>& node,
            bool eagerFlush) -> std::unique_ptr<Operator> {
          auto manager = weakManager.lock();
          VELOX_CHECK_NOT_NULL(manager, "Output buffer manager has expired");
          return build(operatorId, ctx, node, eagerFlush, manager);
        }));
  }

 private:
  OutputTransportEntry(
      std::shared_ptr<OutputBufferManager> manager,
      PartitionedOutputFactory makeOutputOperator)
      : manager(std::move(manager)),
        makeOutputOperator(std::move(makeOutputOperator)) {}
};

/// Thread-safe output transport registry. Query-scoped methods use the registry
/// installed on QueryCtx, or the global registry if none is installed. A query
/// registry falls back only to the parent passed to create().
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

  /// Returns the entry visible to 'queryCtx', or nullptr.
  static std::shared_ptr<OutputTransportEntry> tryGet(
      const core::QueryCtx& queryCtx,
      const std::string& id);

  /// Returns the transport entry registered under 'id' in the global registry,
  /// or nullptr. Ignores per-query overrides; use the QueryCtx overload to
  /// honor them.
  static std::shared_ptr<OutputTransportEntry> tryGet(const std::string& id);

  /// Returns all transports visible to 'queryCtx' as (id, entry) pairs.
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
  /// Backs the QueryCtx-scoped getAll().
  static std::vector<
      std::pair<std::string, std::shared_ptr<OutputTransportEntry>>>
  snapshot(const core::QueryCtx& queryCtx);
};

} // namespace facebook::velox::exec
