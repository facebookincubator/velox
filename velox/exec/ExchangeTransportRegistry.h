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
#include "velox/common/base/Exceptions.h"
// make<TClient>() downcasts the abstract client, so the complete type is
// needed here, not just the forward declaration in ExchangeFactory.h.
#include "velox/exec/ExchangeClient.h"
#include "velox/exec/ExchangeFactory.h"

namespace facebook::velox::core {
class QueryCtx;
} // namespace facebook::velox::core

namespace facebook::velox::exec {

/// Registry value pairing an exchange client factory with the factories that
/// build its matching exchange operators, keyed by transport id. Registering
/// them together ensures a transport's operators and client cannot diverge.
/// Build entries with make(), which binds the operators to the concrete client
/// type and rejects null halves; direct construction is for tests passing real
/// values.
struct ExchangeTransportEntry {
  /// Creates this transport's exchange client for one pipeline of one task.
  ExchangeClientFactory makeClient;

  /// Builds this transport's Exchange operator, bound to a client from
  /// 'makeClient'.
  ExchangeOperatorFactory makeExchangeOperator;

  /// Builds this transport's MergeExchange operator, bound to a client from
  /// 'makeClient'. Null when the transport does not support merge exchange;
  /// Task fails fast if a MergeExchangeNode names such a transport.
  ExchangeOperatorFactory makeMergeExchangeOperator;

  /// Constructs an entry from a client factory and the operator builders bound
  /// to it.
  ExchangeTransportEntry(
      ExchangeClientFactory makeClient,
      ExchangeOperatorFactory makeExchangeOperator,
      ExchangeOperatorFactory makeMergeExchangeOperator = nullptr)
      : makeClient(std::move(makeClient)),
        makeExchangeOperator(std::move(makeExchangeOperator)),
        makeMergeExchangeOperator(std::move(makeMergeExchangeOperator)) {}

  /// Preferred way to build an entry: pairs a client factory with operator
  /// builders that receive the concrete client type that factory produces, so
  /// an operator can't be wired to a client from a different transport. Pass
  /// 'buildMergeExchange' as nullptr when the transport cannot merge.
  template <typename TClient>
  static std::shared_ptr<ExchangeTransportEntry> make(
      std::function<std::shared_ptr<TClient>(
          const ExchangeClientContext& context)> makeClient,
      std::function<std::unique_ptr<Operator>(
          int32_t operatorId,
          DriverCtx* ctx,
          const std::shared_ptr<const core::ExchangeNode>& node,
          const std::shared_ptr<TClient>& client)> buildExchange,
      std::function<std::unique_ptr<Operator>(
          int32_t operatorId,
          DriverCtx* ctx,
          const std::shared_ptr<const core::ExchangeNode>& node,
          const std::shared_ptr<TClient>& client)> buildMergeExchange =
          nullptr) {
    VELOX_CHECK(
        makeClient != nullptr, "Exchange transport client factory is null");
    VELOX_CHECK(
        buildExchange != nullptr,
        "Exchange transport operator builder is null");

    ExchangeClientFactory clientFactory =
        [makeClient =
             std::move(makeClient)](const ExchangeClientContext& context)
        -> std::shared_ptr<ExchangeClient> { return makeClient(context); };

    ExchangeOperatorFactory exchangeOperatorFactory =
        [buildExchange = std::move(buildExchange)](
            int32_t operatorId,
            DriverCtx* ctx,
            const std::shared_ptr<const core::ExchangeNode>& node,
            std::shared_ptr<ExchangeClient> client)
        -> std::unique_ptr<Operator> {
      auto typedClient = std::dynamic_pointer_cast<TClient>(client);
      VELOX_CHECK_NOT_NULL(
          typedClient,
          "Exchange client was not created by this transport's client factory");
      return buildExchange(operatorId, ctx, node, typedClient);
    };

    ExchangeOperatorFactory mergeExchangeOperatorFactory{nullptr};
    if (buildMergeExchange != nullptr) {
      mergeExchangeOperatorFactory =
          [buildMergeExchange = std::move(buildMergeExchange)](
              int32_t operatorId,
              DriverCtx* ctx,
              const std::shared_ptr<const core::ExchangeNode>& node,
              std::shared_ptr<ExchangeClient> client)
          -> std::unique_ptr<Operator> {
        auto typedClient = std::dynamic_pointer_cast<TClient>(client);
        VELOX_CHECK_NOT_NULL(
            typedClient,
            "Exchange client was not created by this transport's client "
            "factory");
        return buildMergeExchange(operatorId, ctx, node, typedClient);
      };
    }

    return std::make_shared<ExchangeTransportEntry>(
        std::move(clientFactory),
        std::move(exchangeOperatorFactory),
        std::move(mergeExchangeOperatorFactory));
  }
};

/// Manages exchange transport registration and lookup, keyed by transport id.
/// Each entry pairs an exchange client factory with the factories that build
/// its matching exchange operators. All methods are thread-safe.
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
class ExchangeTransportRegistry {
 public:
  using Registry = ScopedRegistry<std::string, ExchangeTransportEntry>;

  /// Registry key for per-query exchange transport overrides on QueryCtx.
  static constexpr std::string_view kRegistryKey = "exchangeTransports";

  /// Returns the global registry (root scope).
  static Registry& global();

  /// Creates a per-query registry. If 'parent' is provided, lookups fall back
  /// to it. Pass nullptr for isolation mode (no fallback).
  static std::shared_ptr<Registry> create(const Registry* parent = nullptr);

  /// Returns the transport entry registered under 'id' for 'queryCtx'
  /// (per-query override, then global registry), or nullptr.
  static std::shared_ptr<ExchangeTransportEntry> tryGet(
      const core::QueryCtx& queryCtx,
      const std::string& id);

  /// Returns the transport entry registered under 'id' in the global registry,
  /// or nullptr. Ignores per-query overrides; use the QueryCtx overload to
  /// honor them.
  static std::shared_ptr<ExchangeTransportEntry> tryGet(const std::string& id);

  /// Returns all transports visible to 'queryCtx' as (id, entry) pairs
  /// (per-query override merged over the global registry).
  static std::vector<
      std::pair<std::string, std::shared_ptr<ExchangeTransportEntry>>>
  getAll(const core::QueryCtx& queryCtx);

  /// Returns all registered transports from the global registry, as
  /// (id, entry) pairs.
  static std::vector<
      std::pair<std::string, std::shared_ptr<ExchangeTransportEntry>>>
  getAll();

  /// Clears the per-query transport overrides; global registrations remain.
  static void unregisterAll(const core::QueryCtx& queryCtx);

  /// Clears all registered transports, keeping the built-in in-memory default.
  static void unregisterAll();

 private:
  // Returns the (id, entry) pairs visible to 'queryCtx' -- the per-query
  // override merged with the global registry. Backs the QueryCtx-scoped
  // getAll().
  static std::vector<
      std::pair<std::string, std::shared_ptr<ExchangeTransportEntry>>>
  snapshot(const core::QueryCtx& queryCtx);
};

} // namespace facebook::velox::exec
