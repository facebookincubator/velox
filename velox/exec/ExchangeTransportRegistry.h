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
// make<TClient>() downcasts the client and therefore needs its complete type.
#include "velox/exec/ExchangeClient.h"
#include "velox/exec/ExchangeFactory.h"

namespace facebook::velox::core {
class QueryCtx;
} // namespace facebook::velox::core

namespace facebook::velox::exec {

/// Pairs a transport's client factory with its operator factories. Use make()
/// to bind the operator factories to the concrete client type.
struct ExchangeTransportEntry {
  /// Creates this transport's exchange client for one pipeline of one task.
  const ExchangeClientFactory makeClient;

  /// Builds this transport's Exchange operator, bound to a client from
  /// 'makeClient'.
  const ExchangeOperatorFactory makeExchangeOperator;

  /// Builds this transport's operator for a MergeExchangeNode. Null if merge
  /// exchange is unsupported. The resulting driver pipeline, including any
  /// DriverAdapter changes, must emit rows ordered by the node's sortingKeys()
  /// and sortingOrders().
  const ExchangeOperatorFactory makeMergeExchangeOperator;

  /// Pairs a client factory with operator builders that receive its concrete
  /// client type. Leave 'buildMergeExchange' null if merging is unsupported.
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

    return std::shared_ptr<ExchangeTransportEntry>(new ExchangeTransportEntry(
        std::move(clientFactory),
        std::move(exchangeOperatorFactory),
        std::move(mergeExchangeOperatorFactory)));
  }

 private:
  ExchangeTransportEntry(
      ExchangeClientFactory makeClient,
      ExchangeOperatorFactory makeExchangeOperator,
      ExchangeOperatorFactory makeMergeExchangeOperator)
      : makeClient(std::move(makeClient)),
        makeExchangeOperator(std::move(makeExchangeOperator)),
        makeMergeExchangeOperator(std::move(makeMergeExchangeOperator)) {}
};

/// Thread-safe exchange transport registry. Query-scoped methods use the
/// registry installed on QueryCtx, or the global registry if none is installed.
/// A query registry falls back only to the parent passed to create().
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

  /// Returns the entry visible to 'queryCtx', or nullptr.
  static std::shared_ptr<ExchangeTransportEntry> tryGet(
      const core::QueryCtx& queryCtx,
      const std::string& id);

  /// Returns the transport entry registered under 'id' in the global registry,
  /// or nullptr. Ignores per-query overrides; use the QueryCtx overload to
  /// honor them.
  static std::shared_ptr<ExchangeTransportEntry> tryGet(const std::string& id);

  /// Returns all transports visible to 'queryCtx' as (id, entry) pairs.
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
  /// Backs the QueryCtx-scoped getAll().
  static std::vector<
      std::pair<std::string, std::shared_ptr<ExchangeTransportEntry>>>
  snapshot(const core::QueryCtx& queryCtx);
};

} // namespace facebook::velox::exec
