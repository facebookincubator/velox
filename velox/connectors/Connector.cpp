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

#include "velox/connectors/Connector.h"

#include "velox/common/EnumDefine.h"

#include <memory>
#include <string>

#include "velox/common/ScopedRegistry.h"
#include "velox/common/base/Exceptions.h"
#include "velox/connectors/ConnectorRegistryInternal.h"

namespace facebook::velox::connector {

std::unique_ptr<ConnectorQueryCtx> ConnectorQueryCtx::Builder::build() {
  VELOX_CHECK(!built_, "ConnectorQueryCtx::Builder can only build once");
  built_ = true;
  return std::unique_ptr<ConnectorQueryCtx>(new ConnectorQueryCtx(
      operatorPool_,
      connectorPool_,
      sessionProperties_,
      spillConfig_,
      prefixSortConfig_,
      std::move(expressionEvaluator_),
      cache_,
      queryId_,
      taskId_,
      planNodeId_,
      driverId_,
      sessionTimezone_,
      adjustTimestampToTimezone_,
      std::move(cancellationToken_),
      std::move(tokenProvider_),
      std::move(customPools_)));
}

ConnectorQueryCtx::ConnectorQueryCtx(
    memory::MemoryPool* operatorPool,
    memory::MemoryPool* connectorPool,
    const config::ConfigBase* sessionProperties,
    const common::SpillConfig* spillConfig,
    common::PrefixSortConfig prefixSortConfig,
    std::unique_ptr<core::ExpressionEvaluator> expressionEvaluator,
    cache::AsyncDataCache* cache,
    const std::string& queryId,
    const std::string& taskId,
    const std::string& planNodeId,
    int driverId,
    const std::string& sessionTimezone,
    bool adjustTimestampToTimezone,
    folly::CancellationToken cancellationToken,
    std::shared_ptr<filesystems::TokenProvider> tokenProvider)
    : ConnectorQueryCtx(
          operatorPool,
          connectorPool,
          sessionProperties,
          spillConfig,
          std::move(prefixSortConfig),
          std::move(expressionEvaluator),
          cache,
          queryId,
          taskId,
          planNodeId,
          driverId,
          sessionTimezone,
          adjustTimestampToTimezone,
          std::move(cancellationToken),
          std::move(tokenProvider),
          {}) {}

ConnectorQueryCtx::ConnectorQueryCtx(
    memory::MemoryPool* operatorPool,
    memory::MemoryPool* connectorPool,
    const config::ConfigBase* sessionProperties,
    const common::SpillConfig* spillConfig,
    common::PrefixSortConfig prefixSortConfig,
    std::unique_ptr<core::ExpressionEvaluator> expressionEvaluator,
    cache::AsyncDataCache* cache,
    const std::string& queryId,
    const std::string& taskId,
    const std::string& planNodeId,
    int driverId,
    const std::string& sessionTimezone,
    bool adjustTimestampToTimezone,
    folly::CancellationToken cancellationToken,
    std::shared_ptr<filesystems::TokenProvider> tokenProvider,
    std::unordered_map<std::string, memory::MemoryPool*> customPools)
    : operatorPool_(operatorPool),
      connectorPool_(connectorPool),
      sessionProperties_(sessionProperties),
      spillConfig_(spillConfig),
      prefixSortConfig_(prefixSortConfig),
      expressionEvaluator_(std::move(expressionEvaluator)),
      cache_(cache),
      scanId_(fmt::format("{}.{}", taskId, planNodeId)),
      queryId_(queryId),
      taskId_(taskId),
      driverId_(driverId),
      planNodeId_(planNodeId),
      sessionTimezone_(sessionTimezone),
      adjustTimestampToTimezone_(adjustTimestampToTimezone),
      cancellationToken_(std::move(cancellationToken)),
      fsTokenProvider_(std::move(tokenProvider)),
      customPools_(std::move(customPools)) {
  VELOX_CHECK_NOT_NULL(sessionProperties);
}

ScopedRegistry<std::string, Connector>& connectors() {
  static ScopedRegistry<std::string, Connector> instance;
  return instance;
}

bool registerConnector(const std::shared_ptr<Connector>& connector) {
  connectors().insert(connector->connectorId(), connector);
  return true;
}

bool unregisterConnector(const std::string& connectorId) {
  return connectors().erase(connectorId);
}

std::shared_ptr<Connector> getConnector(const std::string& connectorId) {
  auto connector = connectors().find(connectorId);
  VELOX_CHECK_NOT_NULL(
      connector, "Connector with ID is not registered: {}", connectorId);
  return connector;
}

bool hasConnector(const std::string& connectorId) {
  return connectors().find(connectorId) != nullptr;
}

bool DataSink::Stats::empty() const {
  return numWrittenBytes == 0 && numWrittenFiles == 0 &&
      writerRuntimeStats.empty() && spillStats.empty();
}

std::string DataSink::Stats::toString() const {
  return fmt::format(
      "numWrittenBytes {} numWrittenFiles {} {}",
      succinctBytes(numWrittenBytes),
      numWrittenFiles,
      spillStats.toString());
}

folly::Synchronized<
    std::unordered_map<std::string_view, std::weak_ptr<cache::ScanTracker>>>
    Connector::trackers_;

// static
void Connector::unregisterTracker(cache::ScanTracker* tracker) {
  trackers_.withWLock([&](auto& trackers) { trackers.erase(tracker->id()); });
}

std::shared_ptr<cache::ScanTracker> Connector::getTracker(
    const std::string& scanId,
    int32_t loadQuantum) {
  return trackers_.withWLock([&](auto& trackers) -> auto {
    auto it = trackers.find(scanId);
    if (it == trackers.end()) {
      auto newTracker = std::make_shared<cache::ScanTracker>(
          scanId, unregisterTracker, loadQuantum);
      trackers[newTracker->id()] = newTracker;
      return newTracker;
    }
    std::shared_ptr<cache::ScanTracker> tracker = it->second.lock();
    if (!tracker) {
      tracker = std::make_shared<cache::ScanTracker>(
          scanId, unregisterTracker, loadQuantum);
      trackers[tracker->id()] = tracker;
    }
    return tracker;
  });
}

namespace {
const folly::F14FastMap<CommitStrategy, std::string_view>&
commitStrategyNames() {
  static const folly::F14FastMap<CommitStrategy, std::string_view> kNames = {
      {CommitStrategy::kNoCommit, "NO_COMMIT"},
      {CommitStrategy::kTaskCommit, "TASK_COMMIT"},
  };
  return kNames;
}
} // namespace

VELOX_DEFINE_ENUM_NAME(CommitStrategy, commitStrategyNames);

folly::dynamic ColumnHandle::serializeBase(std::string_view name) {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = name;
  return obj;
}

folly::dynamic ColumnHandle::serialize() const {
  return serializeBase("ColumnHandle");
}

folly::dynamic ConnectorTableHandle::serializeBase(
    std::string_view name) const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = name;
  obj["connectorId"] = connectorId_;
  return obj;
}

folly::dynamic ConnectorTableHandle::serialize() const {
  return serializeBase("ConnectorTableHandle");
}

} // namespace facebook::velox::connector
