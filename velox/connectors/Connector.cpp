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

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "velox/common/ScopedRegistry.h"
#include "velox/common/base/Exceptions.h"
#include "velox/connectors/ConnectorRegistryInternal.h"

namespace facebook::velox::connector {

namespace {
int64_t totalSplitWeight(
    const std::vector<std::shared_ptr<ConnectorSplit>>& splits) {
  int64_t total = 0;
  for (const auto& split : splits) {
    VELOX_USER_CHECK_NOT_NULL(split);
    VELOX_USER_CHECK_GE(split->splitWeight, 0);
    if (total < std::numeric_limits<int64_t>::max()) {
      total = split->splitWeight > std::numeric_limits<int64_t>::max() - total
          ? std::numeric_limits<int64_t>::max()
          : total + split->splitWeight;
    }
  }
  return total;
}

bool allSplitsCacheable(
    const std::vector<std::shared_ptr<ConnectorSplit>>& splits) {
  return std::all_of(splits.begin(), splits.end(), [](const auto& split) {
    return split != nullptr && split->cacheable;
  });
}
} // namespace

ConnectorSplitBatch::ConnectorSplitBatch(
    const std::string& connectorId,
    std::vector<std::shared_ptr<ConnectorSplit>> _splits)
    : ConnectorSplit(
          connectorId,
          totalSplitWeight(_splits),
          allSplitsCacheable(_splits)),
      splits(std::move(_splits)) {
  VELOX_USER_CHECK(!splits.empty(), "A split batch must not be empty");
  for (const auto& split : splits) {
    VELOX_USER_CHECK_NOT_NULL(split);
    VELOX_USER_CHECK(
        dynamic_cast<const ConnectorSplitBatch*>(split.get()) == nullptr,
        "Nested split batches are not supported");
    VELOX_USER_CHECK(
        split->dataSource == nullptr,
        "Preloaded splits cannot be added to a split batch");
    VELOX_USER_CHECK_EQ(
        split->connectorId,
        connectorId,
        "All splits in a batch must use the same connector ID");
  }
  const auto hint = splits.front()->batchSizeHint;
  if (std::all_of(splits.begin(), splits.end(), [hint](const auto& split) {
        return split->batchSizeHint == hint;
      })) {
    batchSizeHint = hint;
  }
}

std::string ConnectorSplitBatch::toString() const {
  return fmt::format("Split batch: {} splits", splits.size());
}

uint64_t ConnectorSplitBatch::size() const {
  uint64_t total = 0;
  for (const auto& split : splits) {
    const auto splitSize = split->size();
    if (splitSize > std::numeric_limits<uint64_t>::max() - total) {
      return std::numeric_limits<uint64_t>::max();
    }
    total += splitSize;
  }
  return total;
}

void DataSource::addSplit(
    const std::vector<std::shared_ptr<ConnectorSplit>>& splits) {
  VELOX_CHECK_EQ(
      splits.size(), 1, "Data source does not support split batches");
  addSplit(splits.front());
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
