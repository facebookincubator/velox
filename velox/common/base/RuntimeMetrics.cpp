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

#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/SuccinctPrinter.h"

#include <folly/Synchronized.h>
#include <folly/container/F14Set.h>

namespace facebook::velox {

namespace {

folly::Synchronized<folly::F14FastSet<std::string>>&
operatorAggregatedMetrics() {
  static folly::Synchronized<folly::F14FastSet<std::string>> registry;
  return registry;
}

const folly::F14FastSet<std::string>& defaultOperatorAggregatedMetrics() {
  static const folly::F14FastSet<std::string> metrics{
      "cacheWaitWallNanos",
      "coalescedSsdLoadWallNanos",
      "coalescedStorageLoadWallNanos",
      "dataSourceAddSplitWallNanos",
      "dataSourceLazyCpuNanos",
      "dataSourceLazyWallNanos",
      "dataSourceLazyInputBytes",
      "dataSourceReadWallNanos",
      "driverCpuTimeNanos",
      "flushTimes",
      "ioWaitWallNanos",
      "prefetchBytes",
      "preloadSplitPrepareTimeNanos",
      "preloadedSplits",
      "ramReadBytes",
      "readyPreloadedSplits",
      "rpcCongestionWindowFinal",
      "rpcPeakInFlight",
      "rpcBaselineRttNanos",
      "rpcRttMinWallNanos",
      "rpcRttMaxWallNanos",
      "rpcStreamingMode",
      "queuedWallNanos",
      "storageReadWallNanos",
      "storageReadBytes",
      "ssdCacheReadWallNanos",
      "waitForPreloadSplitNanos",
  };
  return metrics;
}

} // namespace

void RuntimeMetric::addValue(int64_t value) {
  sum += value;
  count++;
  min = std::min(min, value);
  max = std::max(max, value);
}

void RuntimeMetric::aggregate() {
  count = std::min(count, static_cast<uint64_t>(1));
  min = max = sum;
}

void OperatorAggregatedMetrics::add(std::string name) {
  VELOX_CHECK(
      !defaultOperatorAggregatedMetrics().contains(name),
      "Metric is built in for operator aggregation: {}",
      name);
  operatorAggregatedMetrics().withWLock([&](auto& metrics) {
    const auto inserted = metrics.insert(name).second;
    VELOX_CHECK(
        inserted,
        "Metric is already registered for operator aggregation: {}",
        name);
  });
}

void OperatorAggregatedMetrics::remove(const std::string& name) {
  VELOX_CHECK(
      !defaultOperatorAggregatedMetrics().contains(name),
      "Metric is built in for operator aggregation: {}",
      name);
  operatorAggregatedMetrics().withWLock([&](auto& metrics) {
    VELOX_CHECK_EQ(
        metrics.erase(name),
        1,
        "Metric is not registered for operator aggregation: {}",
        name);
  });
}

bool OperatorAggregatedMetrics::contains(const std::string& name) {
  if (defaultOperatorAggregatedMetrics().contains(name)) {
    return true;
  }
  return operatorAggregatedMetrics().withRLock(
      [&](const auto& metrics) { return metrics.contains(name); });
}

void RuntimeMetric::merge(const RuntimeCounter& value) {
  VELOX_CHECK_EQ(unit, value.unit, "Unit mismatch for runtime stat");
  addValue(value.value);
}

void RuntimeMetric::merge(const RuntimeMetric& other)
#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
    __attribute__((__no_sanitize__("signed-integer-overflow")))
#endif
#endif
{
  VELOX_CHECK_EQ(unit, other.unit);
  sum += other.sum;
  count += other.count;
  min = std::min(min, other.min);
  max = std::max(max, other.max);
}

void RuntimeMetric::printMetric(std::ostream& stream) const {
  switch (unit) {
    case RuntimeCounter::Unit::kNanos:
      stream << " sum: " << succinctNanos(sum) << ", count: " << count
             << ", min: " << succinctNanos(min)
             << ", max: " << succinctNanos(max)
             << ", avg: " << succinctNanos(count == 0 ? 0 : sum / count);
      break;
    case RuntimeCounter::Unit::kBytes:
      stream << " sum: " << succinctBytes(sum) << ", count: " << count
             << ", min: " << succinctBytes(min)
             << ", max: " << succinctBytes(max)
             << ", avg: " << succinctBytes(count == 0 ? 0 : sum / count);
      break;
    case RuntimeCounter::Unit::kNone:
      [[fallthrough]];
    default:
      stream << " sum: " << sum << ", count: " << count << ", min: " << min
             << ", max: " << max << ", avg: " << (count == 0 ? 0 : sum / count);
  }
}

std::string RuntimeMetric::toString() const {
  switch (unit) {
    case RuntimeCounter::Unit::kNanos:
      return fmt::format(
          "sum:{}, count:{}, min:{}, max:{}, avg: {}",
          succinctNanos(sum),
          count,
          succinctNanos(min),
          succinctNanos(max),
          succinctNanos(count == 0 ? 0 : sum / count));
    case RuntimeCounter::Unit::kBytes:
      return fmt::format(
          "sum:{}, count:{}, min:{}, max:{}, avg: {}",
          succinctBytes(sum),
          count,
          succinctBytes(min),
          succinctBytes(max),
          succinctBytes(count == 0 ? 0 : sum / count));
    case RuntimeCounter::Unit::kNone:
      [[fallthrough]];
    default:
      return fmt::format(
          "sum:{}, count:{}, min:{}, max:{}, avg: {}",
          sum,
          count,
          min,
          max,
          count == 0 ? 0 : sum / count);
  }
}

// Thread local runtime stat writers.
static thread_local BaseRuntimeStatWriter* localRuntimeStatWriter;

void setThreadLocalRunTimeStatWriter(BaseRuntimeStatWriter* writer) {
  localRuntimeStatWriter = writer;
}

BaseRuntimeStatWriter* getThreadLocalRunTimeStatWriter() {
  return localRuntimeStatWriter;
}

void addThreadLocalRuntimeStat(
    std::string_view name,
    const RuntimeCounter& value) {
  if (localRuntimeStatWriter) {
    localRuntimeStatWriter->addRuntimeStat(name, value);
  }
}

void setThreadLocalRuntimeStat(
    std::string_view name,
    const RuntimeMetric& metric) {
  if (localRuntimeStatWriter) {
    localRuntimeStatWriter->setRuntimeStat(name, metric);
  }
}

} // namespace facebook::velox
