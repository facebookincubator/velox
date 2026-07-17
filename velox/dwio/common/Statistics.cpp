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

#include "velox/dwio/common/Statistics.h"

#include <fmt/format.h>

namespace facebook::velox::dwio::common {
namespace {

constexpr std::string_view kUnknown = "unknown";

template <typename T>
std::string toStringOr(
    const std::optional<T>& value,
    std::string_view fallback) {
  if (value.has_value()) {
    return folly::to<std::string>(*value);
  }
  return std::string{fallback};
}

void mergeRuntimeMetric(
    std::string name,
    const RuntimeMetric& metric,
    std::unordered_map<std::string, RuntimeMetric>& result) {
  auto [it, inserted] = result.emplace(std::move(name), metric);
  if (!inserted) {
    it->second.merge(metric);
  }
}

} // namespace

bool KeyInfo::operator==(const KeyInfo& other) const {
  return intKey == other.intKey && bytesKey == other.bytesKey;
}

std::string KeyInfo::toString() const {
  if (intKey.has_value()) {
    return folly::to<std::string>(*intKey);
  }
  if (bytesKey.has_value()) {
    return *bytesKey;
  }
  VELOX_UNREACHABLE("Illegal null key info");
}

size_t KeyInfoHash::operator()(const KeyInfo& keyInfo) const {
  if (keyInfo.intKey.has_value()) {
    return folly::Hash{}(*keyInfo.intKey);
  }
  if (keyInfo.bytesKey.has_value()) {
    return folly::Hash{}(*keyInfo.bytesKey);
  }
  VELOX_UNREACHABLE("Illegal null key info");
}

void ColumnStatistics::setNumDistinct(int64_t count) {
  VELOX_CHECK(!numDistinct_.has_value(), "numDistinct_ can be set only once.");
  numDistinct_ = count;
}

bool ColumnStatistics::isAllNull() const {
  return valueCount_.has_value() && valueCount_.value() == 0;
}

std::string ColumnStatistics::toString() const {
  return folly::to<std::string>(
      "RawSize: ",
      toStringOr(rawSize_, kUnknown),
      ", Size: ",
      toStringOr(size_, kUnknown),
      ", Values: ",
      toStringOr(valueCount_, kUnknown),
      ", hasNull: ",
      (hasNull_ ? (hasNull_.value() ? "yes" : "no") : kUnknown));
}

std::string BinaryColumnStatistics::toString() const {
  return folly::to<std::string>(
      ColumnStatistics::toString(),
      ", Length: ",
      toStringOr(length_, kUnknown));
}

std::optional<uint64_t> BooleanColumnStatistics::getFalseCount() const {
  auto valueCount = getNumberOfValues();
  return trueCount_.has_value() && valueCount.has_value()
      ? std::optional{valueCount.value() - trueCount_.value()}
      : std::nullopt;
}

std::string BooleanColumnStatistics::toString() const {
  return folly::to<std::string>(
      ColumnStatistics::toString(),
      ", trueCount: ",
      toStringOr(trueCount_, kUnknown));
}

std::string DoubleColumnStatistics::toString() const {
  return folly::to<std::string>(
      ColumnStatistics::toString(),
      ", min: ",
      toStringOr(min_, kUnknown),
      ", max: ",
      toStringOr(max_, kUnknown),
      ", sum: ",
      toStringOr(sum_, kUnknown));
}

std::string IntegerColumnStatistics::toString() const {
  return folly::to<std::string>(
      ColumnStatistics::toString(),
      ", min: ",
      toStringOr(min_, kUnknown),
      ", max: ",
      toStringOr(max_, kUnknown),
      ", sum: ",
      toStringOr(sum_, kUnknown));
}

std::string TimestampColumnStatistics::toString() const {
  return folly::to<std::string>(
      ColumnStatistics::toString(),
      ", min: ",
      toStringOr(min_, kUnknown),
      ", max: ",
      toStringOr(max_, kUnknown));
}

std::string StringColumnStatistics::toString() const {
  return folly::to<std::string>(
      ColumnStatistics::toString(),
      ", min: ",
      toStringOr(min_, kUnknown),
      ", max: ",
      toStringOr(max_, kUnknown),
      ", length: ",
      toStringOr(length_, kUnknown));
}

std::string MapColumnStatistics::toString() const {
  std::vector<std::string> values{};
  values.reserve(entryStatistics_.size());
  for (const auto& entry : entryStatistics_) {
    auto& stats = *entry.second;
    values.push_back(
        fmt::format(
            "{{ Key: {}, Stats: {},}}",
            entry.first.toString(),
            stats.toString()));
  }
  std::string repr;
  folly::join(",", values, repr);
  return folly::to<std::string>(ColumnStatistics::toString(), repr);
}

void DecodingStats::merge(const DecodingStats& other) {
  decompressCPUTimeNanos.merge(other.decompressCPUTimeNanos);
  decodeCPUTimeNanos.merge(other.decodeCPUTimeNanos);
}

void DecodingStats::toRuntimeMetrics(
    std::string_view prefix,
    std::unordered_map<std::string, RuntimeMetric>& result) const {
  const auto addCounter = [&](std::string_view name, const auto& counter) {
    if (counter.count() == 0) {
      return;
    }
    mergeRuntimeMetric(
        fmt::format("{}.{}", prefix, name),
        RuntimeMetric{
            saturateCast(counter.sum()),
            counter.count(),
            saturateCast(counter.min()),
            saturateCast(counter.max()),
            RuntimeCounter::Unit::kNanos},
        result);
  };
  addCounter("decompressCPUTimeNanos", decompressCPUTimeNanos);
  addCounter("decodeCPUTimeNanos", decodeCPUTimeNanos);
}

void ColumnStats::accumulateStat(
    const std::pair<std::string_view, RuntimeCounter::Unit>& stat,
    int64_t value) {
  auto [it, inserted] = columnMetrics.try_emplace(stat.first);
  if (inserted) {
    it->second.unit = stat.second;
  } else {
    VELOX_CHECK_EQ(it->second.unit, stat.second);
  }
  it->second.addValue(value);
}

ColumnStats& SplitStats::getOrCreateColumnStats(
    uint32_t nodeId,
    TypeKind typeKind) {
  auto [it, inserted] = columnStats.try_emplace(nodeId, typeKind);
  if (!inserted) {
    VELOX_CHECK_EQ(it->second.typeKind, typeKind);
  }
  return it->second;
}

DecodingStats* SplitStats::decodingStats(uint32_t nodeId) {
  const auto it = columnStats.find(nodeId);
  return it != columnStats.end() && it->second.decodingStats
      ? &*it->second.decodingStats
      : nullptr;
}

void SplitStats::initColumnStatsCollection(
    const TypeWithId& schema,
    const RowReaderOptions& options) {
  registerColumnStats(schema, options.collectColumnCpuMetrics());
}

void ColumnStats::mergeFrom(const ColumnStats& other) {
  VELOX_CHECK_EQ(typeKind, other.typeKind);
  for (const auto& [name, metric] : other.columnMetrics) {
    auto [it, inserted] = columnMetrics.emplace(name, metric);
    if (!inserted) {
      it->second.merge(metric);
    }
  }
  if (other.decodingStats) {
    if (!decodingStats) {
      decodingStats.emplace();
    }
    decodingStats->merge(*other.decodingStats);
  }
}

void ColumnStats::toRuntimeMetrics(
    std::string_view prefix,
    std::unordered_map<std::string, RuntimeMetric>& result) const {
  for (const auto& [name, metric] : columnMetrics) {
    mergeRuntimeMetric(fmt::format("{}.{}", prefix, name), metric, result);
  }
  if (decodingStats) {
    decodingStats->toRuntimeMetrics(prefix, result);
  }
}

void SplitStats::accumulateStat(
    const std::pair<std::string_view, RuntimeCounter::Unit>& stat,
    int64_t value) {
  auto [it, inserted] = splitMetrics.try_emplace(stat.first);
  if (inserted) {
    it->second.unit = stat.second;
  } else {
    VELOX_CHECK_EQ(it->second.unit, stat.second);
  }
  it->second.addValue(value);
}

void RuntimeStats::mergeFrom(const SplitStats& split) {
  auto& target = formatSpecificStats[split.format];
  for (const auto& [name, metric] : split.splitMetrics) {
    auto [it, inserted] = target.emplace(name, metric);
    if (!inserted) {
      VELOX_CHECK_EQ(it->second.unit, metric.unit);
      it->second.merge(metric);
    }
  }
  for (const auto& [nodeId, stats] : split.columnStats) {
    auto it =
        columnStats[nodeId].try_emplace(split.format, stats.typeKind).first;
    it->second.mergeFrom(stats);
  }
}

void SplitStats::registerColumnStats(
    const TypeWithId& node,
    bool collectDecodingStats) {
  auto& stats = getOrCreateColumnStats(node.id(), node.type()->kind());
  if (collectDecodingStats && !stats.decodingStats) {
    stats.decodingStats.emplace();
  }
  for (uint32_t i = 0; i < node.size(); ++i) {
    if (const auto* child = node.childAt(i).get()) {
      registerColumnStats(*child, collectDecodingStats);
    }
  }
}

std::unordered_map<std::string, RuntimeMetric>
RuntimeStats::toRuntimeMetricMap() const {
  std::unordered_map<std::string, RuntimeMetric> result;
  for (const auto& [name, metric] : unitLoaderStats.stats()) {
    result.emplace(name, RuntimeMetric(metric.sum, metric.unit));
  }
  if (skippedSplits > 0) {
    result.emplace("skippedSplits", RuntimeMetric(skippedSplits));
  }
  if (processedSplits > 0) {
    result.emplace("processedSplits", RuntimeMetric(processedSplits));
  }
  if (skippedSplitBytes > 0) {
    result.emplace(
        "skippedSplitBytes",
        RuntimeMetric(skippedSplitBytes, RuntimeCounter::Unit::kBytes));
  }
  if (skippedStrides > 0) {
    result.emplace("skippedStrides", RuntimeMetric(skippedStrides));
  }
  if (processedStrides > 0) {
    result.emplace("processedStrides", RuntimeMetric(processedStrides));
  }
  if (footerBufferOverread > 0) {
    result.emplace(
        "footerBufferOverread",
        RuntimeMetric(footerBufferOverread, RuntimeCounter::Unit::kBytes));
  }
  if (footerBufferUnderread > 0) {
    result.emplace(
        "footerBufferUnderread",
        RuntimeMetric(footerBufferUnderread, RuntimeCounter::Unit::kBytes));
  }
  if (footerCacheHit > 0) {
    result.emplace("footerCacheHit", RuntimeMetric(footerCacheHit));
  }
  if (numStripes > 0) {
    result.emplace("numStripes", RuntimeMetric(numStripes));
  }
  for (const auto& [format, metrics] : formatSpecificStats) {
    for (const auto& [name, metric] : metrics) {
      result.emplace(
          fmt::format("{}.{}", FileFormatName::toName(format), name), metric);
    }
  }
  for (const auto& [nodeId, statsByFormat] : columnStats) {
    for (const auto& [format, stats] : statsByFormat) {
      const auto formatPrefix = FileFormatName::toName(format);
      const auto typeName = TypeKindName::toName(stats.typeKind);
      const auto formatAndColumnPrefix =
          fmt::format("{}.column_{}.{}", formatPrefix, nodeId, typeName);
      stats.toRuntimeMetrics(formatPrefix, result);
      stats.toRuntimeMetrics(formatAndColumnPrefix, result);
    }
  }
  return result;
}
} // namespace facebook::velox::dwio::common
