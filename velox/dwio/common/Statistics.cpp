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
      (min_.has_value() ? min_.value().toString() : "unknown"),
      ", max: ",
      (max_.has_value() ? max_.value().toString() : "unknown"));
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

DecodingStats* DecodingStatsSet::getOrCreate(
    uint32_t nodeId,
    TypeKind typeKind) {
  auto locked = map_.wlock();
  auto it = locked->find(nodeId);
  if (it == locked->end()) {
    it = locked->emplace(nodeId, std::make_unique<DecodingStats>(typeKind))
             .first;
  }
  return it->second.get();
}

void DecodingStatsSet::mergeFrom(const DecodingStatsSet& other) {
  auto srcLocked = other.map_.rlock();
  auto dstLocked = map_.wlock();
  for (const auto& [nodeId, srcStats] : *srcLocked) {
    auto it = dstLocked->find(nodeId);
    if (it == dstLocked->end()) {
      it = dstLocked->emplace(nodeId, std::make_unique<DecodingStats>()).first;
      it->second->typeKind = srcStats->typeKind;
    }
    it->second->merge(*srcStats);
  }
}

void DecodingStatsSet::toRuntimeMetrics(
    std::unordered_map<std::string, RuntimeMetric>& result) const {
  auto statsLocked = map_.rlock();
  for (const auto& [nodeId, stats] : *statsLocked) {
    // Export decompression timing.
    const auto& decompressCounter = stats->decompressCPUTimeNanos;
    if (decompressCounter.count() > 0) {
      result.emplace(
          fmt::format(
              "column_{}.{}.decompressCPUTimeNanos",
              nodeId,
              TypeKindName::toName(stats->typeKind)),
          RuntimeMetric{
              saturateCast(decompressCounter.sum()),
              decompressCounter.count(),
              saturateCast(decompressCounter.min()),
              saturateCast(decompressCounter.max()),
              RuntimeCounter::Unit::kNanos});
    }
    // Export decode timing.
    const auto& decodeCounter = stats->decodeCPUTimeNanos;
    if (decodeCounter.count() > 0) {
      result.emplace(
          fmt::format(
              "column_{}.{}.decodeCPUTimeNanos",
              nodeId,
              TypeKindName::toName(stats->typeKind)),
          RuntimeMetric{
              saturateCast(decodeCounter.sum()),
              decodeCounter.count(),
              saturateCast(decodeCounter.min()),
              saturateCast(decodeCounter.max()),
              RuntimeCounter::Unit::kNanos});
    }
  }
}

void ColumnReaderStatistics::initColumnStatsCollection(
    const TypeWithId& schema,
    const RowReaderOptions& options) {
  if (!options.collectColumnCpuMetrics()) {
    return;
  }
  decodingStatsSet.emplace();
  registerDecodingStatsImpl(schema);
}

void ColumnReaderStatistics::mergeFrom(const ColumnReaderStatistics& other) {
  flattenStringDictionaryValues += other.flattenStringDictionaryValues;
  pageLoadTimeNs.merge(other.pageLoadTimeNs);
  if (other.decodingStatsSet) {
    if (!decodingStatsSet) {
      decodingStatsSet.emplace();
    }
    decodingStatsSet->mergeFrom(*other.decodingStatsSet);
  }
}

void ColumnReaderStatistics::toRuntimeMetrics(
    std::unordered_map<std::string, RuntimeMetric>& result) const {
  if (flattenStringDictionaryValues > 0) {
    result.emplace(
        "flattenStringDictionaryValues",
        RuntimeMetric(flattenStringDictionaryValues));
  }
  if (pageLoadTimeNs.sum() > 0) {
    result.emplace(
        "pageLoadTimeNanos",
        RuntimeMetric(
            pageLoadTimeNs.sum(),
            pageLoadTimeNs.count(),
            pageLoadTimeNs.min(),
            pageLoadTimeNs.max(),
            RuntimeCounter::Unit::kNanos));
  }
  if (decodingStatsSet) {
    decodingStatsSet->toRuntimeMetrics(result);
  }
}

void ColumnReaderStatistics::registerDecodingStatsImpl(const TypeWithId& node) {
  decodingStatsSet->getOrCreate(node.id(), node.type()->kind());
  for (uint32_t i = 0; i < node.size(); ++i) {
    if (const auto* child = node.childAt(i).get()) {
      registerDecodingStatsImpl(*child);
    }
  }
}

std::unordered_map<std::string, RuntimeMetric>
RuntimeStatistics::toRuntimeMetricMap() const {
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
  if (parquetFooterEstimatedBytes > 0) {
    result.emplace(
        "parquetFooterEstimatedBytes",
        RuntimeMetric(
            parquetFooterEstimatedBytes, RuntimeCounter::Unit::kBytes));
  }
  columnReaderStats.toRuntimeMetrics(result);
  return result;
}
} // namespace facebook::velox::dwio::common
