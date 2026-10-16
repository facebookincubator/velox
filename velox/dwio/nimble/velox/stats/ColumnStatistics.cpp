/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/velox/stats/ColumnStatistics.h"

#include <bit>
#include <cmath>
#include <optional>
#include <utility>

#include "velox/common/base/XxHashInline.h"
#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/velox/stats/ColumnStatsUtils.h"
#include "velox/functions/lib/HllAccumulator.h"

namespace facebook::nimble {

namespace {

// Storage-agnostic HLL accumulator: values are pre-hashed with XXH64 (the
// HllAsFinalResult=false path) and fed via insertHash, so the element type
// parameter is irrelevant.
using HllAcc = velox::common::hll::
    HllAccumulator<int64_t, false, velox::HashStringAllocator>;

// 2^11 = 2048 buckets (~2.3% standard error), matching the DWRF writers. HLLs
// with different bucket counts cannot be merged.
constexpr int8_t kHllIndexBitLength = 11;

// Hashes match the Java DWRF writers (XxHash64.hash(long) over the value
// widened to 64 bits, floats via floatToIntBits / doubleToLongBits with
// canonical NaN) so sketches from either format merge without NDV drift.
uint64_t hllHashLong(int64_t value) {
  return XXH64(&value, sizeof(value), 0);
}

template <typename T>
uint64_t hllHash(T value) {
  if constexpr (std::is_integral_v<T>) {
    return hllHashLong(static_cast<int64_t>(value));
  } else if constexpr (std::is_same_v<T, float>) {
    const int32_t bits =
        std::isnan(value) ? 0x7fc00000 : std::bit_cast<int32_t>(value);
    return hllHashLong(bits);
  } else {
    static_assert(std::is_same_v<T, double>);
    return hllHashLong(
        std::isnan(value) ? int64_t{0x7ff8000000000000}
                          : std::bit_cast<int64_t>(value));
  }
}

template <typename T>
void addFloatingPointValuesScalar(
    std::optional<double>& min,
    std::optional<double>& max,
    std::span<T> values) {
  if (UNLIKELY(!min.has_value())) {
    min = static_cast<double>(values.front());
  }
  if (UNLIKELY(!max.has_value())) {
    max = static_cast<double>(values.front());
  }

  for (const auto& value : values) {
    auto current = static_cast<double>(value);
    min = min > current ? std::make_optional(current) : min;
    max = max < current ? std::make_optional(current) : max;
  }
}

} // namespace

ColumnStatistics::ColumnStatistics(
    uint64_t valueCount,
    uint64_t nullCount,
    uint64_t logicalSize,
    uint64_t physicalSize)
    : valueCount_{valueCount},
      nullCount_{nullCount},
      logicalSize_{logicalSize},
      physicalSize_{physicalSize} {}

uint64_t ColumnStatistics::getValueCount() const {
  return valueCount_;
}

uint64_t ColumnStatistics::getNullCount() const {
  return nullCount_;
}

uint64_t ColumnStatistics::getLogicalSize() const {
  return logicalSize_;
}

uint64_t ColumnStatistics::getPhysicalSize() const {
  return physicalSize_;
}

StatType ColumnStatistics::getType() const {
  return StatType::DEFAULT;
}

std::unique_ptr<ColumnStatistics> ColumnStatistics::clone() const {
  auto cloned = std::make_unique<ColumnStatistics>(
      getValueCount(), getNullCount(), getLogicalSize(), getPhysicalSize());
  cloned->ndvSketch_ = ndvSketch_;
  return cloned;
}

std::optional<int64_t> ColumnStatistics::getNdvCardinality() const {
  const auto& sketch = getNdvSketch();
  if (!sketch.has_value()) {
    return std::nullopt;
  }
  const char* data = sketch->data();
  namespace hll = velox::common::hll;
  if (hll::SparseHlls::canDeserialize(data)) {
    return hll::SparseHlls::cardinality(data);
  }
  if (hll::DenseHlls::canDeserialize(data)) {
    return hll::DenseHlls::cardinality(data);
  }
  return std::nullopt;
}

/* static */ std::optional<std::string> ColumnStatistics::mergeNdvSketches(
    const std::optional<std::string>& lhs,
    const std::optional<std::string>& rhs,
    velox::memory::MemoryPool* pool) {
  if (!lhs.has_value()) {
    return rhs;
  }
  if (!rhs.has_value()) {
    return lhs;
  }
  velox::HashStringAllocator allocator{pool};
  auto accumulator = HllAcc::deserialize(lhs->data(), &allocator);
  accumulator->mergeWith(
      velox::StringView{rhs->data(), static_cast<int32_t>(rhs->size())},
      &allocator);
  std::string merged(static_cast<size_t>(accumulator->serializedSize()), '\0');
  accumulator->serialize(merged.data());
  return merged;
}

std::unique_ptr<velox::dwio::common::ColumnStatistics>
ColumnStatistics::toCommonStatistics() const {
  const auto valueCount = std::make_optional<uint64_t>(getValueCount());
  const auto hasNull = std::make_optional<bool>(getNullCount() > 0);
  const auto rawSize = std::make_optional<uint64_t>(getLogicalSize());
  const auto size = std::make_optional<uint64_t>(getPhysicalSize());

  std::unique_ptr<velox::dwio::common::ColumnStatistics> result;
  switch (getType()) {
    case StatType::INTEGRAL: {
      const auto* intStats = as<const IntegralStatistics>();
      NIMBLE_DCHECK(
          intStats != nullptr, "Failed to cast to IntegralStatistics");
      result = std::make_unique<velox::dwio::common::IntegerColumnStatistics>(
          valueCount,
          hasNull,
          rawSize,
          size,
          intStats->getMin(),
          intStats->getMax(),
          std::nullopt);
      break;
    }
    case StatType::FLOATING_POINT: {
      const auto* fpStats = as<const FloatingPointStatistics>();
      NIMBLE_DCHECK(
          fpStats != nullptr, "Failed to cast to FloatingPointStatistics");
      result = std::make_unique<velox::dwio::common::DoubleColumnStatistics>(
          valueCount,
          hasNull,
          rawSize,
          size,
          fpStats->getMin(),
          fpStats->getMax(),
          std::nullopt);
      break;
    }
    case StatType::STRING: {
      const auto* strStats = as<const StringStatistics>();
      NIMBLE_DCHECK(strStats != nullptr, "Failed to cast to StringStatistics");
      result = std::make_unique<velox::dwio::common::StringColumnStatistics>(
          valueCount,
          hasNull,
          rawSize,
          size,
          strStats->getMin(),
          strStats->getMax(),
          std::nullopt);
      break;
    }
    case StatType::DEFAULT:
    case StatType::DEDUPLICATED:
      result = std::make_unique<velox::dwio::common::ColumnStatistics>(
          valueCount, hasNull, rawSize, size);
      break;
  }

  if (const auto cardinality = getNdvCardinality()) {
    result->setNumDistinct(*cardinality);
  }
  return result;
}

StringStatistics::StringStatistics(
    uint64_t valueCount,
    uint64_t nullCount,
    uint64_t logicalSize,
    uint64_t physicalSize,
    std::optional<std::string> min,
    std::optional<std::string> max)
    : ColumnStatistics(valueCount, nullCount, logicalSize, physicalSize),
      min_{std::move(min)},
      max_{std::move(max)} {}

std::optional<std::string> StringStatistics::getMin() const {
  return min_;
}

std::optional<std::string> StringStatistics::getMax() const {
  return max_;
}

StatType StringStatistics::getType() const {
  return StatType::STRING;
}

std::unique_ptr<ColumnStatistics> StringStatistics::clone() const {
  auto cloned = std::make_unique<StringStatistics>(
      getValueCount(),
      getNullCount(),
      getLogicalSize(),
      getPhysicalSize(),
      getMin(),
      getMax());
  cloned->setNdvSketch(ndvSketch_);
  return cloned;
}

IntegralStatistics::IntegralStatistics(
    uint64_t valueCount,
    uint64_t nullCount,
    uint64_t logicalSize,
    uint64_t physicalSize,
    std::optional<int64_t> min,
    std::optional<int64_t> max)
    : ColumnStatistics(valueCount, nullCount, logicalSize, physicalSize),
      min_{std::move(min)},
      max_{std::move(max)} {}

std::optional<int64_t> IntegralStatistics::getMin() const {
  return min_;
}

std::optional<int64_t> IntegralStatistics::getMax() const {
  return max_;
}

StatType IntegralStatistics::getType() const {
  return StatType::INTEGRAL;
}

std::unique_ptr<ColumnStatistics> IntegralStatistics::clone() const {
  auto cloned = std::make_unique<IntegralStatistics>(
      getValueCount(),
      getNullCount(),
      getLogicalSize(),
      getPhysicalSize(),
      getMin(),
      getMax());
  cloned->setNdvSketch(ndvSketch_);
  return cloned;
}

FloatingPointStatistics::FloatingPointStatistics(
    uint64_t valueCount,
    uint64_t nullCount,
    uint64_t logicalSize,
    uint64_t physicalSize,
    std::optional<double> min,
    std::optional<double> max)
    : ColumnStatistics(valueCount, nullCount, logicalSize, physicalSize),
      min_{std::move(min)},
      max_{std::move(max)} {}

std::optional<double> FloatingPointStatistics::getMin() const {
  return min_;
}

std::optional<double> FloatingPointStatistics::getMax() const {
  return max_;
}

StatType FloatingPointStatistics::getType() const {
  return StatType::FLOATING_POINT;
}

std::unique_ptr<ColumnStatistics> FloatingPointStatistics::clone() const {
  auto cloned = std::make_unique<FloatingPointStatistics>(
      getValueCount(),
      getNullCount(),
      getLogicalSize(),
      getPhysicalSize(),
      getMin(),
      getMax());
  cloned->setNdvSketch(ndvSketch_);
  return cloned;
}

DeduplicatedColumnStatistics::DeduplicatedColumnStatistics(
    ColumnStatistics* baseStatistics,
    uint64_t dedupedCount,
    uint64_t dedupedLogicalSize)
    : baseStatistics_{baseStatistics},
      dedupedCount_{dedupedCount},
      dedupedLogicalSize_{dedupedLogicalSize} {}

DeduplicatedColumnStatistics::DeduplicatedColumnStatistics(
    std::unique_ptr<ColumnStatistics> baseStatistics,
    uint64_t dedupedCount,
    uint64_t dedupedLogicalSize)
    : ownedBaseStatistics_{std::move(baseStatistics)},
      baseStatistics_{ownedBaseStatistics_.get()},
      dedupedCount_{dedupedCount},
      dedupedLogicalSize_{dedupedLogicalSize} {}

uint64_t DeduplicatedColumnStatistics::getValueCount() const {
  return baseStatistics_ ? baseStatistics_->getValueCount() : 0;
}

uint64_t DeduplicatedColumnStatistics::getNullCount() const {
  return baseStatistics_ ? baseStatistics_->getNullCount() : 0;
}

uint64_t DeduplicatedColumnStatistics::getLogicalSize() const {
  return baseStatistics_ ? baseStatistics_->getLogicalSize() : 0;
}

uint64_t DeduplicatedColumnStatistics::getPhysicalSize() const {
  return baseStatistics_ ? baseStatistics_->getPhysicalSize() : 0;
}

const std::optional<std::string>& DeduplicatedColumnStatistics::getNdvSketch()
    const {
  return baseStatistics_ ? baseStatistics_->getNdvSketch() : ndvSketch_;
}

void DeduplicatedColumnStatistics::setNdvSketch(
    std::optional<std::string> sketch) {
  if (baseStatistics_ != nullptr) {
    baseStatistics_->setNdvSketch(std::move(sketch));
    return;
  }
  ndvSketch_ = std::move(sketch);
}

uint64_t DeduplicatedColumnStatistics::getDedupedCount() const {
  return dedupedCount_;
}

uint64_t DeduplicatedColumnStatistics::getDedupedLogicalSize() const {
  return dedupedLogicalSize_;
}

StatType DeduplicatedColumnStatistics::getType() const {
  return StatType::DEDUPLICATED;
}

std::unique_ptr<ColumnStatistics> DeduplicatedColumnStatistics::clone() const {
  return std::make_unique<DeduplicatedColumnStatistics>(
      baseStatistics_ == nullptr ? std::make_unique<ColumnStatistics>()
                                 : baseStatistics_->clone(),
      getDedupedCount(),
      getDedupedLogicalSize());
}

ColumnStatistics* DeduplicatedColumnStatistics::getBaseStatistics() {
  return baseStatistics_;
}

const ColumnStatistics* DeduplicatedColumnStatistics::getBaseStatistics()
    const {
  return baseStatistics_;
}

StatisticsCollector::StatisticsCollector()
    : stats_{std::make_unique<ColumnStatistics>()} {}

StatisticsCollector::~StatisticsCollector() = default;

void StatisticsCollector::enableHll(velox::memory::MemoryPool* pool) {
  hllPool_ = pool;
  resetHll();
}

void StatisticsCollector::resetHll() {
  hllDirty_ = false;
  // Destroy the accumulator before its backing allocator: the HLL holds
  // allocator-owned storage, so releasing the allocator first would leave the
  // accumulator's destructor accessing freed memory.
  hll_.reset();
  hllAllocator_.reset();
  if (hllPool_ == nullptr) {
    return;
  }
  hllAllocator_ = std::make_unique<velox::HashStringAllocator>(hllPool_);
  hll_ = std::make_unique<HllAcc>(kHllIndexBitLength, hllAllocator_.get());
}

void StatisticsCollector::insertHllHash(uint64_t hash) {
  if (hll_) {
    hll_->insertHash(hash);
    hllDirty_ = true;
  }
}

void StatisticsCollector::flushHll() const {
  if (!hllDirty_) {
    return;
  }
  std::string serialized(static_cast<size_t>(hll_->serializedSize()), '\0');
  hll_->serialize(serialized.data());
  stats_->setNdvSketch(std::move(serialized));
  hllDirty_ = false;
}

uint64_t StatisticsCollector::getValueCount() const {
  return stats_->getValueCount();
}

uint64_t StatisticsCollector::getNullCount() const {
  return stats_->getNullCount();
}

uint64_t StatisticsCollector::getLogicalSize() const {
  return stats_->getLogicalSize();
}

uint64_t StatisticsCollector::getPhysicalSize() const {
  return stats_->getPhysicalSize();
}

StatType StatisticsCollector::getType() const {
  return stats_->getType();
}

ColumnStatistics* StatisticsCollector::getStatsView() {
  flushHll();
  return stats_.get();
}

const ColumnStatistics* StatisticsCollector::getStatsView() const {
  flushHll();
  return stats_.get();
}

void StatisticsCollector::addValues(std::span<bool> values) {
  addLogicalSize(values.size());
}

void StatisticsCollector::addCounts(uint64_t totalCount, uint64_t nullCount) {
  stats_->valueCount_ += (totalCount - nullCount);
  stats_->nullCount_ += nullCount;
}

void StatisticsCollector::addLogicalSize(uint64_t logicalSize) {
  stats_->logicalSize_ += logicalSize;
}

void StatisticsCollector::addPhysicalSize(uint64_t physicalSize) {
  stats_->physicalSize_ += physicalSize;
}

void StatisticsCollector::merge(const StatisticsCollector& other) {
  auto* otherStats = other.getStatsView();
  stats_->valueCount_ += otherStats->getValueCount();
  stats_->nullCount_ += otherStats->getNullCount();
  stats_->logicalSize_ += otherStats->getLogicalSize();
  stats_->physicalSize_ += otherStats->getPhysicalSize();
  if (hll_ && other.hll_) {
    hll_->mergeWith(*other.hll_);
    hllDirty_ = true;
  }
}

void StatisticsCollector::reset() {
  stats_ = std::make_unique<ColumnStatistics>();
  resetHll();
}

/* static */ std::unique_ptr<StatisticsCollector> StatisticsCollector::create(
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& type,
    velox::memory::MemoryPool* hllPool) {
  switch (type->type()->kind()) {
    case velox::TypeKind::TINYINT:
    case velox::TypeKind::SMALLINT:
    case velox::TypeKind::INTEGER:
    case velox::TypeKind::BIGINT: {
      auto collector = std::make_unique<IntegralStatisticsCollector>();
      if (hllPool != nullptr) {
        collector->enableHll(hllPool);
      }
      return collector;
    }
    case velox::TypeKind::REAL:
    case velox::TypeKind::DOUBLE: {
      auto collector = std::make_unique<FloatingPointStatisticsCollector>();
      if (hllPool != nullptr) {
        collector->enableHll(hllPool);
      }
      return collector;
    }
    case velox::TypeKind::VARCHAR:
    case velox::TypeKind::VARBINARY: {
      auto collector = std::make_unique<StringStatisticsCollector>();
      if (hllPool != nullptr) {
        collector->enableHll(hllPool);
      }
      return collector;
    }
    case velox::TypeKind::BOOLEAN:
    case velox::TypeKind::TIMESTAMP:
    case velox::TypeKind::HUGEINT:
    case velox::TypeKind::ROW:
    case velox::TypeKind::ARRAY:
    case velox::TypeKind::MAP:
      return std::make_unique<StatisticsCollector>();
    default:
      NIMBLE_UNSUPPORTED(
          fmt::format("Unsupported schema kind: {}.", type->type()->kind()));
  }
}

IntegralStatisticsCollector::IntegralStatisticsCollector() {
  stats_ = std::make_unique<IntegralStatistics>();
}

IntegralStatistics* IntegralStatisticsCollector::integralStats() {
  return static_cast<IntegralStatistics*>(stats_.get());
}

template <typename T>
void IntegralStatisticsCollector::addValues(std::span<T> values) {
  static_assert(std::is_integral_v<T>);
  addLogicalSize(values.size() * sizeof(T));

  if (values.empty()) {
    return;
  }

  auto* stats = integralStats();
  const auto minMax = findMinMax(values);
  const auto min = static_cast<int64_t>(minMax.min);
  const auto max = static_cast<int64_t>(minMax.max);
  if (!stats->min_.has_value() || min < *stats->min_) {
    stats->min_ = min;
  }
  if (!stats->max_.has_value() || max > *stats->max_) {
    stats->max_ = max;
  }

  if (hll_) {
    for (const auto& value : values) {
      insertHllHash(hllHash(value));
    }
  }
}

void IntegralStatisticsCollector::reset() {
  stats_ = std::make_unique<IntegralStatistics>();
  resetHll();
}

void IntegralStatisticsCollector::merge(const StatisticsCollector& other) {
  auto* otherStats = other.getStatsView();
  NIMBLE_CHECK(
      otherStats->getType() == StatType::INTEGRAL ||
          otherStats->getType() == StatType::DEFAULT,
      "Merging stats with mismatched types.");
  StatisticsCollector::merge(other);
  if (otherStats->getType() == StatType::INTEGRAL) {
    const auto* otherIntegralStats = otherStats->as<IntegralStatistics>();
    auto* stats = integralStats();
    if (otherIntegralStats->getMin().has_value()) {
      if (!stats->min_.has_value() ||
          *otherIntegralStats->getMin() < *stats->min_) {
        stats->min_ = *otherIntegralStats->getMin();
      }
    }
    if (otherIntegralStats->getMax().has_value()) {
      if (!stats->max_.has_value() ||
          *otherIntegralStats->getMax() > *stats->max_) {
        stats->max_ = *otherIntegralStats->getMax();
      }
    }
  }
}

// Explicit template instantiations for IntegralStatisticsCollector::addValues
template void IntegralStatisticsCollector::addValues<int8_t>(std::span<int8_t>);
template void IntegralStatisticsCollector::addValues<int16_t>(
    std::span<int16_t>);
template void IntegralStatisticsCollector::addValues<int32_t>(
    std::span<int32_t>);
template void IntegralStatisticsCollector::addValues<int64_t>(
    std::span<int64_t>);

FloatingPointStatisticsCollector::FloatingPointStatisticsCollector() {
  stats_ = std::make_unique<FloatingPointStatistics>();
}

FloatingPointStatistics*
FloatingPointStatisticsCollector::floatingPointStats() {
  return static_cast<FloatingPointStatistics*>(stats_.get());
}

template <typename T>
void FloatingPointStatisticsCollector::addValues(std::span<T> values) {
  static_assert(std::is_floating_point_v<T>);
  addLogicalSize(values.size() * sizeof(T));

  if (values.empty()) {
    return;
  }

  auto* stats = floatingPointStats();

  if (hll_) {
    for (const auto& value : values) {
      insertHllHash(hllHash(value));
    }
  }

  const auto minMax = findMinMax(values);
  if (!minMax.has_value()) {
    addFloatingPointValuesScalar(stats->min_, stats->max_, values);
    return;
  }
  const auto min = static_cast<double>(minMax->min);
  const auto max = static_cast<double>(minMax->max);
  if (!stats->min_.has_value() || min < *stats->min_) {
    stats->min_ = min;
  }
  if (!stats->max_.has_value() || max > *stats->max_) {
    stats->max_ = max;
  }
}

void FloatingPointStatisticsCollector::reset() {
  stats_ = std::make_unique<FloatingPointStatistics>();
  resetHll();
}

void FloatingPointStatisticsCollector::merge(const StatisticsCollector& other) {
  auto* otherStats = other.getStatsView();
  NIMBLE_CHECK(
      otherStats->getType() == StatType::FLOATING_POINT ||
          otherStats->getType() == StatType::DEFAULT,
      "Merging stats with mismatched types.");
  StatisticsCollector::merge(other);
  if (otherStats->getType() == StatType::FLOATING_POINT) {
    const auto* otherFloatStats = otherStats->as<FloatingPointStatistics>();
    auto* stats = floatingPointStats();
    if (otherFloatStats->getMin().has_value()) {
      if (!stats->min_.has_value() ||
          *otherFloatStats->getMin() < *stats->min_) {
        stats->min_ = *otherFloatStats->getMin();
      }
    }
    if (otherFloatStats->getMax().has_value()) {
      if (!stats->max_.has_value() ||
          *otherFloatStats->getMax() > *stats->max_) {
        stats->max_ = *otherFloatStats->getMax();
      }
    }
  }
}

// Explicit template instantiations for
// FloatingPointStatisticsCollector::addValues
template void FloatingPointStatisticsCollector::addValues<float>(
    std::span<float>);
template void FloatingPointStatisticsCollector::addValues<double>(
    std::span<double>);

StringStatisticsCollector::StringStatisticsCollector() {
  stats_ = std::make_unique<StringStatistics>();
}

StringStatistics* StringStatisticsCollector::stringStats() {
  return static_cast<StringStatistics*>(stats_.get());
}

void StringStatisticsCollector::addValues(std::span<std::string_view> values) {
  if (values.empty()) {
    return;
  }
  // findMinMaxAndTotalSize returns views into the writer's value stream, which
  // is recycled per stripe. Assign rather than construct so a monotonic column
  // reuses the capacity it already has.
  const auto minMax = findMinMaxAndTotalSize(values);

  // Adding once per batch is equivalent: every override is additive.
  addLogicalSize(minMax.totalSize);

  auto* stats = stringStats();
  if (!stats->min_.has_value()) {
    stats->min_.emplace(minMax.min);
  } else if (minMax.min < *stats->min_) {
    stats->min_->assign(minMax.min);
  }
  if (!stats->max_.has_value()) {
    stats->max_.emplace(minMax.max);
  } else if (minMax.max > *stats->max_) {
    stats->max_->assign(minMax.max);
  }

  if (hll_) {
    for (const auto& value : values) {
      insertHllHash(XXH64(value.data(), value.size(), 0));
    }
  }
}

void StringStatisticsCollector::reset() {
  stats_ = std::make_unique<StringStatistics>();
  resetHll();
}

void StringStatisticsCollector::merge(const StatisticsCollector& other) {
  auto* otherStats = other.getStatsView();
  NIMBLE_CHECK(
      otherStats->getType() == StatType::STRING ||
          otherStats->getType() == StatType::DEFAULT,
      "Merging stats with mismatched types.");
  StatisticsCollector::merge(other);
  if (otherStats->getType() == StatType::STRING) {
    const auto* otherStringStats = otherStats->as<StringStatistics>();
    auto* stats = stringStats();
    if (otherStringStats->getMin().has_value()) {
      if (!stats->min_.has_value() ||
          *otherStringStats->getMin() < *stats->min_) {
        stats->min_ = *otherStringStats->getMin();
      }
    }
    if (otherStringStats->getMax().has_value()) {
      if (!stats->max_.has_value() ||
          *otherStringStats->getMax() > *stats->max_) {
        stats->max_ = *otherStringStats->getMax();
      }
    }
  }
}

DeduplicatedStatisticsCollector::DeduplicatedStatisticsCollector(
    std::unique_ptr<StatisticsCollector> baseCollector)
    : baseCollector_{std::move(baseCollector)} {
  // Initialize stats_ with the base collector's stats view and default dedup
  // values
  stats_ = std::make_unique<DeduplicatedColumnStatistics>(
      baseCollector_->getStatsView(), 0, 0);
  dedupedStats_ = stats_->as<DeduplicatedColumnStatistics>();
}

/* static */ std::unique_ptr<DeduplicatedStatisticsCollector>
DeduplicatedStatisticsCollector::wrap(
    std::unique_ptr<StatisticsCollector> baseCollector) {
  return std::make_unique<DeduplicatedStatisticsCollector>(
      std::move(baseCollector));
}

ColumnStatistics* DeduplicatedStatisticsCollector::getStatsView() {
  NIMBLE_DCHECK_NOT_NULL(baseCollector_);
  // Brings the wrapped collector's NDV sketch up to date.
  baseCollector_->getStatsView();
  // Return this as DeduplicatedColumnStatistics to resolve diamond ambiguity.
  return stats_.get();
}

const ColumnStatistics* DeduplicatedStatisticsCollector::getStatsView() const {
  NIMBLE_DCHECK_NOT_NULL(baseCollector_);
  std::as_const(*baseCollector_).getStatsView();
  return stats_.get();
}

StatisticsCollector* DeduplicatedStatisticsCollector::getBaseCollector() const {
  NIMBLE_DCHECK_NOT_NULL(baseCollector_);
  return baseCollector_.get();
}

std::unique_ptr<StatisticsCollector>
DeduplicatedStatisticsCollector::releaseBaseCollector() {
  auto released = std::move(baseCollector_);
  dedupedStats_ = nullptr;
  stats_.reset();
  return released;
}

void DeduplicatedStatisticsCollector::recordDeduplicatedStats(
    uint64_t count,
    uint64_t deduplicatedSize) {
  NIMBLE_DCHECK_NOT_NULL(baseCollector_);
  NIMBLE_DCHECK_NOT_NULL(dedupedStats_);
  dedupedStats_->dedupedCount_ += count;
  dedupedStats_->dedupedLogicalSize_ += deduplicatedSize;
}

void DeduplicatedStatisticsCollector::addCounts(
    uint64_t totalCount,
    uint64_t nullCount) {
  NIMBLE_DCHECK_NOT_NULL(baseCollector_);
  baseCollector_->addCounts(totalCount, nullCount);
}

void DeduplicatedStatisticsCollector::addLogicalSize(uint64_t logicalSize) {
  NIMBLE_DCHECK_NOT_NULL(baseCollector_);
  baseCollector_->addLogicalSize(logicalSize);
}

void DeduplicatedStatisticsCollector::addPhysicalSize(uint64_t physicalSize) {
  NIMBLE_DCHECK_NOT_NULL(baseCollector_);
  baseCollector_->addPhysicalSize(physicalSize);
}

void DeduplicatedStatisticsCollector::merge(const StatisticsCollector& other) {
  auto* otherStats = other.getStatsView();
  NIMBLE_CHECK(
      otherStats->getType() == StatType::DEDUPLICATED,
      "Merging stats with mismatched types.");
  const auto* otherDedupStats = otherStats->as<DeduplicatedColumnStatistics>();
  dedupedStats_->dedupedCount_ += otherDedupStats->getDedupedCount();
  dedupedStats_->dedupedLogicalSize_ +=
      otherDedupStats->getDedupedLogicalSize();
  const auto& otherDedup =
      dynamic_cast<const DeduplicatedStatisticsCollector&>(other);
  baseCollector_->merge(*otherDedup.baseCollector_);
}

void DeduplicatedStatisticsCollector::reset() {
  baseCollector_->reset();
  stats_ = std::make_unique<DeduplicatedColumnStatistics>(
      baseCollector_->getStatsView(), 0, 0);
  dedupedStats_ = stats_->as<DeduplicatedColumnStatistics>();
}

SharedStatisticsCollector::SharedStatisticsCollector(
    std::unique_ptr<StatisticsCollector> baseCollector)
    : baseCollector_{std::move(baseCollector)} {
  // Don't initialize stats_ since we delegate to baseCollector_
  stats_ = nullptr;
}

/* static */ std::unique_ptr<SharedStatisticsCollector>
SharedStatisticsCollector::wrap(
    std::unique_ptr<StatisticsCollector> baseCollector) {
  return std::make_unique<SharedStatisticsCollector>(std::move(baseCollector));
}

ColumnStatistics* SharedStatisticsCollector::getStatsView() {
  std::lock_guard<std::mutex> l{mutex_};
  return baseCollector_->getStatsView();
}

const ColumnStatistics* SharedStatisticsCollector::getStatsView() const {
  std::lock_guard<std::mutex> l{mutex_};
  return baseCollector_->getStatsView();
}

ColumnStatistics* SharedStatisticsCollector::getBaseStatistics() const {
  return baseCollector_->getStatsView();
}

uint64_t SharedStatisticsCollector::getValueCount() const {
  std::lock_guard<std::mutex> l{mutex_};
  return baseCollector_->getValueCount();
}

uint64_t SharedStatisticsCollector::getNullCount() const {
  std::lock_guard<std::mutex> l{mutex_};
  return baseCollector_->getNullCount();
}

uint64_t SharedStatisticsCollector::getLogicalSize() const {
  std::lock_guard<std::mutex> l{mutex_};
  return baseCollector_->getLogicalSize();
}

uint64_t SharedStatisticsCollector::getPhysicalSize() const {
  std::lock_guard<std::mutex> l{mutex_};
  return baseCollector_->getPhysicalSize();
}

StatType SharedStatisticsCollector::getType() const {
  std::lock_guard<std::mutex> l{mutex_};
  return baseCollector_->getType();
}

void SharedStatisticsCollector::addCounts(
    uint64_t totalCount,
    uint64_t nullCount) {
  std::lock_guard<std::mutex> l{mutex_};
  baseCollector_->addCounts(totalCount, nullCount);
}

void SharedStatisticsCollector::addLogicalSize(uint64_t logicalSize) {
  std::lock_guard<std::mutex> l{mutex_};
  baseCollector_->addLogicalSize(logicalSize);
}

void SharedStatisticsCollector::addPhysicalSize(uint64_t physicalSize) {
  std::lock_guard<std::mutex> l{mutex_};
  baseCollector_->addPhysicalSize(physicalSize);
}

void SharedStatisticsCollector::updateBaseCollector(
    std::function<void(StatisticsCollector*)> updateFunc) {
  std::lock_guard<std::mutex> l{mutex_};
  updateFunc(baseCollector_.get());
}

void SharedStatisticsCollector::merge(const StatisticsCollector& other) {
  std::lock_guard<std::mutex> l{mutex_};
  baseCollector_->merge(other);
}

void SharedStatisticsCollector::reset() {
  std::lock_guard<std::mutex> l{mutex_};
  baseCollector_->reset();
}

} // namespace facebook::nimble
