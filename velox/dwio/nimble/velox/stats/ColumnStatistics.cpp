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

#include <optional>
#include <utility>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/velox/stats/ColumnStatsUtils.h"

namespace facebook::nimble {

namespace {

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
  return std::make_unique<ColumnStatistics>(
      getValueCount(), getNullCount(), getLogicalSize(), getPhysicalSize());
}

std::unique_ptr<velox::dwio::common::ColumnStatistics>
ColumnStatistics::toCommonStatistics() const {
  const auto valueCount = std::make_optional<uint64_t>(getValueCount());
  const auto hasNull = std::make_optional<bool>(getNullCount() > 0);
  const auto rawSize = std::make_optional<uint64_t>(getLogicalSize());
  const auto size = std::make_optional<uint64_t>(getPhysicalSize());

  switch (getType()) {
    case StatType::INTEGRAL: {
      const auto* intStats = as<const IntegralStatistics>();
      NIMBLE_DCHECK(
          intStats != nullptr, "Failed to cast to IntegralStatistics");
      return std::make_unique<velox::dwio::common::IntegerColumnStatistics>(
          valueCount,
          hasNull,
          rawSize,
          size,
          intStats->getMin(),
          intStats->getMax(),
          std::nullopt);
    }
    case StatType::FLOATING_POINT: {
      const auto* fpStats = as<const FloatingPointStatistics>();
      NIMBLE_DCHECK(
          fpStats != nullptr, "Failed to cast to FloatingPointStatistics");
      return std::make_unique<velox::dwio::common::DoubleColumnStatistics>(
          valueCount,
          hasNull,
          rawSize,
          size,
          fpStats->getMin(),
          fpStats->getMax(),
          std::nullopt);
    }
    case StatType::STRING: {
      const auto* strStats = as<const StringStatistics>();
      NIMBLE_DCHECK(strStats != nullptr, "Failed to cast to StringStatistics");
      return std::make_unique<velox::dwio::common::StringColumnStatistics>(
          valueCount,
          hasNull,
          rawSize,
          size,
          strStats->getMin(),
          strStats->getMax(),
          std::nullopt);
    }
    case StatType::DEFAULT:
    case StatType::DEDUPLICATED:
      break;
  }

  return std::make_unique<velox::dwio::common::ColumnStatistics>(
      valueCount, hasNull, rawSize, size);
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
  return std::make_unique<StringStatistics>(
      getValueCount(),
      getNullCount(),
      getLogicalSize(),
      getPhysicalSize(),
      getMin(),
      getMax());
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
  return std::make_unique<IntegralStatistics>(
      getValueCount(),
      getNullCount(),
      getLogicalSize(),
      getPhysicalSize(),
      getMin(),
      getMax());
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
  return std::make_unique<FloatingPointStatistics>(
      getValueCount(),
      getNullCount(),
      getLogicalSize(),
      getPhysicalSize(),
      getMin(),
      getMax());
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
  return stats_.get();
}

const ColumnStatistics* StatisticsCollector::getStatsView() const {
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
}

void StatisticsCollector::reset() {
  stats_ = std::make_unique<ColumnStatistics>();
}

/* static */ std::unique_ptr<StatisticsCollector> StatisticsCollector::create(
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& type) {
  switch (type->type()->kind()) {
    case velox::TypeKind::TINYINT:
    case velox::TypeKind::SMALLINT:
    case velox::TypeKind::INTEGER:
    case velox::TypeKind::BIGINT:
      return std::make_unique<IntegralStatisticsCollector>();
    case velox::TypeKind::REAL:
    case velox::TypeKind::DOUBLE:
      return std::make_unique<FloatingPointStatisticsCollector>();
    case velox::TypeKind::VARCHAR:
    case velox::TypeKind::VARBINARY:
      return std::make_unique<StringStatisticsCollector>();
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
}

void IntegralStatisticsCollector::reset() {
  stats_ = std::make_unique<IntegralStatistics>();
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
  if (!values.empty()) {
    auto* stats = stringStats();
    if (UNLIKELY(!stats->min_.has_value())) {
      stats->min_ = std::string(values.front());
    }
    if (UNLIKELY(!stats->max_.has_value())) {
      stats->max_ = std::string(values.front());
    }

    for (const auto& value : values) {
      addLogicalSize(value.size());
      stats->min_ = stats->min_ > value ? std::make_optional(std::string(value))
                                        : stats->min_;
      stats->max_ = stats->max_ < value ? std::make_optional(std::string(value))
                                        : stats->max_;
    }
  }
}

void StringStatisticsCollector::reset() {
  stats_ = std::make_unique<StringStatistics>();
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
  // Return this as DeduplicatedColumnStatistics to resolve diamond ambiguity.
  return stats_.get();
}

const ColumnStatistics* DeduplicatedStatisticsCollector::getStatsView() const {
  NIMBLE_DCHECK_NOT_NULL(baseCollector_);
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
