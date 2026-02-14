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

#include "velox/dwio/dwrf/writer/StatisticsBuilder.h"

#include "velox/dwio/common/Arena.h"

namespace facebook::velox::dwrf {

using dwio::common::ArenaCreate;

namespace {

static bool isValidLength(const std::optional<uint64_t>& length) {
  return length.has_value() &&
      length.value() <= std::numeric_limits<int64_t>::max();
}

template <typename T>
static void mergeCount(std::optional<T>& to, const std::optional<T>& from) {
  if (to.has_value()) {
    if (from.has_value()) {
      to.value() += from.value();
    } else {
      to.reset();
    }
  }
}

template <typename T>
static void mergeMin(std::optional<T>& to, const std::optional<T>& from) {
  if (to.has_value()) {
    if (!from.has_value()) {
      to.reset();
    } else if (from.value() < to.value()) {
      to = from;
    }
  }
}

template <typename T>
static void mergeMax(std::optional<T>& to, const std::optional<T>& from) {
  if (to.has_value()) {
    if (!from.has_value()) {
      to.reset();
    } else if (from.value() > to.value()) {
      to = from;
    }
  }
}

} // namespace

void StatisticsBuilder::merge(
    const dwio::common::ColumnStatistics& other,
    bool ignoreSize) {
  // Merge valueCount_ only if both sides have it. Otherwise, reset.
  mergeCount(valueCount_, other.getNumberOfValues());

  // Merge hasNull_. Follow below rule:
  // self / other => result
  // true / any => true
  // unknown / true => true
  // unknown / unknown or false => unknown
  // false / unknown => unknown
  // false / false => false
  // false / true => true
  if (!hasNull_.has_value() || !hasNull_.value()) {
    auto otherHasNull = other.hasNull();
    if (otherHasNull.has_value()) {
      if (otherHasNull.value()) {
        // other is true, set to true
        hasNull_ = true;
      }
      // when other is false, no change is needed
    } else if (hasNull_.has_value()) {
      // self value is false and other is unknown, set to unknown
      hasNull_.reset();
    }
  }
  // Merge rawSize_ the way similar to valueCount_
  mergeCount(rawSize_, other.getRawSize());
  if (!ignoreSize) {
    // Merge size
    mergeCount(size_, other.getSize());
  }
  if (hll_) {
    auto* otherBuilder = dynamic_cast<const StatisticsBuilder*>(&other);
    VELOX_CHECK_NOT_NULL(otherBuilder);
    VELOX_CHECK_NOT_NULL(otherBuilder->hll_);
    hll_->mergeWith(*otherBuilder->hll_);
  }
}

void StatisticsBuilder::toProto(ColumnStatisticsWriteWrapper& stats) const {
  if (hasNull_.has_value()) {
    stats.setHasNull(hasNull_.value());
  }
  if (valueCount_.has_value()) {
    stats.setNumberOfValues(valueCount_.value());
  }
  if (rawSize_.has_value()) {
    stats.setRawSize(rawSize_.value());
  }
  if (size_.has_value()) {
    stats.setSize(size_.value());
  }
}

std::unique_ptr<dwio::common::ColumnStatistics> StatisticsBuilder::build()
    const {
  auto columnStatistics = ArenaCreate<proto::ColumnStatistics>(arena_.get());
  auto stats = ColumnStatisticsWriteWrapper(columnStatistics);
  toProto(stats);

  StatsContext context{WriterVersion_CURRENT};
  auto result = buildColumnStatisticsFromProto(
      ColumnStatisticsWrapper(columnStatistics), context);
  // We do not alter the proto since this is part of the file format
  // and the file format. The distinct count does not exist in the
  // file format but is added here for use in on demand sampling.
  if (hll_) {
    result->setNumDistinct(hll_->cardinality());
  }
  return result;
}

std::unique_ptr<StatisticsBuilder> StatisticsBuilder::create(
    const Type& type,
    const StatisticsBuilderOptions& options) {
  switch (type.kind()) {
    case TypeKind::BOOLEAN:
      return std::make_unique<BooleanStatisticsBuilder>(options);
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
      return std::make_unique<IntegerStatisticsBuilder>(options);
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
      return std::make_unique<DoubleStatisticsBuilder>(options);
    case TypeKind::VARCHAR:
      return std::make_unique<StringStatisticsBuilder>(options);
    case TypeKind::VARBINARY:
      return std::make_unique<BinaryStatisticsBuilder>(options);
    case TypeKind::MAP:
      // For now we only capture map stats for flatmaps, which are
      // top level maps only.
      // However, we don't need to create a different builder type here
      // because the serialized stats will fall back to default type if we don't
      // call the map specific update methods.
      return std::make_unique<MapStatisticsBuilder>(type, options);
    default:
      return std::make_unique<StatisticsBuilder>(options);
  }
}

void StatisticsBuilder::createTree(
    std::vector<std::unique_ptr<StatisticsBuilder>>& statBuilders,
    const Type& type,
    const StatisticsBuilderOptions& options) {
  auto kind = type.kind();
  switch (kind) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
      statBuilders.push_back(StatisticsBuilder::create(type, options));
      break;

    case TypeKind::ARRAY: {
      statBuilders.push_back(StatisticsBuilder::create(type, options));
      const auto& arrayType = dynamic_cast<const ArrayType&>(type);
      createTree(statBuilders, *arrayType.elementType(), options);
      break;
    }

    case TypeKind::MAP: {
      statBuilders.push_back(StatisticsBuilder::create(type, options));
      const auto& mapType = dynamic_cast<const MapType&>(type);
      createTree(statBuilders, *mapType.keyType(), options);
      createTree(statBuilders, *mapType.valueType(), options);
      break;
    }

    case TypeKind::ROW: {
      statBuilders.push_back(StatisticsBuilder::create(type, options));
      const auto& rowType = dynamic_cast<const RowType&>(type);
      for (const auto& childType : rowType.children()) {
        createTree(statBuilders, *childType, options);
      }
      break;
    }
    default:
      DWIO_RAISE("Not supported type: ", kind);
      break;
  }
  return;
};

void BooleanStatisticsBuilder::merge(
    const dwio::common::ColumnStatistics& other,
    bool ignoreSize) {
  StatisticsBuilder::merge(other, ignoreSize);
  auto stats =
      dynamic_cast<const dwio::common::BooleanColumnStatistics*>(&other);
  if (!stats) {
    // We only care about the case when type specific stats is missing yet
    // it has non-null values.
    if (!isEmpty(other) && trueCount_.has_value()) {
      trueCount_.reset();
    }
    return;
  }

  // Now the case when both sides have type specific stats
  mergeCount(trueCount_, stats->getTrueCount());
}

void BooleanStatisticsBuilder::toProto(
    ColumnStatisticsWriteWrapper& stats) const {
  StatisticsBuilder::toProto(stats);
  // Serialize type specific stats only if there is non-null values
  if (!isEmpty(*this) && trueCount_.has_value()) {
    auto bStats = stats.mutableBucketStatistics();
    DWIO_ENSURE_EQ(bStats.countSize(), 0);
    bStats.addCount(trueCount_.value());
  }
}

void IntegerStatisticsBuilder::merge(
    const dwio::common::ColumnStatistics& other,
    bool ignoreSize) {
  StatisticsBuilder::merge(other, ignoreSize);
  auto stats =
      dynamic_cast<const dwio::common::IntegerColumnStatistics*>(&other);
  if (!stats) {
    // We only care about the case when type specific stats is missing yet
    // it has non-null values.
    if (!isEmpty(other)) {
      min_.reset();
      max_.reset();
      sum_.reset();
    }
    return;
  }

  // Now the case when both sides have type specific stats
  mergeMin(min_, stats->getMinimum());
  mergeMax(max_, stats->getMaximum());
  mergeWithOverflowCheck(sum_, stats->getSum());
}

void IntegerStatisticsBuilder::toProto(
    ColumnStatisticsWriteWrapper& stats) const {
  StatisticsBuilder::toProto(stats);
  // Serialize type specific stats only if there is non-null values
  if (!isEmpty(*this) &&
      (min_.has_value() || max_.has_value() || sum_.has_value())) {
    auto iStats = stats.mutableIntegerStatistics();
    if (min_.has_value()) {
      iStats.setMinimum(min_.value());
    }
    if (max_.has_value()) {
      iStats.setMaximum(max_.value());
    }
    if (sum_.has_value()) {
      iStats.setSum(sum_.value());
    }
  }
}

void DoubleStatisticsBuilder::merge(
    const dwio::common::ColumnStatistics& other,
    bool ignoreSize) {
  StatisticsBuilder::merge(other, ignoreSize);
  auto stats =
      dynamic_cast<const dwio::common::DoubleColumnStatistics*>(&other);
  if (!stats) {
    // We only care about the case when type specific stats is missing yet
    // it has non-null values.
    if (!isEmpty(other)) {
      clear();
    }
    return;
  }

  // Now the case when both sides have type specific stats
  mergeMin(min_, stats->getMinimum());
  mergeMax(max_, stats->getMaximum());
  mergeCount(sum_, stats->getSum());
  if (sum_.has_value() && std::isnan(sum_.value())) {
    sum_.reset();
  }
}

void DoubleStatisticsBuilder::toProto(
    ColumnStatisticsWriteWrapper& stats) const {
  StatisticsBuilder::toProto(stats);
  // Serialize type specific stats only if there is non-null values
  if (!isEmpty(*this) &&
      (min_.has_value() || max_.has_value() || sum_.has_value())) {
    auto dStats = stats.mutableDoubleStatistics();
    if (min_.has_value()) {
      dStats.setMinimum(min_.value());
    }
    if (max_.has_value()) {
      dStats.setMaximum(max_.value());
    }
    if (sum_.has_value()) {
      dStats.setSum(sum_.value());
    }
  }
}

void StringStatisticsBuilder::merge(
    const dwio::common::ColumnStatistics& other,
    bool ignoreSize) {
  // min_/max_ is not initialized with default that can be compared against
  // easily. So we need to capture whether self is empty and handle
  // differently.
  auto isSelfEmpty = isEmpty(*this);
  StatisticsBuilder::merge(other, ignoreSize);
  auto stats =
      dynamic_cast<const dwio::common::StringColumnStatistics*>(&other);
  if (!stats) {
    // We only care about the case when type specific stats is missing yet
    // it has non-null values.
    if (!isEmpty(other)) {
      min_.reset();
      max_.reset();
      length_.reset();
    }
    return;
  }

  // If the other stats is empty, there is nothing to merge at string stats
  // level.
  if (isEmpty(other)) {
    return;
  }

  if (isSelfEmpty) {
    min_ = stats->getMinimum();
    max_ = stats->getMaximum();
  } else {
    mergeMin(min_, stats->getMinimum());
    mergeMax(max_, stats->getMaximum());
  }

  mergeWithOverflowCheck(length_, stats->getTotalLength());
}

void StringStatisticsBuilder::toProto(
    ColumnStatisticsWriteWrapper& stats) const {
  StatisticsBuilder::toProto(stats);
  // If string value is too long, drop it and fall back to basic stats
  if (!isEmpty(*this) &&
      (shouldKeep(min_) || shouldKeep(max_) || isValidLength(length_))) {
    auto dStats = stats.mutableStringStatistics();
    if (isValidLength(length_)) {
      dStats.setSum(length_.value());
    }

    if (shouldKeep(min_)) {
      dStats.setMinimum(min_.value());
    }

    if (shouldKeep(max_)) {
      dStats.setMaximum(max_.value());
    }
  }
}

void BinaryStatisticsBuilder::merge(
    const dwio::common::ColumnStatistics& other,
    bool ignoreSize) {
  StatisticsBuilder::merge(other, ignoreSize);
  auto stats =
      dynamic_cast<const dwio::common::BinaryColumnStatistics*>(&other);
  if (!stats) {
    // We only care about the case when type specific stats is missing yet
    // it has non-null values.
    if (!isEmpty(other) && length_.has_value()) {
      length_.reset();
    }
    return;
  }

  mergeWithOverflowCheck(length_, stats->getTotalLength());
}

void BinaryStatisticsBuilder::toProto(
    ColumnStatisticsWriteWrapper& stats) const {
  StatisticsBuilder::toProto(stats);
  // Serialize type specific stats only if there is non-null values
  if (!isEmpty(*this) && isValidLength(length_)) {
    auto bStats = stats.mutableBinaryStatistics();
    bStats.setSum(length_.value());
  }
}

void MapStatisticsBuilder::merge(
    const dwio::common::ColumnStatistics& other,
    bool ignoreSize) {
  StatisticsBuilder::merge(other, ignoreSize);
  auto stats = dynamic_cast<const dwio::common::MapColumnStatistics*>(&other);
  if (!stats) {
    // We only care about the case when type specific stats is missing yet
    // it has non-null values.
    if (!isEmpty(other) && !entryStatistics_.empty()) {
      entryStatistics_.clear();
    }
    return;
  }

  for (const auto& entry : stats->getEntryStatistics()) {
    getKeyStats(entry.first).merge(*entry.second, ignoreSize);
  }
}

void MapStatisticsBuilder::toProto(ColumnStatisticsWriteWrapper& stats) const {
  StatisticsBuilder::toProto(stats);
  if (!isEmpty(*this) && !entryStatistics_.empty()) {
    auto mapStats = stats.mutableMapStatistics();
    for (const auto& entry : entryStatistics_) {
      auto entryStatistics = mapStats->add_stats();
      const auto& key = entry.first;
      // Sets the corresponding key. Leave null keys null.
      if (key.intKey.has_value()) {
        entryStatistics->mutable_key()->set_intkey(key.intKey.value());
      } else if (key.bytesKey.has_value()) {
        entryStatistics->mutable_key()->set_byteskey(key.bytesKey.value());
      }
      auto c = ColumnStatisticsWriteWrapper(entryStatistics->mutable_stats());
      dynamic_cast<const StatisticsBuilder&>(*entry.second).toProto(c);
    }
  }
}
} // namespace facebook::velox::dwrf
