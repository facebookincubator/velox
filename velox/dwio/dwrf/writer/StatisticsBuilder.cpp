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

#include "velox/dwio/dwrf/writer/StatisticsBuilder.h"

#include "velox/dwio/common/Arena.h"

namespace facebook::velox::dwrf {

using dwio::common::ArenaCreate;

namespace {

bool isValidLength(const std::optional<uint64_t>& length) {
  return length.has_value() &&
      length.value() <= std::numeric_limits<int64_t>::max();
}

// Serializes base ColumnStatistics fields to proto.
void baseToProto(
    const dwio::common::ColumnStatistics& builder,
    ColumnStatisticsWriteWrapper& stats) {
  if (builder.hasNull().has_value()) {
    stats.setHasNull(builder.hasNull().value());
  }
  if (builder.getNumberOfValues().has_value()) {
    stats.setNumberOfValues(builder.getNumberOfValues().value());
  }
  if (builder.getRawSize().has_value()) {
    stats.setRawSize(builder.getRawSize().value());
  }
  if (builder.getSize().has_value()) {
    stats.setSize(builder.getSize().value());
  }
}

} // namespace

void StatisticsBuilder::toProto(ColumnStatisticsWriteWrapper& stats) const {
  baseToProto(*this, stats);
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
}

void BooleanStatisticsBuilder::toProto(
    ColumnStatisticsWriteWrapper& stats) const {
  baseToProto(*this, stats);
  if (!isAllNull() && trueCount_.has_value()) {
    auto bStats = stats.mutableBucketStatistics();
    DWIO_ENSURE_EQ(bStats.countSize(), 0);
    bStats.addCount(trueCount_.value());
  }
}

void IntegerStatisticsBuilder::toProto(
    ColumnStatisticsWriteWrapper& stats) const {
  baseToProto(*this, stats);
  if (!isAllNull() &&
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

void DoubleStatisticsBuilder::toProto(
    ColumnStatisticsWriteWrapper& stats) const {
  baseToProto(*this, stats);
  if (!isAllNull() &&
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

void StringStatisticsBuilder::toProto(
    ColumnStatisticsWriteWrapper& stats) const {
  baseToProto(*this, stats);
  if (!isAllNull() &&
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

void BinaryStatisticsBuilder::toProto(
    ColumnStatisticsWriteWrapper& stats) const {
  baseToProto(*this, stats);
  if (!isAllNull() && isValidLength(length_)) {
    auto bStats = stats.mutableBinaryStatistics();
    bStats.setSum(length_.value());
  }
}

dwio::common::KeyInfo MapStatisticsBuilder::constructKey(
    const dwrf::proto::KeyInfo& keyInfo) {
  if (keyInfo.has_intkey()) {
    return dwio::common::KeyInfo{keyInfo.intkey()};
  } else if (keyInfo.has_byteskey()) {
    return dwio::common::KeyInfo{keyInfo.byteskey()};
  }
  VELOX_UNREACHABLE("Illegal null key info");
}

void MapStatisticsBuilder::addValues(
    const dwrf::proto::KeyInfo& keyInfo,
    const StatisticsBuilder& stats) {
  auto& keyStats = getKeyStats(MapStatisticsBuilder::constructKey(keyInfo));
  keyStats.merge(stats, /*ignoreSize=*/true);
}

void MapStatisticsBuilder::incrementSize(
    const dwrf::proto::KeyInfo& keyInfo,
    uint64_t size) {
  auto& keyStats = getKeyStats(MapStatisticsBuilder::constructKey(keyInfo));
  keyStats.ensureSize();
  keyStats.incrementSize(size);
}

void MapStatisticsBuilder::merge(
    const dwio::common::ColumnStatistics& other,
    bool ignoreSize) {
  StatisticsBuilder::merge(other, ignoreSize);
  auto stats = dynamic_cast<const dwio::common::MapColumnStatistics*>(&other);
  if (!stats) {
    if (!other.isAllNull() && !entryStatistics_.empty()) {
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
  if (!isAllNull() && !entryStatistics_.empty()) {
    auto mapStats = stats.mutableMapStatistics();
    for (const auto& entry : entryStatistics_) {
      auto entryStatistics = mapStats->add_stats();
      const auto& key = entry.first;
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
