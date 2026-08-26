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
#include "velox/dwio/nimble/velox/stats/VectorizedStatistics.h"

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

namespace facebook::nimble {

namespace {
constexpr static uint16_t kVectorizedStatsVersion = 0;
}

template <typename T>
void TypedVectorizedStatistic<T>::append(std::optional<T> value) {
  if (value.has_value()) {
    isValid_.emplace_back(true);
    values_.emplace_back(value.value());
  } else {
    isValid_.emplace_back(false);
  }
}

// Template specialization for string_view to properly store string data
// The base template would create dangling string_views when passed temporary
// strings, so we need to copy the string data to a persistent buffer.
template <>
inline void TypedVectorizedStatistic<std::string_view>::append(
    std::optional<std::string_view> value) {
  if (value.has_value()) {
    isValid_.emplace_back(true);
    // Copy the string data to a velox buffer so it persists
    const auto& sv = value.value();
    auto& buffer = stringBuffers_.emplace_back(
        velox::AlignedBuffer::allocate<char>(sv.size(), isValid_.pool()));
    std::memcpy(buffer->asMutable<char>(), sv.data(), sv.size());
    values_.emplace_back(std::string_view{buffer->as<char>(), sv.size()});
  } else {
    isValid_.emplace_back(false);
  }
}

template <typename T>
std::optional<T> TypedVectorizedStatistic<T>::valueAt(size_t idx) const {
  NIMBLE_CHECK(idx < isValid_.size());
  if (!isValid_[idx]) {
    return std::nullopt;
  }
  // Compact storage: values_ only contains valid entries.
  // Count how many valid entries exist before this index.
  size_t validIdx = 0;
  for (size_t i = 0; i < idx; ++i) {
    if (isValid_[i]) {
      ++validIdx;
    }
  }
  NIMBLE_CHECK(validIdx < values_.size());
  return values_[validIdx];
}

template <typename T>
std::string_view TypedVectorizedStatistic<T>::serialize(
    const EncodingSelectionPolicyCreator& encodingSelectionPolicyCreator,
    nimble::Buffer& buffer) {
  auto policy = std::unique_ptr<EncodingSelectionPolicy<T>>(
      static_cast<EncodingSelectionPolicy<T>*>(
          encodingSelectionPolicyCreator(TypeTraits<T>::dataType).release()));
  return EncodingFactory::encodeNullable<T>(
      std::move(policy), values_, isValid_, buffer);
}

template <typename T>
void TypedVectorizedStatistic<T>::deserializeFrom(
    std::string_view payload,
    velox::memory::MemoryPool& pool) {
  auto encoding = EncodingFactory().create(
      pool,
      payload,
      [&](uint32_t totalLength) {
        auto& buffer = stringBuffers_.emplace_back(
            velox::AlignedBuffer::allocate<char>(totalLength, &pool));
        return buffer->template asMutable<void>();
      },
      Encoding::Options{});
  NIMBLE_CHECK_NOT_NULL(encoding);
  const auto& rowCount = encoding->rowCount();
  velox::BufferPtr nullsBuffer =
      velox::allocateNulls(rowCount, &pool, velox::bits::kNull);
  auto nulls = nullsBuffer->asMutable<uint64_t>();

  // Create a temporary buffer to hold the materialized values
  std::vector<T> tempValues(rowCount);

  if (encoding->isNullable()) {
    uint32_t nonNullCount = encoding->materializeNullable(
        rowCount, tempValues.data(), [&]() { return nulls; });
    isValid_.resize(rowCount);
    // When all values are valid, materializeNullable doesn't populate the nulls
    // bitmap (it remains all zeros). We need to check the return value to
    // determine if all values are valid.
    if (nonNullCount == rowCount) {
      // All values are valid
      for (size_t i = 0; i < rowCount; ++i) {
        isValid_[i] = true;
      }
    } else {
      // Populate isValid_ from the nulls bitmap
      for (size_t i = 0; i < rowCount; ++i) {
        isValid_[i] = velox::bits::isBitSet(nulls, i);
      }
    }
  } else {
    encoding->materialize(rowCount, tempValues.data());
    // For non-nullable encodings, all values are valid
    isValid_.resize(rowCount, true);
  }

  // Copy valid values to values_ (compact storage)
  for (size_t i = 0; i < rowCount; ++i) {
    if (isValid_[i]) {
      values_.emplace_back(tempValues[i]);
    }
  }
}

// Template specialization for string_view to copy string data
// so it persists after encoding goes out of scope
template <>
inline void TypedVectorizedStatistic<std::string_view>::deserializeFrom(
    std::string_view payload,
    velox::memory::MemoryPool& pool) {
  auto encoding = EncodingFactory().create(
      pool,
      payload,
      [&](uint32_t totalLength) {
        auto& buffer = stringBuffers_.emplace_back(
            velox::AlignedBuffer::allocate<char>(totalLength, &pool));
        return buffer->template asMutable<void>();
      },
      Encoding::Options{});
  NIMBLE_CHECK_NOT_NULL(encoding);
  const auto rowCount = encoding->rowCount();
  velox::BufferPtr nullsBuffer =
      velox::allocateNulls(rowCount, &pool, velox::bits::kNull);
  auto nulls = nullsBuffer->asMutable<uint64_t>();

  // Create a temporary vector to hold the materialized string_views
  std::vector<std::string_view> tempValues(rowCount);

  if (encoding->isNullable()) {
    uint32_t nonNullCount = encoding->materializeNullable(
        rowCount, tempValues.data(), [&]() { return nulls; });
    isValid_.resize(rowCount);
    // When all values are valid, materializeNullable doesn't populate the nulls
    // bitmap. Check return value.
    if (nonNullCount == rowCount) {
      for (size_t i = 0; i < rowCount; ++i) {
        isValid_[i] = true;
      }
    } else {
      for (size_t i = 0; i < rowCount; ++i) {
        isValid_[i] = velox::bits::isBitSet(nulls, i);
      }
    }
  } else {
    encoding->materialize(rowCount, tempValues.data());
    isValid_.resize(rowCount, true);
  }

  // Copy valid string data to persistent buffers (compact storage)
  for (size_t i = 0; i < rowCount; ++i) {
    if (isValid_[i]) {
      // Copy string data to a persistent buffer
      const auto& sv = tempValues[i];
      auto& buffer = stringBuffers_.emplace_back(
          velox::AlignedBuffer::allocate<char>(sv.size(), &pool));
      std::memcpy(buffer->asMutable<char>(), sv.data(), sv.size());
      values_.emplace_back(std::string_view{buffer->as<char>(), sv.size()});
    }
  }
}

/* static */ std::unique_ptr<VectorizedStatistic> VectorizedStatistic::create(
    StatStreamType streamType,
    velox::memory::MemoryPool* pool) {
  std::unique_ptr<VectorizedStatistic> stat;
  switch (streamType) {
    case StatStreamType::VALUE_COUNT:
    case StatStreamType::NULL_COUNT:
    case StatStreamType::LOGICAL_SIZE:
    case StatStreamType::PHYSICAL_SIZE:
    case StatStreamType::DEDUPLICATED_VALUE_COUNT:
    case StatStreamType::DEDUPLICATED_LOGICAL_SIZE: {
      stat = std::make_unique<TypedVectorizedStatistic<uint64_t>>(pool);
      break;
    }
    case StatStreamType::INTEGRAL_MIN:
    case StatStreamType::INTEGRAL_MAX: {
      stat = std::make_unique<TypedVectorizedStatistic<int64_t>>(pool);
      break;
    }
    case StatStreamType::FLOATING_POINT_MIN:
    case StatStreamType::FLOATING_POINT_MAX: {
      stat = std::make_unique<TypedVectorizedStatistic<double>>(pool);
      break;
    }
    case StatStreamType::STRING_MIN:
    case StatStreamType::STRING_MAX: {
      stat = std::make_unique<TypedVectorizedStatistic<std::string_view>>(pool);
      break;
    }
    default:
      return nullptr;
  }
  if (stat) {
    stat->type_ = streamType;
  }
  return stat;
}

// Explicit template instantiations for TypedVectorizedStatistic
template class TypedVectorizedStatistic<uint64_t>;
template class TypedVectorizedStatistic<int64_t>;
template class TypedVectorizedStatistic<double>;
template class TypedVectorizedStatistic<std::string_view>;

std::vector<StatStreamType> VectorizedFileStats::collateStatStreamTypes(
    const std::vector<ColumnStatistics*>& columnStats) {
  // All column stats have default stats (valueCount, nullCount, etc.)
  if (!columnStats.empty()) {
    statTypes_.insert(StatType::DEFAULT);
  }
  for (auto& stat : columnStats) {
    auto statType = stat->getType();
    statTypes_.insert(statType);
  }
  return getStatStreamTypes();
}

std::vector<StatStreamType> VectorizedFileStats::getStatStreamTypes() const {
  std::vector<StatStreamType> statStreamTypes;
  for (auto& statType : statTypes_) {
    switch (statType) {
      case StatType::DEFAULT:
        statStreamTypes.push_back(StatStreamType::VALUE_COUNT);
        statStreamTypes.push_back(StatStreamType::NULL_COUNT);
        statStreamTypes.push_back(StatStreamType::LOGICAL_SIZE);
        statStreamTypes.push_back(StatStreamType::PHYSICAL_SIZE);
        break;
      case StatType::STRING:
        statStreamTypes.push_back(StatStreamType::STRING_MIN);
        statStreamTypes.push_back(StatStreamType::STRING_MAX);
        break;
      case StatType::INTEGRAL:
        statStreamTypes.push_back(StatStreamType::INTEGRAL_MIN);
        statStreamTypes.push_back(StatStreamType::INTEGRAL_MAX);
        break;
      case StatType::FLOATING_POINT:
        statStreamTypes.push_back(StatStreamType::FLOATING_POINT_MIN);
        statStreamTypes.push_back(StatStreamType::FLOATING_POINT_MAX);
        break;
      case StatType::DEDUPLICATED:
        statStreamTypes.push_back(StatStreamType::DEDUPLICATED_VALUE_COUNT);
        statStreamTypes.push_back(StatStreamType::DEDUPLICATED_LOGICAL_SIZE);
        break;
      default:
        NIMBLE_UNSUPPORTED(
            fmt::format(
                "Unsupported stat type: {}.", static_cast<uint8_t>(statType)));
    }
  }
  return statStreamTypes;
}

void VectorizedFileStats::createVectorizedStats(
    const std::vector<ColumnStatistics*>& columnStats,
    velox::memory::MemoryPool* pool) {
  auto statStreamTypes = collateStatStreamTypes(columnStats);

  for (auto statStreamType : statStreamTypes) {
    statStreams_.emplace(
        statStreamType, VectorizedStatistic::create(statStreamType, pool));
  }

  for (const auto& columnStat : columnStats) {
    addColumnStat(columnStat);
  }
}

VectorizedFileStats::VectorizedFileStats(
    const std::vector<ColumnStatistics*>& columnStats,
    velox::memory::MemoryPool* pool) {
  createVectorizedStats(columnStats, pool);
}

void VectorizedFileStats::addColumnStat(ColumnStatistics* stat) {
  // All column statistics have the base DEFAULT stats
  statStreams_.at(StatStreamType::VALUE_COUNT)
      ->as<DefaultVectorizedStatistics>()
      ->append(stat->getValueCount());
  statStreams_.at(StatStreamType::NULL_COUNT)
      ->as<DefaultVectorizedStatistics>()
      ->append(stat->getNullCount());
  statStreams_.at(StatStreamType::LOGICAL_SIZE)
      ->as<DefaultVectorizedStatistics>()
      ->append(stat->getLogicalSize());
  statStreams_.at(StatStreamType::PHYSICAL_SIZE)
      ->as<DefaultVectorizedStatistics>()
      ->append(stat->getPhysicalSize());

  // Add type-specific stats
  switch (stat->getType()) {
    case StatType::DEFAULT: {
      break;
    }
    case StatType::STRING: {
      auto stringColumnStat = stat->as<StringStatistics>();
      statStreams_.at(StatStreamType::STRING_MIN)
          ->as<StringVectorizedStatistics>()
          ->append(stringColumnStat->getMin());
      statStreams_.at(StatStreamType::STRING_MAX)
          ->as<StringVectorizedStatistics>()
          ->append(stringColumnStat->getMax());
      break;
    }
    case StatType::INTEGRAL: {
      auto integralColumnStat = stat->as<IntegralStatistics>();
      statStreams_.at(StatStreamType::INTEGRAL_MIN)
          ->as<IntegralVectorizedStatistics>()
          ->append(integralColumnStat->getMin());
      statStreams_.at(StatStreamType::INTEGRAL_MAX)
          ->as<IntegralVectorizedStatistics>()
          ->append(integralColumnStat->getMax());
      break;
    }
    case StatType::FLOATING_POINT: {
      auto floatingPointColumnStat = stat->as<FloatingPointStatistics>();
      statStreams_.at(StatStreamType::FLOATING_POINT_MIN)
          ->as<FloatingPointVectorizedStatistics>()
          ->append(floatingPointColumnStat->getMin());
      statStreams_.at(StatStreamType::FLOATING_POINT_MAX)
          ->as<FloatingPointVectorizedStatistics>()
          ->append(floatingPointColumnStat->getMax());
      break;
    }
    case StatType::DEDUPLICATED: {
      auto deduplicatedColumnStat = stat->as<DeduplicatedColumnStatistics>();
      statStreams_.at(StatStreamType::DEDUPLICATED_VALUE_COUNT)
          ->as<DefaultVectorizedStatistics>()
          ->append(deduplicatedColumnStat->getDedupedCount());
      statStreams_.at(StatStreamType::DEDUPLICATED_LOGICAL_SIZE)
          ->as<DefaultVectorizedStatistics>()
          ->append(deduplicatedColumnStat->getDedupedLogicalSize());

      break;
    }
    default:
      NIMBLE_UNSUPPORTED(
          fmt::format(
              "Unsupported stat type: {}.",
              static_cast<uint8_t>(stat->getType())));
  }
}

// VectorizedFileStats layout
// version: 1 bytes
// stat stream type section:
//   - 1 byte count (uint8_t)
//   - 1 byte bitWidth
//   - packed bits (ceil(count * bitWidth / 8) + 7 bytes slop)
// lengths encoding flag: 1 byte (0 = unencoded)
// stat stream lengths: 8 bytes each, one per stat stream type
// stat streams: independent encodings
std::string_view VectorizedFileStats::serialize(nimble::Buffer& buffer) {
  uint64_t totalLength = 0;
  // add version
  totalLength += sizeof(uint16_t);

  // Calculate bit width needed for stat stream types
  // StatStreamType values are 0-11, so 4 bits is enough
  constexpr int kStatStreamTypeBitWidth = 4;
  const auto statStreamCount = static_cast<uint8_t>(statStreams_.size());

  // stat stream type section layout:
  // 1 byte count + 1 byte bitWidth + FixedBitArray::bufferSize(count, bitWidth)
  // + 1 byte encoding flag + 8 bytes per length
  totalLength += sizeof(uint8_t) + sizeof(uint8_t) +
      FixedBitArray::bufferSize(statStreamCount, kStatStreamTypeBitWidth) +
      sizeof(uint8_t) + statStreams_.size() * sizeof(uint64_t);

  // Serialize each stat stream
  std::vector<std::string_view> encodedStatStreams{};
  for (auto& statStream : statStreams_) {
    auto encoded =
        statStream.second->serialize(encodingSelectionPolicyCreator_, buffer);
    encodedStatStreams.push_back(encoded);
    totalLength += encoded.size();
  }
  auto writeBuf = buffer.reserve(totalLength);
  auto pos = writeBuf;
  encoding::write<uint16_t>(kVectorizedStatsVersion, pos);

  // Write stat stream types header
  encoding::write<uint8_t>(statStreamCount, pos);
  encoding::write<uint8_t>(kStatStreamTypeBitWidth, pos);

  // Write packed stat stream types using FixedBitArray
  auto fbaDataSize =
      FixedBitArray::bufferSize(statStreamCount, kStatStreamTypeBitWidth);
  std::memset(pos, 0, fbaDataSize);
  FixedBitArray fba(pos, kStatStreamTypeBitWidth);
  uint8_t idx = 0;
  for (auto& statStream : statStreams_) {
    fba.set(idx++, static_cast<uint8_t>(statStream.first));
  }
  pos += fbaDataSize;

  // Write the byte for whether the stat stream lengths are fixed bit width
  // encoded. (It's not always worth encoding the lengths).
  encoding::write<uint8_t>(0, pos);

  for (auto& encodedStat : encodedStatStreams) {
    encoding::writeUint64(encodedStat.size(), pos);
  }

  // Stat streams
  for (auto& encodedStat : encodedStatStreams) {
    std::memcpy(pos, encodedStat.data(), encodedStat.size());
    pos += encodedStat.size();
  }
  NIMBLE_CHECK_EQ(pos - writeBuf, totalLength);
  return std::string_view{writeBuf, totalLength};
}

/* static */ std::unique_ptr<VectorizedFileStats>
VectorizedFileStats::deserialize(
    const std::string_view payload,
    velox::memory::MemoryPool& pool) {
  auto pos = payload.data();
  auto totalLength = payload.size();

  auto version = encoding::read<uint16_t>(pos);
  NIMBLE_CHECK_EQ(version, kVectorizedStatsVersion);

  std::vector<StatStreamType> statStreamTypes;
  auto count = encoding::read<uint8_t>(pos);
  auto bitWidth = encoding::read<uint8_t>(pos);
  auto fbaDataSize = FixedBitArray::bufferSize(count, bitWidth);
  FixedBitArray fba(const_cast<char*>(pos), bitWidth);
  statStreamTypes.reserve(count);
  for (uint8_t i = 0; i < count; ++i) {
    statStreamTypes.push_back(static_cast<StatStreamType>(fba.get(i)));
  }
  pos += fbaDataSize;

  const auto statStreamCount = statStreamTypes.size();

  // Read the encoding flag byte (currently always 0 = unencoded)
  auto lengthsEncoded = encoding::read<uint8_t>(pos);
  NIMBLE_CHECK_EQ(lengthsEncoded, 0, "Encoded lengths not yet supported");

  std::vector<uint64_t> streamLengths;
  while (streamLengths.size() < statStreamCount) {
    streamLengths.push_back(encoding::readUint64(pos));
  }
  return std::unique_ptr<VectorizedFileStats>(new VectorizedFileStats(
      statStreamTypes,
      streamLengths,
      std::string_view{pos, totalLength - (pos - payload.data())},
      pool));
}

VectorizedFileStats::VectorizedFileStats(
    const std::vector<StatStreamType>& streamTypes,
    const std::vector<uint64_t>& streamLengths,
    std::string_view statStreams,
    velox::memory::MemoryPool& pool) {
  NIMBLE_CHECK_EQ(streamTypes.size(), streamLengths.size());
  auto pos = statStreams.data();
  for (size_t i = 0; i < streamTypes.size(); ++i) {
    if (auto vectorizedStat = deserializeVectorizedStatistic(
            streamTypes[i], std::string_view{pos, streamLengths[i]}, pool)) {
      statStreams_.emplace(streamTypes[i], std::move(vectorizedStat));
      switch (streamTypes[i]) {
        case StatStreamType::VALUE_COUNT:
        case StatStreamType::NULL_COUNT:
        case StatStreamType::LOGICAL_SIZE:
        case StatStreamType::PHYSICAL_SIZE:
          statTypes_.insert(StatType::DEFAULT);
          break;
        case StatStreamType::INTEGRAL_MIN:
        case StatStreamType::INTEGRAL_MAX:
          statTypes_.insert(StatType::INTEGRAL);
          break;
        case StatStreamType::FLOATING_POINT_MIN:
        case StatStreamType::FLOATING_POINT_MAX:
          statTypes_.insert(StatType::FLOATING_POINT);
          break;
        case StatStreamType::STRING_MIN:
        case StatStreamType::STRING_MAX:
          statTypes_.insert(StatType::STRING);
          break;
        case StatStreamType::DEDUPLICATED_VALUE_COUNT:
        case StatStreamType::DEDUPLICATED_LOGICAL_SIZE:
          statTypes_.insert(StatType::DEDUPLICATED);
          break;
      }
    }
    pos += streamLengths[i];
  }
}

std::unique_ptr<VectorizedStatistic>
VectorizedFileStats::deserializeVectorizedStatistic(
    StatStreamType streamType,
    std::string_view payload,
    velox::memory::MemoryPool& pool) {
  auto stat = VectorizedStatistic::create(streamType, &pool);
  if (UNLIKELY(!stat)) {
    LOG(INFO) << "Ignoring unrecognized stat type: "
              << static_cast<uint8_t>(streamType);
    return stat;
  }
  stat->deserializeFrom(payload, pool);
  return stat;
}

struct SchemaInfo {
  std::vector<uint32_t> flatmapNodeIds;
  std::vector<uint32_t> slidingWindowMapNodeIds;
  std::vector<uint32_t> arrayWithOffsetsNodeIds;

  bool isFlatmapNode(uint32_t nodeId) const {
    return std::find(flatmapNodeIds.begin(), flatmapNodeIds.end(), nodeId) !=
        flatmapNodeIds.end();
  }

  bool isSlidingWindowMapNode(uint32_t nodeId) const {
    return std::find(
               slidingWindowMapNodeIds.begin(),
               slidingWindowMapNodeIds.end(),
               nodeId) != slidingWindowMapNodeIds.end();
  }

  bool isArrayWithOffsetsNode(uint32_t nodeId) const {
    return std::find(
               arrayWithOffsetsNodeIds.begin(),
               arrayWithOffsetsNodeIds.end(),
               nodeId) != arrayWithOffsetsNodeIds.end();
  }
};

namespace {

void populateSchemaInfo(
    const velox::TypePtr& schema,
    const std::shared_ptr<const nimble::Type>& nimbleType,
    uint32_t& nodeId,
    SchemaInfo& schemaInfo) {
  uint32_t currentNodeId = nodeId++;

  switch (schema->kind()) {
    case velox::TypeKind::TINYINT:
    case velox::TypeKind::SMALLINT:
    case velox::TypeKind::INTEGER:
    case velox::TypeKind::BIGINT:
    case velox::TypeKind::REAL:
    case velox::TypeKind::DOUBLE:
    case velox::TypeKind::VARCHAR:
    case velox::TypeKind::VARBINARY:
    case velox::TypeKind::BOOLEAN:
    case velox::TypeKind::TIMESTAMP:
      break;

    case velox::TypeKind::ROW: {
      auto childCount = schema->size();
      for (size_t i = 0; i < childCount; ++i) {
        populateSchemaInfo(
            schema->childAt(i),
            nimbleType->asRow().childAt(i),
            nodeId,
            schemaInfo);
      }
      break;
    }

    case velox::TypeKind::ARRAY: {
      if (nimbleType->isArrayWithOffsets()) {
        schemaInfo.arrayWithOffsetsNodeIds.push_back(currentNodeId);
        populateSchemaInfo(
            schema->childAt(0),
            nimbleType->asArrayWithOffsets().elements(),
            nodeId,
            schemaInfo);
      } else {
        populateSchemaInfo(
            schema->childAt(0),
            nimbleType->asArray().elements(),
            nodeId,
            schemaInfo);
      }
      break;
    }

    case velox::TypeKind::MAP: {
      if (nimbleType->isFlatMap()) {
        schemaInfo.flatmapNodeIds.push_back(currentNodeId);
      } else if (nimbleType->isSlidingWindowMap()) {
        schemaInfo.slidingWindowMapNodeIds.push_back(currentNodeId);
        const auto& slidingWindowMap = nimbleType->asSlidingWindowMap();
        populateSchemaInfo(
            schema->childAt(0), slidingWindowMap.keys(), nodeId, schemaInfo);
        populateSchemaInfo(
            schema->childAt(1), slidingWindowMap.values(), nodeId, schemaInfo);
      } else {
        const auto& map = nimbleType->asMap();
        populateSchemaInfo(schema->childAt(0), map.keys(), nodeId, schemaInfo);
        populateSchemaInfo(
            schema->childAt(1), map.values(), nodeId, schemaInfo);
      }
      break;
    }

    default:
      NIMBLE_UNSUPPORTED(
          fmt::format("Unsupported schema kind: {}.", schema->kind()));
  }
}

SchemaInfo getSchemaInfo(
    const velox::TypePtr& schema,
    const std::shared_ptr<const nimble::Type>& nimbleType) {
  SchemaInfo schemaInfo;
  uint32_t nodeId = 0;
  populateSchemaInfo(schema, nimbleType, nodeId, schemaInfo);
  return schemaInfo;
}

void traverseSchema(
    const velox::TypePtr& schema,
    const SchemaInfo& schemaInfo,
    const std::map<StatStreamType, std::unique_ptr<VectorizedStatistic>>&
        statStreams,
    std::unordered_map<StatType, size_t>& statTypeCounts,
    std::vector<std::unique_ptr<ColumnStatistics>>& columnStats) {
  auto statIdx = columnStats.size();
  auto valueCount = statStreams.at(StatStreamType::VALUE_COUNT)
                        ->as<DefaultVectorizedStatistics>()
                        ->valueAt(statIdx);
  auto nullCount = statStreams.at(StatStreamType::NULL_COUNT)
                       ->as<DefaultVectorizedStatistics>()
                       ->valueAt(statIdx);
  auto logicalSize = statStreams.at(StatStreamType::LOGICAL_SIZE)
                         ->as<DefaultVectorizedStatistics>()
                         ->valueAt(statIdx);
  auto physicalSize = statStreams.at(StatStreamType::PHYSICAL_SIZE)
                          ->as<DefaultVectorizedStatistics>()
                          ->valueAt(statIdx);
  NIMBLE_CHECK(valueCount.has_value());
  NIMBLE_CHECK(nullCount.has_value());
  NIMBLE_CHECK(logicalSize.has_value());
  NIMBLE_CHECK(physicalSize.has_value());
  switch (schema->kind()) {
    case velox::TypeKind::TINYINT:
    case velox::TypeKind::SMALLINT:
    case velox::TypeKind::INTEGER:
    case velox::TypeKind::BIGINT: {
      auto [intStatCountIter, _] =
          statTypeCounts.emplace(StatType::INTEGRAL, 0);
      auto intStatIdx = (intStatCountIter->second)++;
      auto integralMinStatIter = statStreams.find(StatStreamType::INTEGRAL_MIN);
      auto integralMaxStatIter = statStreams.find(StatStreamType::INTEGRAL_MAX);
      columnStats.emplace_back(
          std::make_unique<IntegralStatistics>(
              valueCount.value(),
              nullCount.value(),
              logicalSize.value(),
              physicalSize.value(),
              integralMinStatIter == statStreams.end()
                  ? std::nullopt
                  : integralMinStatIter->second
                        ->as<IntegralVectorizedStatistics>()
                        ->valueAt(intStatIdx),
              integralMinStatIter == statStreams.end()
                  ? std::nullopt
                  : integralMaxStatIter->second
                        ->as<IntegralVectorizedStatistics>()
                        ->valueAt(intStatIdx)));
      break;
    }
    case velox::TypeKind::REAL:
    case velox::TypeKind::DOUBLE: {
      auto [floatStatCountIter, _] =
          statTypeCounts.emplace(StatType::FLOATING_POINT, 0);
      auto floatStatIdx = (floatStatCountIter->second)++;
      auto floatMinStatIter =
          statStreams.find(StatStreamType::FLOATING_POINT_MIN);
      auto floatMaxStatIter =
          statStreams.find(StatStreamType::FLOATING_POINT_MAX);
      columnStats.emplace_back(
          std::make_unique<FloatingPointStatistics>(
              valueCount.value(),
              nullCount.value(),
              logicalSize.value(),
              physicalSize.value(),
              floatMinStatIter == statStreams.end()
                  ? std::nullopt
                  : floatMinStatIter->second
                        ->as<FloatingPointVectorizedStatistics>()
                        ->valueAt(floatStatIdx),
              floatMaxStatIter == statStreams.end()
                  ? std::nullopt
                  : floatMaxStatIter->second
                        ->as<FloatingPointVectorizedStatistics>()
                        ->valueAt(floatStatIdx)));
      break;
    }
    case velox::TypeKind::VARCHAR:
    case velox::TypeKind::VARBINARY: {
      auto [stringStatCountIter, _] =
          statTypeCounts.emplace(StatType::STRING, 0);
      auto stringStatIdx = (stringStatCountIter->second)++;
      auto stringMinStatIter = statStreams.find(StatStreamType::STRING_MIN);
      auto stringMaxStatIter = statStreams.find(StatStreamType::STRING_MAX);
      auto toOptionalString =
          [](std::optional<std::string_view> sv) -> std::optional<std::string> {
        if (sv.has_value()) {
          return std::string(sv.value());
        }
        return std::nullopt;
      };
      columnStats.emplace_back(
          std::make_unique<StringStatistics>(
              valueCount.value(),
              nullCount.value(),
              logicalSize.value(),
              physicalSize.value(),
              stringMinStatIter == statStreams.end()
                  ? std::nullopt
                  : toOptionalString(stringMinStatIter->second
                                         ->as<StringVectorizedStatistics>()
                                         ->valueAt(stringStatIdx)),
              stringMaxStatIter == statStreams.end()
                  ? std::nullopt
                  : toOptionalString(stringMaxStatIter->second
                                         ->as<StringVectorizedStatistics>()
                                         ->valueAt(stringStatIdx))));
      break;
    }
    case velox::TypeKind::BOOLEAN:
    case velox::TypeKind::TIMESTAMP: {
      columnStats.emplace_back(
          std::make_unique<ColumnStatistics>(
              valueCount.value(),
              nullCount.value(),
              logicalSize.value(),
              physicalSize.value()));
      break;
    }
    case velox::TypeKind::ROW: {
      columnStats.emplace_back(
          std::make_unique<ColumnStatistics>(
              valueCount.value(),
              nullCount.value(),
              logicalSize.value(),
              physicalSize.value()));
      auto childCount = schema->size();
      for (size_t i = 0; i < childCount; ++i) {
        traverseSchema(
            schema->childAt(i),
            schemaInfo,
            statStreams,
            statTypeCounts,
            columnStats);
      }
      break;
    }
    case velox::TypeKind::ARRAY: {
      columnStats.emplace_back(
          std::make_unique<ColumnStatistics>(
              valueCount.value(),
              nullCount.value(),
              logicalSize.value(),
              physicalSize.value()));
      traverseSchema(
          schema->childAt(0),
          schemaInfo,
          statStreams,
          statTypeCounts,
          columnStats);
      break;
    }
    case velox::TypeKind::MAP: {
      columnStats.emplace_back(
          std::make_unique<ColumnStatistics>(
              valueCount.value(),
              nullCount.value(),
              logicalSize.value(),
              physicalSize.value()));
      traverseSchema(
          schema->childAt(0),
          schemaInfo,
          statStreams,
          statTypeCounts,
          columnStats);
      traverseSchema(
          schema->childAt(1),
          schemaInfo,
          statStreams,
          statTypeCounts,
          columnStats);
      break;
    }
    default: {
      NIMBLE_UNSUPPORTED(
          fmt::format("Unsupported schema kind: {}.", schema->kind()));
    }
  }
}

} // namespace

std::vector<std::unique_ptr<ColumnStatistics>>
VectorizedFileStats::toColumnStatistics(
    const velox::TypePtr& schema,
    const std::shared_ptr<const nimble::Type>& nimbleType) {
  std::vector<std::unique_ptr<ColumnStatistics>> columnStats;
  std::unordered_map<StatType, size_t> statTypeCounts;
  auto schemaInfo = getSchemaInfo(schema, nimbleType);
  traverseSchema(schema, schemaInfo, statStreams_, statTypeCounts, columnStats);
  return columnStats;
}

} // namespace facebook::nimble
