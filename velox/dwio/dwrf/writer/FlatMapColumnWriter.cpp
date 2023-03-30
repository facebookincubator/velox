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

#include "velox/dwio/dwrf/writer/FlatMapColumnWriter.h"
#include <velox/dwio/dwrf/writer/StatisticsBuilder.h>
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::dwrf {

namespace {

template <typename T>
T getKey(const std::string& val) {
  try {
    return folly::to<T>(val);
  } catch (const folly::ConversionError& e) {
    DWIO_RAISE(e.what());
  }
}

template <>
StringView getKey<StringView>(const std::string& val) {
  return StringView(val);
}

template <typename KeyType>
std::vector<KeyType> parseKeys(const std::vector<std::string>& stringKeys) {
  std::vector<KeyType> keys(stringKeys.size());
  for (auto i = 0; i < stringKeys.size(); i++) {
    keys[i] = getKey<KeyType>(stringKeys[i]);
  }
  return keys;
}

} // namespace

using dwio::common::TypeWithId;
using proto::KeyInfo;

template <TypeKind K>
FlatMapColumnWriter<K>::FlatMapColumnWriter(
    WriterContext& context,
    const TypeWithId& type,
    const uint32_t sequence)
    : BaseColumnWriter{context, type, sequence, nullptr},
      keyType_{*type.childAt(0)},
      valueType_{*type.childAt(1)},
      maxKeyCount_{context_.getConfig(Config::MAP_FLAT_MAX_KEYS)},
      collectMapStats_{context.getConfig(Config::MAP_STATISTICS)} {
  auto options = StatisticsBuilderOptions::fromConfig(context.getConfigs());
  keyFileStatsBuilder_ =
      std::unique_ptr<typename TypeInfo<K>::StatisticsBuilder>(
          dynamic_cast<typename TypeInfo<K>::StatisticsBuilder*>(
              StatisticsBuilder::create(*keyType_.type, options).release()));
  valueFileStatsBuilder_ = ValueStatisticsBuilder::create(context_, valueType_);
  reset();
  const auto structColumnKeys =
      context.getConfig(Config::MAP_FLAT_COLS_STRUCT_KEYS);
  if (!structColumnKeys.empty()) {
    if constexpr (std::is_same_v<KeyType, StringView>) {
      const auto& keys = structColumnKeys[type.column];
      std::copy(keys.cbegin(), keys.cend(), std::back_inserter(stringKeys_));
      structKeys_ = parseKeys<KeyType>(stringKeys_);
    } else {
      structKeys_ = parseKeys<KeyType>(structColumnKeys[type.column]);
    }
  }
}

template <TypeKind K>
void FlatMapColumnWriter<K>::setEncoding(
    proto::ColumnEncoding& encoding) const {
  BaseColumnWriter::setEncoding(encoding);
  encoding.set_kind(proto::ColumnEncoding_Kind::ColumnEncoding_Kind_MAP_FLAT);
}

template <TypeKind K>
void FlatMapColumnWriter<K>::flush(
    std::function<proto::ColumnEncoding&(uint32_t)> encodingFactory,
    std::function<void(proto::ColumnEncoding&)> encodingOverride) {
  BaseColumnWriter::flush(encodingFactory, encodingOverride);

  for (auto& pair : valueWriters_) {
    pair.second.flush(encodingFactory);
  }

  if (collectMapStats_) {
    // Need to record and clear per stripe due to sequence to key instability
    auto& physicalSizeAgg = dynamic_cast<MapPhysicalSizeAggregator&>(
        context_.getPhysicalSizeAggregator(id_));
    // Only top level value column streams have keys in their encodings. We
    // could only identify streams from nested value column with sequence.
    folly::F14FastMap<uint32_t, const proto::KeyInfo&> sequenceToKey{};
    for (auto& pair : valueWriters_) {
      auto& valueWriter = pair.second;
      sequenceToKey.emplace(
          valueWriter.getSequence(), valueWriter.getKeyInfo());
    }
    physicalSizeAgg.prepare(
        sequenceToKey,
        dynamic_cast<MapStatisticsBuilder*>(fileStatsBuilder_.get()));
  }

  // Reset is being called after flush, so no need to explicitly
  // reset internal stripe state
}

template <TypeKind K>
void FlatMapColumnWriter<K>::createIndexEntry() {
  // Aggregate value writer index stats into map writer index stats before
  // merging into file stats.
  auto& mapStatsBuilder =
      dynamic_cast<MapStatisticsBuilder&>(*indexStatsBuilder_);
  for (auto& pair : valueWriters_) {
    pair.second.createIndexEntry(*valueFileStatsBuilder_, mapStatsBuilder);
  }
  BaseColumnWriter::createIndexEntry();
  rowsInStrides_.push_back(rowsInCurrentStride_);
  rowsInCurrentStride_ = 0;
}

template <TypeKind K>
uint64_t FlatMapColumnWriter<K>::writeFileStats(
    std::function<proto::ColumnStatistics&(uint32_t)> statsFactory) const {
  auto& stats = statsFactory(id_);
  fileStatsBuilder_->toProto(stats);
  uint64_t size = context_.getPhysicalSizeAggregator(id_).getResult();

  auto& keyStats = statsFactory(keyType_.id);
  keyFileStatsBuilder_->toProto(keyStats);
  auto keySize = context_.getPhysicalSizeAggregator(keyType_.id).getResult();
  keyStats.set_size(keySize);

  valueFileStatsBuilder_->writeFileStats(statsFactory);
  stats.set_size(size);
  return size;
}

template <TypeKind K>
void FlatMapColumnWriter<K>::clearNodes() {
  // TODO: If we are writing shared dictionaries, the shared dictionary data
  // stream should be reused. Even when the stream is not actually used, we
  // should keep it around in case of future encoding decision changes.
  context_.removeAllIntDictionaryEncodersOnNode([this](uint32_t nodeId) {
    return nodeId >= valueType_.id && nodeId <= valueType_.maxId;
  });

  context_.removeStreams([this](const DwrfStreamIdentifier& identifier) {
    return identifier.encodingKey().node >= valueType_.id &&
        identifier.encodingKey().node <= valueType_.maxId &&
        (identifier.kind() == StreamKind::StreamKind_DICTIONARY_DATA ||
         identifier.encodingKey().sequence > 0);
  });
}

template <TypeKind K>
void FlatMapColumnWriter<K>::reset() {
  BaseColumnWriter::reset();
  clearNodes();
  valueWriters_.clear();
  rowsInStrides_.clear();
  rowsInCurrentStride_ = 0;
}

KeyInfo getKeyInfo(int64_t key) {
  KeyInfo keyInfo;
  keyInfo.set_intkey(key);
  return keyInfo;
}

KeyInfo getKeyInfo(StringView key) {
  KeyInfo keyInfo;
  keyInfo.set_byteskey(key.data(), key.size());
  return keyInfo;
}

template <TypeKind K>
ValueWriter& FlatMapColumnWriter<K>::getValueWriter(
    KeyType key,
    uint32_t inMapSize) {
  auto it = valueWriters_.find(key);
  if (it != valueWriters_.end()) {
    return it->second;
  }

  if (valueWriters_.size() >= maxKeyCount_) {
    DWIO_RAISE(fmt::format(
        "Too many map keys requested in (node {}, column {}). Allowed: {}",
        id_,
        type_.column,
        maxKeyCount_));
  }

  auto keyInfo = getKeyInfo(key);

  it = valueWriters_
           .emplace(
               std::piecewise_construct,
               std::forward_as_tuple(key),
               std::forward_as_tuple(
                   valueWriters_.size() + 1, /* sequence */
                   keyInfo,
                   this->context_,
                   this->valueType_,
                   inMapSize))
           .first;

  ValueWriter& valueWriter = it->second;

  auto& mapStatsBuilder =
      dynamic_cast<MapStatisticsBuilder&>(*fileStatsBuilder_);
  // Back fill previous strides with not-in-map indication
  for (auto& rows : rowsInStrides_) {
    valueWriter.backfill(rows);
    valueWriter.createIndexEntry(*valueFileStatsBuilder_, mapStatsBuilder);
  }

  // Back fill current (partial) stride with not-in-map indication
  valueWriter.backfill(rowsInCurrentStride_);

  return valueWriter;
}

template <TypeKind K>
uint32_t updateKeyStatistics(
    typename TypeInfo<K>::StatisticsBuilder& keyStatsBuilder,
    typename TypeTraits<K>::NativeType value) {
  keyStatsBuilder.addValues(value);
  return sizeof(typename TypeTraits<K>::NativeType);
}

template <>
uint32_t updateKeyStatistics<TypeKind::VARCHAR>(
    StringStatisticsBuilder& keyStatsBuilder,
    StringView value) {
  auto size = value.size();
  keyStatsBuilder.addValues(folly::StringPiece{value.data(), size});
  return size;
}

template <>
uint32_t updateKeyStatistics<TypeKind::VARBINARY>(
    BinaryStatisticsBuilder& keyStatsBuilder,
    StringView value) {
  auto size = value.size();
  keyStatsBuilder.addValues(size);
  return size;
}

namespace {

// Adapters to handle flat or decoded vector using same interfaces. For usages
// when type is not important (ie. we don't call valueAt()), we can rely on the
// default type.
template <typename T = int8_t>
class Flat {
 public:
  explicit Flat(const VectorPtr& vector)
      : vector_{vector}, nulls_{vector_->rawNulls()} {
    auto casted = vector->asFlatVector<T>();
    values_ = casted ? casted->rawValues() : nullptr;
  }

  bool hasNulls() const {
    return vector_->mayHaveNulls();
  }

  bool isNullAt(vector_size_t index) const {
    return bits::isBitNull(nulls_, index);
  }

  T valueAt(vector_size_t index) const {
    return values_[index];
  }

  vector_size_t index(vector_size_t index) const {
    return index;
  }

 private:
  const VectorPtr& vector_;
  const uint64_t* nulls_;
  const T* values_;
};

template <typename T = int8_t>
class Decoded {
 public:
  explicit Decoded(const DecodedVector& decoded) : decoded_{decoded} {}

  bool hasNulls() const {
    return decoded_.mayHaveNulls();
  }

  bool isNullAt(vector_size_t index) const {
    return decoded_.isNullAt(index);
  }

  T valueAt(vector_size_t index) const {
    return decoded_.valueAt<T>(index);
  }

  vector_size_t index(vector_size_t index) const {
    return decoded_.index(index);
  }

 private:
  const DecodedVector& decoded_;
};

template <typename Map, typename MapOp>
uint64_t
iterateMaps(const common::Ranges& ranges, const Map& map, const MapOp& mapOp) {
  uint64_t nullCount = 0;
  if (map.hasNulls()) {
    for (auto& index : ranges) {
      if (map.isNullAt(index)) {
        ++nullCount;
      } else {
        mapOp(map.index(index));
      }
    }
  } else {
    for (auto& index : ranges) {
      mapOp(map.index(index));
    }
  }
  return nullCount;
}

} // namespace

template <TypeKind K>
uint64_t FlatMapColumnWriter<K>::write(
    const VectorPtr& slice,
    const common::Ranges& ranges) {
  switch (slice->typeKind()) {
    case TypeKind::MAP:
      return writeMap(slice, ranges);
    case TypeKind::ROW:
      if (!structKeys_.empty()) {
        return writeRow(slice, ranges);
      }
      DWIO_RAISE(
          "Writing RowVector as flatmap with empty keys. Keys must be specified on writer creation.");
    default:
      DWIO_RAISE(
          "Unexpected vector type. Should be MapVector or RowVector (can be encoded)");
  }
}

template <TypeKind K>
uint64_t FlatMapColumnWriter<K>::writeMap(
    const VectorPtr& slice,
    const common::Ranges& ranges) {
  // Define variables captured and used by below lambdas.
  const vector_size_t* offsets;
  const vector_size_t* lengths;
  uint64_t rawSize = 0;
  uint64_t mapCount = 0;
  common::Ranges keyRanges;

  // Lambda that iterates keys of a map and records the offsets to write to
  // particular value node.
  auto processMap = [&](uint64_t offsetIndex, const auto& keysVector) {
    auto begin = offsets[offsetIndex];
    auto end = begin + lengths[offsetIndex];

    for (auto i = begin; i < end; ++i) {
      auto key = keysVector.valueAt(i);
      ValueWriter& valueWriter = getValueWriter(key, ranges.size());
      valueWriter.addOffset(i, mapCount);
      auto keySize = updateKeyStatistics<K>(*keyFileStatsBuilder_, key);
      keyFileStatsBuilder_->increaseRawSize(keySize);
      rawSize += keySize;
    }

    ++mapCount;
  };

  // Lambda that calculates child ranges
  auto computeKeyRanges = [&](uint64_t offsetIndex) {
    auto begin = offsets[offsetIndex];
    keyRanges.add(begin, begin + lengths[offsetIndex]);
  };

  // Lambda that process the batch
  auto processBatch = [&](const auto& map, const auto& mapSlice) {
    auto& mapKeys = mapSlice->mapKeys();
    auto keysFlat = mapKeys->template asFlatVector<KeyType>();
    if (keysFlat) {
      // Keys are flat
      Flat<KeyType> keysVector{mapKeys};
      return iterateMaps(
          ranges, map, [&](auto offset) { processMap(offset, keysVector); });
    } else {
      // Keys are encoded. Decode.
      iterateMaps(ranges, map, computeKeyRanges);
      auto localDecodedKeys = decode(mapKeys, keyRanges);
      auto& decodedKeys = localDecodedKeys.get();
      Decoded<KeyType> keysVector{decodedKeys};
      return iterateMaps(
          ranges, map, [&](auto offset) { processMap(offset, keysVector); });
    }
  };

  // Reset all existing value writer buffers
  // This includes existing value writers that are not used in this batch
  // (their buffers will be set to empty buffers)
  for (auto& pair : valueWriters_) {
    pair.second.resizeBuffers(ranges.size());
  }

  // Fill value buffers per key
  uint64_t nullCount = 0;
  const MapVector* mapSlice = slice->as<MapVector>();
  if (mapSlice) {
    // Map is flat
    writeNulls(slice, ranges);
    offsets = mapSlice->rawOffsets();
    lengths = mapSlice->rawSizes();
    nullCount = processBatch(Flat{slice}, mapSlice);
  } else {
    // Map is encoded. Decode.
    auto localDecodedMap = decode(slice, ranges);
    auto& decodedMap = localDecodedMap.get();
    writeNulls(decodedMap, ranges);
    mapSlice = decodedMap.base()->template as<MapVector>();
    DWIO_ENSURE(mapSlice, "unexpected vector type");

    offsets = mapSlice->rawOffsets();
    lengths = mapSlice->rawSizes();
    nullCount = processBatch(Decoded{decodedMap}, mapSlice);
  }

  auto& values = mapSlice->mapValues();
  // Write all accumulated buffers (this includes value writers that weren't
  // used in this write. This is how we backfill unused value writers)
  for (auto& pair : valueWriters_) {
    rawSize += pair.second.writeBuffers(values, mapCount);
  }

  if (nullCount > 0) {
    indexStatsBuilder_->setHasNull();
    rawSize += nullCount * NULL_SIZE;
  }
  rowsInCurrentStride_ += mapCount;
  indexStatsBuilder_->increaseValueCount(mapCount);
  indexStatsBuilder_->increaseRawSize(rawSize);
  return rawSize;
}

template <typename Row>
common::Ranges getNonNullRanges(const common::Ranges& ranges, const Row& row) {
  common::Ranges nonNullRanges;
  if (row.hasNulls()) {
    for (auto& index : ranges) {
      if (!row.isNullAt(index)) {
        nonNullRanges.add(row.index(index), row.index(index) + 1);
      }
    }
  } else {
    nonNullRanges = ranges;
  }
  return nonNullRanges;
}

template <TypeKind K>
uint64_t FlatMapColumnWriter<K>::writeRow(
    const VectorPtr& slice,
    const common::Ranges& ranges) {
  uint64_t rawSize = 0;
  common::Ranges nonNullRanges;

  const RowVector* rowSlice = slice->as<RowVector>();
  if (rowSlice) {
    // Row is flat
    writeNulls(slice, ranges);
    nonNullRanges = getNonNullRanges(ranges, Flat{slice});
  } else {
    // Row is encoded. Decode.
    auto& decodedRow = decode(slice, ranges).get();
    writeNulls(decodedRow, ranges);
    rowSlice = decodedRow.base()->template as<RowVector>();
    // may verify in write() before entering writeRow() -- still needed?
    DWIO_ENSURE(rowSlice, "unexpected vector type");
    nonNullRanges = getNonNullRanges(ranges, Decoded{decodedRow});
  }
  DWIO_ENSURE(
      structKeys_.size() == rowSlice->childrenSize(),
      "Writing RowVector as flatmap with mismatched number of keys and columns.");

  BufferPtr inMapBuffer = AlignedBuffer::allocate<char>(
      nonNullRanges.size(),
      &context_.getMemoryPool(MemoryUsageCategory::GENERAL),
      1);

  for (size_t i = 0; i < structKeys_.size(); i++) {
    // looping each non-null row for now; no batch updateKeyStatistics()
    for (size_t j = 0; j < nonNullRanges.size(); j++) {
      auto keySize =
          updateKeyStatistics<K>(*keyFileStatsBuilder_, structKeys_[i]);
      keyFileStatsBuilder_->increaseRawSize(keySize);
      rawSize += keySize;
    }

    ValueWriter& valueWriter =
        getValueWriter(structKeys_[i], nonNullRanges.size());
    valueWriter.writeBuffers(rowSlice->childAt(i), nonNullRanges, inMapBuffer);
  }

  size_t numNullRows = ranges.size() - nonNullRanges.size();
  if (numNullRows > 0) {
    indexStatsBuilder_->setHasNull();
    rawSize += numNullRows * NULL_SIZE;
  }
  rowsInCurrentStride_ += nonNullRanges.size();
  indexStatsBuilder_->increaseValueCount(nonNullRanges.size());
  indexStatsBuilder_->increaseRawSize(rawSize);
  return rawSize;
}

template class FlatMapColumnWriter<TypeKind::TINYINT>;
template class FlatMapColumnWriter<TypeKind::SMALLINT>;
template class FlatMapColumnWriter<TypeKind::INTEGER>;
template class FlatMapColumnWriter<TypeKind::BIGINT>;
template class FlatMapColumnWriter<TypeKind::VARCHAR>;
template class FlatMapColumnWriter<TypeKind::VARBINARY>;

} // namespace facebook::velox::dwrf
