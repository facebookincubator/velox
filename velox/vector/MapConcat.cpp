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
#include "velox/vector/MapConcat.h"

#include <folly/container/F14Map.h>

#include "velox/common/base/SimdUtil.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatMapVector.h"

namespace facebook::velox {
namespace {

struct UpdateSource {
  vector_size_t entryIndex;
  int8_t sourceIndex;
};

template <typename T>
class UpdateMapRow {
 public:
  // Returns true if the key was newly inserted, false if it overwrote an
  // existing entry.
  bool insert(
      const DecodedVector* decoded,
      vector_size_t entryIndex,
      int8_t sourceIndex) {
    auto [iter, inserted] = values_.insert_or_assign(
        decoded->valueAt<T>(entryIndex), UpdateSource{entryIndex, sourceIndex});
    return inserted;
  }

  template <typename F>
  void forEachEntry(F&& func) const {
    for (auto& [_, source] : values_) {
      func(source);
    }
  }

  size_t size() const {
    return values_.size();
  }

  void reserve(size_t capacity) {
    values_.reserve(capacity);
  }

  void clear() {
    values_.clear();
  }

 private:
  folly::F14FastMap<T, UpdateSource> values_;
};

template <>
class UpdateMapRow<void> {
 public:
  bool insert(
      const DecodedVector* decoded,
      vector_size_t entryIndex,
      int8_t sourceIndex) {
    auto [iter, inserted] = references_.insert_or_assign(
        Reference{decoded->base(), decoded->index(entryIndex)},
        UpdateSource{entryIndex, sourceIndex});
    return inserted;
  }

  template <typename F>
  void forEachEntry(F&& func) const {
    for (auto& [_, source] : references_) {
      func(source);
    }
  }

  size_t size() const {
    return references_.size();
  }

  void reserve(size_t capacity) {
    references_.reserve(capacity);
  }

  void clear() {
    references_.clear();
  }

 private:
  struct Reference {
    const BaseVector* base;
    vector_size_t index;

    bool operator==(const Reference& other) const {
      return base->equalValueAt(other.base, index, other.index);
    }
  };

  struct ReferenceHasher {
    uint64_t operator()(const Reference& key) const {
      return key.base->hashValueAt(key.index);
    }
  };

  folly::F14FastMap<Reference, UpdateSource, ReferenceHasher> references_;
};

// Returns the error message string for a duplicate key.
std::string duplicateKeyMessage(
    const DecodedVector& keys,
    vector_size_t entryIndex) {
  return keys.base()->toString(keys.index(entryIndex));
}

template <TypeKind kKeyTypeKind>
MapVectorPtr mapConcatImpl(
    memory::MemoryPool* pool,
    const TypePtr& outputType,
    std::span<DecodedVector* const> inputs,
    const SelectivityVector& rows,
    const MapConcatConfig& config) {
  const auto numInputs = inputs.size();
  const auto numRows = rows.end();

  // Step 1: Compute output nulls.
  BufferPtr newNulls;
  if (!config.emptyForNull) {
    for (size_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
      auto* nulls = inputs[inputIdx]->nulls(&rows);
      if (!nulls) {
        continue;
      }
      if (!newNulls) {
        newNulls = allocateNulls(numRows, pool);
        auto* raw = newNulls->asMutable<uint64_t>();
        bits::copyBits(nulls, 0, raw, 0, numRows);
      } else {
        bits::andBits(newNulls->asMutable<uint64_t>(), nulls, 0, numRows);
      }
    }
  }

  // Step 2: Decode keys for all inputs.
  const auto& expectedKeyType = outputType->asMap().keyType();
  std::vector<DecodedVector> keys;
  keys.reserve(numInputs);
  for (size_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    auto* mapVector = inputs[inputIdx]->base()->asChecked<MapVector>();
    auto& mapKeys = mapVector->mapKeys();
    VELOX_CHECK(
        *expectedKeyType == *mapKeys->type(),
        "Map key type mismatch: {} vs {}",
        expectedKeyType->toString(),
        mapKeys->type()->toString());
    keys.emplace_back(*mapKeys);
  }

  // Step 3: Allocate output metadata.
  auto newOffsets = allocateOffsets(numRows, pool);
  auto* rawNewOffsets = newOffsets->asMutable<vector_size_t>();
  auto newSizes = allocateSizes(numRows, pool);
  auto* rawNewSizes = newSizes->asMutable<vector_size_t>();

  // Step 4: Per-row merge.
  std::vector<std::vector<BaseVector::CopyRange>> ranges(numInputs);
  UpdateMapRow<typename TypeTraits<kKeyTypeKind>::NativeType> mapRow;
  vector_size_t numEntries = 0;

  rows.applyToSelected([&](vector_size_t row) {
    rawNewOffsets[row] = numEntries;

    // Check null propagation.
    if (newNulls && bits::isBitNull(newNulls->as<uint64_t>(), row)) {
      rawNewSizes[row] = 0;
      return;
    }

    // Count total entries and non-empty inputs for this row.
    vector_size_t totalEntries{0};
    int nonEmptyCount{0};
    int nonEmptyIdx{-1};
    for (size_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
      if (config.emptyForNull && inputs[inputIdx]->isNullAt(row)) {
        continue;
      }
      auto decodedIndex = inputs[inputIdx]->index(row);
      auto* mapVector = inputs[inputIdx]->base()->asUnchecked<MapVector>();
      auto size = mapVector->sizeAt(decodedIndex);
      if (size > 0) {
        ++nonEmptyCount;
        nonEmptyIdx = inputIdx;
      }
      totalEntries += size;
    }

    if (totalEntries == 0) {
      rawNewSizes[row] = 0;
      return;
    }

    // Fast path: only one input has entries for this row.  No dedup possible.
    if (nonEmptyCount == 1) {
      auto decodedIndex = inputs[nonEmptyIdx]->index(row);
      auto* mapVector = inputs[nonEmptyIdx]->base()->asUnchecked<MapVector>();
      auto offset = mapVector->offsetAt(decodedIndex);
      ranges[nonEmptyIdx].push_back({offset, numEntries, totalEntries});
      rawNewSizes[row] = totalEntries;
      numEntries += totalEntries;
      return;
    }

    // Build hash map for this row.
    mapRow.reserve(totalEntries);
    for (size_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
      if (config.emptyForNull && inputs[inputIdx]->isNullAt(row)) {
        continue;
      }
      auto decodedIndex = inputs[inputIdx]->index(row);
      auto* mapVector = inputs[inputIdx]->base()->asUnchecked<MapVector>();
      auto offset = mapVector->offsetAt(decodedIndex);
      auto size = mapVector->sizeAt(decodedIndex);
      for (vector_size_t entryIdx = 0; entryIdx < size; ++entryIdx) {
        auto entryOffset = offset + entryIdx;
        VELOX_CHECK(
            !keys[inputIdx].isNullAt(entryOffset), "Map key cannot be null");
        bool inserted = mapRow.insert(&keys[inputIdx], entryOffset, inputIdx);
        if (config.throwOnDuplicateKeys && !inserted) {
          VELOX_USER_FAIL(
              "Duplicate map key {} was found.",
              duplicateKeyMessage(keys[inputIdx], entryOffset));
        }
      }
    }

    // If no dedup happened, use bulk copy ranges (much faster than
    // per-entry ranges).
    if (mapRow.size() == totalEntries) {
      mapRow.clear();
      for (size_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
        if (config.emptyForNull && inputs[inputIdx]->isNullAt(row)) {
          continue;
        }
        auto decodedIndex = inputs[inputIdx]->index(row);
        auto* mapVector = inputs[inputIdx]->base()->asUnchecked<MapVector>();
        auto offset = mapVector->offsetAt(decodedIndex);
        auto size = mapVector->sizeAt(decodedIndex);
        if (size > 0) {
          ranges[inputIdx].push_back({offset, numEntries, size});
          numEntries += size;
        }
      }
      rawNewSizes[row] = totalEntries;
      return;
    }

    // Dedup happened — collect deduplicated entries into per-entry copy
    // ranges.
    vector_size_t newSize{0};
    mapRow.forEachEntry([&](UpdateSource source) {
      ranges[source.sourceIndex].push_back(
          {source.entryIndex, numEntries + newSize, 1});
      ++newSize;
    });
    mapRow.clear();
    rawNewSizes[row] = newSize;
    numEntries += newSize;
  });

  // Step 5: Build result vectors.
  auto newKeys =
      BaseVector::create(outputType->asMap().keyType(), numEntries, pool);
  auto newValues =
      BaseVector::create(outputType->asMap().valueType(), numEntries, pool);
  for (size_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    auto* mapVector = inputs[inputIdx]->base()->asUnchecked<MapVector>();
    newKeys->copyRanges(mapVector->mapKeys().get(), ranges[inputIdx]);
    newValues->copyRanges(mapVector->mapValues().get(), ranges[inputIdx]);
  }

  return std::make_shared<MapVector>(
      pool,
      outputType,
      std::move(newNulls),
      numRows,
      std::move(newOffsets),
      std::move(newSizes),
      std::move(newKeys),
      std::move(newValues));
}

// A key at an index of a keys vector, hashed and compared by value, so that
// keys of different vectors can be looked up in one hash table.
struct KeyAt {
  const BaseVector* keys;
  column_index_t index;

  bool operator==(const KeyAt& other) const {
    return keys->equalValueAt(other.keys, index, other.index);
  }
};

struct KeyAtHasher {
  uint64_t operator()(const KeyAt& key) const {
    return key.keys->hashValueAt(key.index);
  }
};

// Returns the result rows where a flat map holds the key at a channel, or null
// when that is every selected row. The map rows are the selected rows where the
// map is not null, or null when that is every row. A wrapped map also passes
// its dictionary indices over the result rows, with index 0 in the rows it does
// not contribute to, so that simd::gatherBits() reads only its in-map bits.
BufferPtr selectedInMap(
    const DecodedVector& decoded,
    const FlatMapVector& flatMap,
    const BufferPtr& mapRows,
    const BufferPtr& indices,
    column_index_t channel,
    vector_size_t numRows,
    memory::MemoryPool* pool) {
  const auto* rawInMap = flatMap.rawInMapsAt(channel);
  if (rawInMap == nullptr || decoded.isConstantMapping()) {
    return mapRows;
  }
  if (indices == nullptr && mapRows == nullptr) {
    return flatMap.inMapsAt(channel);
  }
  auto inMap = AlignedBuffer::allocate<bool>(numRows, pool);
  auto* rawSourceInMap = inMap->asMutable<uint64_t>();
  if (indices == nullptr) {
    bits::andBits(
        rawSourceInMap, mapRows->as<uint64_t>(), rawInMap, 0, numRows);
  } else {
    simd::gatherBits(
        rawInMap,
        folly::Range<const int32_t*>(indices->as<int32_t>(), numRows),
        rawSourceInMap);
    if (mapRows != nullptr) {
      bits::andBits(rawSourceInMap, mapRows->as<uint64_t>(), 0, numRows);
    }
  }
  return inMap;
}

// Returns the values of a key of a flat map, wrapped as the map is so that they
// are indexed by result row.
VectorPtr wrapValues(
    DecodedVector& decoded,
    const DecodedVector::DictionaryWrapping& wrapping,
    const VectorPtr& values,
    const SelectivityVector& rows,
    memory::MemoryPool* pool) {
  if (decoded.isConstantMapping()) {
    return decoded.wrap(values, *pool, rows);
  }
  if (wrapping.indices == nullptr) {
    return values;
  }
  return BaseVector::wrapInDictionary(
      wrapping.nulls, wrapping.indices, rows.end(), values);
}

// Checks that every map is a flat map with the result's key type and returns
// the total number of their distinct keys, which bounds the result's. Unless
// null maps count as empty, also sets 'resultNulls' to the rows where every map
// is not null, or leaves it null when no map has nulls.
vector_size_t estimateKeySizeAndCheckTypes(
    std::span<DecodedVector* const> maps,
    const SelectivityVector& rows,
    const MapConcatConfig& config,
    const TypePtr& keyType,
    memory::MemoryPool* pool,
    BufferPtr& resultNulls) {
  const auto numRows = rows.end();
  vector_size_t maxNumKeys{0};
  for (auto* map : maps) {
    const auto* flatMap = map->base()->asChecked<FlatMapVector>();
    const auto& mapKeyType = flatMap->distinctKeys()->type();
    VELOX_CHECK(
        *keyType == *mapKeyType,
        "Map key type mismatch: {} vs {}",
        keyType->toString(),
        mapKeyType->toString());
    maxNumKeys += flatMap->numDistinctKeys();
    if (config.emptyForNull) {
      continue;
    }
    const auto* nulls = map->nulls(&rows);
    if (nulls == nullptr) {
      continue;
    }
    if (resultNulls == nullptr) {
      resultNulls = allocateNulls(numRows, pool);
      bits::copyBits(nulls, 0, resultNulls->asMutable<uint64_t>(), 0, numRows);
    } else {
      bits::andBits(resultNulls->asMutable<uint64_t>(), nulls, 0, numRows);
    }
  }
  return maxNumKeys;
}

// Merges the values of a key of a flat map into the result's, where the map's
// values win in the rows where both hold the key. Returns whether the result's
// values and in-map are still a map's, which the result must not write into.
bool mergeChannel(
    DecodedVector& decoded,
    const DecodedVector::DictionaryWrapping& wrapping,
    const VectorPtr& currentValues,
    BufferPtr currentInMap,
    bool isOriginal,
    VectorPtr& resultValues,
    BufferPtr& resultInMap,
    const SelectivityVector& rows,
    const TypePtr& valueType,
    SelectivityVector& rowsToCopy,
    memory::MemoryPool* pool) {
  const auto numRows = rows.end();
  const auto* selectedRowBits = rows.asRange().bits();
  const auto* currentKeyRows =
      currentInMap != nullptr ? currentInMap->as<uint64_t>() : selectedRowBits;
  const auto* resultKeyRows =
      resultInMap != nullptr ? resultInMap->as<uint64_t>() : selectedRowBits;

  // When this map holds the key in every row the result does, its values win
  // in all of them, so the result shares them instead.
  if (bits::isSubset(resultKeyRows, currentKeyRows, rows.begin(), rows.end())) {
    resultValues = wrapValues(decoded, wrapping, currentValues, rows, pool);
    resultInMap = std::move(currentInMap);
    return true;
  }

  // The first merge of a key copies its shared values into values the result
  // owns. Later merges write into those.
  if (isOriginal) {
    rowsToCopy.setFromBits(resultKeyRows, numRows);
    rowsToCopy.deselectNonNulls(currentKeyRows, rows.begin(), rows.end());
    auto mergedValues = BaseVector::create(valueType, numRows, pool);
    mergedValues->copy(resultValues.get(), rowsToCopy, nullptr);
    resultValues = std::move(mergedValues);

    auto mergedInMap = AlignedBuffer::allocate<bool>(numRows, pool);
    bits::copyBits(
        resultKeyRows, 0, mergedInMap->asMutable<uint64_t>(), 0, numRows);
    resultInMap = std::move(mergedInMap);
  }
  rowsToCopy.setFromBits(currentKeyRows, numRows);
  resultValues->copy(
      currentValues.get(),
      rowsToCopy,
      wrapping.indices != nullptr ? wrapping.indices->as<vector_size_t>()
                                  : nullptr);
  bits::orBits(
      resultInMap->asMutable<uint64_t>(),
      currentKeyRows,
      rows.begin(),
      rows.end());
  return false;
}

// Concatenates the maps one at a time. A key the result does not have yet
// shares the map's values. A key it already has is copied into values of its
// own the first time, and later maps copy over it, so the last map wins.
// Output keys are in the order they are first seen. When they are exactly the
// keys of the first flat map, the result keeps its distinct keys and key index.
FlatMapVectorPtr flatMapConcat(
    memory::MemoryPool* pool,
    const TypePtr& outputType,
    std::span<DecodedVector* const> maps,
    const SelectivityVector& rows,
    const MapConcatConfig& config) {
  const auto numMaps = maps.size();
  const auto numRows = rows.end();
  const auto& keyType = outputType->asMap().keyType();
  const auto& valueType = outputType->asMap().valueType();
  BufferPtr resultNulls;
  const auto maxNumKeys = estimateKeySizeAndCheckTypes(
      maps, rows, config, keyType, pool, resultNulls);
  const auto* selectedRowBits = rows.asRange().bits();
  BufferPtr resultRowsBuffer;
  if (resultNulls != nullptr) {
    resultRowsBuffer = AlignedBuffer::allocate<bool>(numRows, pool);
    bits::andBits(
        resultRowsBuffer->asMutable<uint64_t>(),
        selectedRowBits,
        resultNulls->as<uint64_t>(),
        0,
        numRows);
  } else if (!rows.isAllSelected()) {
    resultRowsBuffer = AlignedBuffer::allocate<bool>(numRows, pool);
    bits::copyBits(
        selectedRowBits,
        0,
        resultRowsBuffer->asMutable<uint64_t>(),
        0,
        numRows);
  }

  auto keys = BaseVector::create(keyType, maxNumKeys, pool);
  vector_size_t numKeys{0};
  // Channels of the output keys by hash.
  folly::F14FastMap<KeyAt, vector_size_t, KeyAtHasher> keyChannels;
  keyChannels.reserve(maxNumKeys);
  std::vector<VectorPtr> mapValues;
  mapValues.reserve(maxNumKeys);
  std::vector<BufferPtr> inMaps;
  inMaps.reserve(maxNumKeys);
  // Whether the values and in-map of each output key are still a map's,
  // which the result must not write into.
  std::vector<bool> isOriginal;
  isOriginal.reserve(maxNumKeys);
  vector_size_t numKeysKept{0};
  SelectivityVector rowsToCopy;
  for (size_t i = 0; i < numMaps; ++i) {
    auto& decoded = *maps[i];
    const auto& flatMap = *decoded.base()->asUnchecked<FlatMapVector>();
    auto mapRows = resultRowsBuffer;
    const auto* rawNulls = decoded.nulls(&rows);
    if (config.emptyForNull && rawNulls != nullptr) {
      mapRows = AlignedBuffer::allocate<bool>(numRows, pool);
      bits::andBits(
          mapRows->asMutable<uint64_t>(),
          selectedRowBits,
          rawNulls,
          0,
          numRows);
    }
    DecodedVector::DictionaryWrapping wrapping;
    if (!decoded.isIdentityMapping() && !decoded.isConstantMapping()) {
      wrapping = decoded.dictionaryWrapping(*pool, rows);
      if (mapRows != nullptr) {
        auto* rawIndices = wrapping.indices->asMutable<vector_size_t>();
        bits::forEachUnsetBit(
            mapRows->as<uint64_t>(), 0, numRows, [&](vector_size_t row) {
              rawIndices[row] = 0;
            });
      }
    }
    const auto& mapKeys = flatMap.distinctKeys();
    for (column_index_t currentChannel = 0; currentChannel < mapKeys->size();
         ++currentChannel) {
      if (decoded.isConstantMapping()) {
        const auto* rawInMap = flatMap.rawInMapsAt(currentChannel);
        if (rawInMap != nullptr &&
            !bits::isBitSet(rawInMap, decoded.index(0))) {
          continue;
        }
      }
      auto currentInMap = selectedInMap(
          decoded,
          flatMap,
          mapRows,
          wrapping.indices,
          currentChannel,
          numRows,
          pool);
      const auto* currentKeyRows = currentInMap != nullptr
          ? currentInMap->as<uint64_t>()
          : selectedRowBits;
      // Skips a key whose values no selected row uses, before looking it up or
      // touching its values, so that the result has no empty keys.
      if (bits::findFirstBit(currentKeyRows, rows.begin(), rows.end()) >= 0) {
        const auto& currentValues = flatMap.mapValuesAt(currentChannel);
        auto [it, inserted] = keyChannels.try_emplace(
            KeyAt{mapKeys.get(), currentChannel}, numKeys);

        // The concatenated maps share this key, so their values must be merged.
        if (!inserted) {
          const auto resultChannel = it->second;
          const auto* resultKeyRows = inMaps[resultChannel] != nullptr
              ? inMaps[resultChannel]->as<uint64_t>()
              : selectedRowBits;
          if (config.throwOnDuplicateKeys &&
              bits::hasIntersection(
                  resultKeyRows, currentKeyRows, rows.begin(), rows.end())) {
            VELOX_USER_FAIL(
                "Duplicate map key {} was found.",
                keys->toString(resultChannel));
          }

          isOriginal[resultChannel] = mergeChannel(
              decoded,
              wrapping,
              currentValues,
              std::move(currentInMap),
              isOriginal[resultChannel],
              mapValues[resultChannel],
              inMaps[resultChannel],
              rows,
              valueType,
              rowsToCopy,
              pool);
        } else {
          keys->copy(mapKeys.get(), numKeys++, currentChannel, 1);
          mapValues.push_back(
              wrapValues(decoded, wrapping, currentValues, rows, pool));
          inMaps.push_back(std::move(currentInMap));
          isOriginal.push_back(true);
          if (i == 0) {
            ++numKeysKept;
          }
        }
      }
    }
  }

  keys->resize(numKeys);
  const auto& firstFlatMap = *maps[0]->base()->asUnchecked<FlatMapVector>();
  const bool firstKeys = numKeysKept == firstFlatMap.numDistinctKeys() &&
      numKeys == firstFlatMap.numDistinctKeys();
  auto result = std::make_shared<FlatMapVector>(
      pool,
      outputType,
      std::move(resultNulls),
      numRows,
      firstKeys ? firstFlatMap.distinctKeys() : keys,
      std::move(mapValues),
      std::move(inMaps));
  if (firstKeys) {
    result->shareKeyIndex(firstFlatMap);
  }
  return result;
}

} // namespace

VectorPtr mapConcat(
    memory::MemoryPool* pool,
    const TypePtr& outputType,
    std::span<DecodedVector* const> inputs,
    const SelectivityVector& rows,
    const MapConcatConfig& config) {
  VELOX_CHECK_GT(inputs.size(), 0);
  VELOX_CHECK_LT(inputs.size(), std::numeric_limits<int8_t>::max());
  if (inputs[0]->base()->encoding() == VectorEncoding::Simple::FLAT_MAP) {
    return flatMapConcat(pool, outputType, inputs, rows, config);
  }
  const auto& keyType = outputType->asMap().keyType();
  // Route to UpdateMapRow<void> (BaseVector::equalValueAt/hashValueAt) when the
  // key type provides custom comparison, so that custom hash and equality are
  // respected for dedup.
  if (keyType->providesCustomComparison()) {
    return mapConcatImpl<TypeKind::ROW>(pool, outputType, inputs, rows, config);
  }
  return VELOX_DYNAMIC_TYPE_DISPATCH(
      mapConcatImpl, keyType->kind(), pool, outputType, inputs, rows, config);
}

} // namespace facebook::velox
