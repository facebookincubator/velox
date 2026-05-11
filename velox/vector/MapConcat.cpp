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

#include "velox/vector/DecodedVector.h"

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
  void forEachEntry(F&& func) {
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
  void forEachEntry(F&& func) {
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

} // namespace

MapVectorPtr mapConcat(
    memory::MemoryPool* pool,
    const TypePtr& outputType,
    std::span<DecodedVector* const> inputs,
    const SelectivityVector& rows,
    const MapConcatConfig& config) {
  VELOX_CHECK_GT(inputs.size(), 0);
  VELOX_CHECK_LT(inputs.size(), std::numeric_limits<int8_t>::max());
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
