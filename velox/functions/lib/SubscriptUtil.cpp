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

#include <cstdint>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>

#include "velox/common/memory/MemoryPool.h"
#include "velox/functions/lib/SubscriptUtil.h"
#include "velox/type/Type.h"
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::velox::functions {

namespace {

template <typename T>
inline bool isPrimitiveEqual(const T& lhs, const T& rhs) {
  if constexpr (std::is_floating_point_v<T>) {
    return util::floating_point::NaNAwareEquals<T>{}(lhs, rhs);
  } else {
    return lhs == rhs;
  }
}

template <TypeKind Kind>
struct SimpleType {
  using type = typename TypeTraits<Kind>::NativeType;
};

template <>
struct SimpleType<TypeKind::VARBINARY> {
  using type = Varbinary;
};

template <>
struct SimpleType<TypeKind::VARCHAR> {
  using type = Varchar;
};

/// Decode arguments and transform result into a dictionaryVector where the
/// dictionary maintains a mapping from a given row to the index of the input
/// map value vector. This allows us to ensure that element_at is zero-copy.
template <TypeKind kind>
VectorPtr applyMapTyped(
    bool triggerCaching,
    std::shared_ptr<detail::LookupTableBase>& cachedLookupTablePtr,
    const SelectivityVector& rows,
    const DecodedVector& decodedMap,
    const VectorPtr& indexArg,
    exec::EvalCtx& context) {
  static constexpr vector_size_t kMinCachedMapSize = 100;
  using TKey = typename TypeTraits<kind>::NativeType;

  detail::LookupTable<TKey>* typedLookupTable = nullptr;
  if (triggerCaching) {
    if (!cachedLookupTablePtr) {
      cachedLookupTablePtr =
          std::make_shared<detail::LookupTable<TKey>>(*context.pool());
    }

    typedLookupTable = cachedLookupTablePtr->typedTable<TKey>();
  }

  auto* pool = context.pool();
  BufferPtr indices = allocateIndices(rows.end(), pool);
  auto rawIndices = indices->asMutable<vector_size_t>();

  // Create nulls for lazy initialization.
  NullsBuilder nullsBuilder(rows.end(), pool);

  // Get base MapVector.
  // TODO: Optimize the case when indices are identity.
  auto baseMap = decodedMap.base()->as<MapVector>();
  auto mapIndices = decodedMap.indices();

  // Get map keys.
  auto mapKeys = baseMap->mapKeys();
  exec::LocalSelectivityVector allElementRows(context, mapKeys->size());
  allElementRows->setAll();
  exec::LocalDecodedVector mapKeysHolder(context, *mapKeys, *allElementRows);
  auto decodedMapKeys = mapKeysHolder.get();

  // Get index vector (second argument).
  exec::LocalDecodedVector indexHolder(context, *indexArg, rows);
  auto decodedIndices = indexHolder.get();

  auto rawSizes = baseMap->rawSizes();
  auto rawOffsets = baseMap->rawOffsets();

  // Lambda that does the search for a key, for each row.
  auto processRow = [&](vector_size_t row, TKey searchKey) {
    size_t mapIndex = mapIndices[row];
    auto size = rawSizes[mapIndex];
    size_t offsetStart = rawOffsets[mapIndex];
    size_t offsetEnd = offsetStart + size;
    bool found = false;

    if (triggerCaching && size >= kMinCachedMapSize) {
      VELOX_DCHECK_NOT_NULL(typedLookupTable);

      // Create map for mapIndex if not created.
      if (!typedLookupTable->containsMapAtIndex(mapIndex)) {
        typedLookupTable->ensureMapAtIndex(mapIndex);
        // Materialize the map at index row.
        auto& map = typedLookupTable->getMapAtIndex(mapIndex);
        for (size_t offset = offsetStart; offset < offsetEnd; ++offset) {
          map.emplace(decodedMapKeys->valueAt<TKey>(offset), offset);
        }
      }

      auto& map = typedLookupTable->getMapAtIndex(mapIndex);

      // Fast lookup.
      auto value = map.find(searchKey);
      if (value != map.end()) {
        rawIndices[row] = value->second;
        found = true;
      }

    } else {
      // Search map without caching.
      for (size_t offset = offsetStart; offset < offsetEnd; ++offset) {
        if (isPrimitiveEqual<TKey>(
                decodedMapKeys->valueAt<TKey>(offset), searchKey)) {
          rawIndices[row] = offset;
          found = true;
          break;
        }
      }
    }

    // Handle NULLs.
    if (!found) {
      nullsBuilder.setNull(row);
    }
  };

  // When second argument ("at") is a constant.
  if (decodedIndices->isConstantMapping()) {
    auto searchKey = decodedIndices->valueAt<TKey>(0);
    rows.applyToSelected(
        [&](vector_size_t row) { processRow(row, searchKey); });
  }

  // When the second argument ("at") is also a variable vector.
  else {
    rows.applyToSelected([&](vector_size_t row) {
      auto searchKey = decodedIndices->valueAt<TKey>(row);
      processRow(row, searchKey);
    });
  }

  // Subscript into empty maps always returns NULLs. Check added at the end to
  // ensure user error checks for indices are not skipped.
  if (baseMap->mapValues()->size() == 0) {
    return BaseVector::createNullConstant(
        baseMap->mapValues()->type(), rows.end(), context.pool());
  }

  // Subscript can pass along very large elements vectors that can hold onto
  // memory and copy operations on them can further put memory pressure. We
  // try to flatten them if the dictionary layer is much smaller than the
  // elements vector.
  return BaseVector::wrapInDictionary(
      nullsBuilder.build(),
      indices,
      rows.end(),
      baseMap->mapValues(),
      true /*flattenIfRedundant*/);
}

// Returns the result for a key that is the same in every row: the key's
// values, with nulls in the rows whose map lacks it, as projected by the flat
// map without copying.
VectorPtr applyFlatMapConstantKey(
    const SelectivityVector& rows,
    DecodedVector& decodedMap,
    const FlatMapVector& flatMap,
    const VectorPtr& key,
    exec::EvalCtx& context) {
  // A key the map does not have is null in every row.
  auto values = flatMap.projectKey(key, 0);
  if (values == nullptr) {
    return BaseVector::createNullConstant(
        flatMap.valueType(), rows.end(), context.pool());
  }

  // Unwrapped map: the rows of the key's values line up with the result.
  if (decodedMap.isIdentityMapping()) {
    return values;
  }

  // Wrapped map: the key's values take the wrapping of the map.
  return decodedMap.wrap(std::move(values), *context.pool(), rows);
}

// A range to copy from the map values of the key at `channel`.
struct KeyCopyRange {
  column_index_t channel;
  BaseVector::CopyRange range;
};

// Returns one copy range per row whose map contains its key, and nulls the
// other rows of `result`. `isMapNullAt(row)` tells whether the map of `row` is
// null; the caller picks it once, so the loop has no other check.
template <typename TIsMapNullAt>
std::vector<KeyCopyRange> collectKeyRanges(
    const SelectivityVector& rows,
    const DecodedVector& decodedMap,
    const FlatMapVector& flatMap,
    const std::vector<column_index_t>& channels,
    const TIsMapNullAt& isMapNullAt,
    BaseVector& result) {
  std::vector<KeyCopyRange> keyRanges;
  keyRanges.reserve(rows.countSelected());
  uint64_t* rawNulls = nullptr;
  rows.applyToSelected([&](vector_size_t row) {
    const auto channel = channels[row];
    const auto index = decodedMap.index(row);
    if (channel != FlatMapVector::kNoChannel && !isMapNullAt(row) &&
        flatMap.isInMap(channel, index)) {
      keyRanges.push_back({channel, {index, row, 1}});
    } else {
      if (rawNulls == nullptr) {
        rawNulls = result.mutableRawNulls();
      }
      bits::setNull(rawNulls, row);
    }
  });
  return keyRanges;
}

// Copies `keyRanges` into `result`. The ranges come in row order, so the keys
// they read from are interleaved; a counting sort first groups them by key, so
// that each key's map values are copied from in a single call instead of once
// per row.
void copyGroupedByKey(
    const FlatMapVector& flatMap,
    const std::vector<KeyCopyRange>& keyRanges,
    BaseVector& result) {
  // Counting sort by key: count each key's ranges, then prefix-sum.
  const auto numKeys = flatMap.numDistinctKeys();
  std::vector<vector_size_t> offsets(numKeys, 0);
  for (const auto& keyRange : keyRanges) {
    ++offsets[keyRange.channel];
  }
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());

  // Place ranges back to front, so offsets[c] ends at key c's first range.
  std::vector<BaseVector::CopyRange> grouped(keyRanges.size());
  for (auto i = keyRanges.size(); i-- > 0;) {
    grouped[--offsets[keyRanges[i].channel]] = keyRanges[i].range;
  }

  // Copy key c's ranges, grouped[offsets[c], offsets[c + 1]), in one call.
  for (column_index_t channel = 0; channel < numKeys; ++channel) {
    const vector_size_t end = channel + 1 < numKeys
        ? offsets[channel + 1]
        : static_cast<vector_size_t>(grouped.size());
    if (offsets[channel] < end) {
      result.copyRanges(
          flatMap.mapValuesAt(channel).get(),
          folly::Range(
              grouped.data() + offsets[channel], grouped.data() + end));
    }
  }
}

// Returns the result for a key that varies per row: copies the value of each
// row from the map values of its key, or null if the row does not contain it.
VectorPtr applyFlatMapPerRowKey(
    const SelectivityVector& rows,
    const DecodedVector& decodedMap,
    const FlatMapVector& flatMap,
    const VectorPtr& keys,
    exec::EvalCtx& context) {
  exec::LocalDecodedVector keysDecoder(context, *keys, rows);
  const auto channels = flatMap.getKeyChannels(rows, *keysDecoder.get());
  auto result =
      BaseVector::create(flatMap.valueType(), rows.end(), context.pool());

  // One range per row whose map contains its key; the other rows are null.
  const auto keyRanges = decodedMap.mayHaveNulls()
      ? collectKeyRanges(
            rows,
            decodedMap,
            flatMap,
            channels,
            [&](vector_size_t row) { return decodedMap.isNullAt(row); },
            *result)
      : collectKeyRanges(
            rows,
            decodedMap,
            flatMap,
            channels,
            [](vector_size_t) { return false; },
            *result);

  copyGroupedByKey(flatMap, keyRanges, *result);
  return result;
}

// Applies subscript to a FlatMapVector, which keeps the values of each key in
// their own vector.
VectorPtr applyFlatMap(
    const SelectivityVector& rows,
    DecodedVector& decodedMap,
    const VectorPtr& elementAt,
    exec::EvalCtx& context) {
  const auto& flatMap = *decodedMap.base()->as<FlatMapVector>();
  if (elementAt->isConstantEncoding()) {
    return applyFlatMapConstantKey(
        rows, decodedMap, flatMap, elementAt, context);
  }
  return applyFlatMapPerRowKey(rows, decodedMap, flatMap, elementAt, context);
}

VectorPtr applyMapComplexType(
    const SelectivityVector& rows,
    const DecodedVector& decodedMap,
    const VectorPtr& indexArg,
    exec::EvalCtx& context,
    bool triggerCaching,
    std::shared_ptr<detail::LookupTableBase>& cachedLookupTablePtr) {
  auto* pool = context.pool();

  // Use indices with the mapValues wrapped in a dictionary vector.
  BufferPtr indices = allocateIndices(rows.end(), pool);
  auto rawIndices = indices->asMutable<vector_size_t>();

  // Create nulls for lazy initialization.
  NullsBuilder nullsBuilder(rows.end(), pool);

  // Get base MapVector
  auto baseMap = decodedMap.base()->as<MapVector>();
  auto mapIndices = decodedMap.indices();

  // Get map keys.
  auto mapKeys = baseMap->mapKeys();
  exec::LocalSelectivityVector allElementRows(context, mapKeys->size());
  allElementRows->setAll();
  exec::LocalDecodedVector mapKeysHolder(context, *mapKeys, *allElementRows);
  auto mapKeysDecoded = mapKeysHolder.get();
  auto mapKeysBase = mapKeysDecoded->base();
  auto mapKeysIndices = mapKeysDecoded->indices();

  // Get index vector (second argument).
  exec::LocalDecodedVector indexHolder(context, *indexArg, rows);
  auto decodedIndices = indexHolder.get();
  auto searchBase = decodedIndices->base();
  auto searchIndices = decodedIndices->indices();

  auto rawSizes = baseMap->rawSizes();
  auto rawOffsets = baseMap->rawOffsets();

  // Fast path for the case of a single map. It may be constant or dictionary
  // encoded. Use hash table for quick search.
  if (baseMap->size() == 1) {
    detail::ComplexKeyHashMap hashMap{detail::MapKeyAllocator(*pool)};
    detail::ComplexKeyHashMap* hashMapPtr = &hashMap;

    if (triggerCaching) {
      if (!cachedLookupTablePtr) {
        cachedLookupTablePtr =
            std::make_shared<detail::LookupTable<void>>(*context.pool());
      }

      detail::LookupTable<void>* typedLookupTable =
          cachedLookupTablePtr->typedTable<void>();

      static constexpr vector_size_t kMapIndex = 0;

      if (!typedLookupTable->containsMapAtIndex(kMapIndex)) {
        typedLookupTable->ensureMapAtIndex(kMapIndex);
      }

      auto& map = typedLookupTable->getMapAtIndex(kMapIndex);
      hashMapPtr = &map;
    }

    if (hashMapPtr->empty()) {
      auto numKeys = rawSizes[0];
      hashMapPtr->reserve(numKeys * 1.3);
      for (auto i = 0; i < numKeys; ++i) {
        const vector_size_t offset = rawOffsets[0] + i;
        hashMapPtr->insert(
            detail::MapKey{mapKeysBase, mapKeysIndices[offset], offset});
      }
    }

    rows.applyToSelected([&](vector_size_t row) {
      VELOX_CHECK_EQ(0, mapIndices[row]);

      auto searchIndex = searchIndices[row];
      auto it = hashMapPtr->find(detail::MapKey{searchBase, searchIndex, row});
      if (it != hashMapPtr->end()) {
        rawIndices[row] = it->index;
      } else {
        nullsBuilder.setNull(row);
      }
    });

  } else {
    // Search the key in each row.
    rows.applyToSelected([&](vector_size_t row) {
      size_t mapIndex = mapIndices[row];
      size_t size = rawSizes[mapIndex];
      size_t offset = rawOffsets[mapIndex];

      bool found = false;
      auto searchIndex = searchIndices[row];
      for (auto i = 0; i < size; i++) {
        if (mapKeysBase->equalValueAt(
                searchBase, mapKeysIndices[offset + i], searchIndex)) {
          rawIndices[row] = offset + i;
          found = true;
          break;
        }
      }

      if (!found) {
        nullsBuilder.setNull(row);
      }
    });
  }

  // Subscript into empty maps always returns NULLs. Check added at the end to
  // ensure user error checks for indices are not skipped.
  if (baseMap->mapValues()->size() == 0) {
    return BaseVector::createNullConstant(
        baseMap->mapValues()->type(), rows.end(), context.pool());
  }

  // Subscript can pass along very large elements vectors that can hold onto
  // memory and copy operations on them can further put memory pressure. We
  // try to flatten them if the dictionary layer is much smaller than the
  // elements vector.
  return BaseVector::wrapInDictionary(
      nullsBuilder.build(),
      indices,
      rows.end(),
      baseMap->mapValues(),
      true /*flattenIfRedundant*/);
}

} // namespace

namespace detail {

VectorPtr MapSubscript::applyMap(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args,
    exec::EvalCtx& context) const {
  auto& mapArg = args[0];
  auto& indexArg = args[1];

  // Ensure map key type and second argument are the same.
  VELOX_CHECK(mapArg->type()->childAt(0)->equivalent(*indexArg->type()));

  // Short-circuit for FlatMapVector encoding. FlatMapVector doesn't need to
  // distinguish between primitive and complex types (where the former requires
  // a type dispatch).
  exec::LocalDecodedVector mapDecoder(context, *mapArg, rows);
  auto decodedMap = mapDecoder.get();
  if (decodedMap->base()->encoding() == VectorEncoding::Simple::FLAT_MAP) {
    return applyFlatMap(rows, *decodedMap, indexArg, context);
  }

  // Regular map encoding with two paths for complex and primitive types.
  bool triggerCaching = shouldTriggerCaching(mapArg);
  if (indexArg->type()->isPrimitiveType() &&
      !indexArg->type()->providesCustomComparison()) {
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        applyMapTyped,
        indexArg->typeKind(),
        triggerCaching,
        lookupTable_,
        rows,
        *decodedMap,
        indexArg,
        context);
  } else {
    // We use applyMapComplexType when the key type is complex, but also when it
    // provides custom comparison operators because the main difference between
    // applyMapComplexType and applyTyped is that applyMapComplexType calls the
    // Vector's equalValueAt method, which calls the Types custom comparison
    // operator internally.
    return applyMapComplexType(
        rows, *decodedMap, indexArg, context, triggerCaching, lookupTable_);
  }
}
} // namespace detail

} // namespace facebook::velox::functions
