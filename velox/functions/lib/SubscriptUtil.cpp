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
#include <type_traits>
#include <utility>

#include "velox/common/memory/MemoryPool.h"
#include "velox/functions/lib/SubscriptUtil.h"
#include "velox/type/Type.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::velox::functions {

namespace {

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
    std::shared_ptr<LookupTableBase>& cachedLookupTablePtr,
    const SelectivityVector& rows,
    const VectorPtr& mapArg,
    const VectorPtr& indexArg,
    exec::EvalCtx& context) {
  static constexpr vector_size_t kMinCachedMapSize = 100;
  using TKey = typename TypeTraits<kind>::NativeType;

  LookupTable<kind>* typedLookupTable = nullptr;
  if (triggerCaching) {
    if (!cachedLookupTablePtr) {
      cachedLookupTablePtr =
          std::make_shared<LookupTable<kind>>(*context.pool());
    }

    typedLookupTable = cachedLookupTablePtr->typedTable<kind>();
  }

  auto* pool = context.pool();
  BufferPtr indices = allocateIndices(rows.end(), pool);
  auto rawIndices = indices->asMutable<vector_size_t>();

  // Create nulls for lazy initialization.
  NullsBuilder nullsBuilder(rows.end(), pool);

  // Get base MapVector.
  // TODO: Optimize the case when indices are identity.
  exec::LocalDecodedVector mapHolder(context, *mapArg, rows);
  auto decodedMap = mapHolder.get();
  auto baseMap = decodedMap->base()->as<MapVector>();
  auto mapIndices = decodedMap->indices();

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
        if (decodedMapKeys->valueAt<TKey>(offset) == searchKey) {
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

  return BaseVector::wrapInDictionary(
      nullsBuilder.build(), indices, rows.end(), baseMap->mapValues());
}

// A flat vector of map keys, an index into that vector and an index into
// the original map keys vector that may have encodings.
struct MapKey {
  const BaseVector* baseVector;
  const vector_size_t baseIndex;
  const vector_size_t index;

  size_t hash() const {
    return baseVector->hashValueAt(baseIndex);
  }

  bool operator==(const MapKey& other) const {
    return baseVector->equalValueAt(
        other.baseVector, baseIndex, other.baseIndex);
  }

  bool operator<(const MapKey& other) const {
    return baseVector->compare(other.baseVector, baseIndex, other.baseIndex) <
        0;
  }
};

struct MapKeyHasher {
  size_t operator()(const MapKey& key) const {
    return key.hash();
  }
};

VectorPtr applyMapComplexType(
    const SelectivityVector& rows,
    const VectorPtr& mapArg,
    const VectorPtr& indexArg,
    exec::EvalCtx& context) {
  auto* pool = context.pool();

  // Use indices with the mapValues wrapped in a dictionary vector.
  BufferPtr indices = allocateIndices(rows.end(), pool);
  auto rawIndices = indices->asMutable<vector_size_t>();

  // Create nulls for lazy initialization.
  NullsBuilder nullsBuilder(rows.end(), pool);

  // Get base MapVector.
  exec::LocalDecodedVector mapHolder(context, *mapArg, rows);
  auto decodedMap = mapHolder.get();
  auto baseMap = decodedMap->base()->as<MapVector>();
  auto mapIndices = decodedMap->indices();

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
  // encoded. Sort map keys, then use binary search.
  if (baseMap->size() == 1) {
    folly::F14FastSet<MapKey, MapKeyHasher> set;
    auto numKeys = rawSizes[0];
    set.reserve(numKeys * 1.3);
    for (auto i = 0; i < numKeys; ++i) {
      set.insert(MapKey{mapKeysBase, mapKeysIndices[i], i});
    }
    rows.applyToSelected([&](vector_size_t row) {
      VELOX_CHECK_EQ(0, mapIndices[row]);

      auto searchIndex = searchIndices[row];
      auto it = set.find(MapKey{searchBase, searchIndex, row});
      if (it != set.end()) {
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

  return BaseVector::wrapInDictionary(
      nullsBuilder.build(), indices, rows.end(), baseMap->mapValues());
}

} // namespace

VectorPtr MapSubscript::applyMap(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args,
    exec::EvalCtx& context) const {
  auto& mapArg = args[0];
  auto& indexArg = args[1];

  // Ensure map key type and second argument are the same.
  VELOX_CHECK(mapArg->type()->childAt(0)->equivalent(*indexArg->type()));

  if (indexArg->type()->isPrimitiveType()) {
    bool triggerCaching = shouldTriggerCaching(mapArg);

    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        applyMapTyped,
        indexArg->typeKind(),
        triggerCaching,
        const_cast<std::shared_ptr<LookupTableBase>&>(lookupTable_),
        rows,
        mapArg,
        indexArg,
        context);
  } else {
    return applyMapComplexType(rows, mapArg, indexArg, context);
  }
}

namespace {
std::exception_ptr makeZeroSubscriptError() {
  try {
    VELOX_USER_FAIL("SQL array indices start at 1");
  } catch (const std::exception&) {
    return std::current_exception();
  }
}

std::exception_ptr makeBadSubscriptError() {
  try {
    VELOX_USER_FAIL("Array subscript out of bounds.");
  } catch (const std::exception&) {
    return std::current_exception();
  }
}

std::exception_ptr makeNegativeSubscriptError() {
  try {
    VELOX_USER_FAIL("Array subscript is negative.");
  } catch (const std::exception&) {
    return std::current_exception();
  }
}
} // namespace

const std::exception_ptr& zeroSubscriptError() {
  static std::exception_ptr error = makeZeroSubscriptError();
  return error;
}

const std::exception_ptr& badSubscriptError() {
  static std::exception_ptr error = makeBadSubscriptError();
  return error;
}

const std::exception_ptr& negativeSubscriptError() {
  static std::exception_ptr error = makeNegativeSubscriptError();
  return error;
}

} // namespace facebook::velox::functions
