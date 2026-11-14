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

#include <folly/container/F14Set.h>

#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/RowsTranslationUtil.h"
#include "velox/type/FloatingPointUtil.h"
#include "velox/vector/FlatMapVector.h"

namespace facebook::velox::functions {
namespace {

// Excluded keys of primitive type, held in a hash set of values.
template <TypeKind kKind>
class PrimitiveExcludedKeys {
  using T = typename TypeTraits<kKind>::NativeType;

 public:
  PrimitiveExcludedKeys(
      const BaseVector& mapKeys,
      const BaseVector& elements,
      exec::EvalCtx& context)
      : mapKeysDecoder_{context}, elementsDecoder_{context} {
    mapKeysDecoder_.get()->decode(mapKeys);
    elementsDecoder_.get()->decode(elements);
    mapKeys_ = mapKeysDecoder_.get();
    elements_ = elementsDecoder_.get();
  }

  void collect(vector_size_t begin, vector_size_t end) {
    keys_.clear();
    for (auto i = begin; i < end; ++i) {
      if (!elements_->isNullAt(i)) {
        keys_.insert(elements_->valueAt<T>(i));
      }
    }
  }

  size_t size() const {
    return keys_.size();
  }

  bool contains(vector_size_t mapKeyIndex) const {
    return keys_.contains(mapKeys_->valueAt<T>(mapKeyIndex));
  }

 private:
  exec::LocalDecodedVector mapKeysDecoder_;
  exec::LocalDecodedVector elementsDecoder_;
  const DecodedVector* mapKeys_;
  const DecodedVector* elements_;
  util::floating_point::HashSetNaNAware<T> keys_;
};

// Excluded keys of any other type, held in a hash set of element indices that
// hashes and compares the vectors' values.
class GenericExcludedKeys {
 public:
  GenericExcludedKeys(
      const BaseVector& mapKeys,
      const BaseVector& elements,
      exec::EvalCtx& /*context*/)
      : mapKeys_{mapKeys}, elements_{elements} {}

  void collect(vector_size_t begin, vector_size_t end) {
    keys_.clear();
    for (auto i = begin; i < end; ++i) {
      if (!elements_.isNullAt(i)) {
        keys_.insert({&elements_, i});
      }
    }
  }

  size_t size() const {
    return keys_.size();
  }

  bool contains(vector_size_t mapKeyIndex) const {
    return keys_.contains({&mapKeys_, mapKeyIndex});
  }

 private:
  struct Key {
    const BaseVector* vector;
    vector_size_t index;
  };

  struct KeyHash {
    size_t operator()(const Key& key) const {
      return key.vector->hashValueAt(key.index);
    }
  };

  struct KeyEqual {
    bool operator()(const Key& left, const Key& right) const {
      return left.vector->equalValueAt(right.vector, left.index, right.index);
    }
  };

  const BaseVector& mapKeys_;
  const BaseVector& elements_;
  folly::F14FastSet<Key, KeyHash, KeyEqual> keys_;
};

// Returns true for the key types whose values are compared natively. Others,
// including types with custom comparison, are compared through the vector.
bool hasNativeComparison(const TypePtr& keyType) {
  if (keyType->providesCustomComparison()) {
    return false;
  }
  switch (keyType->kind()) { // NOLINT(clang-diagnostic-switch-enum)
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
      return true;
    default:
      return false;
  }
}

VectorPtr wrapEntries(
    const BufferPtr& indices,
    vector_size_t size,
    const VectorPtr& entries,
    memory::MemoryPool* pool) {
  if (size == 0) {
    return BaseVector::create(entries->type(), 0, pool);
  }
  return BaseVector::wrapInDictionary(
      nullptr, indices, size, entries, /*flattenIfRedundant=*/true);
}

// Returns 'input' as is when no row loses a key.
template <typename TExcludedKeys>
void applyMap(
    const VectorPtr& input,
    const DecodedVector& decodedMap,
    const DecodedVector& decodedKeys,
    const SelectivityVector& rows,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) {
  const auto& map = *decodedMap.base()->asUnchecked<MapVector>();
  const auto* rawOffsets = map.rawOffsets();
  const auto* rawSizes = map.rawSizes();
  const auto& keysArray = *decodedKeys.base()->asUnchecked<ArrayVector>();

  TExcludedKeys excluded(*map.mapKeys(), *keysArray.elements(), context);
  const auto collectExcluded = [&](vector_size_t row) {
    const auto keysRow = decodedKeys.index(row);
    const auto offset = keysArray.offsetAt(keysRow);
    excluded.collect(offset, offset + keysArray.sizeAt(keysRow));
  };
  const bool constantKeys = decodedKeys.isConstantMapping();
  if (constantKeys) {
    collectExcluded(rows.begin());
  }

  size_t numEntries{0};
  rows.applyToSelected([&](vector_size_t row) {
    numEntries += rawSizes[decodedMap.index(row)];
  });
  VELOX_USER_CHECK_LE(
      numEntries,
      std::numeric_limits<vector_size_t>::max(),
      "map_except input too large: {} entries exceeds maximum {}",
      numEntries,
      std::numeric_limits<vector_size_t>::max());

  auto* pool = context.pool();
  auto offsets = allocateOffsets(rows.end(), pool);
  auto sizes = allocateSizes(rows.end(), pool);
  auto indices = allocateIndices(static_cast<vector_size_t>(numEntries), pool);
  auto* rawNewOffsets = offsets->asMutable<vector_size_t>();
  auto* rawNewSizes = sizes->asMutable<vector_size_t>();
  auto* rawIndices = indices->asMutable<vector_size_t>();

  vector_size_t numKept{0};
  bool anyRemoved{false};
  rows.applyToSelected([&](vector_size_t row) {
    const auto mapRow = decodedMap.index(row);
    const auto offset = rawOffsets[mapRow];
    const auto size = rawSizes[mapRow];
    rawNewOffsets[row] = numKept;
    if (!constantKeys && size > 0) {
      collectExcluded(row);
    }
    // Map keys are unique, so once every excluded key is found the rest of
    // the row is kept without lookups.
    auto toExclude = size > 0 ? excluded.size() : 0;
    for (auto i = offset; i < offset + size; ++i) {
      if (toExclude > 0 && excluded.contains(i)) {
        --toExclude;
        continue;
      }
      rawIndices[numKept++] = i;
    }
    rawNewSizes[row] = numKept - rawNewOffsets[row];
    anyRemoved |= rawNewSizes[row] != size;
  });

  if (!anyRemoved) {
    context.moveOrCopyResult(input, rows, result);
    return;
  }

  auto localResult = std::make_shared<MapVector>(
      pool,
      outputType,
      nullptr,
      rows.end(),
      std::move(offsets),
      std::move(sizes),
      wrapEntries(indices, numKept, map.mapKeys(), pool),
      wrapEntries(indices, numKept, map.mapValues(), pool));
  context.moveOrCopyResult(localResult, rows, result);
}

template <TypeKind kKind>
void applyMapPrimitive(
    const VectorPtr& input,
    const DecodedVector& decodedMap,
    const DecodedVector& decodedKeys,
    const SelectivityVector& rows,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) {
  applyMap<PrimitiveExcludedKeys<kKind>>(
      input, decodedMap, decodedKeys, rows, outputType, context, result);
}

// Removes the keys from a MapVector, comparing them natively or through the
// vector depending on the key type.
void applyMapForKeyType(
    const VectorPtr& input,
    const DecodedVector& decodedMap,
    const DecodedVector& decodedKeys,
    const SelectivityVector& rows,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) {
  const auto& keyType = outputType->childAt(0);
  if (hasNativeComparison(keyType)) {
    VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        applyMapPrimitive,
        keyType->kind(),
        input,
        decodedMap,
        decodedKeys,
        rows,
        outputType,
        context,
        result);
  } else {
    applyMap<GenericExcludedKeys>(
        input, decodedMap, decodedKeys, rows, outputType, context, result);
  }
}

// Returns the flat map channel of each element of 'keys' in 'elementRows',
// indexed like the elements, or kNoChannel for null keys and keys the flat map
// does not have.
std::vector<column_index_t> findChannels(
    const FlatMapVector& flatMap,
    const ArrayVector& keys,
    const SelectivityVector& elementRows,
    exec::EvalCtx& context) {
  exec::LocalDecodedVector decodedElements(
      context, *keys.elements(), elementRows);
  return flatMap.getKeyChannels(elementRows, *decodedElements.get());
}

// Copies 'decoded's row mapping for 'rows', leaving every other row on the zero
// allocateIndices() starts from.
BufferPtr copyIndices(
    const DecodedVector& decoded,
    const SelectivityVector& rows,
    memory::MemoryPool* pool) {
  auto indices = allocateIndices(decoded.size(), pool);
  auto* rawIndices = indices->asMutable<vector_size_t>();
  rows.applyToSelected(
      [&](vector_size_t row) { rawIndices[row] = decoded.index(row); });
  return indices;
}

// True when every selected row reads the flat map row of the same index, as
// for an unwrapped flat map or a dictionary that only adds nulls. The flat
// map's in-map buffers can then be shared.
bool sharesRowSpace(
    const DecodedVector& decodedMap,
    const SelectivityVector& rows) {
  if (decodedMap.isIdentityMapping()) {
    return true;
  }
  if (decodedMap.isConstantMapping()) {
    return false;
  }
  return rows.testSelected(
      [&](vector_size_t row) { return decodedMap.index(row) == row; });
}

// Handles a constant list of excluded keys. Every flat map row drops the same
// channels, so the result keeps the input's map values and in-map buffers for
// the others and drops the excluded channels entirely. A wrapped flat map has
// the result wrapped the same way.
void applyFlatMapConstantKeys(
    const VectorPtr& map,
    const DecodedVector& decodedMap,
    bool sameRows,
    const DecodedVector& decodedKeys,
    const FlatMapVector& flatMap,
    const SelectivityVector& rows,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) {
  const auto& keysArray = *decodedKeys.base()->asUnchecked<ArrayVector>();
  const auto keysIndex = decodedKeys.index(rows.begin());
  const auto begin = keysArray.offsetAt(keysIndex);
  const auto end = begin + keysArray.sizeAt(keysIndex);
  SelectivityVector elementRows(end, false);
  elementRows.setValidRange(begin, end, true);
  elementRows.updateBounds();
  const auto channels = findChannels(flatMap, keysArray, elementRows, context);

  const auto numDistinctKeys = flatMap.numDistinctKeys();
  std::vector<bool> excluded(numDistinctKeys, false);
  auto numKept = numDistinctKeys;
  for (auto i = begin; i < end; ++i) {
    const auto channel = channels[i];
    if (channel != FlatMapVector::kNoChannel && !excluded[channel]) {
      excluded[channel] = true;
      --numKept;
    }
  }

  if (numKept == numDistinctKeys) {
    context.moveOrCopyResult(map, rows, result);
    return;
  }

  VectorPtr distinctKeys;
  std::vector<VectorPtr> mapValues;
  std::vector<BufferPtr> inMaps;
  // With every key removed, the result is a flat map without keys.
  if (numKept > 0) {
    auto keyIndices = allocateIndices(numKept, context.pool());
    auto* rawKeyIndices = keyIndices->asMutable<vector_size_t>();
    mapValues.reserve(numKept);
    inMaps.reserve(numKept);
    for (column_index_t channel = 0; channel < numDistinctKeys; ++channel) {
      if (excluded[channel]) {
        continue;
      }
      rawKeyIndices[mapValues.size()] = static_cast<vector_size_t>(channel);
      mapValues.push_back(flatMap.mapValuesAt(channel));
      // A missing in-map buffer means the key is present in every row, the
      // same convention FlatMapVector::isInMap() follows.
      if (channel < flatMap.inMaps().size()) {
        inMaps.push_back(flatMap.inMaps()[channel]);
      } else {
        inMaps.emplace_back(nullptr);
      }
    }
    distinctKeys = BaseVector::wrapInDictionary(
        BufferPtr(nullptr),
        std::move(keyIndices),
        numKept,
        flatMap.distinctKeys());
  }

  VectorPtr localResult = std::make_shared<FlatMapVector>(
      context.pool(),
      outputType,
      flatMap.nulls(),
      flatMap.size(),
      std::move(distinctKeys),
      std::move(mapValues),
      std::move(inMaps));
  if (!sameRows) {
    localResult = BaseVector::wrapInDictionary(
        BufferPtr(nullptr),
        copyIndices(decodedMap, rows, context.pool()),
        decodedMap.size(),
        std::move(localResult));
  }

  context.moveOrCopyResult(localResult, rows, result);
}

// Handles flat map inputs. With per-row excluded keys, finding what to remove
// visits only the (channel, row) pairs an excluded key hits. In the flat map's
// row space, only the in-map buffers of those channels are copied; a wrapped
// flat map has every in-map buffer rebuilt in the output row space.
void applyFlatMap(
    const VectorPtr& map,
    const DecodedVector& decodedMap,
    const DecodedVector& decodedKeys,
    const SelectivityVector& rows,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) {
  const auto& flatMap = *decodedMap.base()->asUnchecked<FlatMapVector>();

  // A flat map without keys has nothing to remove, and its key index cannot
  // be built.
  if (flatMap.numDistinctKeys() == 0) {
    context.moveOrCopyResult(map, rows, result);
    return;
  }

  // The flat map key index does not match NaN keys to each other. Flat maps
  // rarely have floating point keys, so these are converted to a MapVector.
  const auto keyKind = flatMap.keyType()->kind();
  if (keyKind == TypeKind::REAL || keyKind == TypeKind::DOUBLE) {
    VectorPtr converted = flatMap.toMapVector();
    if (!decodedMap.isIdentityMapping()) {
      converted = BaseVector::wrapInDictionary(
          BufferPtr(nullptr),
          copyIndices(decodedMap, rows, context.pool()),
          decodedMap.size(),
          std::move(converted));
    }
    exec::LocalDecodedVector mapDecoder(context, *converted, rows);
    applyMapForKeyType(
        map, *mapDecoder.get(), decodedKeys, rows, outputType, context, result);
    return;
  }

  const bool sameRows = sharesRowSpace(decodedMap, rows);
  if (decodedKeys.isConstantMapping()) {
    applyFlatMapConstantKeys(
        map,
        decodedMap,
        sameRows,
        decodedKeys,
        flatMap,
        rows,
        outputType,
        context,
        result);
    return;
  }

  const auto numDistinctKeys = flatMap.numDistinctKeys();
  // In the flat map's row space, in-map buffers stay shareable. Otherwise they
  // are rebuilt in the output row space.
  const auto numRows = sameRows ? flatMap.size() : decodedMap.size();
  const auto& keysArray = *decodedKeys.base()->asUnchecked<ArrayVector>();
  // map_except has default null behavior, so no row in 'rows' has a null
  // array.
  const auto channels = findChannels(
      flatMap,
      keysArray,
      toElementRows(
          keysArray.elements()->size(),
          rows,
          &keysArray,
          nullptr,
          decodedKeys.indices()),
      context);

  // In the flat map's row space, the in-map buffer of a channel that loses a
  // key is copied once and its bits are cleared in place. Otherwise, the rows
  // that lose the key are recorded and applied while rebuilding.
  std::vector<BufferPtr> changed(numDistinctKeys);
  bool anyRemoved = false;
  rows.applyToSelected([&](vector_size_t row) {
    const auto mapRow = decodedMap.index(row);
    const auto keysIndex = decodedKeys.index(row);
    const auto begin = keysArray.offsetAt(keysIndex);
    const auto end = begin + keysArray.sizeAt(keysIndex);
    for (auto i = begin; i < end; ++i) {
      const auto channel = channels[i];
      if (channel == FlatMapVector::kNoChannel ||
          !flatMap.isInMap(channel, mapRow)) {
        continue;
      }
      BufferPtr& channelBits = changed[channel];
      if (channelBits == nullptr) {
        if (!sameRows) {
          channelBits =
              AlignedBuffer::allocate<bool>(numRows, context.pool(), false);
        } else if (
            channel < flatMap.inMaps().size() &&
            flatMap.inMaps()[channel] != nullptr) {
          channelBits =
              AlignedBuffer::copy(context.pool(), flatMap.inMaps()[channel]);
        } else {
          channelBits =
              AlignedBuffer::allocate<bool>(numRows, context.pool(), true);
        }
      }
      if (sameRows) {
        bits::clearBit(channelBits->asMutable<uint64_t>(), row);
      } else {
        bits::setBit(channelBits->asMutable<uint64_t>(), row);
      }
      anyRemoved = true;
    }
  });

  if (!anyRemoved) {
    context.moveOrCopyResult(map, rows, result);
    return;
  }

  std::vector<VectorPtr> mapValues;
  std::vector<BufferPtr> inMaps;
  BufferPtr nulls;
  if (sameRows) {
    mapValues = flatMap.mapValues();
    inMaps = flatMap.inMaps();
    inMaps.resize(numDistinctKeys);
    for (column_index_t channel = 0; channel < numDistinctKeys; ++channel) {
      if (changed[channel] != nullptr) {
        inMaps[channel] = std::move(changed[channel]);
      }
    }
    nulls = flatMap.nulls();
  } else {
    const auto indices = copyIndices(decodedMap, rows, context.pool());
    mapValues.reserve(numDistinctKeys);
    inMaps.reserve(numDistinctKeys);
    for (column_index_t channel = 0; channel < numDistinctKeys; ++channel) {
      const auto& values = flatMap.mapValuesAt(channel);
      mapValues.push_back(
          values == nullptr
              ? nullptr
              : BaseVector::wrapInDictionary(
                    BufferPtr(nullptr), indices, numRows, values));

      const auto* baseInMap = flatMap.rawInMapsAt(channel);
      const auto* rawRemoved = changed[channel] != nullptr
          ? changed[channel]->as<uint64_t>()
          : nullptr;
      if (baseInMap == nullptr && rawRemoved == nullptr) {
        inMaps.emplace_back(nullptr);
        continue;
      }
      auto inMap =
          AlignedBuffer::allocate<bool>(numRows, context.pool(), false);
      auto* rawInMap = inMap->asMutable<uint64_t>();
      rows.applyToSelected([&](vector_size_t row) {
        if ((baseInMap == nullptr ||
             bits::isBitSet(baseInMap, decodedMap.index(row))) &&
            (rawRemoved == nullptr || !bits::isBitSet(rawRemoved, row))) {
          bits::setBit(rawInMap, row);
        }
      });
      inMaps.push_back(std::move(inMap));
    }
    // Top level nulls are left to the engine: map_except is a default-null
    // function, so no row in 'rows' has a null map.
  }

  auto localResult = std::make_shared<FlatMapVector>(
      context.pool(),
      outputType,
      std::move(nulls),
      numRows,
      flatMap.distinctKeys(),
      std::move(mapValues),
      std::move(inMaps));
  localResult->shareKeyIndex(flatMap);

  context.moveOrCopyResult(localResult, rows, result);
}

// map_except(map(K, V), array(K)) -> map(K, V)
//
// Returns the map without the keys listed in the array; null elements of the
// array are ignored. For a MapVector, the result reuses the input's keys and
// values, wrapped in a dictionary that skips the removed entries. For a
// FlatMapVector, it reuses the input's channels and clears the in-map bits of
// the removed entries.
class MapExceptFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2);
    exec::DecodedArgs decodedArgs(rows, args, context);
    const auto& decodedMap = *decodedArgs.at(0);
    const auto& decodedKeys = *decodedArgs.at(1);

    const auto encoding = decodedMap.base()->encoding();
    if (encoding == VectorEncoding::Simple::MAP) {
      applyMapForKeyType(
          args[0], decodedMap, decodedKeys, rows, outputType, context, result);
    } else if (encoding == VectorEncoding::Simple::FLAT_MAP) {
      applyFlatMap(
          args[0], decodedMap, decodedKeys, rows, outputType, context, result);
    } else {
      VELOX_UNSUPPORTED("map_except not supported for encoding: {}", encoding);
    }
  }
};

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  // map(K,V), array(K) -> map(K,V)
  return {exec::FunctionSignatureBuilder()
              .comparableTypeVariable("K")
              .typeVariable("V")
              .returnType("map(K,V)")
              .argumentType("map(K,V)")
              .argumentType("array(K)")
              .build()};
}

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_map_except,
    signatures(),
    std::make_unique<MapExceptFunction>());

} // namespace facebook::velox::functions
