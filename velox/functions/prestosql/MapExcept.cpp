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
#include "velox/type/FloatingPointUtil.h"

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

// map_except(map(K, V), array(K)) -> map(K, V)
//
// Returns the map without the keys listed in the array; null elements of the
// array are ignored. The result reuses the input's keys and values, wrapped in
// a dictionary that skips the removed entries.
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

    if (decodedMap.base()->encoding() != VectorEncoding::Simple::MAP) {
      VELOX_UNSUPPORTED(
          "map_except not supported for encoding: {}",
          decodedMap.base()->encoding());
    }

    const auto& keyType = outputType->childAt(0);
    if (hasNativeComparison(keyType)) {
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          applyMapPrimitive,
          keyType->kind(),
          args[0],
          decodedMap,
          decodedKeys,
          rows,
          outputType,
          context,
          result);
    } else {
      applyMap<GenericExcludedKeys>(
          args[0], decodedMap, decodedKeys, rows, outputType, context, result);
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
