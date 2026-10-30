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
#include "velox/functions/sparksql/Map.h"

#include <velox/common/base/Exceptions.h>
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/sparksql/LastWin.h"
#include "velox/functions/sparksql/MapFromArrays.h"

// The description of the map function in Spark
// https://kontext.tech/article/586/spark-sql-map-functions
//
// Example:
// Select map(1,'a',2,'b',3,'c');
// map(1, a, 2, b, 3, c)
//
// Result:
// {1:"a",2:"b",3:"c"}

namespace facebook::velox::functions::sparksql {

namespace {

// Decoded inputs and output buffers threaded through the per-row build loop of
// map_from_arrays and transform_keys.
struct MapFromArraysBuffers {
  const DecodedVector* decodedKeys;
  const DecodedVector* decodedValues;
  const ArrayVectorBase* keysArray;
  const ArrayVectorBase* valuesArray;
  const DecodedVector* decodedKeyElements;
  bool throwExceptionOnDuplicateMapKeys;
  vector_size_t* rawOffsets;
  vector_size_t* rawSizes;
  vector_size_t* rawKeysIndices;
  vector_size_t* rawValuesIndices;
};

// Fills the offsets, sizes and key/value index buffers of 'buffers' for 'rows'
// and returns the number of map entries written.
vector_size_t buildMapEntries(
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const MapFromArraysBuffers& buffers) {
  LastWin<SparkMapKeyEquality> lastWin;
  vector_size_t numEntries{0};
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    const auto keysIndex = buffers.decodedKeys->index(row);
    const auto valuesIndex = buffers.decodedValues->index(row);
    const auto numKeys = buffers.keysArray->sizeAt(keysIndex);
    VELOX_USER_CHECK_EQ(
        numKeys,
        buffers.valuesArray->sizeAt(valuesIndex),
        "The key array and value array of MapData must have the same length.");

    const auto keysOffset = buffers.keysArray->offsetAt(keysIndex);
    const auto valuesOffset = buffers.valuesArray->offsetAt(valuesIndex);

    // A row that throws below leaves the entry count unchanged, so its size
    // stays zero.
    buffers.rawOffsets[row] = numEntries;
    for (const auto& entry : lastWin.resolve(
             *buffers.decodedKeyElements,
             keysOffset,
             numKeys,
             buffers.throwExceptionOnDuplicateMapKeys)) {
      buffers.rawKeysIndices[numEntries] = keysOffset + entry.key;
      buffers.rawValuesIndices[numEntries] = valuesOffset + entry.value;
      ++numEntries;
    }
    buffers.rawSizes[row] = numEntries - buffers.rawOffsets[row];
  });
  return numEntries;
}

// Builds the result buffers when the key array is constant across rows, so slot
// assignment is computed once and a row writes only its value indices.
vector_size_t buildMapEntriesForConstantKeys(
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const MapFromArraysBuffers& buffers) {
  const auto keysIndex = buffers.decodedKeys->index(rows.begin());
  const auto numKeys = buffers.keysArray->sizeAt(keysIndex);
  const auto keysOffset = buffers.keysArray->offsetAt(keysIndex);

  LastWin<SparkMapKeyEquality> lastWin;
  std::vector<KeptPositions> entries;
  try {
    entries = lastWin.resolve(
        *buffers.decodedKeyElements,
        keysOffset,
        numKeys,
        buffers.throwExceptionOnDuplicateMapKeys);
  } catch (const VeloxException&) {
    // A bad key fails every row, since they all share the key array.
    context.setErrors(rows, std::current_exception());
    return 0;
  }

  const auto numDistinctKeys = entries.size();
  vector_size_t numEntries{0};
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    const auto valuesIndex = buffers.decodedValues->index(row);
    VELOX_USER_CHECK_EQ(
        numKeys,
        buffers.valuesArray->sizeAt(valuesIndex),
        "The key array and value array of MapData must have the same length.");

    const auto valuesOffset = buffers.valuesArray->offsetAt(valuesIndex);
    buffers.rawOffsets[row] = numEntries;
    buffers.rawSizes[row] = numDistinctKeys;
    for (const auto& entry : entries) {
      buffers.rawKeysIndices[numEntries] = keysOffset + entry.key;
      buffers.rawValuesIndices[numEntries] = valuesOffset + entry.value;
      ++numEntries;
    }
  });
  return numEntries;
}

} // namespace

std::shared_ptr<MapVector> buildDeduplicatedMap(
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    vector_size_t size,
    const TypePtr& outputType,
    const DecodedVector& decodedKeys,
    const DecodedVector& decodedValues,
    const DecodedVector& decodedKeyElements,
    const VectorPtr& keyElements,
    const VectorPtr& valueElements,
    bool throwOnDuplicateKey) {
  const auto* keysArray = decodedKeys.base()->asUnchecked<ArrayVectorBase>();
  const auto* valuesArray =
      decodedValues.base()->asUnchecked<ArrayVectorBase>();

  // Deduplication can only shrink a row, so the total number of keys is an
  // upper bound on the number of entries in the result.
  vector_size_t maxNumEntries{0};
  rows.applyToSelected([&](vector_size_t row) {
    maxNumEntries += keysArray->sizeAt(decodedKeys.index(row));
  });

  auto* pool = context.pool();
  BufferPtr offsets = allocateOffsets(size, pool);
  BufferPtr sizes = allocateSizes(size, pool);
  BufferPtr keysIndices = allocateIndices(maxNumEntries, pool);
  BufferPtr valuesIndices = allocateIndices(maxNumEntries, pool);
  const MapFromArraysBuffers buffers{
      .decodedKeys = &decodedKeys,
      .decodedValues = &decodedValues,
      .keysArray = keysArray,
      .valuesArray = valuesArray,
      .decodedKeyElements = &decodedKeyElements,
      .throwExceptionOnDuplicateMapKeys = throwOnDuplicateKey,
      .rawOffsets = offsets->asMutable<vector_size_t>(),
      .rawSizes = sizes->asMutable<vector_size_t>(),
      .rawKeysIndices = keysIndices->asMutable<vector_size_t>(),
      .rawValuesIndices = valuesIndices->asMutable<vector_size_t>(),
  };
  const auto numEntries = decodedKeys.isConstantMapping()
      ? buildMapEntriesForConstantKeys(rows, context, buffers)
      : buildMapEntries(rows, context, buffers);

  return std::make_shared<MapVector>(
      pool,
      outputType,
      nullptr,
      size,
      std::move(offsets),
      std::move(sizes),
      BaseVector::wrapInDictionary(
          nullptr, std::move(keysIndices), numEntries, keyElements),
      BaseVector::wrapInDictionary(
          nullptr, std::move(valuesIndices), numEntries, valueElements));
}

void checkNullAndDuplicateKeys(
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const ArrayVectorBase& keysArray,
    const DecodedVector& decodedKeyElements) {
  // Only runs under EXCEPTION, so a repeat always throws.
  LastWin<SparkMapKeyEquality> lastWin;
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    lastWin.resolve(
        decodedKeyElements,
        keysArray.offsetAt(row),
        keysArray.sizeAt(row),
        /*throwOnDuplicateKey=*/true);
  });
}

namespace {

// Returns the key and value arguments as two array vectors holding one array
// per selected row, with one element per key-value pair. When every key
// argument is constant, the keys are a single array wrapped in a constant
// vector.
std::pair<VectorPtr, VectorPtr> toArrays(
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    vector_size_t mapSize,
    bool constantKeys,
    memory::MemoryPool* pool) {
  const auto& keyType = args[0]->type();
  const auto& valueType = args[1]->type();
  const vector_size_t numElements{rows.countSelected() * mapSize};

  BufferPtr offsets = allocateOffsets(rows.end(), pool);
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  BufferPtr sizes = allocateSizes(rows.end(), pool);
  auto* rawSizes = sizes->asMutable<vector_size_t>();
  vector_size_t offset{0};
  rows.applyToSelected([&](vector_size_t row) {
    rawOffsets[row] = offset;
    rawSizes[row] = mapSize;
    offset += mapSize;
  });

  auto keys =
      BaseVector::create(keyType, constantKeys ? mapSize : numElements, pool);
  auto values = BaseVector::create(valueType, numElements, pool);
  SelectivityVector targetRows(numElements, false);
  std::vector<vector_size_t> toSourceRow(numElements);
  for (vector_size_t i = 0; i < mapSize; ++i) {
    rows.applyToSelected([&](vector_size_t row) {
      targetRows.setValid(rawOffsets[row] + i, true);
      toSourceRow[rawOffsets[row] + i] = row;
    });
    targetRows.updateBounds();
    if (constantKeys) {
      keys->copy(args[i * 2].get(), i, rows.begin(), 1);
    } else {
      keys->copy(args[i * 2].get(), targetRows, toSourceRow.data());
    }
    values->copy(args[i * 2 + 1].get(), targetRows, toSourceRow.data());
    targetRows.clearAll();
  }

  VectorPtr keyArrays;
  if (constantKeys) {
    BufferPtr constantOffsets = allocateOffsets(1, pool);
    BufferPtr constantSizes = allocateSizes(1, pool);
    constantSizes->asMutable<vector_size_t>()[0] = mapSize;
    keyArrays = BaseVector::wrapInConstant(
        rows.end(),
        0,
        std::make_shared<ArrayVector>(
            pool,
            ARRAY(keyType),
            nullptr,
            1,
            std::move(constantOffsets),
            std::move(constantSizes),
            std::move(keys)));
  } else {
    keyArrays = std::make_shared<ArrayVector>(
        pool,
        ARRAY(keyType),
        nullptr,
        rows.end(),
        offsets,
        sizes,
        std::move(keys));
  }
  return {
      std::move(keyArrays),
      std::make_shared<ArrayVector>(
          pool,
          ARRAY(valueType),
          nullptr,
          rows.end(),
          std::move(offsets),
          std::move(sizes),
          std::move(values))};
}

class MapFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_USER_CHECK_GE(
        args.size(), 2, "Map function must take at least two arguments");
    VELOX_USER_CHECK_EQ(
        args.size() % 2,
        0,
        "Map function must take an even number of arguments");
    const vector_size_t mapSize = args.size() / 2;

    auto keyType = args[0]->type();
    auto valueType = args[1]->type();

    // Check key and value types
    for (auto i = 0; i < mapSize; i++) {
      VELOX_USER_CHECK(
          args[i * 2]->type()->equivalent(*keyType),
          "All the key arguments in Map function must be the same!");
      VELOX_USER_CHECK(
          args[i * 2 + 1]->type()->equivalent(*valueType),
          "All the value arguments in Map function must be the same!");
    }

    bool throwExceptionOnDuplicateMapKeys{false};
    if (auto* ctx = context.execCtx()->queryCtx()) {
      throwExceptionOnDuplicateMapKeys =
          ctx->queryConfig().throwExceptionOnDuplicateMapKeys();
    }

    bool constantKeys{true};
    for (vector_size_t i = 0; i < mapSize; ++i) {
      constantKeys = constantKeys && args[i * 2]->isConstantEncoding();
    }

    auto* pool = context.pool();
    auto [keys, values] = toArrays(rows, args, mapSize, constantKeys, pool);
    std::vector<VectorPtr> arrays{std::move(keys), std::move(values)};
    MapFromArraysFunction(throwExceptionOnDuplicateMapKeys)
        .apply(rows, arrays, outputType, context, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // One signature per pair count up to 10, then a variadic one for more.
    // K, V, K, V, ... -> map(K,V)
    std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
    constexpr int kNumberOfSignatures = 10;
    signatures.reserve(kNumberOfSignatures + 1);
    for (int i = 1; i <= kNumberOfSignatures; i++) {
      auto builder = exec::FunctionSignatureBuilder()
                         .knownTypeVariable("K")
                         .typeVariable("V")
                         .returnType("map(K,V)");
      for (int arg = 0; arg < i; arg++) {
        builder.argumentType("K").argumentType("V");
      }
      signatures.push_back(builder.build());
    }
    signatures.push_back(
        exec::FunctionSignatureBuilder()
            .knownTypeVariable("K")
            .typeVariable("V")
            .returnType("map(K,V)")
            .argumentType("K")
            .argumentType("V")
            .variableArity("any")
            .build());
    return signatures;
  }
};
} // namespace

void registerMapFunction(const std::string& name) {
  exec::registerVectorFunction(
      name,
      MapFunction::signatures(),
      std::make_unique<MapFunction>(),
      exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build());
}

} // namespace facebook::velox::functions::sparksql
