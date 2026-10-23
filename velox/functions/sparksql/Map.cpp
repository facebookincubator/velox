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
#include <fmt/core.h>

#include <velox/common/base/Exceptions.h>
#include "velox/expression/DecodedArgs.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"
#include "velox/functions/sparksql/LastWin.h"
#include "velox/vector/VectorTypeUtils.h"

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

// Throws on a null key or a repeated key. Used by the zero-copy path, which
// builds no entry indices.
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

// Can only take the fast path if keys and values have an equal number of arrays
// and the offsets and sizes of these arrays match 1:1. The map must be well
// formed for all elements, also ones not in 'rows' in apply(). This is because
// canonicalize() will touch all elements in any case.
bool canTakeFastPath(
    const ArrayVector& keys,
    const ArrayVector& values,
    const SelectivityVector& rows) {
  VELOX_CHECK_GE(keys.size(), rows.end());
  VELOX_CHECK_GE(values.size(), rows.end());
  if (keys.size() != values.size()) {
    return false;
  }
  for (vector_size_t row = 0; row < keys.size(); ++row) {
    if (keys.isNullAt(row)) {
      continue;
    }

    if (values.isNullAt(row)) {
      return false;
    }

    if (keys.offsetAt(row) != values.offsetAt(row) ||
        keys.sizeAt(row) != values.sizeAt(row)) {
      return false;
    }
  }
  return true;
}

// Implements Spark's map_from_arrays(array(K), array(V)) -> map(K,V). Entries
// are inserted in array order. A repeated key raises DUPLICATED_MAP_KEY, or
// under LAST_WIN overwrites the value of its first occurrence in place, so the
// key keeps that first position.
class MapFromArraysFunction : public exec::VectorFunction {
 public:
  explicit MapFromArraysFunction(bool throwExceptionOnDuplicateMapKeys)
      : throwExceptionOnDuplicateMapKeys_{throwExceptionOnDuplicateMapKeys} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2);

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto* decodedKeys = decodedArgs.at(0);
    auto* decodedValues = decodedArgs.at(1);

    auto* keysArray = decodedKeys->base()->as<ArrayVector>();
    auto* valuesArray = decodedValues->base()->as<ArrayVector>();
    const auto& keysElements = keysArray->elements();
    const auto keyElementRows = toElementRows(
        keysElements->size(),
        rows,
        keysArray,
        decodedKeys->nulls(&rows),
        decodedKeys->indices());
    exec::LocalDecodedVector decodedKeyElements(
        context, *keysElements, keyElementRows);

    // Under EXCEPTION no row can shrink, so the result can reference the input
    // arrays instead of building entry indices.
    if (throwExceptionOnDuplicateMapKeys_ && decodedKeys->isIdentityMapping() &&
        decodedValues->isIdentityMapping() &&
        canTakeFastPath(*keysArray, *valuesArray, rows)) {
      exec::LocalSelectivityVector remainingRows(context, rows);
      checkNullAndDuplicateKeys(
          *remainingRows, context, *keysArray, *decodedKeyElements);
      context.deselectErrors(*remainingRows);

      auto mapVector = std::make_shared<MapVector>(
          context.pool(),
          outputType,
          keysArray->nulls(),
          rows.end(),
          keysArray->offsets(),
          keysArray->sizes(),
          keysElements,
          valuesArray->elements());
      context.moveOrCopyResult(mapVector, *remainingRows, result);
      return;
    }

    // Deduplication can only shrink a row, so the total number of keys is an
    // upper bound on the number of entries in the result.
    vector_size_t maxNumEntries{0};
    rows.applyToSelected([&](vector_size_t row) {
      maxNumEntries += keysArray->sizeAt(decodedKeys->index(row));
    });

    auto* pool = context.pool();
    BufferPtr offsets = allocateOffsets(rows.end(), pool);
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    BufferPtr sizes = allocateSizes(rows.end(), pool);
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    BufferPtr keysIndices = allocateIndices(maxNumEntries, pool);
    auto* rawKeysIndices = keysIndices->asMutable<vector_size_t>();
    BufferPtr valuesIndices = allocateIndices(maxNumEntries, pool);
    auto* rawValuesIndices = valuesIndices->asMutable<vector_size_t>();

    exec::LocalSelectivityVector remainingRows(context, rows);

    const MapFromArraysBuffers buffers{
        .decodedKeys = decodedKeys,
        .decodedValues = decodedValues,
        .keysArray = keysArray,
        .valuesArray = valuesArray,
        .decodedKeyElements = decodedKeyElements.get(),
        .throwExceptionOnDuplicateMapKeys = throwExceptionOnDuplicateMapKeys_,
        .rawOffsets = rawOffsets,
        .rawSizes = rawSizes,
        .rawKeysIndices = rawKeysIndices,
        .rawValuesIndices = rawValuesIndices,
    };
    const auto numEntries = decodedKeys->isConstantMapping()
        ? buildMapEntriesForConstantKeys(*remainingRows, context, buffers)
        : buildMapEntries(*remainingRows, context, buffers);

    context.deselectErrors(*remainingRows);

    auto mapVector = std::make_shared<MapVector>(
        context.pool(),
        outputType,
        nullptr,
        rows.end(),
        std::move(offsets),
        std::move(sizes),
        BaseVector::wrapInDictionary(
            nullptr, std::move(keysIndices), numEntries, keysElements),
        BaseVector::wrapInDictionary(
            nullptr,
            std::move(valuesIndices),
            numEntries,
            valuesArray->elements()));
    context.moveOrCopyResult(mapVector, *remainingRows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // array(K), array(V) -> map(K,V)
    return {
        exec::FunctionSignatureBuilder()
            .knownTypeVariable("K")
            .typeVariable("V")
            .returnType("map(K,V)")
            .argumentType("array(K)")
            .argumentType("array(V)")
            .build(),
    };
  }

 private:
  const bool throwExceptionOnDuplicateMapKeys_;
};

// Returns the key or value arguments as an array vector holding one array
// per selected row, with one element per key-value pair. When the arguments
// are constant, returns a single array wrapped in a constant vector.
VectorPtr toArrays(
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    vector_size_t first,
    vector_size_t mapSize,
    bool constant,
    memory::MemoryPool* pool) {
  const auto& type = args[first]->type();
  const vector_size_t numArrays = constant ? 1 : rows.end();
  const vector_size_t numElements =
      (constant ? 1 : rows.countSelected()) * mapSize;
  auto elements = BaseVector::create(type, numElements, pool);
  BufferPtr offsets = allocateOffsets(numArrays, pool);
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  BufferPtr sizes = allocateSizes(numArrays, pool);
  auto* rawSizes = sizes->asMutable<vector_size_t>();

  if (constant) {
    for (vector_size_t i = 0; i < mapSize; ++i) {
      elements->copy(args[first + i * 2].get(), i, rows.begin(), 1);
    }
    rawSizes[0] = mapSize;
    return BaseVector::wrapInConstant(
        rows.end(),
        0,
        std::make_shared<ArrayVector>(
            pool,
            ARRAY(type),
            nullptr,
            1,
            std::move(offsets),
            std::move(sizes),
            std::move(elements)));
  }

  vector_size_t offset{0};
  rows.applyToSelected([&](vector_size_t row) {
    rawOffsets[row] = offset;
    rawSizes[row] = mapSize;
    offset += mapSize;
  });
  SelectivityVector targetRows(numElements, false);
  std::vector<vector_size_t> toSourceRow(numElements);
  for (vector_size_t i = 0; i < mapSize; ++i) {
    rows.applyToSelected([&](vector_size_t row) {
      targetRows.setValid(rawOffsets[row] + i, true);
      toSourceRow[rawOffsets[row] + i] = row;
    });
    targetRows.updateBounds();
    elements->copy(args[first + i * 2].get(), targetRows, toSourceRow.data());
    targetRows.clearAll();
  }
  return std::make_shared<ArrayVector>(
      pool,
      ARRAY(type),
      nullptr,
      numArrays,
      std::move(offsets),
      std::move(sizes),
      std::move(elements));
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
    std::vector<VectorPtr> arrays{
        toArrays(rows, args, 0, mapSize, constantKeys, pool),
        toArrays(rows, args, 1, mapSize, false, pool),
    };
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

std::shared_ptr<exec::VectorFunction> makeMapFromArrays(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
    const core::QueryConfig& config) {
  return std::make_shared<MapFromArraysFunction>(
      config.throwExceptionOnDuplicateMapKeys());
}

// Implements Spark's transform_keys(map(K1,V), function(K1,V,K2)) ->
// map(K2,V). Pairs each key the lambda returns with the original value, in
// entry order. A repeated key raises DUPLICATED_MAP_KEY, or under LAST_WIN
// overwrites the value of its first occurrence in place, so the key keeps that
// first position.
//
// Default null behavior does not apply: the lambda may capture other columns,
// and a null capture does not make the result null. A null map still yields a
// null map.
class TransformKeysFunction : public exec::VectorFunction {
 public:
  explicit TransformKeysFunction(bool throwExceptionOnDuplicateMapKeys)
      : throwExceptionOnDuplicateMapKeys_{throwExceptionOnDuplicateMapKeys} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2, "transform_keys requires two arguments");

    exec::LocalDecodedVector mapDecoder(context, *args[0], rows);
    auto flatMap = flattenMap(rows, args[0], *mapDecoder.get());

    const auto numKeys = flatMap->mapKeys()->size();
    const std::vector<VectorPtr> lambdaArgs = {
        flatMap->mapKeys(), flatMap->mapValues()};
    SelectivityVector validRowsInReusedResult =
        toElementRows<MapVector>(numKeys, rows, flatMap.get());

    VectorPtr transformedKeys;
    applyLambdaToElements<MapVector>(
        args[1],
        rows,
        numKeys,
        flatMap,
        lambdaArgs,
        validRowsInReusedResult,
        context,
        transformedKeys);

    // A null map has no keys to check and stays null in the result.
    exec::LocalSelectivityVector remainingRows(context, rows);
    context.deselectErrors(*remainingRows);
    if (flatMap->mayHaveNulls()) {
      remainingRows->deselectNulls(
          flatMap->rawNulls(), rows.begin(), rows.end());
    }
    exec::LocalDecodedVector decodedKeys(
        context,
        *transformedKeys,
        toElementRows<MapVector>(numKeys, *remainingRows, flatMap.get()));

    // Under EXCEPTION no row can shrink, so the result keeps the offsets and
    // sizes of the input map.
    if (throwExceptionOnDuplicateMapKeys_) {
      checkNullAndDuplicateKeys(
          *remainingRows, context, *flatMap, *decodedKeys);
      context.deselectErrors(*remainingRows);

      auto mapVector = std::make_shared<MapVector>(
          context.pool(),
          outputType,
          addNullsForUnselectedRows(flatMap, *remainingRows),
          rows.end(),
          flatMap->offsets(),
          flatMap->sizes(),
          transformedKeys,
          flatMap->mapValues());
      context.moveOrCopyResult(mapVector, rows, result);
      return;
    }

    // Deduplication can only shrink a row, so the number of keys is an upper
    // bound on the number of entries in the result.
    vector_size_t maxNumEntries{0};
    remainingRows->applyToSelected(
        [&](vector_size_t row) { maxNumEntries += flatMap->sizeAt(row); });

    auto* pool = context.pool();
    BufferPtr offsets = allocateOffsets(rows.end(), pool);
    BufferPtr sizes = allocateSizes(rows.end(), pool);
    BufferPtr keysIndices = allocateIndices(maxNumEntries, pool);
    BufferPtr valuesIndices = allocateIndices(maxNumEntries, pool);

    // The offsets and sizes of the map locate both the transformed keys and the
    // values.
    exec::LocalDecodedVector decodedFlatMap(context, *flatMap, *remainingRows);
    const MapFromArraysBuffers buffers{
        .decodedKeys = decodedFlatMap.get(),
        .decodedValues = decodedFlatMap.get(),
        .keysArray = flatMap.get(),
        .valuesArray = flatMap.get(),
        .decodedKeyElements = decodedKeys.get(),
        .throwExceptionOnDuplicateMapKeys = false,
        .rawOffsets = offsets->asMutable<vector_size_t>(),
        .rawSizes = sizes->asMutable<vector_size_t>(),
        .rawKeysIndices = keysIndices->asMutable<vector_size_t>(),
        .rawValuesIndices = valuesIndices->asMutable<vector_size_t>(),
    };
    const auto numEntries = buildMapEntries(*remainingRows, context, buffers);
    context.deselectErrors(*remainingRows);

    auto mapVector = std::make_shared<MapVector>(
        pool,
        outputType,
        addNullsForUnselectedRows(flatMap, *remainingRows),
        rows.end(),
        std::move(offsets),
        std::move(sizes),
        BaseVector::wrapInDictionary(
            nullptr, std::move(keysIndices), numEntries, transformedKeys),
        BaseVector::wrapInDictionary(
            nullptr,
            std::move(valuesIndices),
            numEntries,
            flatMap->mapValues()));
    context.moveOrCopyResult(mapVector, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // map(K1,V), function(K1,V,K2) -> map(K2,V)
    // K2 is orderable, which rejects a key type that is or contains a map, as
    // Spark does.
    return {
        exec::FunctionSignatureBuilder()
            .typeVariable("K1")
            .orderableTypeVariable("K2")
            .typeVariable("V")
            .returnType("map(K2,V)")
            .argumentType("map(K1,V)")
            .argumentType("function(K1,V,K2)")
            .build(),
    };
  }

 private:
  const bool throwExceptionOnDuplicateMapKeys_;
};

std::shared_ptr<exec::VectorFunction> makeTransformKeys(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
    const core::QueryConfig& config) {
  return std::make_shared<TransformKeysFunction>(
      config.throwExceptionOnDuplicateMapKeys());
}
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_map,
    MapFunction::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<MapFunction>());

// Default null behavior applies: Spark's MapFromArrays is NullIntolerant, so a
// null key or value array yields a null map.
VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_map_from_arrays,
    MapFromArraysFunction::signatures(),
    makeMapFromArrays);

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION_WITH_METADATA(
    udf_transform_keys,
    TransformKeysFunction::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    makeTransformKeys);
} // namespace facebook::velox::functions::sparksql
