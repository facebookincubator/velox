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
#include <map>

#include <velox/common/base/Exceptions.h>
#include "velox/expression/DecodedArgs.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
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

constexpr const char* kNullKeyErrorMessage = "Cannot use null as map key!";

constexpr const char* kDuplicateKeyErrorMessage =
    "Duplicate map key ({}) was found.";

void setKeysAndValuesResult(
    vector_size_t mapSize,
    std::vector<VectorPtr>& args,
    const VectorPtr& keysResult,
    const VectorPtr& valuesResult,
    const int32_t* offsets,
    const int32_t* sizes,
    exec::EvalCtx& context,
    const SelectivityVector& rows) {
  exec::LocalDecodedVector decoded(context);
  SelectivityVector targetRows(keysResult->size(), false);
  std::vector<vector_size_t> targetIdx(rows.size(), 0);
  std::vector<vector_size_t> toSourceRow(keysResult->size());
  for (vector_size_t i = 0; i < mapSize; i++) {
    decoded.get()->decode(*args[i * 2], rows);
    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      VELOX_USER_CHECK(!decoded->isNullAt(row), kNullKeyErrorMessage);
      const auto offset = offsets[row];
      const auto size = sizes[row];
      bool duplicate = false;
      if (size < mapSize) {
        // Check if the current key at position i is duplicated in any later
        // position. When a duplicate is found, mark this occurrence as
        // duplicate and skip further checks. This implements the LAST_WIN
        // policy where only the last occurrence of any key is kept.
        for (vector_size_t j = i + 1; j < mapSize; j++) {
          if (args[i * 2]->equalValueAt(args[j * 2].get(), row, row)) {
            duplicate = true;
            break;
          }
        }
      }
      if (size == mapSize || !duplicate) {
        targetRows.setValid(offset + targetIdx[row], true);
        toSourceRow[offset + targetIdx[row]] = row;
        targetIdx[row]++;
      }
    });
    targetRows.updateBounds();
    keysResult->copy(args[i * 2].get(), targetRows, toSourceRow.data());
    valuesResult->copy(args[i * 2 + 1].get(), targetRows, toSourceRow.data());
    targetRows.clearAll();
  }
}

class MapFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_USER_CHECK(
        args.size() >= 2 && args.size() % 2 == 0,
        "Map function must take an even number of arguments");
    auto mapSize = args.size() / 2;

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

    // Initializing input
    context.ensureWritable(
        rows, std::make_shared<MapType>(keyType, valueType), result);

    auto mapResult = result->as<MapVector>();
    auto sizes = mapResult->mutableSizes(rows.end());
    auto rawSizes = sizes->asMutable<int32_t>();
    auto offsets = mapResult->mutableOffsets(rows.end());
    auto rawOffsets = offsets->asMutable<int32_t>();

    // Setting keys and value elements
    auto& keysResult = mapResult->mapKeys();
    auto& valuesResult = mapResult->mapValues();
    const auto baseOffset =
        std::max<vector_size_t>(keysResult->size(), valuesResult->size());
    vector_size_t offset = baseOffset;

    bool throwExceptionOnDuplicateMapKeys = false;
    if (auto* ctx = context.execCtx()->queryCtx()) {
      throwExceptionOnDuplicateMapKeys =
          ctx->queryConfig().throwExceptionOnDuplicateMapKeys();
    }

    // Check for duplicate keys and set size & offsets.
    rows.applyToSelected([&](vector_size_t row) {
      vector_size_t duplicateCnt = 0;
      for (vector_size_t i = 0; i < mapSize; i++) {
        for (vector_size_t j = i + 1; j < mapSize; j++) {
          if (args[i * 2]->equalValueAt(args[j * 2].get(), row, row)) {
            if (throwExceptionOnDuplicateMapKeys) {
              auto duplicateKey = args[i * 2]->toString(row);
              VELOX_USER_FAIL(kDuplicateKeyErrorMessage, duplicateKey);
            }
            // The key at position i is superseded by a later occurrence and
            // will be dropped by setKeysAndValuesResult (LAST_WIN). Count this
            // occurrence once and stop; otherwise a key repeated N times would
            // be counted C(N,2) times, under-sizing the result map (0 for N=3,
            // negative for N>=4) and causing an out-of-bounds write.
            duplicateCnt++;
            break;
          }
        }
      }
      rawSizes[row] = mapSize - duplicateCnt;
      rawOffsets[row] = offset;
      offset += mapSize - duplicateCnt;
    });

    keysResult->resize(offset);
    valuesResult->resize(offset);
    setKeysAndValuesResult(
        mapSize,
        args,
        keysResult,
        valuesResult,
        rawOffsets,
        rawSizes,
        context,
        rows);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // Support up to 10 key-value pairs (20 arguments) for MAP() function.
    // array(K), array(V) -> map(K,V)
    std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
    constexpr int kNumberOfSignatures = 10;
    signatures.reserve(kNumberOfSignatures);
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
    return signatures;
  }
};

// Decoded inputs and output buffers threaded through the per-row build loop of
// map_from_arrays.
struct MapFromArraysBuffers {
  const DecodedVector* decodedKeys;
  const DecodedVector* decodedValues;
  const ArrayVector* keysArray;
  const ArrayVector* valuesArray;
  const BaseVector* keysElements;
  bool throwExceptionOnDuplicateMapKeys;
  vector_size_t* rawOffsets;
  vector_size_t* rawSizes;
  vector_size_t* rawKeysIndices;
  vector_size_t* rawValuesIndices;
};

// Assigns each distinct key of a row the index of the entry slot claimed by the
// key's first occurrence, ordering keys by BaseVector::compare.
class OrderedKeySlots {
 public:
  explicit OrderedKeySlots(const BaseVector& keys)
      : slotByKey_{KeyComparator{&keys}} {}

  void startRow() {
    slotByKey_.clear();
    numSlots_ = 0;
  }

  std::optional<vector_size_t> findOrInsert(vector_size_t elementIndex) {
    const auto [it, inserted] = slotByKey_.emplace(elementIndex, numSlots_);
    if (!inserted) {
      return it->second;
    }
    ++numSlots_;
    return std::nullopt;
  }

 private:
  // The default compare flags use the same kNullAsValue handling as
  // BaseVector::equalValueAt, so equality here matches the dedup contract.
  struct KeyComparator {
    const BaseVector* keys;

    bool operator()(vector_size_t lhs, vector_size_t rhs) const {
      return keys->compare(keys, lhs, rhs) < 0;
    }
  };

  std::map<vector_size_t, vector_size_t, KeyComparator> slotByKey_;
  vector_size_t numSlots_{0};
};

// The per-key checks every build path shares: rejects a null key, and a
// repeated one under EXCEPTION.
class KeyChecker {
 public:
  KeyChecker(
      const BaseVector& keysElements,
      bool throwExceptionOnDuplicateMapKeys)
      : keysElements_{keysElements},
        mayHaveNullKeys_{keysElements.mayHaveNulls()},
        throwExceptionOnDuplicateMapKeys_{throwExceptionOnDuplicateMapKeys} {}

  // Returns the slot a repeated key already owns, or nullopt when the key is
  // new and has claimed the next slot.
  std::optional<vector_size_t> findOrInsert(
      OrderedKeySlots& keySlots,
      vector_size_t elementIndex) const {
    if (mayHaveNullKeys_) {
      VELOX_USER_CHECK(
          !keysElements_.isNullAt(elementIndex), kNullKeyErrorMessage);
    }

    const auto slot = keySlots.findOrInsert(elementIndex);
    if (slot.has_value()) {
      VELOX_USER_CHECK(
          !throwExceptionOnDuplicateMapKeys_,
          kDuplicateKeyErrorMessage,
          keysElements_.toString(elementIndex));
    }
    return slot;
  }

 private:
  const BaseVector& keysElements_;
  const bool mayHaveNullKeys_;
  const bool throwExceptionOnDuplicateMapKeys_;
};

// Fills the offsets, sizes and key/value index buffers of 'buffers' for 'rows'
// and returns the number of map entries written.
vector_size_t buildMapEntries(
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const MapFromArraysBuffers& buffers) {
  OrderedKeySlots keySlots{*buffers.keysElements};
  const KeyChecker keyChecker{
      *buffers.keysElements, buffers.throwExceptionOnDuplicateMapKeys};
  vector_size_t numEntries = 0;
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

    // A row that throws below leaves 'numEntries' unchanged, so the slots it
    // partially filled are reused by the next row and its size stays zero.
    buffers.rawOffsets[row] = numEntries;
    keySlots.startRow();

    vector_size_t numDistinctKeys = 0;
    for (vector_size_t i = 0; i < numKeys; ++i) {
      const auto elementIndex = keysOffset + i;
      if (const auto slot = keyChecker.findOrInsert(keySlots, elementIndex)) {
        buffers.rawValuesIndices[numEntries + slot.value()] = valuesOffset + i;
        continue;
      }

      buffers.rawKeysIndices[numEntries + numDistinctKeys] = elementIndex;
      buffers.rawValuesIndices[numEntries + numDistinctKeys] = valuesOffset + i;
      ++numDistinctKeys;
    }

    buffers.rawSizes[row] = numDistinctKeys;
    numEntries += numDistinctKeys;
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

  // Slot occupied by the key at each position of the constant array, and the
  // key element index occupying each slot.
  std::vector<vector_size_t> slotOfPosition(numKeys);
  std::vector<vector_size_t> slotKeyIndices(numKeys);
  vector_size_t numDistinctKeys = 0;
  const KeyChecker keyChecker{
      *buffers.keysElements, buffers.throwExceptionOnDuplicateMapKeys};
  OrderedKeySlots keySlots{*buffers.keysElements};
  keySlots.startRow();
  try {
    for (vector_size_t i = 0; i < numKeys; ++i) {
      const auto elementIndex = keysOffset + i;
      if (const auto slot = keyChecker.findOrInsert(keySlots, elementIndex)) {
        slotOfPosition[i] = slot.value();
        continue;
      }

      slotKeyIndices[numDistinctKeys] = elementIndex;
      slotOfPosition[i] = numDistinctKeys;
      ++numDistinctKeys;
    }
  } catch (const VeloxException&) {
    // A bad key fails every row, since they all share the key array.
    context.setErrors(rows, std::current_exception());
    return 0;
  }

  vector_size_t numEntries = 0;
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    const auto valuesIndex = buffers.decodedValues->index(row);
    VELOX_USER_CHECK_EQ(
        numKeys,
        buffers.valuesArray->sizeAt(valuesIndex),
        "The key array and value array of MapData must have the same length.");

    const auto valuesOffset = buffers.valuesArray->offsetAt(valuesIndex);
    buffers.rawOffsets[row] = numEntries;
    buffers.rawSizes[row] = numDistinctKeys;
    for (vector_size_t slot = 0; slot < numDistinctKeys; ++slot) {
      buffers.rawKeysIndices[numEntries + slot] = slotKeyIndices[slot];
    }
    // Input order, so under LAST_WIN a repeated key's later value overwrites
    // the earlier one.
    for (vector_size_t i = 0; i < numKeys; ++i) {
      buffers.rawValuesIndices[numEntries + slotOfPosition[i]] =
          valuesOffset + i;
    }
    numEntries += numDistinctKeys;
  });
  return numEntries;
}

// Throws on a null key or a repeated key. Used by the zero-copy path, which
// builds no entry indices.
void checkNullAndDuplicateKeys(
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const ArrayVector& keysArray) {
  const auto& keysElements = keysArray.elements();
  // Only runs under EXCEPTION, so a repeat always throws.
  const KeyChecker keyChecker{
      *keysElements, /*throwExceptionOnDuplicateMapKeys=*/true};
  OrderedKeySlots keySlots{*keysElements};
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    const auto numKeys = keysArray.sizeAt(row);
    const auto keysOffset = keysArray.offsetAt(row);
    keySlots.startRow();

    for (vector_size_t i = 0; i < numKeys; ++i) {
      keyChecker.findOrInsert(keySlots, keysOffset + i);
    }
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

    // Under EXCEPTION no row can shrink, so the result can reference the input
    // arrays instead of building entry indices.
    if (throwExceptionOnDuplicateMapKeys_ && decodedKeys->isIdentityMapping() &&
        decodedValues->isIdentityMapping() &&
        canTakeFastPath(*keysArray, *valuesArray, rows)) {
      exec::LocalSelectivityVector remainingRows(context, rows);
      checkNullAndDuplicateKeys(*remainingRows, context, *keysArray);
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
    vector_size_t maxNumEntries = 0;
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
        .keysElements = keysElements.get(),
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

std::shared_ptr<exec::VectorFunction> makeMapFromArrays(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
    const core::QueryConfig& config) {
  return std::make_shared<MapFromArraysFunction>(
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
} // namespace facebook::velox::functions::sparksql
