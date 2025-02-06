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
#include <common/base/Exceptions.h>
#include <core/QueryConfig.h>
#include <vector/ComplexVector.h>
#include <vector/DecodedVector.h>
#include <vector/TypeAliases.h>
#include <unordered_set>

#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"

// Returns a map created using the given key/value arrays.
// See documentation at https://spark.apache.org/docs/latest/api/sql/#map_from_arrays
//
// Example:
// Select map_from_arrays(array(1,2,3), array('a','b','c'));
//
// Result:
// {1:"a",2:"b",3:"c"}

namespace facebook::velox::functions::sparksql {
namespace {

template <core::SparkMapKeyDedupPolicy Policy>
class MapFromArraysFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedKeys = decodedArgs.at(0);
    auto decodedValues = decodedArgs.at(1);

    auto keyIndices = decodedKeys->indices();
    auto valueIndices = decodedValues->indices();

    auto keysArray = decodedKeys->base()->as<ArrayVector>();
    auto valuesArray = decodedValues->base()->as<ArrayVector>();

    auto keysElements = keysArray->elements();

    vector_size_t originalElements = 0;
    rows.applyToSelected([&](vector_size_t row) {
      originalElements += keysArray->sizeAt(keyIndices[row]);
    });

    BufferPtr offsets = allocateOffsets(rows.end(), context.pool());
    auto rawOffsets = offsets->asMutable<vector_size_t>();

    BufferPtr sizes = allocateSizes(rows.end(), context.pool());
    auto rawSizes = sizes->asMutable<vector_size_t>();

    BufferPtr keysIndices = allocateIndices(originalElements, context.pool());
    auto rawKeysIndices = keysIndices->asMutable<vector_size_t>();

    BufferPtr valuesIndices = allocateIndices(originalElements, context.pool());
    auto rawValuesIndices = valuesIndices->asMutable<vector_size_t>();

    vector_size_t offset = 0;
    vector_size_t totalElements = 0;

    rows.applyToSelected([&](vector_size_t row) {
      auto numKeys = keysArray->sizeAt(keyIndices[row]);
      auto keysOffset = keysArray->offsetAt(keyIndices[row]);

      auto numValues = valuesArray->sizeAt(valueIndices[row]);
      auto valuesOffset = valuesArray->offsetAt(valueIndices[row]);

      VELOX_USER_CHECK_EQ(
        numKeys,
        numValues,
        "{}",
        "Key and value arrays must be the same length");

      checkNullsInKey(numKeys, keysOffset, keysElements);
      auto dedupIndices = deduplicateKeys(numKeys, keysOffset, keysElements);
      for (auto i = 0; i < numKeys; ++i) {
        if (dedupIndices.find(keysOffset + i) != dedupIndices.end()) {
          rawKeysIndices[totalElements] = keysOffset + i;
          rawValuesIndices[totalElements] = valuesOffset + i;
          ++totalElements;
        }
      }

      rawOffsets[row] = offset;
      rawSizes[row] = dedupIndices.size();
      offset += dedupIndices.size();
    });

    auto wrappedKeys = BaseVector::wrapInDictionary(
      nullptr, keysIndices, totalElements, keysArray->elements());
    auto wrappedValues = BaseVector::wrapInDictionary(
      nullptr, valuesIndices, totalElements, valuesArray->elements());
    auto mapVector = std::make_shared<MapVector>(
        context.pool(),
        outputType,
        nullptr,
        rows.end(),
        offsets,
        sizes,
        wrappedKeys,
        wrappedValues);
    context.moveOrCopyResult(mapVector, rows, result);
  }

 private:
  void checkNullsInKey(
    vector_size_t size,
    vector_size_t offset,
    const VectorPtr& keysElements) const {
    for (auto i = 0; i < size; ++i) {
      VELOX_USER_CHECK(
          !keysElements->isNullAt(offset + i), "map key cannot be null");

      VELOX_USER_CHECK(
          !keysElements->containsNullAt(offset + i),
          "{}: {}",
          "map key cannot be indeterminate",
          keysElements->toString(offset + i));
    }
  }

  std::unordered_set<vector_size_t> deduplicateKeys(
    vector_size_t numKeys,
    vector_size_t keysOffset,
    const VectorPtr& keysElements) const {
      std::vector<vector_size_t> sortedIndices(numKeys);
      std::iota(sortedIndices.begin(), sortedIndices.end(), keysOffset);
      keysElements->sortIndices(sortedIndices, CompareFlags());

      std::unordered_set<vector_size_t> dedupIndices;
      for (auto i = 0; i < sortedIndices.size()-1; ++i) {
        auto isDuplicatedKey = keysElements->equalValueAt(
          keysElements.get(), sortedIndices[i], sortedIndices[i + 1]);

        if constexpr (Policy == core::SparkMapKeyDedupPolicy::EXCEPTION) {
          if (isDuplicatedKey) {
            auto duplicateKey = keysElements->wrappedVector()->toString(
              keysElements->wrappedIndex(sortedIndices[i]));
            VELOX_USER_FAIL("Duplicate map keys ({}) are not allowed", duplicateKey);
          }
        }

        if (!isDuplicatedKey) {
          dedupIndices.insert(sortedIndices[i]);
        }
      }
      dedupIndices.insert(sortedIndices[sortedIndices.size() - 1]);

      return dedupIndices;
    }
};

std::unique_ptr<exec::VectorFunction> createMapFromArrayFunction(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  const auto mapKeyDedupPolicy = config.sparkMapKeyDedupPolicy();
  if (mapKeyDedupPolicy == core::SparkMapKeyDedupPolicy::EXCEPTION) {
    return std::make_unique<MapFromArraysFunction<core::SparkMapKeyDedupPolicy::EXCEPTION>>();
  } else if (mapKeyDedupPolicy == core::SparkMapKeyDedupPolicy::LAST_WIN) {
    return std::make_unique<MapFromArraysFunction<core::SparkMapKeyDedupPolicy::LAST_WIN>>();
  } else {
    VELOX_UNREACHABLE();
  }
}

static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  // array(K), array(V) -> map(K,V)
  return {exec::FunctionSignatureBuilder()
              .typeVariable("K")
              .typeVariable("V")
              .returnType("map(K,V)")
              .argumentType("array(K)")
              .argumentType("array(V)")
              .build()};
}

} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_map_from_arrays,
    signatures(),
    createMapFromArrayFunction);
} // facebook::velox::functions::sparksql