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
#include "velox/functions/sparksql/MapFromEntries.h"

#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/sparksql/LastWin.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Implements Spark's map_from_entries, which builds a map from an array of
// key-value structs. Entries keep their input order. A repeated key keeps its
// first position and its last value, or is an error under the exception
// policy. A null entry makes the map null, and a null key is an error.
class MapFromEntriesFunction : public exec::VectorFunction {
 public:
  explicit MapFromEntriesFunction(bool throwOnDuplicateKey)
      : throwOnDuplicateKey_{throwOnDuplicateKey} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 1);
    auto* pool = context.pool();
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto* decodedArray = decodedArgs.at(0);
    auto* arrayVector = decodedArray->base()->as<ArrayVector>();
    exec::LocalDecodedVector decodedEntries(context);
    decodedEntries.get()->decode(*arrayVector->elements());
    const auto* entries = decodedEntries->base()->as<RowVector>();
    const auto& keys = entries->childAt(0);
    const auto numElements = arrayVector->elements()->size();
    BufferPtr entryNulls = allocateNulls(numElements, pool);
    auto* rawEntryNulls = entryNulls->asMutable<uint64_t>();
    BufferPtr entryIndices = allocateIndices(numElements, pool);
    auto* rawEntryIndices = entryIndices->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < numElements; ++i) {
      if (decodedEntries->isNullAt(i)) {
        bits::setNull(rawEntryNulls, i);
      } else {
        rawEntryIndices[i] = decodedEntries->index(i);
      }
    }
    const auto wrappedKeys = BaseVector::wrapInDictionary(
        std::move(entryNulls), std::move(entryIndices), numElements, keys);
    const DecodedVector decodedKeys{*wrappedKeys};

    vector_size_t maxNumEntries{0};
    rows.applyToSelected([&](vector_size_t row) {
      maxNumEntries += arrayVector->sizeAt(decodedArray->index(row));
    });

    BufferPtr offsets = allocateOffsets(rows.end(), pool);
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    BufferPtr sizes = allocateSizes(rows.end(), pool);
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    BufferPtr nulls;
    BufferPtr keyIndices = allocateIndices(maxNumEntries, pool);
    auto* rawKeyIndices = keyIndices->asMutable<vector_size_t>();
    BufferPtr valueIndices = allocateIndices(maxNumEntries, pool);
    auto* rawValueIndices = valueIndices->asMutable<vector_size_t>();

    LastWin<SparkMapKeyEquality> lastWin;
    vector_size_t numEntries{0};
    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      const auto arrayIndex = decodedArray->index(row);
      const auto offset = arrayVector->offsetAt(arrayIndex);
      const auto size = arrayVector->sizeAt(arrayIndex);
      rawOffsets[row] = numEntries;
      for (vector_size_t i = 0; i < size; ++i) {
        if (decodedEntries->isNullAt(offset + i)) {
          if (nulls == nullptr) {
            nulls = allocateNulls(rows.end(), pool);
          }
          bits::setNull(nulls->asMutable<uint64_t>(), row);
          return;
        }
      }

      for (const auto& entry :
           lastWin.resolve(decodedKeys, offset, size, throwOnDuplicateKey_)) {
        rawKeyIndices[numEntries] = decodedEntries->index(offset + entry.key);
        rawValueIndices[numEntries] =
            decodedEntries->index(offset + entry.value);
        ++numEntries;
      }
      rawSizes[row] = numEntries - rawOffsets[row];
    });

    exec::LocalSelectivityVector remainingRows(context, rows);
    context.deselectErrors(*remainingRows);
    auto mapVector = std::make_shared<MapVector>(
        pool,
        outputType,
        nulls,
        rows.end(),
        std::move(offsets),
        std::move(sizes),
        BaseVector::wrapInDictionary(
            nullptr, std::move(keyIndices), numEntries, keys),
        BaseVector::wrapInDictionary(
            nullptr, std::move(valueIndices), numEntries, entries->childAt(1)));
    context.moveOrCopyResult(mapVector, *remainingRows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        // array(row(K,V)) -> map(K,V)
        exec::FunctionSignatureBuilder()
            .typeVariable("K")
            .typeVariable("V")
            .returnType("map(K,V)")
            .argumentType("array(row(K,V))")
            .build(),
    };
  }

 private:
  const bool throwOnDuplicateKey_;
};

std::shared_ptr<exec::VectorFunction> makeMapFromEntries(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
    const core::QueryConfig& config) {
  return std::make_shared<MapFromEntriesFunction>(
      config.throwExceptionOnDuplicateMapKeys());
}
} // namespace

void registerMapFromEntriesFunction(const std::string& name) {
  exec::registerStatefulVectorFunction(
      name, MapFromEntriesFunction::signatures(), makeMapFromEntries);
}

} // namespace facebook::velox::functions::sparksql
