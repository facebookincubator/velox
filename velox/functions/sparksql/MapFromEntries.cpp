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

#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/RowsTranslationUtil.h"
#include "velox/functions/sparksql/Map.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Implements Spark's map_from_entries.
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
    exec::LocalDecodedVector decodedArray(context, *args[0], rows);
    auto* arrayVector = decodedArray->base()->as<ArrayVector>();
    const auto numElements = arrayVector->elements()->size();
    const auto elementRows = toElementRows(
        numElements,
        rows,
        arrayVector,
        decodedArray->nulls(&rows),
        decodedArray->indices());
    exec::LocalDecodedVector decodedEntries(
        context, *arrayVector->elements(), elementRows);
    const auto* entries = decodedEntries->base()->as<RowVector>();
    const auto& keys = entries->childAt(0);
    BufferPtr entryNulls = allocateNulls(numElements, pool, bits::kNull);
    auto* rawEntryNulls = entryNulls->asMutable<uint64_t>();
    BufferPtr entryIndices = allocateIndices(numElements, pool);
    auto* rawEntryIndices = entryIndices->asMutable<vector_size_t>();
    elementRows.applyToSelected([&](vector_size_t i) {
      if (!decodedEntries->isNullAt(i)) {
        bits::clearNull(rawEntryNulls, i);
        rawEntryIndices[i] = decodedEntries->index(i);
      }
    });
    const auto wrappedKeys = BaseVector::wrapInDictionary(
        entryNulls, entryIndices, numElements, keys);
    exec::LocalDecodedVector decodedKeys(context, *wrappedKeys, elementRows);

    exec::LocalSelectivityVector entryRows(context, rows);
    BufferPtr nulls;
    if (decodedEntries->mayHaveNulls()) {
      std::vector<vector_size_t> nullRows;
      entryRows->applyToSelected([&](vector_size_t row) {
        const auto arrayIndex = decodedArray->index(row);
        const auto offset = arrayVector->offsetAt(arrayIndex);
        const auto size = arrayVector->sizeAt(arrayIndex);
        for (vector_size_t i = 0; i < size; ++i) {
          if (decodedEntries->isNullAt(offset + i)) {
            nullRows.push_back(row);
            return;
          }
        }
      });
      if (!nullRows.empty()) {
        nulls = allocateNulls(rows.end(), pool);
        for (const auto row : nullRows) {
          bits::setNull(nulls->asMutable<uint64_t>(), row);
          entryRows->setValid(row, false);
        }
        entryRows->updateBounds();
      }
    }

    auto mapVector = buildDeduplicatedMap(
        *entryRows,
        context,
        rows.end(),
        outputType,
        *decodedArray,
        *decodedArray,
        *decodedKeys,
        wrappedKeys,
        BaseVector::wrapInDictionary(
            std::move(entryNulls),
            std::move(entryIndices),
            numElements,
            entries->childAt(1)),
        throwOnDuplicateKey_);
    if (nulls != nullptr) {
      mapVector->setNulls(nulls);
    }

    exec::LocalSelectivityVector remainingRows(context, rows);
    context.deselectErrors(*remainingRows);
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
  // Whether a repeated key is an error rather than keeping its last value.
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
