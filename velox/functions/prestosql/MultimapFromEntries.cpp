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

#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/RowsTranslationUtil.h"
namespace facebook::velox::functions {
namespace {
/// See documentation at
/// https://prestodb.io/docs/current/functions/map.html#multimap_from_entries
///
/// Zero element copy:
/// In order to prevent copies of key/values elements, the function reuses the
/// internal children vectors from the original RowVector.
///
/// 0) Break down the original RowVector into `keyVector` and `valueVector`.
/// 1) Process the rows: validate each row and calculate how many entries/values
/// present in the final map, as well as populate `newSizes` and `newOffsets`
/// which controls how many entries each row has in the final map.
/// 2) Process the rows (excluding the ones had errors): deduplicate the keys
/// and assign indices, which will be used to wrap the `valueVector` into the
/// `elements` of the ArrayVector, and eventually becomes the ValueVector of
/// the final MapVector. Since all indices of the same key point to the same
/// key, only first indice will be used for each key, which will be used to wrap
/// the `keyVector` into the KeyVector of the final MapVector.
/// 3) Wrap KeyVector and ValueVector respectively, then combine them as the
/// output MapVector.
class MultimapFromEntriesFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 1);
    auto& arg = args[0];
    auto localResult =
        applyFlat(rows, arg->as<ArrayVector>(), outputType, context);

    context.moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {// array(row(K,V)) -> map(K,array(V))
            exec::FunctionSignatureBuilder()
                .knownTypeVariable("K")
                .typeVariable("V")
                .returnType("map(K,array(V))")
                .argumentType("array(row(K,V))")
                .build()};
  }

 private:
  VectorPtr applyFlat(
      const SelectivityVector& rows,
      const ArrayVector* arrayVector,
      const TypePtr& outputType,
      exec::EvalCtx& context) const {
    auto& elementsVector = arrayVector->elements();
    exec::LocalDecodedVector decodedElementsVector(context);
    decodedElementsVector.get()->decode(*elementsVector);
    auto elementRowVector = decodedElementsVector->base()->as<RowVector>();
    auto elementRowIndices = decodedElementsVector->indices();

    // Get keys/values.
    auto keyVector = elementRowVector->childAt(0);
    auto valueVector = elementRowVector->childAt(1);
    exec::LocalDecodedVector mapKeysHolder(context, *keyVector, rows);
    auto mapKeysDecoded = mapKeysHolder.get();
    auto mapKeysBase = mapKeysDecoded->base();
    auto mapKeysIndices = mapKeysDecoded->indices();

    const auto numRows = rows.end();
    auto pool = context.pool();

    // Allocate new vectors for length and offsets.
    BufferPtr newSizes = allocateSizes(numRows, pool);
    BufferPtr newOffsets = allocateOffsets(numRows, pool);
    auto* rawSizes = newSizes->asMutable<vector_size_t>();
    auto* rawOffsets = newOffsets->asMutable<vector_size_t>();

    // Count the numbers of values and entries.
    // totalValueCount: the number of elements in the final value array.
    // totalEntryCount: the number of keys in the result map.
    int32_t totalValueCount = 0;
    vector_size_t totalEntryCount = 0;
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto size = arrayVector->sizeAt(row);
      const auto offset = arrayVector->offsetAt(row);
      if (size == 0) {
        rawOffsets[row] = totalEntryCount;
        return;
      }
      if (elementRowVector->mayHaveNulls() || mapKeysBase->mayHaveNulls()) {
        // Validate if map entries/keys contain null.
        for (auto i = 0; i < size; ++i) {
          if (elementRowVector->isNullAt(elementRowIndices[offset + i])) {
            VELOX_USER_FAIL("map entry cannot be null");
          }
          if (keyVector->isNullAt(elementRowIndices[offset + i])) {
            VELOX_USER_FAIL("map key cannot be null");
          }
        }
      }

      std::vector<vector_size_t> arrayIndicesInRange(size);
      std::iota(arrayIndicesInRange.begin(), arrayIndicesInRange.end(), offset);
      std::vector<vector_size_t> elementRowIndicesSorted(size);
      for (auto i = 0; i < size; i++) {
        elementRowIndicesSorted[i] = elementRowIndices[arrayIndicesInRange[i]];
      }
      keyVector->sortIndices(elementRowIndicesSorted, CompareFlags());
      vector_size_t entryCount = 0;
      for (auto i = 0; i < size; i++) {
        if (i == 0 ||
            !mapKeysBase->equalValueAt(
                mapKeysBase,
                mapKeysIndices[elementRowIndicesSorted[i]],
                mapKeysIndices[elementRowIndicesSorted[i - 1]])) {
          entryCount++;
        }
      }

      rawSizes[row] = entryCount;
      rawOffsets[row] = totalEntryCount;
      totalValueCount += size;
      totalEntryCount += entryCount;
    });

    // Allocate new vectors for indices, lengths and offsets.
    vector_size_t keyIndiceCursor = -1;
    BufferPtr keyIndices = allocateIndices(totalEntryCount, pool);
    auto* rawKeyIndices = keyIndices->asMutable<vector_size_t>();

    vector_size_t valueIndiceCursor = 0;
    BufferPtr valueIndices = allocateIndices(totalValueCount, pool);
    BufferPtr valueSizes = allocateSizes(totalEntryCount, pool);
    BufferPtr valueOffsets = allocateOffsets(totalEntryCount, pool);
    auto* rawValueIndices = valueIndices->asMutable<vector_size_t>();
    auto* rawValueSizes = valueSizes->asMutable<vector_size_t>();
    auto* rawValueOffsets = valueOffsets->asMutable<vector_size_t>();

    // When context.throwOnError is false, rows with null key/entry should be
    // deselected and not be processed further.
    SelectivityVector remainingRows = rows;
    context.deselectErrors(remainingRows);
    context.applyToSelectedNoThrow(remainingRows, [&](vector_size_t row) {
      const auto size = arrayVector->sizeAt(row);
      const auto offset = arrayVector->offsetAt(row);
      if (size == 0) {
        return;
      }

      std::vector<vector_size_t> arrayIndicesInRange(size);
      std::iota(arrayIndicesInRange.begin(), arrayIndicesInRange.end(), offset);
      std::vector<vector_size_t> elementRowIndicesSorted(size);
      for (auto i = 0; i < size; i++) {
        elementRowIndicesSorted[i] = elementRowIndices[arrayIndicesInRange[i]];
      }
      keyVector->sortIndices(elementRowIndicesSorted, CompareFlags());
      for (auto i = 0; i < size; i++) {
        if (i == 0 ||
            !mapKeysBase->equalValueAt(
                mapKeysBase,
                mapKeysIndices[elementRowIndicesSorted[i]],
                mapKeysIndices[elementRowIndicesSorted[i - 1]])) {
          keyIndiceCursor++;
          rawValueOffsets[keyIndiceCursor] = valueIndiceCursor;
          rawKeyIndices[keyIndiceCursor] = elementRowIndicesSorted[i];
        }
        rawValueSizes[keyIndiceCursor]++;
        rawValueIndices[valueIndiceCursor++] = elementRowIndicesSorted[i];
      }
    });

    // Wrap result key/value vectors in dictionary.
    auto valueElements =
        BaseVector::transpose(valueIndices, std::move(valueVector));
    auto resultValueVector = std::make_shared<ArrayVector>(
        pool,
        ARRAY(valueElements->type()),
        nullptr,
        totalEntryCount,
        valueOffsets,
        valueSizes,
        std::move(valueElements));
    auto resultKeyVector =
        BaseVector::transpose(keyIndices, std::move(keyVector));

    return std::make_shared<MapVector>(
        pool,
        outputType,
        nullptr,
        numRows,
        newOffsets,
        newSizes,
        resultKeyVector,
        resultValueVector);
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_multimap_from_entries,
    MultimapFromEntriesFunction::signatures(),
    std::make_unique<MultimapFromEntriesFunction>());
} // namespace facebook::velox::functions
