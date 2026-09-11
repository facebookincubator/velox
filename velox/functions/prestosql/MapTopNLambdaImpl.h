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
#pragma once

#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"

#include <queue>

namespace facebook::velox::functions {

enum class MapTopNMode { kKeys, kValues };

template <MapTopNMode Mode>
class MapTopNLambdaFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 3);

    // Flatten input map.
    exec::LocalDecodedVector mapDecoder(context, *args[0], rows);
    auto& decodedMap = *mapDecoder.get();
    auto flatMap = flattenMap(rows, args[0], decodedMap);
    VELOX_CHECK_NOT_NULL(flatMap);

    auto mapKeys = flatMap->mapKeys();
    auto mapValues = flatMap->mapValues();
    auto numKeys = mapKeys->size();
    const VectorPtr& sourceVector =
        (Mode == MapTopNMode::kKeys) ? mapKeys : mapValues;

    // Decode the n parameter.
    exec::LocalDecodedVector nDecoder(context, *args[1], rows);
    auto& decodedN = *nDecoder.get();

    // A null map or a null 'n' produces a null result. Every loop below skips
    // these rows.
    const auto isNullRow = [&](vector_size_t row) {
      return decodedMap.isNullAt(row) || decodedN.isNullAt(row);
    };

    exec::LocalSelectivityVector remainingRows(context, rows);
    context.applyToSelectedNoThrow(*remainingRows, [&](vector_size_t row) {
      if (isNullRow(row)) {
        return;
      }
      int64_t n = decodedN.valueAt<int64_t>(row);
      VELOX_USER_CHECK_GE(n, 0, "n must be greater than or equal to 0");
    });
    context.deselectErrors(*remainingRows);

    // All rows errored; emit an all-null result via moveOrCopyResult so a
    // sibling IF/CASE branch's already-populated rows are not overwritten.
    if (!remainingRows->hasSelections()) {
      auto nullArray = std::make_shared<ArrayVector>(
          context.pool(),
          outputType,
          nullptr,
          rows.end(),
          allocateOffsets(rows.end(), context.pool()),
          allocateSizes(rows.end(), context.pool()),
          BaseVector::create(
              outputType->asArray().elementType(), 0, context.pool()));
      rows.applyToSelected(
          [&](vector_size_t row) { nullArray->setNull(row, true); });
      context.moveOrCopyResult(nullArray, rows, result);
      return;
    }

    // Rows with n == 0 return an empty array, so their elements are excluded
    // from the ranking below: nothing reads their transformed values, and the
    // lambda must not raise errors for a result the query discards. The
    // 2-argument functions short-circuit n == 0 the same way.
    exec::LocalSelectivityVector rowsToRank(context, *remainingRows);
    remainingRows->applyToSelected([&](vector_size_t row) {
      if (isNullRow(row) || decodedN.valueAt<int64_t>(row) == 0) {
        rowsToRank->setValid(row, false);
      }
    });
    rowsToRank->updateBounds();

    // Positions of the elements to emit, ordered per row. Initialized to
    // identity so that rows the ranking loop skips (maps with a single entry)
    // still find their only element at the map's offset.
    BufferPtr indices = allocateIndices(numKeys, context.pool());
    auto* rawIndices = indices->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < numKeys; ++i) {
      rawIndices[i] = i;
    }

    if (rowsToRank->hasSelections()) {
      SelectivityVector validRowsInReusedResult =
          toElementRows<MapVector>(numKeys, *rowsToRank, flatMap.get());

      VectorPtr transformedElements;

      // A NULL lambda ranks by the keys or the values themselves, matching the
      // 2-argument simple-function behavior.
      const bool isLambdaNull = args[2]->type()->kind() == TypeKind::UNKNOWN;
      if (isLambdaNull) {
        transformedElements = sourceVector;
      } else {
        std::vector<VectorPtr> lambdaArgs = {mapKeys, mapValues};
        applyLambdaToElements<MapVector>(
            args[2],
            *rowsToRank,
            numKeys,
            flatMap,
            lambdaArgs,
            validRowsInReusedResult,
            context,
            transformedElements);
      }

      // Rows where the lambda raised errors must not be ranked: their
      // transformed values were never written.
      context.deselectErrors(*remainingRows);
      rowsToRank->intersect(*remainingRows);

      // Decode the values the elements are ranked by.
      exec::LocalDecodedVector decodedTransformed(
          context, *transformedElements, validRowsInReusedResult);
      auto* baseTransformed = decodedTransformed->base();

      // Decode the keys, which break ties on equal ranking values. The lambda
      // signature does not constrain the key type to be orderable, so keys
      // that cannot be ordered (a map-typed key, for instance) fall back to
      // the element's position in the input.
      exec::LocalDecodedVector decodedKeys(
          context, *mapKeys, validRowsInReusedResult);
      auto* baseKeys = decodedKeys->base();
      const bool keysOrderable = mapKeys->type()->isOrderable();

      CompareFlags flags{
          .nullsFirst = true,
          .ascending = true,
          .nullHandlingMode =
              CompareFlags::NullHandlingMode::kNullAsIndeterminate,
      };

      // Min-heap comparator: returns true if left > right by the transform
      // value. The heap keeps the smallest seen value at the top so we can
      // evict it when a larger value arrives, maintaining the top-n by value in
      // descending order. Top-level nulls are handled here so that compare()
      // only fires on nested nulls inside non-null values (where
      // kNullAsIndeterminate throws).
      struct GreaterThanComparator {
        const DecodedVector* decodedTransformed;
        const BaseVector* baseTransformed;
        const DecodedVector* decodedKeys;
        const BaseVector* baseKeys;
        bool keysOrderable;
        CompareFlags flags;

        // Ranks two elements whose transform values are equal by their keys, in
        // descending order. Keys are unique within a map, so this totally
        // orders the entries and keeps the result independent of the order in
        // which the map happens to store them. Matches the tie-break in
        // map_top_n and map_keys_by_top_n_values. Keys that are not orderable
        // cannot be ranked, so those tie-break on the input position instead.
        bool greaterByKey(vector_size_t leftIdx, vector_size_t rightIdx) const {
          if (!keysOrderable) {
            return leftIdx > rightIdx;
          }

          auto result = baseKeys->compare(
              baseKeys,
              decodedKeys->index(leftIdx),
              decodedKeys->index(rightIdx),
              flags);
          return result.value() > 0;
        }

        bool operator()(vector_size_t leftIdx, vector_size_t rightIdx) const {
          if (leftIdx == rightIdx) {
            return false;
          }

          bool leftNull = decodedTransformed->isNullAt(leftIdx);
          bool rightNull = decodedTransformed->isNullAt(rightIdx);

          // Nulls sort last in descending top-n, so a null is never "greater"
          // than a non-null.
          if (leftNull && rightNull) {
            return greaterByKey(leftIdx, rightIdx);
          }
          if (leftNull) {
            return false;
          }
          if (rightNull) {
            return true;
          }

          auto leftTransformedIdx = decodedTransformed->index(leftIdx);
          auto rightTransformedIdx = decodedTransformed->index(rightIdx);

          // Under kNullAsIndeterminate ordering, compare() throws on any null
          // it encounters (nested nulls inside non-null values), so it always
          // returns a value here.
          auto result = baseTransformed->compare(
              baseTransformed, leftTransformedIdx, rightTransformedIdx, flags);

          if (result.value() == 0) {
            return greaterByKey(leftIdx, rightIdx);
          }

          return result.value() > 0;
        }
      };

      GreaterThanComparator comparator{
          decodedTransformed.get(),
          baseTransformed,
          decodedKeys.get(),
          baseKeys,
          keysOrderable,
          flags};

      // Use applyToSelectedNoThrow so that compare()-thrown errors are captured
      // per row (letting try() convert them to nulls) instead of escaping.
      context.applyToSelectedNoThrow(*rowsToRank, [&](vector_size_t row) {
        auto mapOffset = flatMap->offsetAt(row);
        auto mapSize = flatMap->sizeAt(row);

        // 'rowsToRank' excludes null rows and rows with n == 0, so a map with
        // at most one entry is the only case that needs no ranking.
        if (mapSize <= 1) {
          return;
        }

        int64_t n = decodedN.valueAt<int64_t>(row);
        auto resultSize = static_cast<vector_size_t>(
            std::min(n, static_cast<int64_t>(mapSize)));

        std::priority_queue<
            vector_size_t,
            std::vector<vector_size_t>,
            GreaterThanComparator>
            minHeap(comparator);

        // Build heap with top N elements.
        for (vector_size_t i = 0; i < mapSize; ++i) {
          auto idx = mapOffset + i;
          if (minHeap.size() < static_cast<size_t>(resultSize)) {
            minHeap.push(idx);
          } else if (comparator(idx, minHeap.top())) {
            minHeap.push(idx);
            minHeap.pop();
          }
        }

        // Pop in reverse so the final order is descending by transform value.
        std::vector<vector_size_t> topIndices(minHeap.size());
        auto heapSize = minHeap.size();
        for (int i = heapSize - 1; i >= 0; --i) {
          topIndices[i] = minHeap.top();
          minHeap.pop();
        }

        // Copy back to rawIndices.
        for (size_t i = 0; i < topIndices.size(); ++i) {
          rawIndices[mapOffset + i] = topIndices[i];
        }
      });

      // Drop rows whose heap loop threw errors.
      context.deselectErrors(*remainingRows);
    }

    vector_size_t totalElements = 0;
    remainingRows->applyToSelected([&](vector_size_t row) {
      if (isNullRow(row)) {
        return;
      }
      auto mapSize = flatMap->sizeAt(row);
      int64_t n = decodedN.valueAt<int64_t>(row);
      totalElements = checkedPlus<vector_size_t>(
          totalElements,
          static_cast<vector_size_t>(
              std::min(n, static_cast<int64_t>(mapSize))));
    });

    auto elements = BaseVector::create(
        outputType->asArray().elementType(), totalElements, context.pool());

    auto arrayVector = std::make_shared<ArrayVector>(
        context.pool(),
        outputType,
        nullptr,
        rows.end(),
        allocateOffsets(rows.end(), context.pool()),
        allocateSizes(rows.end(), context.pool()),
        elements);

    auto* rawOffsets =
        arrayVector->mutableOffsets(rows.end())->asMutable<vector_size_t>();
    auto* rawSizes =
        arrayVector->mutableSizes(rows.end())->asMutable<vector_size_t>();

    // Rows that errored out earlier still need their offset/size initialized
    // before we copy elements for the surviving rows.
    rows.applyToSelected([&](vector_size_t row) {
      if (!remainingRows->isValid(row)) {
        arrayVector->setNull(row, true);
        rawOffsets[row] = 0;
        rawSizes[row] = 0;
      }
    });

    vector_size_t elemIdx = 0;
    remainingRows->applyToSelected([&](vector_size_t row) {
      if (isNullRow(row)) {
        arrayVector->setNull(row, true);
        rawOffsets[row] = elemIdx;
        rawSizes[row] = 0;
        return;
      }

      // After flattenMap, the flat map is indexed directly by row number,
      // not by decodedMap.index(row).
      auto mapOffset = flatMap->offsetAt(row);
      auto mapSize = flatMap->sizeAt(row);
      int64_t n = decodedN.valueAt<int64_t>(row);
      auto resultSize = static_cast<vector_size_t>(
          std::min(n, static_cast<int64_t>(mapSize)));

      rawOffsets[row] = elemIdx;
      rawSizes[row] = resultSize;

      for (vector_size_t i = 0; i < resultSize; ++i) {
        elements->copy(
            sourceVector.get(), elemIdx++, rawIndices[mapOffset + i], 1);
      }
    });

    context.moveOrCopyResult(arrayVector, rows, result);
  }
};
} // namespace facebook::velox::functions
