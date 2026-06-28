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

#include "velox/common/base/CheckedArithmetic.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"

#include <queue>

namespace facebook::velox::functions {

enum class MapTopNMode { Keys, Values };

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
        (Mode == MapTopNMode::Keys) ? mapKeys : mapValues;

    // Decode the n parameter.
    exec::LocalDecodedVector nDecoder(context, *args[1], rows);
    auto& decodedN = *nDecoder.get();

    exec::LocalSelectivityVector remainingRows(context, rows);
    context.applyToSelectedNoThrow(*remainingRows, [&](vector_size_t row) {
      if (!decodedN.isNullAt(row)) {
        int64_t n = decodedN.valueAt<int64_t>(row);
        VELOX_USER_CHECK_GE(n, 0, "n must be greater than or equal to 0");
      }
    });
    context.deselectErrors(*remainingRows);

    // If all rows have errors, return a null constant.
    if (!remainingRows->hasSelections()) {
      auto nullConstant = BaseVector::createNullConstant(
          outputType, rows.end(), context.pool());
      context.moveOrCopyResult(nullConstant, rows, result);
      return;
    }

    std::vector<VectorPtr> lambdaArgs = {mapKeys, mapValues};
    SelectivityVector validRowsInReusedResult =
        toElementRows<MapVector>(numKeys, *remainingRows, flatMap.get());

    VectorPtr newElements;

    // Check if the lambda argument is NULL (constant null vector).
    bool isLambdaNull = args[2]->isConstantEncoding() && args[2]->isNullAt(0);

    if (isLambdaNull) {
      // Fall back to natural sort element, matching the 2-arg simple-function
      // behavior.
      newElements = (Mode == MapTopNMode::Keys) ? mapKeys : mapValues;
    } else {
      applyLambdaToElements<MapVector>(
          args[2],
          *remainingRows,
          numKeys,
          flatMap,
          lambdaArgs,
          validRowsInReusedResult,
          context,
          newElements);
    }

    // Deselect rows where the lambda raised errors.
    context.deselectErrors(*remainingRows);

    // Decode the transformed values from the lambda.
    exec::LocalDecodedVector decodedElements(
        context, *newElements, validRowsInReusedResult);
    auto* baseElements = decodedElements->base();
    auto decodedIndices = decodedElements->indices();

    // Build indices buffer for sorting.
    BufferPtr indices = allocateIndices(numKeys, context.pool());
    auto* rawIndices = indices->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < numKeys; ++i) {
      rawIndices[i] = i;
    }

    CompareFlags flags{
        .nullsFirst = true,
        .ascending = true,
        .nullHandlingMode =
            CompareFlags::NullHandlingMode::kNullAsIndeterminate,
    };

    // Min-heap comparator: returns true if left > right by the transform value.
    // The heap keeps the smallest seen value at the top so we can evict it when
    // a larger value arrives, maintaining the top-n by value in descending
    // order. Top-level nulls are handled here so that compare() only fires on
    // nested nulls inside non-null values (where kNullAsIndeterminate throws).
    struct GreaterThanComparator {
      const DecodedVector* decodedElements;
      const BaseVector* baseElements;
      const vector_size_t* decodedIndices;
      CompareFlags flags;

      bool operator()(vector_size_t leftIdx, vector_size_t rightIdx) const {
        if (leftIdx == rightIdx) {
          return false;
        }

        bool leftNull = decodedElements->isNullAt(leftIdx);
        bool rightNull = decodedElements->isNullAt(rightIdx);

        // Nulls sort last in descending top-n, so a null is never "greater"
        // than a non-null. Among two nulls, the later input index wins, to
        // match the reverse-stable order documented at the tie-break below.
        if (leftNull && rightNull) {
          return leftIdx > rightIdx;
        }
        if (leftNull) {
          return false;
        }
        if (rightNull) {
          return true;
        }

        auto leftElementIdx =
            decodedIndices ? decodedIndices[leftIdx] : leftIdx;
        auto rightElementIdx =
            decodedIndices ? decodedIndices[rightIdx] : rightIdx;

        auto result = baseElements->compare(
            baseElements, leftElementIdx, rightElementIdx, flags);

        if (!result.has_value()) {
          VELOX_USER_FAIL("Ordering nulls is not supported");
        }

        // Reverse-stable tie-break to match Presto's map_top_n, which is
        // SLICE(REVERSE(ARRAY_SORT(input, f)), 1, n): among equal values the
        // later input index is treated as "larger", so it is kept by the
        // min-heap and emitted before earlier ties.
        if (result.value() == 0) {
          return leftIdx > rightIdx;
        }

        return result.value() > 0;
      }
    };

    GreaterThanComparator comparator{
        decodedElements.get(), baseElements, decodedIndices, flags};

    // Use applyToSelectedNoThrow so compare() errors are captured per-row.
    context.applyToSelectedNoThrow(*remainingRows, [&](vector_size_t row) {
      if (decodedMap.isNullAt(row) || decodedN.isNullAt(row)) {
        return;
      }

      // After flattenMap, the flat map is indexed directly by row number,
      // not by decodedMap.index(row).
      auto mapOffset = flatMap->offsetAt(row);
      auto mapSize = flatMap->sizeAt(row);

      int64_t n = decodedN.valueAt<int64_t>(row);
      auto resultSize = static_cast<vector_size_t>(
          std::min(n, static_cast<int64_t>(mapSize)));

      if (mapSize <= 1 || resultSize == 0) {
        return;
      }

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

      // Extract from heap in reverse order to get descending order.
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

    vector_size_t totalElements = 0;
    remainingRows->applyToSelected([&](vector_size_t row) {
      if (decodedMap.isNullAt(row) || decodedN.isNullAt(row)) {
        return;
      }
      auto mapSize = flatMap->sizeAt(row);
      int64_t n = decodedN.valueAt<int64_t>(row);
      totalElements += static_cast<vector_size_t>(
          std::min(n, static_cast<int64_t>(mapSize)));
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
      if (decodedMap.isNullAt(row) || decodedN.isNullAt(row)) {
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
