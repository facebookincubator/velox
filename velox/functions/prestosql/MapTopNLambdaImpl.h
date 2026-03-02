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

    std::vector<VectorPtr> lambdaArgs = {mapKeys, mapValues};
    auto newNumElements = mapKeys->size();
    SelectivityVector validRowsInReusedResult =
        toElementRows<MapVector>(newNumElements, *remainingRows, flatMap.get());

    VectorPtr newElements;

    applyLambdaToElements<MapVector>(
        args[2],
        *remainingRows,
        newNumElements,
        flatMap,
        lambdaArgs,
        validRowsInReusedResult,
        context,
        newElements);

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

    CompareFlags flags{.nullsFirst = false, .ascending = true};

    // Define comparator for min-heap: returns true if left > right.
    // This keeps the smallest elements at the top, so we can maintain the
    // largest N elements.
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

        if (leftNull && rightNull) {
          return false;
        }
        if (leftNull) {
          return false;
        }
        if (rightNull) {
          return true;
        }

        auto result = baseElements->compare(
            baseElements,
            decodedIndices[leftIdx],
            decodedIndices[rightIdx],
            flags);

        if (!result.has_value()) {
          VELOX_USER_FAIL("Ordering nulls is not supported");
        }

        return result.value() > 0;
      }
    };

    GreaterThanComparator comparator{
        decodedElements.get(), baseElements, decodedIndices, flags};

    vector_size_t totalElements = 0;
    remainingRows->applyToSelected([&](vector_size_t row) {
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

      totalElements += resultSize;

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

    const VectorPtr& sourceVector =
        (Mode == MapTopNMode::Keys) ? mapKeys : mapValues;

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

    context.moveOrCopyResult(arrayVector, *remainingRows, result);
  }
};
} // namespace facebook::velox::functions
