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
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"

#include <queue>

namespace facebook::velox::functions {
namespace {

/// See documentation at https://prestodb.io/docs/current/functions/map.html
/// Implements the map_top_n_values with transform lambda function.
/// Returns the top N values of the given map sorting its values using the
/// provided transform lambda.
class MapTopNValuesLambdaFunction : public exec::VectorFunction {
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

    // Validate n values.
    rows.applyToSelected([&](vector_size_t row) {
      if (!decodedN.isNullAt(row)) {
        int64_t n = decodedN.valueAt<int64_t>(row);
        VELOX_USER_CHECK_GE(n, 0, "n must be greater than or equal to 0");
      }
    });

    std::vector<VectorPtr> lambdaArgs = {mapKeys, mapValues};
    auto newNumElements = mapKeys->size();
    SelectivityVector validRowsInReusedResult =
        toElementRows<MapVector>(newNumElements, rows, flatMap.get());

    VectorPtr newElements;
    auto elementToTopLevelRows = getElementToTopLevelRows(
        newNumElements, rows, flatMap.get(), context.pool());

    // Loop over lambda functions and apply these to elements of the base array.
    // In most cases there will be only one function and the loop will run once.
    auto it = args[2]->asUnchecked<FunctionVector>()->iterator(&rows);
    while (auto entry = it.next()) {
      auto elementRows =
          toElementRows<MapVector>(newNumElements, *entry.rows, flatMap.get());
      auto wrapCapture = toWrapCapture<MapVector>(
          newNumElements, entry.callable, *entry.rows, flatMap);

      entry.callable->apply(
          elementRows,
          &validRowsInReusedResult,
          wrapCapture,
          &context,
          lambdaArgs,
          elementToTopLevelRows,
          &newElements);
    }

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
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedMap.isNullAt(row) || decodedN.isNullAt(row)) {
        return;
      }

      auto mapIndex = decodedMap.index(row);
      auto mapOffset = flatMap->offsetAt(mapIndex);
      auto mapSize = flatMap->sizeAt(mapIndex);

      if (mapSize <= 1) {
        return;
      }

      int64_t n = decodedN.valueAt<int64_t>(row);
      auto resultSize = std::min(static_cast<vector_size_t>(n), mapSize);

      if (resultSize == 0) {
        return;
      }

      totalElements += resultSize;
      std::priority_queue<
          vector_size_t,
          std::vector<vector_size_t>,
          GreaterThanComparator>
          minHeap(comparator);

      // Build heap with top N elements
      for (vector_size_t i = 0; i < mapSize; ++i) {
        auto idx = mapOffset + i;
        if (minHeap.size() < resultSize) {
          minHeap.push(idx);
        } else if (comparator(idx, minHeap.top())) {
          minHeap.push(idx);
          minHeap.pop();
        }
      }

      // Extract from heap in reverse order to get descending order
      std::vector<vector_size_t> topIndices(minHeap.size());
      auto heapSize = minHeap.size();
      for (int i = heapSize - 1; i >= 0; --i) {
        topIndices[i] = minHeap.top();
        minHeap.pop();
      }

      // Copy back to rawIndices
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

    vector_size_t elemIdx = 0;
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedMap.isNullAt(row) || decodedN.isNullAt(row)) {
        arrayVector->setNull(row, true);
        rawOffsets[row] = elemIdx;
        rawSizes[row] = 0;
        return;
      }

      auto mapIndex = decodedMap.index(row);
      auto mapOffset = flatMap->offsetAt(mapIndex);
      auto mapSize = flatMap->sizeAt(mapIndex);
      int64_t n = decodedN.valueAt<int64_t>(row);

      auto resultSize = std::min(static_cast<vector_size_t>(n), mapSize);
      rawOffsets[row] = elemIdx;
      rawSizes[row] = resultSize;

      for (vector_size_t i = 0; i < resultSize; ++i) {
        elements->copy(
            mapValues.get(), elemIdx++, rawIndices[mapOffset + i], 1);
      }
    });

    context.moveOrCopyResult(arrayVector, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // map_top_n_values(map(K,V), bigint, function(K,V,U)) -> array(V)
    return {exec::FunctionSignatureBuilder()
                .typeVariable("K")
                .typeVariable("V")
                .orderableTypeVariable("U")
                .returnType("array(V)")
                .argumentType("map(K,V)")
                .argumentType("bigint")
                .argumentType("function(K, V, U)")
                .build()};
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_map_top_n_values,
    MapTopNValuesLambdaFunction::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<MapTopNValuesLambdaFunction>());

} // namespace facebook::velox::functions
