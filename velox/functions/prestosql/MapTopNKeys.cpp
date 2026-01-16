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
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/vector/FunctionVector.h"

namespace facebook::velox::functions {
namespace {

/// See documentation at https://prestodb.io/docs/current/functions/map.html
/// Implements the map_top_n_keys with lambda comparator function.
/// Returns the top N keys of the given map sorting its keys using the provided
/// lambda comparator.
class MapTopNKeysLambdaFunction : public exec::VectorFunction {
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

    // Count total pairs needed for comparison across all maps.
    vector_size_t totalPairs = 0;
    rows.applyToSelected([&](vector_size_t row) {
      if (!decodedMap.isNullAt(row)) {
        auto mapIndex = decodedMap.index(row);
        auto mapSize = flatMap->sizeAt(mapIndex);
        if (mapSize > 1) {
          totalPairs += mapSize * (mapSize - 1);
        }
      }
    });

    // Build vectors for all pairs to evaluate in batch.
    std::vector<vector_size_t> leftIndices;
    std::vector<vector_size_t> rightIndices;
    std::vector<vector_size_t> pairToRow;
    leftIndices.reserve(totalPairs);
    rightIndices.reserve(totalPairs);
    pairToRow.reserve(totalPairs);

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedMap.isNullAt(row)) {
        return;
      }

      auto mapIndex = decodedMap.index(row);
      auto mapOffset = flatMap->offsetAt(mapIndex);
      auto mapSize = flatMap->sizeAt(mapIndex);

      // Each element in the map pair it up with other elements.
      for (vector_size_t i = 0; i < mapSize; ++i) {
        for (vector_size_t j = 0; j < mapSize; ++j) {
          if (i != j) {
            leftIndices.push_back(mapOffset + i);
            rightIndices.push_back(mapOffset + j);
            pairToRow.push_back(row);
          }
        }
      }
    });

    // Evaluate comparator lambda on all pairs at once.
    std::map<std::pair<vector_size_t, vector_size_t>, int64_t> comparisonCache;

    if (!leftIndices.empty() && !rightIndices.empty()) {
      auto leftIndicesBuffer =
          allocateIndices(leftIndices.size(), context.pool());
      auto rightIndicesBuffer =
          allocateIndices(rightIndices.size(), context.pool());

      auto* rawLeftIndices = leftIndicesBuffer->asMutable<vector_size_t>();
      auto* rawRightIndices = rightIndicesBuffer->asMutable<vector_size_t>();

      for (size_t i = 0; i < leftIndices.size(); ++i) {
        rawLeftIndices[i] = leftIndices[i];
        rawRightIndices[i] = rightIndices[i];
      }

      auto leftVector = BaseVector::wrapInDictionary(
          nullptr, leftIndicesBuffer, leftIndices.size(), mapKeys);
      auto rightVector = BaseVector::wrapInDictionary(
          nullptr, rightIndicesBuffer, rightIndices.size(), mapKeys);

      std::vector<VectorPtr> lambdaArgs = {leftVector, rightVector};
      SelectivityVector allPairsRows(leftIndices.size());

      VectorPtr lambdaResult;

      // Loop over lambda functions and apply comparator.
      auto it = args[2]->asUnchecked<FunctionVector>()->iterator(&rows);
      while (auto entry = it.next()) {
        entry.callable->apply(
            allPairsRows,
            nullptr,
            nullptr,
            &context,
            lambdaArgs,
            nullptr,
            &lambdaResult);
      }

      // Extract comparison results into cache.
      exec::LocalDecodedVector decodedResult(
          context, *lambdaResult, allPairsRows);
      for (size_t i = 0; i < leftIndices.size(); ++i) {
        if (decodedResult->isNullAt(i)) {
          VELOX_FAIL("Lambda comparator must return either -1, 0, or 1");
        }
        auto cmp = decodedResult->valueAt<int32_t>(i);
        if (cmp != -1 && cmp != 1 && cmp != 0) {
          VELOX_FAIL("Lambda comparator must return either -1, 0, or 1");
        }
        comparisonCache[{leftIndices[i], rightIndices[i]}] = cmp;
      }
    }

    // Build indices buffer for sorting.
    BufferPtr indices = allocateIndices(numKeys, context.pool());
    auto* rawIndices = indices->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < numKeys; ++i) {
      rawIndices[i] = i;
    }

    // Sort each map's keys using the cached comparisons.
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedMap.isNullAt(row)) {
        return;
      }

      auto mapIndex = decodedMap.index(row);
      auto mapOffset = flatMap->offsetAt(mapIndex);
      auto mapSize = flatMap->sizeAt(mapIndex);

      if (mapSize <= 1) {
        return;
      }

      // Sort keys in ascending order (matching Java's array_sort behavior).
      // Use stable_sort to preserve original order when comparator returns 0.
      // Then reverse to get descending order for TOP N.
      std::stable_sort(
          rawIndices + mapOffset,
          rawIndices + mapOffset + mapSize,
          [&comparisonCache](vector_size_t leftIdx, vector_size_t rightIdx) {
            if (leftIdx == rightIdx) {
              return false;
            }

            // Try forward lookup
            auto it = comparisonCache.find({leftIdx, rightIdx});
            if (it != comparisonCache.end()) {
              // If comparator says left < right (negative), put left first
              // (ascending).
              return it->second < 0;
            }
            return false;
          });

      // Reverse the sorted array to get descending order (matching Java's
      // reverse())
      std::reverse(rawIndices + mapOffset, rawIndices + mapOffset + mapSize);
    });

    // Build result array with top N keys.
    vector_size_t totalElements = 0;
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedMap.isNullAt(row) || decodedN.isNullAt(row)) {
        return;
      }

      auto mapIndex = decodedMap.index(row);
      auto mapSize = flatMap->sizeAt(mapIndex);
      int64_t n = decodedN.valueAt<int64_t>(row);

      auto resultSize = std::min(static_cast<vector_size_t>(n), mapSize);
      totalElements += resultSize;
    });

    // Create result array vector.
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

      // Copy top N sorted keys to result.
      for (vector_size_t i = 0; i < resultSize; ++i) {
        elements->copy(mapKeys.get(), elemIdx++, rawIndices[mapOffset + i], 1);
      }
    });

    context.moveOrCopyResult(arrayVector, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // double check java code to make sure all types are same...
    // map_top_n_keys(map(K,V), bigint, function(K,K,integer)) -> array(K)
    return {exec::FunctionSignatureBuilder()
                .typeVariable("K")
                .typeVariable("V")
                .returnType("array(K)")
                .argumentType("map(K,V)")
                .argumentType("bigint")
                .argumentType("function(K,K,integer)")
                .build()};
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_map_top_n_keys,
    MapTopNKeysLambdaFunction::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<MapTopNKeysLambdaFunction>());

} // namespace facebook::velox::functions
