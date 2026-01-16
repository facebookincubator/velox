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
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"

namespace facebook::velox::functions {
namespace {

// Implements array_top_n(array(T), n, function(T, T, bigint)) -> array(T)
// Returns the top n elements of the array based on the given comparator.
// The comparator takes two nullable elements and returns:
//   -1 if first element is less than second
//    0 if elements are equal
//    1 if first element is greater than second
// The result is sorted in descending order according to the comparator.
class ArrayTopNComparatorFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 3);

    // Decode the array argument.
    exec::LocalDecodedVector arrayDecoder(context, *args[0], rows);
    auto& decodedArray = *arrayDecoder.get();

    // Flatten input array.
    auto flatArray = flattenArray(rows, args[0], decodedArray);

    // Decode the 'n' argument.
    exec::LocalDecodedVector nDecoder(context, *args[1], rows);
    auto& decodedN = *nDecoder.get();

    // Get the lambda function.
    auto* functionVector = args[2]->asUnchecked<FunctionVector>();

    // Prepare the result arrays.
    auto pool = context.pool();
    auto elementsVector = flatArray->elements();
    auto elementType = flatArray->type()->childAt(0);

    // Count total elements needed for the result.
    vector_size_t totalResultElements = 0;
    rows.applyToSelected([&](vector_size_t row) {
      if (!flatArray->isNullAt(row)) {
        auto n = decodedN.valueAt<int32_t>(row);
        if (n > 0) {
          auto arraySize = flatArray->sizeAt(row);
          totalResultElements +=
              std::min(static_cast<vector_size_t>(n), arraySize);
        }
      }
    });

    // Create result elements and offsets/sizes.
    BufferPtr resultOffsets = allocateOffsets(rows.end(), pool);
    BufferPtr resultSizes = allocateSizes(rows.end(), pool);
    auto* rawResultOffsets = resultOffsets->asMutable<vector_size_t>();
    auto* rawResultSizes = resultSizes->asMutable<vector_size_t>();

    std::vector<vector_size_t> resultIndices;
    resultIndices.reserve(totalResultElements);

    BufferPtr resultNulls = addNullsForUnselectedRows(flatArray, rows);

    vector_size_t currentOffset = 0;

    // Process each row.
    auto it = functionVector->iterator(&rows);
    while (auto entry = it.next()) {
      entry.rows->applyToSelected([&](vector_size_t row) {
        if (flatArray->isNullAt(row)) {
          rawResultOffsets[row] = currentOffset;
          rawResultSizes[row] = 0;
          return;
        }

        auto n = decodedN.valueAt<int32_t>(row);
        VELOX_USER_CHECK_GE(
            n, 0, "Parameter n: {} to ARRAY_TOP_N is negative", n);

        auto arraySize = flatArray->sizeAt(row);
        auto offset = flatArray->offsetAt(row);

        if (n == 0 || arraySize == 0) {
          rawResultOffsets[row] = currentOffset;
          rawResultSizes[row] = 0;
          return;
        }

        // Build indices for this array, separating nulls from non-nulls.
        std::vector<vector_size_t> nonNullIndices;
        std::vector<vector_size_t> nullIndices;

        for (vector_size_t i = 0; i < arraySize; ++i) {
          auto idx = offset + i;
          if (elementsVector->isNullAt(idx)) {
            nullIndices.push_back(idx);
          } else {
            nonNullIndices.push_back(idx);
          }
        }

        auto numNonNull = nonNullIndices.size();

        // If we have elements to compare, evaluate the lambda.
        if (numNonNull > 1) {
          // Create a comparison cache.
          std::unordered_map<int64_t, int64_t> comparisonCache;

          auto makeKey = [](vector_size_t a, vector_size_t b) -> int64_t {
            return (static_cast<int64_t>(a) << 32) |
                static_cast<int64_t>(static_cast<uint32_t>(b));
          };

          // Build vectors of left and right elements for all pairs.
          std::vector<vector_size_t> leftIndices;
          std::vector<vector_size_t> rightIndices;
          leftIndices.reserve(numNonNull * (numNonNull - 1));
          rightIndices.reserve(numNonNull * (numNonNull - 1));

          for (size_t i = 0; i < numNonNull; ++i) {
            for (size_t j = 0; j < numNonNull; ++j) {
              if (i != j) {
                leftIndices.push_back(nonNullIndices[i]);
                rightIndices.push_back(nonNullIndices[j]);
              }
            }
          }

          if (!leftIndices.empty()) {
            // Create dictionary vectors for left and right elements.
            auto leftIndicesBuffer = allocateIndices(leftIndices.size(), pool);
            auto rightIndicesBuffer =
                allocateIndices(rightIndices.size(), pool);

            auto* rawLeftIndices =
                leftIndicesBuffer->asMutable<vector_size_t>();
            auto* rawRightIndices =
                rightIndicesBuffer->asMutable<vector_size_t>();

            for (size_t i = 0; i < leftIndices.size(); ++i) {
              rawLeftIndices[i] = leftIndices[i];
              rawRightIndices[i] = rightIndices[i];
            }

            auto leftVector = BaseVector::wrapInDictionary(
                nullptr, leftIndicesBuffer, leftIndices.size(), elementsVector);
            auto rightVector = BaseVector::wrapInDictionary(
                nullptr,
                rightIndicesBuffer,
                rightIndices.size(),
                elementsVector);

            // Evaluate the lambda on all pairs.
            std::vector<VectorPtr> lambdaArgs = {leftVector, rightVector};
            SelectivityVector allPairsRows(leftIndices.size());
            SelectivityVector validRows(leftIndices.size());

            // Create element to top level rows mapping (all map to this row).
            BufferPtr elementToTopLevelRows =
                allocateIndices(leftIndices.size(), pool);
            auto* rawElementToTopLevel =
                elementToTopLevelRows->asMutable<vector_size_t>();
            for (size_t i = 0; i < leftIndices.size(); ++i) {
              rawElementToTopLevel[i] = row;
            }

            VectorPtr lambdaResult;
            auto wrapCapture = toWrapCapture<ArrayVector>(
                leftIndices.size(), entry.callable, allPairsRows, flatArray);

            entry.callable->apply(
                allPairsRows,
                &validRows,
                wrapCapture,
                &context,
                lambdaArgs,
                elementToTopLevelRows,
                &lambdaResult);

            // Extract comparison results.
            exec::LocalDecodedVector decodedResult(
                context, *lambdaResult, allPairsRows);

            size_t pairIdx = 0;
            for (size_t i = 0; i < numNonNull; ++i) {
              for (size_t j = 0; j < numNonNull; ++j) {
                if (i != j) {
                  auto key = makeKey(nonNullIndices[i], nonNullIndices[j]);
                  if (decodedResult->isNullAt(pairIdx)) {
                    VELOX_USER_FAIL("Comparator function must not return NULL");
                  }
                  auto cmp = decodedResult->valueAt<int64_t>(pairIdx);
                  VELOX_USER_CHECK(
                      cmp == -1 || cmp == 0 || cmp == 1,
                      "Comparator function must return -1, 0, or 1, got: {}",
                      cmp);
                  comparisonCache[key] = cmp;
                  pairIdx++;
                }
              }
            }

            // Sort using cached comparisons.
            // For descending order (top N), we want larger elements first.
            // Comparator returns 1 if first > second, so for descending sort,
            // we want elements where compare(a, b) > 0 to come first.
            std::sort(
                nonNullIndices.begin(),
                nonNullIndices.end(),
                [&](vector_size_t a, vector_size_t b) {
                  if (a == b) {
                    return false;
                  }
                  auto key = makeKey(a, b);
                  auto cacheIt = comparisonCache.find(key);
                  if (cacheIt != comparisonCache.end()) {
                    // For descending order: if compare(a, b) > 0, a is greater,
                    // so a should come before b.
                    return cacheIt->second > 0;
                  }
                  // Fallback (should not happen if cache is complete).
                  return false;
                });
          }
        }

        // Build result: take top n elements (non-nulls first, then nulls).
        auto resultSize = std::min(static_cast<vector_size_t>(n), arraySize);

        rawResultOffsets[row] = currentOffset;
        rawResultSizes[row] = resultSize;

        vector_size_t added = 0;
        // Add non-null elements first (sorted by comparator).
        for (size_t i = 0; i < nonNullIndices.size() && added < resultSize;
             ++i) {
          resultIndices.push_back(nonNullIndices[i]);
          added++;
        }
        // Add nulls to fill if needed.
        for (size_t i = 0; i < nullIndices.size() && added < resultSize; ++i) {
          resultIndices.push_back(nullIndices[i]);
          added++;
        }

        currentOffset += resultSize;
      });
    }

    // Build the result elements vector using dictionary wrapping.
    VectorPtr resultElements;
    if (resultIndices.empty()) {
      resultElements = BaseVector::create(elementType, 0, pool);
    } else {
      BufferPtr resultIndicesBuffer =
          allocateIndices(resultIndices.size(), pool);
      auto* rawResultIndices = resultIndicesBuffer->asMutable<vector_size_t>();
      for (size_t i = 0; i < resultIndices.size(); ++i) {
        rawResultIndices[i] = resultIndices[i];
      }
      resultElements = BaseVector::wrapInDictionary(
          nullptr, resultIndicesBuffer, resultIndices.size(), elementsVector);
    }

    auto localResult = std::make_shared<ArrayVector>(
        pool,
        flatArray->type(),
        resultNulls,
        rows.end(),
        resultOffsets,
        resultSizes,
        resultElements);

    context.moveOrCopyResult(localResult, rows, result);
  }
};

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  return {
      // array(T), integer, function(T, T, bigint) -> array(T)
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("array(T)")
          .argumentType("array(T)")
          .argumentType("integer")
          .argumentType("function(T,T,bigint)")
          .build(),
  };
}

} // namespace

void registerArrayTopNComparatorFunction(const std::string& prefix) {
  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_array_top_n_comparator, prefix + "array_top_n");
}

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_array_top_n_comparator,
    signatures(),
    std::make_unique<ArrayTopNComparatorFunction>());

} // namespace facebook::velox::functions
