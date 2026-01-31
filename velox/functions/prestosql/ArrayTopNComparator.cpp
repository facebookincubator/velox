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
#include "velox/functions/lib/ArraySort.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"

namespace facebook::velox::functions {
namespace {

// Implements array_top_n(array(T), n, function(T, U)) -> array(T)
// Returns the top n elements of the array sorted in descending order
// according to values computed by the transform function.
// The transform function computes a sorting key for each element.
// Elements are sorted by keys in descending order, with nulls at the end.
class ArrayTopNTransformFunction : public exec::VectorFunction {
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

    // Prepare for lambda application.
    auto pool = context.pool();
    auto elementsVector = flatArray->elements();
    auto elementType = flatArray->type()->childAt(0);
    auto numElements = elementsVector->size();

    // Apply transform lambda to compute sorting keys.
    std::vector<VectorPtr> lambdaArgs = {elementsVector};
    SelectivityVector validRowsInReusedResult =
        toElementRows<ArrayVector>(numElements, rows, flatArray.get());
    VectorPtr transformedKeys;

    auto elementToTopLevelRows =
        getElementToTopLevelRows(numElements, rows, flatArray.get(), pool);

    // Loop over lambda functions and apply these to elements.
    auto it = functionVector->iterator(&rows);
    while (auto entry = it.next()) {
      auto elementRows =
          toElementRows<ArrayVector>(numElements, *entry.rows, flatArray.get());
      auto wrapCapture = toWrapCapture<ArrayVector>(
          numElements, entry.callable, *entry.rows, flatArray);

      entry.callable->apply(
          elementRows,
          &validRowsInReusedResult,
          wrapCapture,
          &context,
          lambdaArgs,
          elementToTopLevelRows,
          &transformedKeys);
    }

    // Sort elements by transformed keys (descending order, nulls last).
    auto sortedIndices = sortElements(
        rows,
        *flatArray,
        *transformedKeys,
        false /* ascending */,
        false /* nullsFirst */,
        context,
        true /* throwOnNestedNull */);
    auto* rawSortedIndices = sortedIndices->as<vector_size_t>();

    // Create result arrays by taking the first n elements from each sorted
    // array.
    BufferPtr resultOffsets = allocateOffsets(rows.end(), pool);
    BufferPtr resultSizes = allocateSizes(rows.end(), pool);
    auto* rawResultOffsets = resultOffsets->asMutable<vector_size_t>();
    auto* rawResultSizes = resultSizes->asMutable<vector_size_t>();

    std::vector<vector_size_t> resultIndices;
    resultIndices.reserve(numElements);

    BufferPtr resultNulls = addNullsForUnselectedRows(flatArray, rows);

    vector_size_t currentOffset = 0;

    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      rawResultOffsets[row] = currentOffset;

      if (flatArray->isNullAt(row)) {
        rawResultSizes[row] = 0;
        return;
      }

      auto n = decodedN.valueAt<int32_t>(row);
      VELOX_USER_CHECK_GE(
          n, 0, "Parameter n: {} to ARRAY_TOP_N is negative", n);

      auto arraySize = flatArray->sizeAt(row);
      auto arrayOffset = flatArray->offsetAt(row);

      if (n == 0 || arraySize == 0) {
        rawResultSizes[row] = 0;
        return;
      }

      // Take the first min(n, arraySize) elements from the sorted indices.
      auto resultSize = std::min(static_cast<vector_size_t>(n), arraySize);
      for (vector_size_t i = 0; i < resultSize; ++i) {
        resultIndices.push_back(rawSortedIndices[arrayOffset + i]);
      }

      rawResultSizes[row] = resultSize;
      currentOffset += resultSize;
    });

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
      // array(T), integer, function(T, U) -> array(T)
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .orderableTypeVariable("U")
          .returnType("array(T)")
          .argumentType("array(T)")
          .argumentType("integer")
          .constantArgumentType("function(T,U)")
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
    std::make_unique<ArrayTopNTransformFunction>());

} // namespace facebook::velox::functions
