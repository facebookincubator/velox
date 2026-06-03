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
#include <queue>

#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"

namespace facebook::velox::functions {
namespace {

/// Implements array_top_n(array(T), bigint, function(T, U)) -> array(T).
/// Returns the top n elements of the array in descending order by the
/// transform lambda's output. Elements whose transform result is null are
/// placed at the end.
class ArrayTopNTransformFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 3);

    exec::LocalDecodedVector arrayDecoder(context, *args[0], rows);
    auto& decodedArray = *arrayDecoder.get();
    auto flatArray = flattenArray(rows, args[0], decodedArray);
    VELOX_CHECK_NOT_NULL(flatArray);

    auto arrayElements = flatArray->elements();
    auto numElements = arrayElements->size();

    exec::LocalDecodedVector nDecoder(context, *args[1], rows);
    auto& decodedN = *nDecoder.get();

    exec::LocalSelectivityVector remainingRows(context, rows);
    context.applyToSelectedNoThrow(*remainingRows, [&](vector_size_t row) {
      if (!decodedN.isNullAt(row)) {
        int64_t n = decodedN.valueAt<int32_t>(row);
        VELOX_USER_CHECK_GE(n, 0, "n must be greater than or equal to 0");
      }
    });
    context.deselectErrors(*remainingRows);

    // If all rows have errors, we still need to create a result vector
    // with all rows set to null to satisfy the output size requirement.
    if (!remainingRows->hasSelections()) {
      auto nullArray = BaseVector::create(outputType, 1, context.pool());
      nullArray->setNull(0, true);
      result = BaseVector::wrapInConstant(rows.end(), 0, nullArray);
      return;
    }

    std::vector<VectorPtr> lambdaArgs = {arrayElements};
    SelectivityVector validRowsInReusedResult = toElementRows<ArrayVector>(
        numElements, *remainingRows, flatArray.get());

    VectorPtr newElements;
    applyLambdaToElements<ArrayVector>(
        args[2],
        *remainingRows,
        numElements,
        flatArray,
        lambdaArgs,
        validRowsInReusedResult,
        context,
        newElements);

    exec::LocalDecodedVector decodedKeys(
        context, *newElements, validRowsInReusedResult);
    auto* baseKeys = decodedKeys->base();
    auto keyIndices = decodedKeys->indices();

    BufferPtr indices = allocateIndices(numElements, context.pool());
    auto* rawIndices = indices->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < numElements; ++i) {
      rawIndices[i] = i;
    }

    CompareFlags flags{
        .nullsFirst = true,
        .ascending = true,
        .nullHandlingMode =
            CompareFlags::NullHandlingMode::kNullAsIndeterminate,
    };

    // Min-heap comparator: returns true if left > right by the transform key.
    // The heap keeps the smallest seen key at the top so we can evict it when
    // a larger key arrives, maintaining the top-n by key in descending order.
    // Top-level nulls are handled here so that compare() only fires on nested
    // nulls inside non-null keys (where kNullAsIndeterminate throws).
    struct GreaterThanComparator {
      const DecodedVector* decodedKeys;
      const BaseVector* baseKeys;
      const vector_size_t* keyIndices;
      CompareFlags flags;

      bool operator()(vector_size_t leftIdx, vector_size_t rightIdx) const {
        if (leftIdx == rightIdx) {
          return false;
        }

        bool leftNull = decodedKeys->isNullAt(leftIdx);
        bool rightNull = decodedKeys->isNullAt(rightIdx);

        // Nulls sort last in descending top-n, so a null is never "greater"
        // than a non-null. Among two nulls, tie-break by index for stable
        // order.
        if (leftNull && rightNull) {
          return leftIdx < rightIdx;
        }
        if (leftNull) {
          return false;
        }
        if (rightNull) {
          return true;
        }

        auto leftKeyIdx = keyIndices ? keyIndices[leftIdx] : leftIdx;
        auto rightKeyIdx = keyIndices ? keyIndices[rightIdx] : rightIdx;

        auto result =
            baseKeys->compare(baseKeys, leftKeyIdx, rightKeyIdx, flags);

        if (!result.has_value()) {
          VELOX_USER_FAIL("Ordering nulls is not supported");
        }

        // Tie-break by original index so that, among elements with equal
        // keys, earlier ones are kept (stable order). The min-heap evicts
        // its top first, so the earlier index must be the "larger" element
        // by this comparator to stay off the top.
        if (result.value() == 0) {
          return leftIdx < rightIdx;
        }

        return result.value() > 0;
      }
    };

    GreaterThanComparator comparator{
        decodedKeys.get(), baseKeys, keyIndices, flags};

    // Use applyToSelectedNoThrow so that compare()-thrown errors are captured
    // per row (letting try() convert them to nulls) instead of escaping.
    context.applyToSelectedNoThrow(*remainingRows, [&](vector_size_t row) {
      if (decodedArray.isNullAt(row) || decodedN.isNullAt(row)) {
        return;
      }

      auto arrayOffset = flatArray->offsetAt(row);
      auto arraySize = flatArray->sizeAt(row);

      int64_t n = decodedN.valueAt<int32_t>(row);
      auto resultSize = static_cast<vector_size_t>(
          std::min(n, static_cast<int64_t>(arraySize)));

      if (arraySize <= 1 || resultSize == 0) {
        return;
      }

      std::priority_queue<
          vector_size_t,
          std::vector<vector_size_t>,
          GreaterThanComparator>
          minHeap(comparator);

      for (vector_size_t i = 0; i < arraySize; ++i) {
        auto idx = arrayOffset + i;
        if (minHeap.size() < static_cast<size_t>(resultSize)) {
          minHeap.push(idx);
        } else if (comparator(idx, minHeap.top())) {
          minHeap.push(idx);
          minHeap.pop();
        }
      }

      // Pop in reverse so the final order is descending by transform key.
      std::vector<vector_size_t> topIndices(minHeap.size());
      auto heapSize = minHeap.size();
      for (int i = heapSize - 1; i >= 0; --i) {
        topIndices[i] = minHeap.top();
        minHeap.pop();
      }

      for (size_t i = 0; i < topIndices.size(); ++i) {
        rawIndices[arrayOffset + i] = topIndices[i];
      }
    });

    // Drop rows whose heap loop threw, so they don't contribute to
    // totalElements or get copied below.
    context.deselectErrors(*remainingRows);

    vector_size_t totalElements = 0;
    remainingRows->applyToSelected([&](vector_size_t row) {
      if (decodedArray.isNullAt(row) || decodedN.isNullAt(row)) {
        return;
      }
      auto arraySize = flatArray->sizeAt(row);
      int64_t n = decodedN.valueAt<int32_t>(row);
      totalElements += static_cast<vector_size_t>(
          std::min(n, static_cast<int64_t>(arraySize)));
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
      if (decodedArray.isNullAt(row) || decodedN.isNullAt(row)) {
        arrayVector->setNull(row, true);
        rawOffsets[row] = elemIdx;
        rawSizes[row] = 0;
        return;
      }

      auto arrayOffset = flatArray->offsetAt(row);
      auto arraySize = flatArray->sizeAt(row);
      int64_t n = decodedN.valueAt<int32_t>(row);
      auto resultSize = static_cast<vector_size_t>(
          std::min(n, static_cast<int64_t>(arraySize)));

      rawOffsets[row] = elemIdx;
      rawSizes[row] = resultSize;

      for (vector_size_t i = 0; i < resultSize; ++i) {
        elements->copy(
            arrayElements.get(), elemIdx++, rawIndices[arrayOffset + i], 1);
      }
    });

    context.moveOrCopyResult(arrayVector, rows, result);
  }
};

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  return {
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .orderableTypeVariable("U")
          .returnType("array(T)")
          .argumentType("array(T)")
          .argumentType("integer")
          .argumentType("function(T, U)")
          .build(),
  };
}

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_array_top_n_transform,
    signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<ArrayTopNTransformFunction>());

} // namespace facebook::velox::functions
