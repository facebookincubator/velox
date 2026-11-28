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

#include "velox/common/base/CheckedArithmetic.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"

namespace facebook::velox::functions {
namespace {

// Implements array_top_n(array(T), integer, function(T, U)) -> array(T).
// Returns the top n elements of the array in descending order by the
// transform lambda's output. Elements whose transform result is null are
// placed at the end.
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

    // A null array or a null 'n' produces a null result. Every loop below skips
    // these rows.
    const auto isNullRow = [&](vector_size_t row) {
      return decodedArray.isNullAt(row) || decodedN.isNullAt(row);
    };

    exec::LocalSelectivityVector remainingRows(context, rows);
    context.applyToSelectedNoThrow(*remainingRows, [&](vector_size_t row) {
      if (isNullRow(row)) {
        return;
      }
      int64_t n = decodedN.valueAt<int32_t>(row);
      VELOX_USER_CHECK_GE(
          n, 0, "Parameter n: {} to ARRAY_TOP_N is negative", n);
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
    // 2-argument function short-circuits n == 0 the same way.
    exec::LocalSelectivityVector rowsToRank(context, *remainingRows);
    remainingRows->applyToSelected([&](vector_size_t row) {
      if (isNullRow(row) || decodedN.valueAt<int32_t>(row) == 0) {
        rowsToRank->setValid(row, false);
      }
    });
    rowsToRank->updateBounds();

    // Positions of the elements to emit, ordered per row. Initialized to
    // identity so that rows the ranking loop skips (arrays with a single
    // element) still find their only element at the array's offset.
    BufferPtr indices = allocateIndices(numElements, context.pool());
    auto* rawIndices = indices->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < numElements; ++i) {
      rawIndices[i] = i;
    }

    if (rowsToRank->hasSelections()) {
      SelectivityVector validRowsInReusedResult =
          toElementRows<ArrayVector>(numElements, *rowsToRank, flatArray.get());

      VectorPtr transformedElements;

      // A NULL lambda ranks by the elements themselves, matching the
      // 2-argument simple-function behavior.
      const bool isLambdaNull = args[2]->type()->kind() == TypeKind::UNKNOWN;
      if (isLambdaNull) {
        transformedElements = arrayElements;
      } else {
        std::vector<VectorPtr> lambdaArgs = {arrayElements};
        applyLambdaToElements<ArrayVector>(
            args[2],
            *rowsToRank,
            numElements,
            flatArray,
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
        CompareFlags flags;

        bool operator()(vector_size_t leftIdx, vector_size_t rightIdx) const {
          if (leftIdx == rightIdx) {
            return false;
          }

          bool leftNull = decodedTransformed->isNullAt(leftIdx);
          bool rightNull = decodedTransformed->isNullAt(rightIdx);

          // Nulls sort last in descending top-n, so a null is never "greater"
          // than a non-null. Two nulls tie, and break like equal values below.
          if (leftNull && rightNull) {
            return leftIdx > rightIdx;
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

          // Arrays have no key to break ties on, unlike map_top_n_keys, so
          // ties break on input position: the later element ranks higher, so
          // the min-heap keeps it and emits it before earlier ties. This
          // makes the result deterministic and is documented in array.rst.
          if (result.value() == 0) {
            return leftIdx > rightIdx;
          }

          return result.value() > 0;
        }
      };

      GreaterThanComparator comparator{
          decodedTransformed.get(), baseTransformed, flags};

      // Use applyToSelectedNoThrow so that compare()-thrown errors are captured
      // per row (letting try() convert them to nulls) instead of escaping.
      context.applyToSelectedNoThrow(*rowsToRank, [&](vector_size_t row) {
        auto arrayOffset = flatArray->offsetAt(row);
        auto arraySize = flatArray->sizeAt(row);

        // 'rowsToRank' excludes null rows and rows with n == 0, so an array
        // with at most one element is the only case that needs no ranking.
        if (arraySize <= 1) {
          return;
        }

        int64_t n = decodedN.valueAt<int32_t>(row);
        auto resultSize = static_cast<vector_size_t>(
            std::min(n, static_cast<int64_t>(arraySize)));

        std::priority_queue<
            vector_size_t,
            std::vector<vector_size_t>,
            GreaterThanComparator>
            minHeap(comparator);

        // Build heap with top N elements.
        for (vector_size_t i = 0; i < arraySize; ++i) {
          auto idx = arrayOffset + i;
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
          rawIndices[arrayOffset + i] = topIndices[i];
        }
      });

      // Drop rows whose heap loop threw, so they don't contribute to
      // totalElements or get copied below.
      context.deselectErrors(*remainingRows);
    }

    vector_size_t totalElements = 0;
    remainingRows->applyToSelected([&](vector_size_t row) {
      if (isNullRow(row)) {
        return;
      }
      auto arraySize = flatArray->sizeAt(row);
      int64_t n = decodedN.valueAt<int32_t>(row);
      totalElements = checkedPlus<vector_size_t>(
          totalElements,
          static_cast<vector_size_t>(
              std::min(n, static_cast<int64_t>(arraySize))));
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

// array_top_n(array(T), integer, function(T, U)) -> array(T)
// array_top_n(array(T), integer, unknown) -> array(T)
// The second signature handles NULL as the third parameter, which should
// behave the same as the 2-argument version.
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
      exec::FunctionSignatureBuilder()
          .orderableTypeVariable("T")
          .returnType("array(T)")
          .argumentType("array(T)")
          .argumentType("integer")
          .argumentType("unknown")
          .build(),
  };
}

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_array_top_n,
    signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<ArrayTopNTransformFunction>());

} // namespace facebook::velox::functions
