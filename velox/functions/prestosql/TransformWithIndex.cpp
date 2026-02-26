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
#include "velox/functions/lib/RowsTranslationUtil.h"
#include "velox/vector/FunctionVector.h"

namespace facebook::velox::functions {
namespace {

// transform_with_index(array(T), function(T, bigint, U)) -> array(U)
//
// Transforms each element of an array using the provided function.
// The lambda function receives both the element and its 1-based index.
class TransformWithIndexFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2);

    // Flatten input array.
    exec::LocalDecodedVector arrayDecoder(context, *args[0], rows);
    auto& decodedArray = *arrayDecoder.get();

    auto flatArray = flattenArray(rows, args[0], decodedArray);

    auto newNumElements = flatArray->elements()->size();

    // Create indices vector (1-based indexing for Presto compatibility)
    auto indices = createIndicesVector(flatArray, rows, context.pool());

    std::vector<VectorPtr> lambdaArgs = {flatArray->elements(), indices};

    SelectivityVector validRowsInReusedResult =
        toElementRows<ArrayVector>(newNumElements, rows, flatArray.get());

    VectorPtr newElements;

    auto elementToTopLevelRows = getElementToTopLevelRows(
        newNumElements, rows, flatArray.get(), context.pool());

    // Loop over lambda functions and apply these to elements of the base array;
    // in most cases there will be only one function and the loop will run once
    auto it = args[1]->asUnchecked<FunctionVector>()->iterator(&rows);
    while (auto entry = it.next()) {
      auto elementRows = toElementRows<ArrayVector>(
          newNumElements, *entry.rows, flatArray.get());
      auto wrapCapture = toWrapCapture<ArrayVector>(
          newNumElements, entry.callable, *entry.rows, flatArray);

      entry.callable->apply(
          elementRows,
          &validRowsInReusedResult,
          wrapCapture,
          &context,
          lambdaArgs,
          elementToTopLevelRows,
          &newElements);
    }

    // Set nulls for rows not present in 'rows'.
    BufferPtr newNulls = addNullsForUnselectedRows(flatArray, rows);

    VectorPtr localResult = std::make_shared<ArrayVector>(
        flatArray->pool(),
        outputType,
        std::move(newNulls),
        rows.end(),
        flatArray->offsets(),
        flatArray->sizes(),
        newElements);
    context.moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // array(T), function(T, bigint, U) -> array(U)
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .typeVariable("U")
                .returnType("array(U)")
                .argumentType("array(T)")
                .argumentType("function(T, bigint, U)")
                .build()};
  }

 private:
  // Creates a vector of 1-based indices for each element in the flattened
  // array. For example, if we have arrays [[a, b, c], [d, e]], the indices
  // will be [1, 2, 3, 1, 2] (1-based for each array).
  static VectorPtr createIndicesVector(
      const std::shared_ptr<ArrayVector>& flatArray,
      const SelectivityVector& rows,
      memory::MemoryPool* pool) {
    const auto numElements = flatArray->elements()->size();
    auto indicesVector =
        BaseVector::create<FlatVector<int64_t>>(BIGINT(), numElements, pool);
    auto* rawIndices = indicesVector->mutableRawValues();

    const auto* rawOffsets = flatArray->rawOffsets();
    const auto* rawSizes = flatArray->rawSizes();
    const auto* rawNulls = flatArray->rawNulls();

    rows.applyToSelected([&](vector_size_t row) {
      if (rawNulls && bits::isBitNull(rawNulls, row)) {
        return;
      }
      const auto offset = rawOffsets[row];
      const auto size = rawSizes[row];
      for (vector_size_t i = 0; i < size; ++i) {
        // Use 1-based indexing for Presto compatibility
        rawIndices[offset + i] = i + 1;
      }
    });

    return indicesVector;
  }
};
} // namespace

/// transform_with_index is null preserving for the array. But since an
/// expr tree with a lambda depends on all named fields, including
/// captures, a null in a capture does not automatically make a
/// null result.

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_transform_with_index,
    TransformWithIndexFunction::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<TransformWithIndexFunction>());

} // namespace facebook::velox::functions
