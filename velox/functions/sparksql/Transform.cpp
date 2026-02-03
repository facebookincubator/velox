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

#include "velox/common/base/BitUtil.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"
#include "velox/vector/FunctionVector.h"

namespace facebook::velox::functions::sparksql {
namespace {

/// Spark's transform function supports both signatures:
/// 1. transform(array, x -> expr) - element only
/// 2. transform(array, (x, i) -> expr) - element + index (Spark-specific)
///
/// See Spark documentation:
/// https://spark.apache.org/docs/latest/api/sql/index.html#transform
class TransformFunction : public exec::VectorFunction {
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

    // Determine if we need to pass index to the lambda.
    // Check the lambda function type to see if it expects 2 input arguments.
    // function(T, U) has 2 children (input T, output U) -> 1 input arg.
    // function(T, integer, U) has 3 children (input T, index integer, output U)
    // -> 2 input args.
    auto functionType = args[1]->type();
    bool withIndex = functionType->size() == 3;

    std::vector<VectorPtr> lambdaArgs = {flatArray->elements()};

    // If lambda expects index, create index vector.
    if (withIndex) {
      auto indexVector = createIndexVector(flatArray, newNumElements, context);
      lambdaArgs.push_back(indexVector);
    }

    SelectivityVector validRowsInReusedResult =
        toElementRows<ArrayVector>(newNumElements, rows, flatArray.get());

    // Transformed elements.
    VectorPtr newElements;

    auto elementToTopLevelRows = getElementToTopLevelRows(
        newNumElements, rows, flatArray.get(), context.pool());

    // Loop over lambda functions and apply these to elements of the base array.
    // In most cases there will be only one function and the loop will run once.
    auto lambdaIt = args[1]->asUnchecked<FunctionVector>()->iterator(&rows);
    while (auto entry = lambdaIt.next()) {
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
    return {
        // Signature 1: array(T), function(T, U) -> array(U) (element only)
        exec::FunctionSignatureBuilder()
            .typeVariable("T")
            .typeVariable("U")
            .returnType("array(U)")
            .argumentType("array(T)")
            .argumentType("function(T, U)")
            .build(),
        // Signature 2: array(T), function(T, integer, U) -> array(U) (element +
        // index). Spark uses IntegerType (32-bit) for the index parameter.
        exec::FunctionSignatureBuilder()
            .typeVariable("T")
            .typeVariable("U")
            .returnType("array(U)")
            .argumentType("array(T)")
            .argumentType("function(T, integer, U)")
            .build()};
  }

 private:
  /// Creates an index vector where each element contains its position within
  /// its respective array. For example, if we have arrays [a, b] and [c, d, e],
  /// the index vector will be [0, 1, 0, 1, 2].
  /// Spark uses IntegerType (32-bit) for the index.
  static VectorPtr createIndexVector(
      const std::shared_ptr<ArrayVector>& flatArray,
      vector_size_t numElements,
      exec::EvalCtx& context) {
    auto* pool = context.pool();
    auto indexVector =
        BaseVector::create<FlatVector<int32_t>>(INTEGER(), numElements, pool);

    auto* rawOffsets = flatArray->rawOffsets();
    auto* rawSizes = flatArray->rawSizes();
    auto* rawNulls = flatArray->rawNulls();
    auto* rawIndices = indexVector->mutableRawValues();

    for (vector_size_t row = 0; row < flatArray->size(); ++row) {
      // Skip null arrays.
      if (rawNulls && bits::isBitNull(rawNulls, row)) {
        continue;
      }
      auto offset = rawOffsets[row];
      auto size = rawSizes[row];
      for (vector_size_t i = 0; i < size; ++i) {
        rawIndices[offset + i] = i;
      }
    }

    return indexVector;
  }
};

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_spark_transform,
    TransformFunction::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<TransformFunction>());

} // namespace facebook::velox::functions::sparksql
