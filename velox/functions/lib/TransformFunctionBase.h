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

#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"

namespace facebook::velox::functions {

/// Base class for array transform functions.
/// Subclasses can override addIndexVector() to provide additional lambda
/// arguments (e.g., element index for Spark's transform function).
class TransformFunctionBase : public exec::VectorFunction {
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

    auto numElements = flatArray->elements()->size();
    std::vector<VectorPtr> lambdaArgs = {flatArray->elements()};

    // Allow subclasses to add additional lambda arguments (e.g., index vector).
    addIndexVector(args, flatArray, numElements, context, lambdaArgs);

    SelectivityVector validRowsInReusedResult =
        toElementRows<ArrayVector>(numElements, rows, flatArray.get());

    // Transformed elements.
    VectorPtr newElements;

    // Apply lambda to elements.
    applyLambdaToElements<ArrayVector>(
        args[1],
        rows,
        numElements,
        flatArray,
        lambdaArgs,
        validRowsInReusedResult,
        context,
        newElements);

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

  /// Returns the base signature: array(T), function(T, U) -> array(U).
  /// Subclasses can call this and add additional signatures.
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .typeVariable("U")
                .returnType("array(U)")
                .argumentType("array(T)")
                .argumentType("function(T, U)")
                .build()};
  }

 protected:
  /// Override this method to add additional arguments to the lambda.
  /// Default implementation does nothing (element-only transform).
  /// @param args The original function arguments.
  /// @param flatArray The flattened input array.
  /// @param numElements Total number of elements across all arrays.
  /// @param context The evaluation context.
  /// @param lambdaArgs Output vector to append additional arguments to.
  virtual void addIndexVector(
      const std::vector<VectorPtr>& /*args*/,
      const ArrayVectorPtr& /*flatArray*/,
      vector_size_t /*numElements*/,
      exec::EvalCtx& /*context*/,
      std::vector<VectorPtr>& /*lambdaArgs*/) const {}
};

} // namespace facebook::velox::functions
