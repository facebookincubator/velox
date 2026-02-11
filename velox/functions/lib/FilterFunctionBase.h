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
#include "velox/functions/lib/RowsTranslationUtil.h"
#include "velox/vector/FunctionVector.h"

namespace facebook::velox::functions {

/// Base class for filter functions (both array and map).
/// Provides a shared doApply() template for evaluating lambda predicates
/// and collecting indices of matching elements.
class FilterFunctionBase : public exec::VectorFunction {
 protected:
  // Applies filter functions to elements of maps or arrays and returns the
  // number of elements that passed the filters. Stores the number of elements
  // in each array or map that passed the filter in resultSizes. Stores the
  // indices of passing elements in selectedIndices. resultOffsets is populated
  // assuming sequential layout of the passing elements.
  template <typename T>
  static vector_size_t doApply(
      const SelectivityVector& rows,
      const std::shared_ptr<T>& input,
      const VectorPtr& lambda,
      const std::vector<VectorPtr>& lambdaArgs,
      exec::EvalCtx& context,
      BufferPtr& resultOffsets,
      BufferPtr& resultSizes,
      BufferPtr& selectedIndices) {
    const auto* inputOffsets = input->rawOffsets();
    const auto* inputSizes = input->rawSizes();

    auto* pool = context.pool();
    resultSizes = allocateSizes(rows.end(), pool);
    resultOffsets = allocateOffsets(rows.end(), pool);
    auto* rawResultSizes = resultSizes->asMutable<vector_size_t>();
    auto* rawResultOffsets = resultOffsets->asMutable<vector_size_t>();

    const auto numElements = lambdaArgs[0]->size();

    selectedIndices = allocateIndices(numElements, pool);
    auto* rawSelectedIndices = selectedIndices->asMutable<vector_size_t>();

    vector_size_t numSelected = 0;

    auto elementToTopLevelRows =
        getElementToTopLevelRows(numElements, rows, input.get(), pool);

    exec::LocalDecodedVector bitsDecoder(context);
    auto iter = lambda->asUnchecked<FunctionVector>()->iterator(&rows);
    while (auto entry = iter.next()) {
      auto elementRows =
          toElementRows<T>(numElements, *entry.rows, input.get());
      auto wrapCapture =
          toWrapCapture<T>(numElements, entry.callable, *entry.rows, input);

      VectorPtr bits;
      entry.callable->apply(
          elementRows,
          nullptr,
          wrapCapture,
          &context,
          lambdaArgs,
          elementToTopLevelRows,
          &bits);
      bitsDecoder.get()->decode(*bits, elementRows);
      entry.rows->applyToSelected([&](vector_size_t row) {
        if (input->isNullAt(row)) {
          return;
        }
        auto size = inputSizes[row];
        auto offset = inputOffsets[row];
        rawResultOffsets[row] = numSelected;
        for (auto i = 0; i < size; ++i) {
          if (!bitsDecoder.get()->isNullAt(offset + i) &&
              bitsDecoder.get()->valueAt<bool>(offset + i)) {
            ++rawResultSizes[row];
            rawSelectedIndices[numSelected] = offset + i;
            ++numSelected;
          }
        }
      });
    }

    selectedIndices->setSize(numSelected * sizeof(vector_size_t));

    return numSelected;
  }
};

/// Array filter function base class.
/// Subclasses can override addIndexVector() to provide additional lambda
/// arguments (e.g., element index for Spark's filter function).
class ArrayFilterFunctionBase : public FilterFunctionBase {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2);
    exec::LocalDecodedVector arrayDecoder(context, *args[0], rows);
    auto& decodedArray = *arrayDecoder.get();

    auto flatArray = flattenArray(rows, args[0], decodedArray);

    VectorPtr elements = flatArray->elements();
    std::vector<VectorPtr> lambdaArgs = {elements};

    // Allow subclasses to add additional lambda arguments (e.g., index vector).
    addIndexVector(args, flatArray, elements->size(), context, lambdaArgs);

    BufferPtr resultSizes;
    BufferPtr resultOffsets;
    BufferPtr selectedIndices;
    auto numSelected = doApply(
        rows,
        flatArray,
        args[1],
        lambdaArgs,
        context,
        resultOffsets,
        resultSizes,
        selectedIndices);

    // Filter can pass along very large elements vectors that can hold onto
    // memory and copy operations on them can further put memory pressure. We
    // try to flatten them if the dictionary layer is much smaller than the
    // elements vector.
    auto wrappedElements = numSelected ? BaseVector::wrapInDictionary(
                                             BufferPtr(nullptr),
                                             std::move(selectedIndices),
                                             numSelected,
                                             std::move(elements),
                                             true /*flattenIfRedundant*/)
                                       : nullptr;
    // Set nulls for rows not present in 'rows'.
    BufferPtr newNulls = addNullsForUnselectedRows(flatArray, rows);
    auto localResult = std::make_shared<ArrayVector>(
        flatArray->pool(),
        flatArray->type(),
        std::move(newNulls),
        rows.end(),
        std::move(resultOffsets),
        std::move(resultSizes),
        wrappedElements);
    context.moveOrCopyResult(localResult, rows, result);
  }

  /// Returns the base signature: array(T), function(T, boolean) -> array(T).
  /// Subclasses can call this and add additional signatures.
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("array(T)")
                .argumentType("array(T)")
                .argumentType("function(T,boolean)")
                .build()};
  }

 protected:
  /// Override this method to add additional arguments to the lambda.
  /// Default implementation does nothing (element-only filter).
  virtual void addIndexVector(
      const std::vector<VectorPtr>& /*args*/,
      const ArrayVectorPtr& /*flatArray*/,
      vector_size_t /*numElements*/,
      exec::EvalCtx& /*context*/,
      std::vector<VectorPtr>& /*lambdaArgs*/) const {}
};

} // namespace facebook::velox::functions
