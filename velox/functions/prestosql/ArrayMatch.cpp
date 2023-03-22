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
#include "velox/vector/FunctionVector.h"

namespace facebook::velox::functions {
namespace {

enum class MatchMethod { kAll = 0, kAny = 1, kNone = 2 };

template <MatchMethod matchMethod>
class MatchFunction : public exec::VectorFunction {
 private:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    exec::LocalDecodedVector arrayDecoder(context, *args[0], rows);
    auto& decodedArray = *arrayDecoder.get();

    auto flatArray = flattenArray(rows, args[0], decodedArray);
    auto offsets = flatArray->rawOffsets();
    auto sizes = flatArray->rawSizes();

    std::vector<VectorPtr> lambdaArgs = {flatArray->elements()};
    auto numElements = flatArray->elements()->size();

    SelectivityVector finalSelection;
    if (!context.isFinalSelection()) {
      finalSelection = toElementRows<ArrayVector>(
          numElements, *context.finalSelection(), flatArray.get());
    }

    VectorPtr matchBits;
    auto elementToTopLevelRows = getElementToTopLevelRows(
        numElements, rows, flatArray.get(), context.pool());

    // Loop over lambda functions and apply these to elements of the base array,
    // in most cases there will be only one function and the loop will run once.
    context.ensureWritable(rows, BOOLEAN(), result);
    auto flatResult = result->asFlatVector<bool>();
    exec::LocalDecodedVector bitsDecoder(context);
    auto it = args[1]->asUnchecked<FunctionVector>()->iterator(&rows);

    while (auto entry = it.next()) {
      ErrorVectorPtr elementErrors;
      auto elementRows =
          toElementRows<ArrayVector>(numElements, *entry.rows, flatArray.get());
      auto wrapCapture = toWrapCapture<ArrayVector>(
          numElements, entry.callable, *entry.rows, flatArray);
      entry.callable->applyNoThrow(
          elementRows,
          finalSelection,
          wrapCapture,
          &context,
          lambdaArgs,
          elementErrors,
          &matchBits);

      bitsDecoder.get()->decode(*matchBits, elementRows);
      entry.rows->applyToSelected([&](vector_size_t row) {
        applyInternal(
            flatResult,
            context,
            row,
            offsets,
            sizes,
            elementErrors,
            bitsDecoder);
      });
    }
  }

  static FOLLY_ALWAYS_INLINE bool hasError(
      const ErrorVectorPtr& errors,
      int idx) {
    return errors && idx < errors->size() && !errors->isNullAt(idx);
  }

 private:
  void applyInternal(
      FlatVector<bool>* flatResult,
      exec::EvalCtx& context,
      vector_size_t row,
      const vector_size_t* offsets,
      const vector_size_t* sizes,
      const ErrorVectorPtr& elementErrors,
      const exec::LocalDecodedVector& bitsDecoder) const {
    auto size = sizes[row];
    auto offset = offsets[row];

    // All, none, and any match have different and similar logic intertwined in
    // terms of the initial value, flip condition, and the result finalization:
    //
    // Initial value:
    //  All is true
    //  Any is false
    //  None is true
    //
    // Flip logic:
    //  All flips once encounter an unmatched element
    //  Any flips once encounter a matched element
    //  None flips once encounter a matched element
    //
    // Result finalization:
    //  All: ignore the error and null if one or more elements are unmatched and
    //  return false Any: ignore the error and null if one or more elements
    //  matched and return true None: ignore the error and null if one or more
    //  elements matched and return false
    auto match = (matchMethod != MatchMethod::kAny);
    auto hasNull = false;
    std::exception_ptr errorPtr{nullptr};
    for (auto i = 0; i < size; ++i) {
      auto idx = offset + i;
      if (hasError(elementErrors, idx)) {
        errorPtr = *std::static_pointer_cast<std::exception_ptr>(
            elementErrors->valueAt(idx));
        continue;
      }

      if (bitsDecoder->isNullAt(idx)) {
        hasNull = true;
      } else if (matchMethod == MatchMethod::kAll) {
        if (!bitsDecoder->valueAt<bool>(idx)) {
          match = !match;
          break;
        }
      } else {
        if (bitsDecoder->valueAt<bool>(idx)) {
          match = !match;
          break;
        }
      }
    }

    if ((matchMethod == MatchMethod::kAny) == match) {
      flatResult->set(row, match);
    } else if (errorPtr) {
      context.setError(row, errorPtr);
    } else if (hasNull) {
      flatResult->setNull(row, true);
    } else {
      flatResult->set(row, match);
    }
  }
};

class AllMatchFunction : public MatchFunction<MatchMethod::kAll> {};
class AnyMatchFunction : public MatchFunction<MatchMethod::kAny> {};
class NoneMatchFunction : public MatchFunction<MatchMethod::kNone> {};

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  // array(T), function(T) -> boolean
  return {exec::FunctionSignatureBuilder()
              .typeVariable("T")
              .returnType("boolean")
              .argumentType("array(T)")
              .argumentType("function(T, boolean)")
              .build()};
}

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_all_match,
    signatures(),
    std::make_unique<AllMatchFunction>());

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_any_match,
    signatures(),
    std::make_unique<AnyMatchFunction>());

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_none_match,
    signatures(),
    std::make_unique<NoneMatchFunction>());

} // namespace facebook::velox::functions
