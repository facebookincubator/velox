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

class AllMatchFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
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
      auto elementRows =
          toElementRows<ArrayVector>(numElements, *entry.rows, flatArray.get());
      auto wrapCapture = toWrapCapture<ArrayVector>(
          numElements, entry.callable, *entry.rows, flatArray);
      bool hasVeloxUserError = false;
      entry.callable->apply(
          elementRows,
          finalSelection,
          wrapCapture,
          &context,
          lambdaArgs,
          elementToTopLevelRows,
          &matchBits);

      bitsDecoder.get()->decode(*matchBits, elementRows);
      entry.rows->applyToSelected([&](vector_size_t row) {
        auto size = sizes[row];
        auto offset = offsets[row];
        auto allMatch = true;
        auto hasNull = false;
        for (auto i = 0; i < size; ++i) {
          auto idx = offset + i;
          if (bitsDecoder->isNullAt(idx)) {
            hasNull = true;
          } else if (!bitsDecoder->valueAt<bool>(idx)) {
            allMatch = false;
            break;
          }
        }

        if (hasNull && allMatch) {
          flatResult->setNull(row, true);
        } else {
          flatResult->set(row, allMatch);
        }
      });
    }
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // array(T), function(T) -> boolean
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("boolean")
                .argumentType("array(T)")
                .argumentType("function(T, boolean)")
                .build()};
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_all_match,
    AllMatchFunction::signatures(),
    std::make_unique<AllMatchFunction>());

} // namespace facebook::velox::functions
