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

#include "velox/functions/sparksql/specialforms/AtLeastNNonNulls.h"
#include "velox/expression/SpecialForm.h"

using namespace facebook::velox::exec;

namespace facebook::velox::functions::sparksql {
namespace {
class AtLeastNNonNullsExpr : public SpecialForm {
 public:
  AtLeastNNonNullsExpr(
      TypePtr type,
      std::vector<ExprPtr>&& inputs,
      bool inputsSupportFlatNoNullsFastPath)
      : SpecialForm(
            std::move(type),
            std::move(inputs),
            AtLeastNNonNullsCallToSpecialForm::kAtLeastNNonNulls,
            inputsSupportFlatNoNullsFastPath,
            false /* trackCpuUsage */) {}

  void evalSpecialForm(
      const SelectivityVector& rows,
      EvalCtx& context,
      VectorPtr& result) override {
    context.ensureWritable(rows, type(), result);
    (*result).clearNulls(rows);
    auto flatResult = result->asFlatVector<bool>();
    LocalSelectivityVector activeRowsHolder(context, rows);
    auto activeRows = activeRowsHolder.get();
    VELOX_DCHECK_NOT_NULL(activeRows);
    // Load 'n' from the first input.
    VectorPtr vector;
    inputs_[0]->eval(*activeRows, context, vector);
    VELOX_CHECK(!context.errors());
    VELOX_USER_CHECK(
        vector->isConstantEncoding(),
        "The first input should be constant encoding.");
    auto constVector = vector->asUnchecked<ConstantVector<int32_t>>();
    VELOX_USER_CHECK(
        !constVector->isNullAt(0), "The first parameter should not be null.");
    const int32_t n = constVector->valueAt(0);
    auto values = flatResult->mutableValues(rows.end())->asMutable<uint64_t>();
    // If 'n' <= 0, set result to all true.
    if (n <= 0) {
      bits::orBits(values, rows.asRange().bits(), rows.begin(), rows.end());
      return;
    } else {
      bits::andWithNegatedBits(
          values, rows.asRange().bits(), rows.begin(), rows.end());
      // If 'n' > inputs_.size() - 1, result should be all false.
      if (n > inputs_.size() - 1) {
        return;
      }
    }
    // Create a temp buffer to track count of non null values for active rows.
    auto nonNullCounts =
        AlignedBuffer::allocate<int32_t>(activeRows->size(), context.pool(), 0);
    auto* rawNonNullCounts = nonNullCounts->asMutable<int32_t>();
    for (int32_t i = 1; i < inputs_.size(); ++i) {
      VectorPtr input;
      inputs_[i]->eval(*activeRows, context, input);
      if (context.errors()) {
        context.deselectErrors(*activeRows);
      }
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          updateResultTyped,
          inputs_[i]->type()->kind(),
          input.get(),
          n,
          context,
          rawNonNullCounts,
          flatResult,
          activeRows);
      if (activeRows->countSelected() == 0) {
        break;
      }
    }
  }

 private:
  void computePropagatesNulls() override {
    propagatesNulls_ = false;
  }

  template <TypeKind Kind>
  void updateResultTyped(
      BaseVector* input,
      int32_t n,
      EvalCtx& context,
      int32_t* rawNonNullCounts,
      FlatVector<bool>* result,
      SelectivityVector* activeRows) {
    using Type = typename TypeTraits<Kind>::NativeType;
    exec::LocalDecodedVector decodedVector(context);
    decodedVector.get()->decode(*input, *activeRows);
    bool updateBounds = false;
    activeRows->applyToSelected([&](auto row) {
      bool nonNull = !decodedVector->isNullAt(row);
      if constexpr (
          std::is_same_v<Type, double> || std::is_same_v<Type, float>) {
        nonNull = nonNull && !std::isnan(decodedVector->valueAt<Type>(row));
      }
      if (nonNull) {
        rawNonNullCounts[row]++;
        if (rawNonNullCounts[row] >= n) {
          updateBounds = true;
          result->set(row, true);
          // Exclude the 'row' from active rows after finding 'n' non-NULL /
          // non-NaN values.
          activeRows->setValid(row, false);
        }
      }
    });
    if (updateBounds) {
      activeRows->updateBounds();
    }
  }
};
} // namespace

TypePtr AtLeastNNonNullsCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  VELOX_USER_CHECK_GT(
      argTypes.size(),
      1,
      "AtLeastNNonNulls expect to receive at least 2 arguments.");
  VELOX_USER_CHECK(
      argTypes[0]->isInteger(),
      "The first input type should be INTEGER but Actual {}.",
      argTypes[0]->toString());
  return BOOLEAN();
}

ExprPtr AtLeastNNonNullsCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<ExprPtr>&& compiledChildren,
    bool /* trackCpuUsage */,
    const core::QueryConfig& /*config*/) {
  return std::make_shared<AtLeastNNonNullsExpr>(
      type,
      std::move(compiledChildren),
      Expr::allSupportFlatNoNullsFastPath(compiledChildren));
}
} // namespace facebook::velox::functions::sparksql
