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
#include "velox/expression/NullIfExpr.h"
#include "velox/expression/ExprConstants.h"

namespace facebook::velox::exec {

NullIfExpr::NullIfExpr(
    std::vector<ExprPtr>&& inputs,
    std::shared_ptr<CastExpr> castExpr,
    bool trackCpuUsage)
    : SpecialForm(
          SpecialFormKind::kNullIf,
          inputs[0]->type(),
          inputs,
          expression::kNullIf,
          /*supportsFlatNoNullsFastPath=*/false,
          trackCpuUsage),
      castExpr_(std::move(castExpr)),
      needsCastA_(
          castExpr_ && !inputs_[0]->type()->equivalent(*castExpr_->type())),
      needsCastB_(
          castExpr_ && !inputs_[1]->type()->equivalent(*castExpr_->type())) {
  VELOX_CHECK_EQ(inputs_.size(), 2, "NULLIF requires exactly 2 inputs");
  if (castExpr_ == nullptr) {
    const auto& a = inputs_[0]->type();
    const auto& b = inputs_[1]->type();
    VELOX_CHECK(
        a->equivalent(*b),
        "NULLIF inputs must have the same type when castExpr is not provided: {} vs {}",
        a->toString(),
        b->toString());
  }
}

void NullIfExpr::evalSpecialForm(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  // Evaluate both inputs.
  VectorPtr aResult;
  inputs_[0]->eval(rows, context, aResult);

  VectorPtr bResult;
  inputs_[1]->eval(rows, context, bResult);

  // Cast to common type for comparison if needed.
  VectorPtr castA;
  VectorPtr castB;
  if (needsCastA_) {
    castExpr_->apply(
        rows, aResult, context, inputs_[0]->type(), castExpr_->type(), castA);
  }
  if (needsCastB_) {
    castExpr_->apply(
        rows, bResult, context, inputs_[1]->type(), castExpr_->type(), castB);
  }

  const auto& compareA = castA ? castA : aResult;
  const auto& compareB = castB ? castB : bResult;

  // Find rows where a does not equal b. These rows return a's value.
  // Rows where a equals b return NULL. Rows where either is null are
  // indeterminate and return a's value (which may itself be null).
  LocalSelectivityVector nonNullRows(context, rows);

  // TODO: Spark uses kNullAsValue which produces different results for complex
  // types with nested nulls, e.g. NULLIF(ARRAY[1, NULL], ARRAY[1, NULL]).
  rows.applyToSelected([&](auto row) {
    auto equal = compareA->equalValueAt(
        compareB.get(),
        row,
        row,
        CompareFlags::NullHandlingMode::kNullAsIndeterminate);
    if (equal.has_value() && equal.value()) {
      nonNullRows->setValid(row, false);
    }
  });
  nonNullRows->updateBounds();

  if (nonNullRows->isSubset(rows) && rows.isSubset(*nonNullRows)) {
    // No matches — return a as is.
    context.moveOrCopyResult(aResult, rows, result);
    return;
  }

  if (!nonNullRows->hasSelections()) {
    // All rows match — return constant null.
    result = BaseVector::createNullConstant(type(), rows.end(), context.pool());
    return;
  }

  // Some rows match — copy only non-null rows from a, set remaining to null.
  BaseVector::ensureWritable(rows, type(), context.pool(), result);
  result->copy(aResult.get(), *nonNullRows, /*toSourceRow=*/nullptr);

  LocalSelectivityVector nullRows(context, rows);
  nullRows->deselect(*nonNullRows);
  result->addNulls(*nullRows);
}

// static
ExprPtr NullIfExpr::create(
    std::vector<ExprPtr>&& inputs,
    const TypePtr& commonType,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_USER_CHECK_EQ(
      inputs.size(),
      2,
      "NULLIF requires exactly 2 arguments, received {}.",
      inputs.size());

  // Create a CastExpr for casting evaluated vectors to commonType. The child
  // expression passed here is not used — we call apply() directly with
  // pre-evaluated vectors. We use CastCallToSpecialForm to get proper hooks.
  std::shared_ptr<CastExpr> castExpr;
  if (*inputs[0]->type() != *commonType || *inputs[1]->type() != *commonType) {
    CastCallToSpecialForm castFactory;
    std::vector<ExprPtr> castInputs = {inputs[0]};
    castExpr =
        std::dynamic_pointer_cast<CastExpr>(castFactory.constructSpecialForm(
            commonType,
            std::move(castInputs),
            /*trackCpuUsage=*/false,
            config));
  }

  return std::make_shared<NullIfExpr>(
      std::move(inputs), std::move(castExpr), trackCpuUsage);
}

} // namespace facebook::velox::exec
