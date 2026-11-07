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

#include "velox/functions/sparksql/specialforms/Pmod.h"

#include <algorithm>

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/SpecialForm.h"
#include "velox/functions/sparksql/Pmod.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Prevent eager argument evaluation and field-null pruning from changing
// Spark's right-to-left evaluation and error precedence.
class PmodExpr : public exec::SpecialForm {
 public:
  PmodExpr(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& inputs,
      bool trackCpuUsage,
      bool failOnError,
      bool checkZeroBeforeLeft)
      : SpecialForm(
            exec::SpecialFormKind::kCustom,
            type,
            std::move(inputs),
            PmodCallToSpecialForm::kPmodWithMode,
            false,
            trackCpuUsage),
        failOnError_(failOnError),
        checkZeroBeforeLeft_(checkZeroBeforeLeft) {}

  bool isConditional() const override {
    return true;
  }

  void evalSpecialForm(
      const SelectivityVector& rows,
      exec::EvalCtx& context,
      VectorPtr& result) override {
    VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        evalTyped, type()->kind(), rows, context, result);
  }

 private:
  void computePropagatesNulls() override {
    // A null left operand must not hide an error in the right operand.
    propagatesNulls_ = false;
  }

  template <TypeKind Kind>
  void evalTyped(
      const SelectivityVector& rows,
      exec::EvalCtx& context,
      VectorPtr& result) {
    using T = typename TypeTraits<Kind>::NativeType;
    if constexpr (
        Kind == TypeKind::TINYINT || Kind == TypeKind::SMALLINT ||
        Kind == TypeKind::INTEGER || Kind == TypeKind::BIGINT ||
        Kind == TypeKind::REAL || Kind == TypeKind::DOUBLE) {
      context.ensureWritable(rows, type(), result);
      result->addNulls(rows);
      exec::LocalSelectivityVector remaining(context, rows);
      auto& activeRows = *remaining.get();

      VectorPtr divisor;
      inputs_[1]->eval(activeRows, context, divisor);
      context.deselectErrors(activeRows);
      if (!activeRows.hasSelections()) {
        return;
      }
      exec::LocalDecodedVector decodedDivisor(context, *divisor, activeRows);
      context.applyToSelectedNoThrow(activeRows, [&](auto row) {
        if (decodedDivisor->isNullAt(row)) {
          activeRows.setValid(row, false);
        } else if (decodedDivisor->valueAt<T>(row) == 0) {
          if (!failOnError_) {
            activeRows.setValid(row, false);
          } else if (checkZeroBeforeLeft_) {
            VELOX_USER_FAIL("Division by zero");
          }
        }
      });
      activeRows.updateBounds();
      context.deselectErrors(activeRows);
      if (!activeRows.hasSelections()) {
        return;
      }

      VectorPtr dividend;
      inputs_[0]->eval(activeRows, context, dividend);
      context.deselectErrors(activeRows);
      if (!activeRows.hasSelections()) {
        return;
      }
      exec::LocalDecodedVector decodedDividend(context, *dividend, activeRows);
      auto* flatResult = result->asFlatVector<T>();
      context.applyToSelectedNoThrow(activeRows, [&](auto row) {
        if (decodedDividend->isNullAt(row)) {
          return;
        }
        const auto divisorValue = decodedDivisor->valueAt<T>(row);
        if (divisorValue == 0) {
          VELOX_USER_FAIL("Division by zero");
        }
        flatResult->set(
            row, computePmod(decodedDividend->valueAt<T>(row), divisorValue));
        flatResult->setNull(row, false);
      });
    } else {
      VELOX_UNREACHABLE();
    }
  }

  const bool failOnError_;
  const bool checkZeroBeforeLeft_;
};

bool constantBoolean(const exec::ExprPtr& expression) {
  auto constant = std::dynamic_pointer_cast<exec::ConstantExpr>(expression);
  VELOX_USER_CHECK_NOT_NULL(
      constant, "pmod_with_mode options must be constant booleans");
  VELOX_USER_CHECK(
      !constant->value()->isNullAt(0),
      "pmod_with_mode options must be non-null");
  return constant->value()->as<SimpleVector<bool>>()->valueAt(0);
}

} // namespace

TypePtr PmodCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  VELOX_USER_CHECK_EQ(
      argTypes.size(), 4, "pmod_with_mode requires 4 arguments");
  VELOX_USER_CHECK(
      *argTypes[0] == *argTypes[1],
      "pmod_with_mode operands must have the same type");
  const std::vector<TypePtr> supported = {
      TINYINT(), SMALLINT(), INTEGER(), BIGINT(), REAL(), DOUBLE()};
  VELOX_USER_CHECK(
      std::any_of(
          supported.begin(),
          supported.end(),
          [&](const auto& type) { return *argTypes[0] == *type; }),
      "pmod_with_mode requires a primitive numeric type");
  VELOX_USER_CHECK(
      argTypes[2]->isBoolean() && argTypes[3]->isBoolean(),
      "pmod_with_mode options must be booleans");
  return argTypes[0];
}

exec::ExprPtr PmodCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& /*config*/) {
  std::vector<TypePtr> argTypes;
  for (const auto& arg : args) {
    argTypes.push_back(arg->type());
  }
  VELOX_USER_CHECK(
      *resolveType(argTypes) == *type,
      "pmod_with_mode result must have the operand type");
  const auto failOnError = constantBoolean(args[2]);
  const auto checkZeroBeforeLeft = constantBoolean(args[3]);
  return std::make_shared<PmodExpr>(
      type, std::move(args), trackCpuUsage, failOnError, checkZeroBeforeLeft);
}

} // namespace facebook::velox::functions::sparksql
