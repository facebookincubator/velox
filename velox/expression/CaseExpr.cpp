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
#include "velox/expression/CaseExpr.h"
#include "velox/expression/CastExpr.h"
#include "velox/expression/ExprConstants.h"
#include "velox/expression/FieldReference.h"
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/expression/VectorFunction.h"
#include "velox/type/TypeCoercer.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::exec {

namespace {

// Returns true when the total number of inputs is consistent with the
// case layout: 1 (subject) + 2*N (when/then pairs) + 0 or 1 (else).
// Equivalently the count after removing the subject must be even (no else) or
// odd (with else).
bool hasElseClause(const std::vector<ExprPtr>& inputs) {
  // total = 1 + 2*N         → (total - 1) is even  → no else
  // total = 1 + 2*N + 1 = 2*(N+1)  → (total - 1) is odd → has else
  return (inputs.size() - 1) % 2 == 1;
}

size_t numCasesFromInputs(const std::vector<ExprPtr>& inputs) {
  bool hasElse = hasElseClause(inputs);
  return (inputs.size() - 1 - (hasElse ? 1 : 0)) / 2;
}

} // namespace

CaseExpr::CaseExpr(
    TypePtr type,
    const std::vector<ExprPtr>& inputs,
    ExprPtr eqExpr,
    RowTypePtr eqInputType,
    bool inputsSupportFlatNoNullsFastPath)
    : CaseExprBase(
          SpecialFormKind::kCase,
          std::move(type),
          inputs,
          expression::kCase,
          /*hasSubject=*/true,
          inputsSupportFlatNoNullsFastPath),
      eqExpr_{std::move(eqExpr)},
      eqInputType_{std::move(eqInputType)} {
  VELOX_CHECK_NOT_NULL(eqExpr_, "CaseExpr requires a non-null eq Expr");
  VELOX_CHECK_NOT_NULL(eqInputType_);
  VELOX_CHECK_GE(
      inputs_.size(),
      3,
      "CaseExpr requires at least 3 inputs: subject, when_value, then_result");

  const auto& subjectType = inputs_[0]->type();
  for (size_t i = 0; i < numCases_; i++) {
    VELOX_CHECK(
        *inputs_[2 * i + 1]->type() == *subjectType,
        "All WHEN values must have the same type as the subject. "
        "Expected {}, got {}.",
        subjectType->toString(),
        inputs_[2 * i + 1]->type()->toString());
  }
}

// static
ExprPtr CaseExpr::create(
    TypePtr type,
    std::vector<ExprPtr> inputs,
    const TypePtr& comparisonType,
    bool /* trackCpuUsage */,
    const core::QueryConfig& config) {
  VELOX_CHECK_GE(
      inputs.size(),
      3,
      "CASE requires at least 3 compiled children: subject, when, then");

  const size_t numCases = numCasesFromInputs(inputs);

  // Cast the subject and each WHEN value to the comparison super-type so the
  // single 'eq' Expr can be applied with a single uniform type.
  CastCallToSpecialForm castForm;
  if (*inputs[0]->type() != *comparisonType) {
    inputs[0] = castForm.constructSpecialForm(
        comparisonType, {inputs[0]}, false, config);
  }
  for (size_t i = 0; i < numCases; ++i) {
    if (*inputs[2 * i + 1]->type() != *comparisonType) {
      inputs[2 * i + 1] = castForm.constructSpecialForm(
          comparisonType, {inputs[2 * i + 1]}, false, config);
    }
  }

  // Build a reusable 'eq' Expr over a two-column synthetic row. At evaluation
  // time evalCondition wraps the cached subject and each WHEN vector into a
  // RowVector matching this type, then calls eqExpr_->eval. The pattern
  // mirrors how LambdaExpr's body Expr is evaluated against a synthetic row
  // of (lambda args + captures): only public Expr APIs are needed and we
  // still get peeling, the registered eq variant's null semantics, and
  // per-row error reporting.
  auto eqInputType = ROW({"c0", "c1"}, {comparisonType, comparisonType});

  std::vector<ExprPtr> eqChildren;
  eqChildren.reserve(2);
  eqChildren.push_back(
      std::make_shared<FieldReference>(
          comparisonType, std::vector<ExprPtr>{}, "c0"));
  eqChildren.push_back(
      std::make_shared<FieldReference>(
          comparisonType, std::vector<ExprPtr>{}, "c1"));

  // Resolve 'eq' for the comparison type, mirroring compileCall: try the
  // VectorFunction registry first (covers the SIMD eq for numeric, date,
  // interval, and decimal types), then fall back to the SimpleFunction
  // registry (which has a Generic<T1>, Generic<T1> eq registration that
  // covers everything else — varchar, varbinary, timestamp, boolean, JSON,
  // IPAddress, complex types, etc.).
  std::shared_ptr<VectorFunction> eqFunction;
  VectorFunctionMetadata eqMetadata;
  const std::vector<TypePtr> eqInputTypes{comparisonType, comparisonType};
  if (auto vectorEq = getVectorFunctionWithMetadata(
          "eq", eqInputTypes, /*constantInputs=*/{}, config)) {
    eqFunction = std::move(vectorEq->first);
    eqMetadata = std::move(vectorEq->second);
  } else if (
      auto simpleEq = simpleFunctions().resolveFunction("eq", eqInputTypes)) {
    eqFunction = simpleEq->createFunction()->createVectorFunction(
        eqInputTypes, /*constantInputs=*/{}, config, /*memoryPool=*/nullptr);
    eqMetadata = simpleEq->metadata();
  } else {
    VELOX_USER_FAIL(
        "case: no 'eq' function registered for type '{}'.",
        comparisonType->toString());
  }
  auto eqExpr = std::make_shared<Expr>(
      BOOLEAN(),
      std::move(eqChildren),
      std::move(eqFunction),
      std::move(eqMetadata),
      "eq",
      /*trackCpuUsage=*/false);
  eqExpr->computeMetadata();

  const bool inputsSupportFlatNoNullsFastPath =
      Expr::allSupportFlatNoNullsFastPath(inputs);

  return std::make_shared<CaseExpr>(
      std::move(type),
      std::move(inputs),
      std::move(eqExpr),
      std::move(eqInputType),
      inputsSupportFlatNoNullsFastPath);
}

void CaseExpr::evalCondition(
    size_t index,
    const SelectivityVector& rows,
    const VectorPtr& subject,
    EvalCtx& context,
    VectorPtr& result) {
  VectorPtr whenVector;
  whenClause(index)->eval(rows, context, whenVector);

  // Compare only rows where the WHEN value didn't fail.
  const SelectivityVector* eqRows = &rows;
  LocalSelectivityVector nonErrorRows(context);
  if (context.errors()) {
    eqRows = nonErrorRows.get(rows);
    context.deselectErrors(*nonErrorRows);
    if (!eqRows->hasSelections()) {
      return;
    }
  }

  if (eqRow_ == nullptr) {
    eqRow_ = std::make_shared<RowVector>(
        context.pool(),
        eqInputType_,
        /*nulls=*/nullptr,
        0,
        std::vector<VectorPtr>(2));
  }
  eqRow_->children()[0] = subject;
  eqRow_->children()[1] = std::move(whenVector);
  eqRow_->unsafeResize(eqRows->end());

  // A fresh EvalCtx per call keeps internal state (peeled fields, encodings,
  // errors) from leaking between clauses.
  EvalCtx eqCtx{context.execCtx(), context.exprSet(), eqRow_.get()};
  *eqCtx.mutableThrowOnError() = context.throwOnError();

  eqExpr_->clearCache();
  eqExpr_->eval(*eqRows, eqCtx, result);

  // Don't keep the subject and the WHEN value alive after this call.
  eqRow_->children()[0].reset();
  eqRow_->children()[1].reset();

  // Report eq errors as errors of the condition.
  if (eqCtx.errors()) {
    eqCtx.moveAppendErrors(*context.errorsPtr());
  }
}

std::string CaseExpr::toSql(std::vector<VectorPtr>* complexConstants) const {
  // Generate standard SQL simple-CASE syntax so that the expression can be
  // faithfully round-tripped through parseCaseExpr → CaseExpr without
  // structural divergence.
  std::stringstream sql;
  sql << "CASE " << inputs_[0]->toSql(complexConstants);
  for (size_t i = 0; i < numCases_; i++) {
    sql << " WHEN " << inputs_[2 * i + 1]->toSql(complexConstants);
    sql << " THEN " << inputs_[2 * i + 2]->toSql(complexConstants);
  }
  if (hasElseClause_) {
    sql << " ELSE " << inputs_.back()->toSql(complexConstants);
  }
  sql << " END";
  return sql.str();
}

// CaseCallToSpecialForm

TypePtr CaseCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  // Layout: subject, (when_val, then_result)+, [else]
  VELOX_CHECK_GE(
      argTypes.size(),
      3,
      "case requires at least 3 arguments: subject, when_value, then_result");

  const bool hasElse = (argTypes.size() % 2 == 0);
  const size_t numCases = (argTypes.size() - 1 - (hasElse ? 1 : 0)) / 2;

  // The result type is the type of the first THEN clause (index 2).
  const TypePtr& resultType = argTypes[2];

  for (size_t i = 0; i < numCases; i++) {
    const auto& thenType = argTypes[2 * i + 2];
    VELOX_CHECK(
        *thenType == *resultType,
        "All THEN clauses in case must have the same type. "
        "Expected {}, got {}.",
        resultType->toString(),
        thenType->toString());
  }

  if (hasElse) {
    VELOX_CHECK(
        *argTypes.back() == *resultType,
        "ELSE clause in case must match THEN clause type. "
        "Expected {}, got {}.",
        resultType->toString(),
        argTypes.back()->toString());
  }

  return resultType;
}

TypePtr CaseCallToSpecialForm::resolveTypeWithCoercions(
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions,
    const TypeCoercer& coercer) {
  coercions.clear();
  coercions.resize(argTypes.size());

  const bool hasElse = (argTypes.size() % 2 == 0);
  const size_t numCases = (argTypes.size() - 1 - (hasElse ? 1 : 0)) / 2;

  // Determine the comparison type: the least-common super-type of the subject
  // and all WHEN values.  The ExprCompiler will insert CastExprs for any
  // argument whose type doesn't already match the comparison type.
  TypePtr comparisonType = argTypes[0]; // start with the subject type
  for (size_t i = 0; i < numCases; i++) {
    const auto& whenType = argTypes[2 * i + 1];
    if (*whenType != *comparisonType) {
      auto common = coercer.leastCommonSuperType(comparisonType, whenType);
      VELOX_CHECK_NOT_NULL(
          common,
          "case: subject type '{}' and WHEN value type '{}' have "
          "no common super-type.",
          comparisonType->toString(),
          whenType->toString());
      comparisonType = common;
    }
  }

  // Emit coercions for subject and WHEN values that differ from comparisonType.
  if (*argTypes[0] != *comparisonType) {
    coercions[0] = comparisonType;
  }
  for (size_t i = 0; i < numCases; i++) {
    if (*argTypes[2 * i + 1] != *comparisonType) {
      coercions[2 * i + 1] = comparisonType;
    }
  }

  // Records the coercion the first 'numClauses' THEN clauses need to reach
  // 'type'. A clause already of that type needs none.
  auto setThenCoercions = [&](size_t numClauses, const TypePtr& type) {
    for (size_t i = 0; i < numClauses; i++) {
      const auto index = 2 * i + 2;
      coercions[index] = *argTypes[index] == *type ? nullptr : type;
    }
  };

  // Resolve the result type from THEN (and optional ELSE) clauses with
  // widening coercions, mirroring the logic in SwitchCallToSpecialForm.
  TypePtr resultType = argTypes[2];
  for (size_t i = 0; i < numCases; i++) {
    const auto& thenType = argTypes[2 * i + 2];
    if (*thenType != *resultType) {
      const auto common = coercer.leastCommonSuperType(resultType, thenType);
      VELOX_CHECK_NOT_NULL(
          common,
          "All THEN clauses in case must have the same type. "
          "Expected {}, but got {}.",
          resultType->toString(),
          thenType->toString());

      resultType = common;
      setThenCoercions(i + 1, resultType);
    }
  }

  if (hasElse) {
    const auto& elseType = argTypes.back();
    if (*elseType != *resultType) {
      const auto common = coercer.leastCommonSuperType(resultType, elseType);
      VELOX_CHECK_NOT_NULL(
          common,
          "ELSE clause in case must match THEN clause type. "
          "Expected {}, but got {}.",
          resultType->toString(),
          elseType->toString());

      resultType = common;
      setThenCoercions(numCases, resultType);
      coercions.back() = *elseType == *resultType ? nullptr : resultType;
    }
  }

  return resultType;
}

ExprPtr CaseCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<ExprPtr>&& compiledChildren,
    bool /* trackCpuUsage */,
    const core::QueryConfig& config) {
  VELOX_CHECK_GE(
      compiledChildren.size(), 3, "case requires at least 3 compiled children");

  const size_t numCases = numCasesFromInputs(compiledChildren);

  // Align the subject and WHEN value types by inserting CastExprs where
  // needed so the 'eq' VectorFunction can be looked up with a single
  // uniform type.
  //
  // Although resolveTypeWithCorsions already computes these coercions, the
  // Velox type resolver (parse::TypeResolver) never calls it. It only calls
  // resolveType, which validates THEN/ELSE types but not subject/WHEN types.
  // The resolver handles mismatches via brute-force implicit casts that
  // permute each argument independently — but that does not guarantee a
  // group of arguments (subject + all WHEN values) converge to a common
  // type. SwitchExpr avoids this because its inputs are already fully-typed
  // boolean conditions (e.g. eq(c0, 7) is resolved before SwitchExpr sees
  // it). CaseExpr receives the raw subject and WHEN values as separate
  // inputs, so it must reconcile their types here.
  TypePtr comparisonType = compiledChildren[0]->type();
  for (size_t i = 0; i < numCases; i++) {
    const auto& whenType = compiledChildren[2 * i + 1]->type();
    if (*whenType != *comparisonType) {
      auto common = TypeCoercer::defaults().leastCommonSuperType(
          comparisonType, whenType);
      VELOX_CHECK_NOT_NULL(
          common,
          "case: subject type '{}' and WHEN value type '{}' have "
          "no common super-type.",
          comparisonType->toString(),
          whenType->toString());
      comparisonType = common;
    }
  }

  // Delegate to CaseExpr::create so that the cast insertion + eq Expr
  // construction logic lives in one place — shared with the kCase ExprKind
  // path in ExprCompiler.
  return CaseExpr::create(
      type,
      std::move(compiledChildren),
      comparisonType,
      /*trackCpuUsage=*/false,
      config);
}

} // namespace facebook::velox::exec
