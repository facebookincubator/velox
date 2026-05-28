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
#include "velox/expression/BooleanMix.h"
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
    : SpecialForm(
          SpecialFormKind::kCase,
          std::move(type),
          inputs,
          expression::kCase,
          hasElseClause(inputs) && inputsSupportFlatNoNullsFastPath,
          false /* trackCpuUsage */),
      numCases_{numCasesFromInputs(inputs)},
      hasElseClause_{hasElseClause(inputs)},
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
  // time evalSpecialForm wraps the cached subject and each WHEN vector into a
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

void CaseExpr::evalSpecialForm(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& finalResult) {
  VectorPtr localResult;
  LocalSelectivityVector remainingRows(context, rows);
  LocalSelectivityVector thenRows(context);

  // Fix finalSelection at 'rows' unless already fixed by an outer expression.
  ScopedFinalSelectionSetter scopedFinalSelectionSetter(context, &rows);

  // Null-propagation pre-pass: if all THEN/ELSE branches propagate nulls from
  // the same input fields, and the subject/WHEN values use a subset of those
  // fields, pre-null and deselect rows where those fields are null.  This
  // avoids evaluating the subject and eq function on rows that are guaranteed
  // to produce null, and prevents errors on null inputs for strict functions.
  if (propagatesNulls_) {
    auto& remaining = *remainingRows.get();
    for (auto* field : distinctFields_) {
      context.ensureFieldLoaded(field->index(context), remaining);
      const auto& vector = context.getField(field->index(context));
      if (vector->mayHaveNulls()) {
        LocalDecodedVector decoded(context, *vector, remaining);
        addNulls(remaining, decoded->nulls(&remaining), context, localResult);
        remaining.deselectNulls(
            decoded->nulls(&remaining), remaining.begin(), remaining.end());
      }
    }
  }

  // Step 1: Evaluate the subject expression exactly once.
  VectorPtr subjectVector;
  inputs_[0]->eval(rows, context, subjectVector);

  // Rows where subject evaluation itself errored out are deselected so that
  // subsequent WHEN checks do not observe stale data at those positions.
  if (context.errors()) {
    context.deselectErrors(*remainingRows);
  }

  // Step 2: Build a reusable two-column RowVector for invoking eqExpr_.
  // The RowVector is shared across WHEN iterations (only children()[1]
  // changes); a fresh EvalCtx is created per iteration so no internal
  // state (peeled fields, encodings, errors) leaks between clauses.
  auto eqRow = std::make_shared<RowVector>(
      context.pool(),
      eqInputType_,
      /*nulls=*/nullptr,
      rows.end(),
      std::vector<VectorPtr>{subjectVector, subjectVector});

  VectorPtr condition;
  const uint64_t* values;

  for (size_t i = 0; i < numCases_; i++) {
    context.releaseVector(condition);

    if (!remainingRows->hasSelections()) {
      break;
    }

    // Evaluate the WHEN value for the remaining (unmatched) rows.
    VectorPtr whenVector;
    inputs_[2 * i + 1]->eval(*remainingRows.get(), context, whenVector);

    if (context.errors()) {
      context.deselectErrors(*remainingRows);
      if (!remainingRows->hasSelections()) {
        break;
      }
    }

    eqRow->children()[1] = whenVector;
    eqRow->unsafeResize(remainingRows->end());
    EvalCtx eqCtx{context.execCtx(), context.exprSet(), eqRow.get()};
    *eqCtx.mutableThrowOnError() = context.throwOnError();

    eqExpr_->clearCache();
    eqExpr_->eval(*remainingRows.get(), eqCtx, condition);

    // Surface eq errors through the parent context so the existing CASE
    // error-handling deselects them and they fall through to the
    // error-null-fill at the end.
    if (eqCtx.errors()) {
      eqCtx.moveAppendErrors(*context.errorsPtr());
      context.deselectErrors(*remainingRows);
      if (!remainingRows->hasSelections()) {
        break;
      }
    }

    // Interpret the boolean condition result.  Nulls are merged to false
    // (mergeNullsToValues = true), matching SQL semantics: eq(x, null) is
    // UNKNOWN and therefore does not select the THEN branch.
    const auto booleanMix = getFlatBool(
        condition.get(),
        *remainingRows.get(),
        context,
        &tempValues_,
        nullptr,
        true /* mergeNullsToValues */,
        &values,
        nullptr);

    switch (booleanMix) {
      case BooleanMix::kAllTrue:
        // Every remaining row matched — evaluate THEN for all of them and stop.
        inputs_[2 * i + 2]->eval(*remainingRows.get(), context, localResult);
        remainingRows->clearAll();
        continue;
      case BooleanMix::kAllNull:
      case BooleanMix::kAllFalse:
        // No rows matched this WHEN — move to the next case.
        continue;
      default: {
        // Mixed: compute the subset of remainingRows where condition is true.
        thenRows.get(remainingRows->end(), false);
        bits::andBits(
            thenRows.get()->asMutableRange().bits(),
            remainingRows->asRange().bits(),
            values,
            0,
            remainingRows->end());
        thenRows.get()->updateBounds();

        if (thenRows.get()->hasSelections()) {
          inputs_[2 * i + 2]->eval(*thenRows.get(), context, localResult);
          remainingRows->deselect(*thenRows.get());
        }
      }
    }
  }

  // Step 4: Handle rows that matched no WHEN branch.
  if (remainingRows->hasSelections()) {
    if (hasElseClause_) {
      inputs_.back()->eval(*remainingRows.get(), context, localResult);
    } else {
      // No ELSE: write NULL for every unmatched row.
      context.ensureWritable(*remainingRows.get(), type(), localResult);
      remainingRows->applyToSelected(
          [&](auto row) { localResult->setNull(row, true); });
    }
  }

  // Step 5: Null-fill rows that produced errors during evaluation.
  // Some rows may have been deselected due to errors in subject, WHEN-value,
  // or eq evaluation.  Ensure those positions are addressable in localResult.
  if (context.errors()) {
    // TODO: Fix decoding function vector issue #6269.
    if (type()->kind() != TypeKind::FUNCTION) {
      LocalSelectivityVector nonErrorRows(context, rows);
      context.deselectErrors(*nonErrorRows);
      addNulls(rows, nonErrorRows->asRange().bits(), context, localResult);
    }
  }

  // TODO: Fix evaluate lambda expression return vector of size 0 issue #6270.
  if (type()->kind() != TypeKind::FUNCTION) {
    VELOX_CHECK_NOT_NULL(localResult);
    VELOX_CHECK_GE(localResult->size(), rows.end());
  }

  context.moveOrCopyResult(localResult, rows, finalResult);
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

void CaseExpr::computePropagatesNulls() {
  // CaseExpr propagates nulls when:
  //   1. Every THEN clause (and the optional ELSE) propagates nulls.
  //   2. Every THEN/ELSE clause references the same set of input fields.
  //   3. The subject and every WHEN value reference only a subset of those
  //      fields.
  //
  // Under these conditions a null in any of the shared fields causes the
  // subject to be null, which in turn causes no WHEN branch to match (since
  // eq(null, x) is always null/false).  With no ELSE the row returns null,
  // and with an ELSE its THEN/ELSE expression also returns null (condition 1).
  // Pre-nulling and deselecting those rows avoids unnecessary work and
  // prevents errors on strict functions.

  // Condition 1 & 2: all THEN clauses propagate nulls and share the same
  // distinct fields.
  for (size_t i = 0; i < numCases_; i++) {
    if (!inputs_[2 * i + 2]->propagatesNulls()) {
      propagatesNulls_ = false;
      return;
    }
  }
  if (hasElseClause_ && !inputs_.back()->propagatesNulls()) {
    propagatesNulls_ = false;
    return;
  }

  const auto& firstThenFields = inputs_[2]->distinctFields();
  for (size_t i = 0; i < numCases_; i++) {
    if (!Expr::isSameFields(
            firstThenFields, inputs_[2 * i + 2]->distinctFields())) {
      propagatesNulls_ = false;
      return;
    }
  }
  if (hasElseClause_) {
    if (!Expr::isSameFields(
            firstThenFields, inputs_.back()->distinctFields())) {
      propagatesNulls_ = false;
      return;
    }
  }

  // Condition 3: subject and all WHEN values use only a subset of the
  // THEN/ELSE fields.
  if (!Expr::isSubsetOfFields(inputs_[0]->distinctFields(), firstThenFields)) {
    propagatesNulls_ = false;
    return;
  }
  for (size_t i = 0; i < numCases_; i++) {
    if (!Expr::isSubsetOfFields(
            inputs_[2 * i + 1]->distinctFields(), firstThenFields)) {
      propagatesNulls_ = false;
      return;
    }
  }

  propagatesNulls_ = true;
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

  // Resolve the result type from THEN (and optional ELSE) clauses with
  // widening coercions, mirroring the logic in SwitchCallToSpecialForm.
  TypePtr resultType = argTypes[2];
  for (size_t i = 0; i < numCases; i++) {
    const auto& thenType = argTypes[2 * i + 2];
    if (*thenType != *resultType) {
      if (coercer.coercible(thenType, resultType)) {
        coercions[2 * i + 2] = resultType;
      } else if (coercer.coercible(resultType, thenType)) {
        resultType = thenType;
        for (size_t j = 0; j < i; j++) {
          coercions[2 * j + 2] = resultType;
        }
      } else {
        VELOX_FAIL(
            "All THEN clauses in case must have the same type. "
            "Expected {}, but got {}.",
            resultType->toString(),
            thenType->toString());
      }
    }
  }

  if (hasElse) {
    const auto& elseType = argTypes.back();
    if (*elseType != *resultType) {
      if (coercer.coercible(elseType, resultType)) {
        coercions.back() = resultType;
      } else if (coercer.coercible(resultType, elseType)) {
        resultType = elseType;
        for (size_t i = 0; i < numCases; i++) {
          coercions[2 * i + 2] = resultType;
        }
      } else {
        VELOX_FAIL(
            "ELSE clause in case must match THEN clause type. "
            "Expected {}, but got {}.",
            resultType->toString(),
            elseType->toString());
      }
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
