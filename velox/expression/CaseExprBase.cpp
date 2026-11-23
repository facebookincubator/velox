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
#include "velox/expression/CaseExprBase.h"
#include "velox/expression/BooleanMix.h"
#include "velox/expression/FieldReference.h"

namespace facebook::velox::exec {
namespace {

// Returns the number of WHEN, THEN and ELSE clauses.
size_t numClauses(const std::vector<ExprPtr>& inputs, bool hasSubject) {
  return inputs.size() - (hasSubject ? 1 : 0);
}

} // namespace

CaseExprBase::CaseExprBase(
    SpecialFormKind kind,
    TypePtr type,
    const std::vector<ExprPtr>& inputs,
    const std::string& name,
    bool hasSubject,
    bool inputsSupportFlatNoNullsFastPath)
    : SpecialForm(
          kind,
          std::move(type),
          inputs,
          name,
          numClauses(inputs, hasSubject) % 2 == 1 &&
              inputsSupportFlatNoNullsFastPath,
          false /* trackCpuUsage */),
      hasSubject_{hasSubject},
      numCases_{numClauses(inputs, hasSubject) / 2},
      hasElseClause_{numClauses(inputs, hasSubject) % 2 == 1} {}

void CaseExprBase::evalSpecialForm(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& finalResult) {
  VectorPtr localResult;
  LocalSelectivityVector remainingRows(context, rows);
  LocalSelectivityVector thenRows(context);

  // Fix finalSelection at 'rows' unless already fixed by an outer expression.
  ScopedFinalSelectionSetter scopedFinalSelectionSetter(context, &rows);

  if (propagatesNulls_) {
    // Rows where any field is null return null. Set these rows to null and
    // deselect them before evaluating the conditions, so that the conditions
    // can't fail on these rows. Null propagation is on only if all THEN and
    // ELSE clauses access the same vectors, so loading lazy vectors here
    // doesn't load extra vectors.
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

  VectorPtr subjectVector;
  if (hasSubject_) {
    subject()->eval(rows, context, subjectVector);

    // Rows where the subject failed don't match any case.
    if (context.errors()) {
      context.deselectErrors(*remainingRows);
    }
  }

  VectorPtr condition;
  const uint64_t* values;

  for (size_t i = 0; i < numCases_; i++) {
    context.releaseVector(condition);

    if (!remainingRows->hasSelections()) {
      break;
    }

    evalCondition(i, *remainingRows, subjectVector, context, condition);

    if (context.errors()) {
      context.deselectErrors(*remainingRows);
      if (!remainingRows->hasSelections()) {
        break;
      }
    }

    // A null condition doesn't select the THEN clause.
    const auto booleanMix = getFlatBool(
        condition.get(),
        *remainingRows,
        context,
        &tempValues_,
        nullptr,
        true /* mergeNullsToValues */,
        &values,
        nullptr);
    switch (booleanMix) {
      case BooleanMix::kAllTrue:
        thenClause(i)->eval(*remainingRows, context, localResult);
        remainingRows->clearAll();
        continue;
      case BooleanMix::kAllNull:
      case BooleanMix::kAllFalse:
        continue;
      default: {
        thenRows.get(remainingRows->end(), false);
        bits::andBits(
            thenRows->asMutableRange().bits(),
            remainingRows->asRange().bits(),
            values,
            0,
            remainingRows->end());
        thenRows->updateBounds();

        if (thenRows->hasSelections()) {
          thenClause(i)->eval(*thenRows, context, localResult);
          remainingRows->deselect(*thenRows);
        }
      }
    }
  }

  // Evaluate the ELSE clause on rows that match no case.
  if (remainingRows->hasSelections()) {
    if (hasElseClause_) {
      elseClause()->eval(*remainingRows, context, localResult);
    } else {
      context.ensureWritable(*remainingRows, type(), localResult);
      remainingRows->applyToSelected(
          [&](auto row) { localResult->setNull(row, true); });
    }
  }

  // Some rows may have not been evaluated by any THEN or ELSE clause because
  // the subject or a condition threw an error on these rows. We set those to
  // nulls to make sure the result vector is addressable at those indices.
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

void CaseExprBase::computePropagatesNulls() {
  // Propagates nulls when all of the following conditions are met:
  // - All THEN clauses and the optional ELSE clause propagate nulls.
  // - All THEN clauses and the optional ELSE clause use the same fields.
  // - The subject and all WHEN clauses use a subset of these fields.
  // A null in any of these fields then makes every THEN and ELSE clause
  // return null, whichever case the row matches.
  propagatesNulls_ = false;

  const auto& fields = thenClause(0)->distinctFields();
  auto propagatesNullsOfFields = [&](const ExprPtr& clause) {
    return clause->propagatesNulls() &&
        Expr::isSameFields(fields, clause->distinctFields());
  };

  for (size_t i = 0; i < numCases_; ++i) {
    if (!propagatesNullsOfFields(thenClause(i)) ||
        !Expr::isSubsetOfFields(whenClause(i)->distinctFields(), fields)) {
      return;
    }
  }

  if (hasElseClause_ && !propagatesNullsOfFields(elseClause())) {
    return;
  }

  if (hasSubject_ &&
      !Expr::isSubsetOfFields(subject()->distinctFields(), fields)) {
    return;
  }

  propagatesNulls_ = true;
}

} // namespace facebook::velox::exec
