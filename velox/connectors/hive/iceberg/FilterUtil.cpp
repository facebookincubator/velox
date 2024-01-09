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
#include "velox/connectors/hive/iceberg/FilterUtil.h"
#include <iostream>

#include "velox/core/ITypedExpr.h"
#include "velox/expression/ConjunctExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/expression/FunctionCallToSpecialForm.h"

namespace facebook::velox::connector::hive::iceberg {

using namespace facebook::velox::exec;
using namespace facebook::velox::core;

// Read 'size' values from 'valuesVector' starting at 'offset', de-duplicate
// remove nulls and sort. Return a list of unique non-null values sorted in
// ascending order and a boolean indicating whether there were any null values.
template <typename T, typename U = T>
std::pair<std::vector<T>, bool> toValues(
    const VectorPtr& valuesVector,
    vector_size_t offset,
    vector_size_t size) {
  auto simpleValues = valuesVector->as<SimpleVector<U>>();

  bool nullAllowed = false;
  std::vector<T> values;
  values.reserve(size);

  for (auto i = offset; i < offset + size; i++) {
    if (simpleValues->isNullAt(i)) {
      nullAllowed = true;
    } else {
      if constexpr (std::is_same_v<U, Timestamp>) {
        values.emplace_back(simpleValues->valueAt(i).toMillis());
      } else {
        values.emplace_back(simpleValues->valueAt(i));
      }
    }
  }

  // In-place sort, remove duplicates, and later std::move to save memory
  std::sort(values.begin(), values.end());
  auto last = std::unique(values.begin(), values.end());
  values.resize(std::distance(values.begin(), last));

  return {std::move(values), nullAllowed};
}

template <typename T>
std::unique_ptr<common::Filter> createNegatedBigintValuesFilter(
    const VectorPtr& valuesVector,
    vector_size_t offset,
    vector_size_t size) {
  auto valuesPair = toValues<int64_t, T>(valuesVector, offset, size);

  const auto& values = valuesPair.first;
  bool hasNull = valuesPair.second;

  return common::createNegatedBigintValues(values, hasNull);
}

std::unique_ptr<common::Filter> createNotInFilter(
    const VectorPtr& elements,
    vector_size_t offset,
    vector_size_t size,
    const TypeKind& type) {
  std::unique_ptr<common::Filter> filter;
  switch (type) {
    case TypeKind::HUGEINT:
      // TODO: createNegatedHugeintValuesFilter is not implemented yet.
      filter = nullptr;
      break;
    case TypeKind::BIGINT:
      filter = createNegatedBigintValuesFilter<int64_t>(elements, offset, size);
      break;
    case TypeKind::INTEGER:
      filter = createNegatedBigintValuesFilter<int32_t>(elements, offset, size);
      break;
    case TypeKind::SMALLINT:
      filter = createNegatedBigintValuesFilter<int16_t>(elements, offset, size);
      break;
    case TypeKind::TINYINT:
      filter = createNegatedBigintValuesFilter<int8_t>(elements, offset, size);
      break;
    case TypeKind::BOOLEAN:
      // Hack: using BIGINT filter for bool, which is essentially "int1_t".
      filter = createNegatedBigintValuesFilter<bool>(elements, offset, size);
      break;
    case TypeKind::TIMESTAMP:
      filter =
          createNegatedBigintValuesFilter<Timestamp>(elements, offset, size);
      break;
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      // TODO: createNegatedBytesValuesFilter is not implemented yet.
      filter = nullptr;
      break;
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
      VELOX_USER_FAIL(
          "Iceberg equality delete column cannot be DOUBLE or FLOAT");
      break;
    case TypeKind::UNKNOWN:
      filter = nullptr;
      break;
    case TypeKind::ARRAY:
      [[fallthrough]];
    case TypeKind::MAP:
      [[fallthrough]];
    case TypeKind::ROW:
      [[fallthrough]];
    default:
      VELOX_UNSUPPORTED(
          "Unsupported in-list type for NOT IN predicate: {}", type);
  }
  return filter;
}

std::unique_ptr<exec::ExprSet> createEqualityDeleteExprSet(
    RowVectorPtr rowVector,
    core::ExpressionEvaluator* expressionEvaluator) {
  auto numDeleteFields = rowVector->childrenSize();
  VELOX_CHECK_GT(
      numDeleteFields,
      0,
      "Iceberg equality delete file should have at least one field.");
  auto numDeletedValues = rowVector->childAt(0)->size();
  auto rowType = asRowType(rowVector->type());

  std::vector<core::TypedExprPtr> conjunctInputs;
  for (int i = 0; i < numDeletedValues; i++) {
    std::vector<core::TypedExprPtr> disconjunctInputs;

    for (int j = 0; j < rowType->size(); j++) {
      auto type = rowType->childAt(j);
      auto name = rowType->nameOf(j);
      auto value = BaseVector::wrapInConstant(1, i, rowVector->childAt(j));

      std::vector<core::TypedExprPtr> isNotEqualInputs;
      isNotEqualInputs.push_back(
          std::make_shared<FieldAccessTypedExpr>(type, name));
      isNotEqualInputs.push_back(std::make_shared<ConstantTypedExpr>(value));
      auto isNotEqualExpr =
          std::make_shared<CallTypedExpr>(BOOLEAN(), isNotEqualInputs, "neq");

      disconjunctInputs.push_back(isNotEqualExpr);
    }

    auto disconjunctNotEqualExpr =
        std::make_shared<CallTypedExpr>(BOOLEAN(), disconjunctInputs, "or");
    conjunctInputs.push_back(disconjunctNotEqualExpr);
  }

  core::TypedExprPtr expression =
      std::make_shared<CallTypedExpr>(BOOLEAN(), conjunctInputs, "and");

  return expressionEvaluator->compile(expression);
}

exec::ExprPtr createEqualityDeleteExpr(RowVectorPtr rowVector) {
  auto numDeletedValues = rowVector->size();
  auto rowType = asRowType(rowVector->type());

  std::vector<ExprPtr> conjunctInputs;
  for (int i = 0; i < numDeletedValues; i++) {
    std::vector<ExprPtr> disconjunctInputs;
    for (int j = 0; j < rowType->size(); j++) {
      auto type = rowType->childAt(j);
      auto name = rowType->nameOf(j);
      disconjunctInputs.push_back(
          std::make_shared<FieldReference>(type, std::vector<ExprPtr>{}, name));
    }

    ExprPtr disconjunctNotEqualExpr = std::make_shared<ConjunctExpr>(
        BOOLEAN(), std::move(disconjunctInputs), false, false);
    conjunctInputs.push_back(disconjunctNotEqualExpr);
  }

  ExprPtr conjunctNotEqualExpr = std::make_shared<ConjunctExpr>(
      BOOLEAN(), std::move(conjunctInputs), true, false);

  return conjunctNotEqualExpr;
}

} // namespace facebook::velox::connector::hive::iceberg