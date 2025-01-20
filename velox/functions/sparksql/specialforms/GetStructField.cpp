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

#include "velox/functions/sparksql/specialforms/GetStructField.h"
#include <type/Type.h>
#include "expression/Expr.h"
#include "vector/ComplexVector.h"
#include "vector/ConstantVector.h"
#include "velox/expression/PeeledEncoding.h"

using namespace facebook::velox::exec;

namespace facebook::velox::functions::sparksql {

void GetStructFieldExpr::evalSpecialForm(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  VectorPtr input;
  VectorPtr ordinalVector;
  inputs_[0]->eval(rows, context, input);
  inputs_[1]->eval(rows, context, ordinalVector);
  VELOX_USER_CHECK(ordinalVector->isConstantEncoding());
  auto ordinal =
      ordinalVector->asUnchecked<ConstantVector<int32_t>>()->valueAt(0);
  auto resultType = std::const_pointer_cast<const Type>(type_);

  LocalSelectivityVector remainingRows(context, rows);
  context.deselectErrors(*remainingRows);

  LocalDecodedVector decoded(context, *input, *remainingRows);

  auto* rawNulls = decoded->nulls();
  if (rawNulls) {
    remainingRows->deselectNulls(
        rawNulls, remainingRows->begin(), remainingRows->end());
  }

  VectorPtr localResult;
  if (!remainingRows->hasSelections()) {
    localResult =
        BaseVector::createNullConstant(resultType, rows.end(), context.pool());
  } else {
    auto rowData = decoded->base()->as<RowVector>();
    if (decoded->isIdentityMapping()) {
      localResult = rowData->childAt(ordinal);
    } else {
      localResult =
          decoded->wrap(rowData->childAt(ordinal), *input, decoded->size());
    }
  }

  context.moveOrCopyResult(localResult, *remainingRows, result);
  context.releaseVector(localResult);

  VELOX_CHECK_NOT_NULL(result);
  if (rawNulls || context.errors()) {
    EvalCtx::addNulls(
        rows, remainingRows->asRange().bits(), context, resultType, result);
  }

  context.releaseVector(input);
  context.releaseVector(ordinalVector);
}

TypePtr GetStructFieldCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& /*argTypes*/) {
  VELOX_FAIL("get_struct_field function does not support type resolution.");
}

ExprPtr GetStructFieldCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& /*config*/) {
  VELOX_USER_CHECK_EQ(args.size(), 2, "get_struct_field expects two argument.");

  VELOX_USER_CHECK_EQ(
      args[0]->type()->kind(),
      TypeKind::ROW,
      "The first argument of get_struct_field should be of row type.");

  VELOX_USER_CHECK_EQ(
      args[1]->type()->kind(),
      TypeKind::INTEGER,
      "The second argument of get_struct_field should be of integer type.");

  return std::make_shared<GetStructFieldExpr>(
      type, std::move(args), trackCpuUsage);
}

} // namespace facebook::velox::functions::sparksql
