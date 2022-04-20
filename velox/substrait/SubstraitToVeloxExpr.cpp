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

#include "velox/substrait/SubstraitToVeloxExpr.h"
#include "velox/substrait/TypeUtils.h"

namespace facebook::velox::substrait {

std::shared_ptr<const core::FieldAccessTypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::FieldReference& sField,
    const RowTypePtr& vPreNodeOutPut) {
  auto typeCase = sField.reference_type_case();
  switch (typeCase) {
    case ::substrait::Expression::FieldReference::ReferenceTypeCase::
        kDirectReference: {
      auto dRef = sField.direct_reference();
      int32_t colIdx = subParser_->parseReferenceSegment(dRef);

      std::vector<TypePtr> vPreNodeColTypes = vPreNodeOutPut->children();
      std::vector<std::string> vPreNodeColNames = vPreNodeOutPut->names();
      int64_t vPreNodeColNums = vPreNodeColNames.size();

      if (colIdx <= vPreNodeColNums) {
        // convert type to row
        return std::make_shared<core::FieldAccessTypedExpr>(
            vPreNodeColTypes[colIdx],
            std::make_shared<core::InputTypedExpr>(vPreNodeColTypes[colIdx]),
            vPreNodeColNames[colIdx]);
      } else {
        VELOX_FAIL("Missing the column with id '{}' .", colIdx);
      }
    }
    default:
      VELOX_NYI(
          "Substrait conversion not supported for Reference '{}'", typeCase);
  }
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::ScalarFunction& sFunc,
    const RowTypePtr& vPreNodeOutPut) {
  std::vector<std::shared_ptr<const core::ITypedExpr>> params;
  params.reserve(sFunc.args().size());
  for (const auto& sArg : sFunc.args()) {
    params.emplace_back(toVeloxExpr(sArg, vPreNodeOutPut));
  }
  auto functionId = sFunc.function_reference();
  auto veloxFunction = subParser_->findVeloxFunction(functionMap_, functionId);
  auto subType = subParser_->parseType(sFunc.output_type());
  auto veloxType = toVeloxType(subType->type);
  return std::make_shared<const core::CallTypedExpr>(
      veloxType, std::move(params), veloxFunction);
}

std::shared_ptr<const core::ConstantTypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::Literal& sLit) {
  auto typeCase = sLit.literal_type_case();
  switch (typeCase) {
    case ::substrait::Expression_Literal::LiteralTypeCase::kFp64:
      return std::make_shared<core::ConstantTypedExpr>(
          velox::variant(velox::variant(sLit.fp64())));
    case ::substrait::Expression_Literal::LiteralTypeCase::kBoolean:
      return std::make_shared<core::ConstantTypedExpr>(variant(sLit.boolean()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI64:
      return std::make_shared<core::ConstantTypedExpr>(variant(sLit.i64()));
    default:
      VELOX_NYI(
          "Substrait conversion not supported for type case '{}'", typeCase);
  }
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::Cast& sCast,
    const RowTypePtr& vPreNodeOutPut) {
  auto subType = subParser_->parseType(sCast.type());
  auto vCastType = toVeloxType(subType->type);
  // TODO add flag in substrait after. now is set false.
  bool nullOnFailure = false;

  std::vector<std::shared_ptr<const core::ITypedExpr>> vCastInputs;
  vCastInputs.reserve(1);
  std::shared_ptr<const core::ITypedExpr> vCastInput =
      toVeloxExpr(sCast.input(), vPreNodeOutPut);
  vCastInputs.emplace_back(vCastInput);

  return std::make_shared<core::CastTypedExpr>(
      vCastType, vCastInputs, nullOnFailure);
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression& sExpr,
    const RowTypePtr& vPreNodeOutPut) {
  std::shared_ptr<const core::ITypedExpr> veloxExpr;
  auto typeCase = sExpr.rex_type_case();
  switch (typeCase) {
    case ::substrait::Expression::RexTypeCase::kLiteral:
      return toVeloxExpr(sExpr.literal());
    case ::substrait::Expression::RexTypeCase::kScalarFunction:
      return toVeloxExpr(sExpr.scalar_function(), vPreNodeOutPut);
    case ::substrait::Expression::RexTypeCase::kSelection:
      return toVeloxExpr(sExpr.selection(), vPreNodeOutPut);
    case ::substrait::Expression::RexTypeCase::kCast:
      return toVeloxExpr(sExpr.cast(), vPreNodeOutPut);
    default:
      VELOX_NYI(
          "Substrait conversion not supported for Expression '{}'", typeCase);
  }
}

} // namespace facebook::velox::substrait
