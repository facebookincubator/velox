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
    const ::substrait::Expression::FieldReference& substraitField,
    const RowTypePtr& inputType) {
  auto typeCase = substraitField.reference_type_case();
  switch (typeCase) {
    case ::substrait::Expression::FieldReference::ReferenceTypeCase::
        kDirectReference: {
      const auto& directRef = substraitField.direct_reference();
      int32_t colIdx = substraitParser_.parseReferenceSegment(directRef);

      const auto& inputTypes = inputType->children();
      const auto& inputNames = inputType->names();
      const int64_t inputSize = inputNames.size();

      if (colIdx <= inputSize) {
        // Convert type to row.
        return std::make_shared<core::FieldAccessTypedExpr>(
            inputTypes[colIdx],
            std::make_shared<core::InputTypedExpr>(inputTypes[colIdx]),
            inputNames[colIdx]);
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
    const ::substrait::Expression::ScalarFunction& substraitFunc,
    const RowTypePtr& inputType) {
  std::vector<std::shared_ptr<const core::ITypedExpr>> params;
  params.reserve(substraitFunc.args().size());
  for (const auto& sArg : substraitFunc.args()) {
    params.emplace_back(toVeloxExpr(sArg, inputType));
  }
  const auto& veloxFunction = substraitParser_.findVeloxFunction(
      functionMap_, substraitFunc.function_reference());
  const auto& veloxType = toVeloxType(
      substraitParser_.parseType(substraitFunc.output_type())->type);

  return std::make_shared<const core::CallTypedExpr>(
      veloxType, std::move(params), veloxFunction);
}

std::shared_ptr<const core::ConstantTypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::Literal& substraitLit) {
  auto typeCase = substraitLit.literal_type_case();
  switch (typeCase) {
    case ::substrait::Expression_Literal::LiteralTypeCase::kBoolean:
      return std::make_shared<core::ConstantTypedExpr>(
          variant(substraitLit.boolean()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI32:
      return std::make_shared<core::ConstantTypedExpr>(
          variant(substraitLit.i32()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI64:
      return std::make_shared<core::ConstantTypedExpr>(
          variant(substraitLit.i64()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kFp64:
      return std::make_shared<core::ConstantTypedExpr>(
          variant(substraitLit.fp64()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kNull: {
      auto veloxType =
          toVeloxType(substraitParser_.parseType(substraitLit.null())->type);
      return std::make_shared<core::ConstantTypedExpr>(
          veloxType, variant::null(veloxType->kind()));
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kString:
      return std::make_shared<core::ConstantTypedExpr>(
          variant(literal.string()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kList: {
      // Will wrap a constant vector with an array vector inside to create the constant expression.
      std::vector<variant> variants;
      variants.reserve(substraitLit.list().values().size());
      VELOX_CHECK(
          substraitLit.list().values().size() > 0,
          "List should have at least one item.");
      std::optional<TypePtr> literalType = std::nullopt;
      for (const auto& literal : substraitLit.list().values()) {
        auto typedVariant = toTypedVariant(literal);
        if (!literalType.has_value()) {
          literalType = typedVariant->variantType;
        }
        variants.emplace_back(typedVariant->veloxVariant);
      }
      VELOX_CHECK(literalType.has_value(), "Type expected.");
      // Create flat vector from the variants.
      VectorPtr vector =
          setVectorFromVariants(literalType.value(), variants, pool_);
      // Create array vector from the flat vector.
      ArrayVectorPtr arrayVector =
          toArrayVector(literalType.value(), vector, pool_);
      // Wrap the array vector into constant vector.
      auto constantVector = BaseVector::wrapInConstant(1, 0, arrayVector);
      auto constantExpr =
          std::make_shared<core::ConstantTypedExpr>(constantVector);
      return constantExpr;
    }
    default:
      VELOX_NYI(
          "Substrait conversion not supported for type case '{}'", typeCase);
  }
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::Cast& castExpr,
    const RowTypePtr& inputType) {
  auto substraitType = substraitParser_.parseType(castExpr.type());
  auto type = toVeloxType(substraitType->type);
  // TODO add flag in substrait after. now is set false.
  bool nullOnFailure = false;

  std::vector<core::TypedExprPtr> inputs{
      toVeloxExpr(castExpr.input(), inputType)};

  return std::make_shared<core::CastTypedExpr>(type, inputs, nullOnFailure);
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression& substraitExpr,
    const RowTypePtr& inputType) {
  std::shared_ptr<const core::ITypedExpr> veloxExpr;
  auto typeCase = substraitExpr.rex_type_case();
  switch (typeCase) {
    case ::substrait::Expression::RexTypeCase::kLiteral:
      return toVeloxExpr(substraitExpr.literal());
    case ::substrait::Expression::RexTypeCase::kScalarFunction:
      return toVeloxExpr(substraitExpr.scalar_function(), inputType);
    case ::substrait::Expression::RexTypeCase::kSelection:
      return toVeloxExpr(substraitExpr.selection(), inputType);
    case ::substrait::Expression::RexTypeCase::kCast:
      return toVeloxExpr(substraitExpr.cast(), inputType);
    default:
      VELOX_NYI(
          "Substrait conversion not supported for Expression '{}'", typeCase);
  }
}

std::shared_ptr<SubstraitVeloxExprConverter::TypedVariant>
SubstraitVeloxExprConverter::toTypedVariant(
    const ::substrait::Expression::Literal& literal) {
  auto typeCase = literal.literal_type_case();
  switch (typeCase) {
    case ::substrait::Expression_Literal::LiteralTypeCase::kBoolean: {
      TypedVariant typedVariant = {variant(literal.boolean()), BOOLEAN()};
      return std::make_shared<TypedVariant>(typedVariant);
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kI32: {
      TypedVariant typedVariant = {variant(literal.i32()), INTEGER()};
      return std::make_shared<TypedVariant>(typedVariant);
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kI64: {
      TypedVariant typedVariant = {variant(literal.i64()), BIGINT()};
      return std::make_shared<TypedVariant>(typedVariant);
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kFp64: {
      TypedVariant typedVariant = {variant(literal.fp64()), DOUBLE()};
      return std::make_shared<TypedVariant>(typedVariant);
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kString: {
      TypedVariant typedVariant = {variant(literal.string()), VARCHAR()};
      return std::make_shared<TypedVariant>(typedVariant);
    }
    default:
      VELOX_NYI("ToVariant not supported for type case '{}'", typeCase);
  }
}

} // namespace facebook::velox::substrait
