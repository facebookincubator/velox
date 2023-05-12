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
#include "velox/vector/FlatVector.h"

using namespace facebook::velox;
namespace {
// Get values for the different supported types.
template <typename T>
T getLiteralValue(const ::substrait::Expression::Literal& /* literal */) {
  VELOX_NYI();
}

template <>
int8_t getLiteralValue(const ::substrait::Expression::Literal& literal) {
  return static_cast<int8_t>(literal.i8());
}

template <>
int16_t getLiteralValue(const ::substrait::Expression::Literal& literal) {
  return static_cast<int16_t>(literal.i16());
}

template <>
int32_t getLiteralValue(const ::substrait::Expression::Literal& literal) {
  return literal.i32();
}

template <>
int64_t getLiteralValue(const ::substrait::Expression::Literal& literal) {
  return literal.i64();
}

template <>
double getLiteralValue(const ::substrait::Expression::Literal& literal) {
  return literal.fp64();
}

template <>
float getLiteralValue(const ::substrait::Expression::Literal& literal) {
  return literal.fp32();
}

template <>
bool getLiteralValue(const ::substrait::Expression::Literal& literal) {
  return literal.boolean();
}

template <>
uint32_t getLiteralValue(const ::substrait::Expression::Literal& literal) {
  return literal.i32();
}

template <>
Timestamp getLiteralValue(const ::substrait::Expression::Literal& literal) {
  return Timestamp::fromMicros(literal.timestamp());
}

template <>
Date getLiteralValue(const ::substrait::Expression::Literal& literal) {
  return Date(literal.date());
}

ArrayVectorPtr makeArrayVector(const VectorPtr& elements) {
  BufferPtr offsets = allocateOffsets(1, elements->pool());
  BufferPtr sizes = allocateOffsets(1, elements->pool());
  sizes->asMutable<vector_size_t>()[0] = elements->size();

  return std::make_shared<ArrayVector>(
      elements->pool(),
      ARRAY(elements->type()),
      nullptr,
      1,
      offsets,
      sizes,
      elements);
}

ArrayVectorPtr makeEmptyArrayVector(memory::MemoryPool* pool) {
  BufferPtr offsets = allocateOffsets(1, pool);
  BufferPtr sizes = allocateOffsets(1, pool);
  return std::make_shared<ArrayVector>(
      pool, ARRAY(UNKNOWN()), nullptr, 1, offsets, sizes, nullptr);
}

template <typename T>
void setLiteralValue(
    const ::substrait::Expression::Literal& literal,
    FlatVector<T>* vector,
    vector_size_t index) {
  if (literal.has_null()) {
    vector->setNull(index, true);
  } else if constexpr (std::is_same_v<T, StringView>) {
    if (literal.has_string()) {
      vector->set(index, StringView(literal.string()));
    } else if (literal.has_var_char()) {
      vector->set(index, StringView(literal.var_char().value()));
    } else {
      VELOX_FAIL("Unexpected string literal");
    }
  } else {
    vector->set(index, getLiteralValue<T>(literal));
  }
}

template <TypeKind kind>
VectorPtr constructFlatVector(
    const ::substrait::Expression::Literal& listLiteral,
    const vector_size_t size,
    const TypePtr& type,
    memory::MemoryPool* pool) {
  VELOX_CHECK(type->isPrimitiveType());
  auto vector = BaseVector::create(type, size, pool);
  using T = typename TypeTraits<kind>::NativeType;
  auto flatVector = vector->as<FlatVector<T>>();

  vector_size_t index = 0;
  for (auto child : listLiteral.list().values()) {
    setLiteralValue(child, flatVector, index++);
  }
  return vector;
}

/// Whether null will be returned on cast failure.
bool isNullOnFailure(
    ::substrait::Expression::Cast::FailureBehavior failureBehavior) {
  switch (failureBehavior) {
    case ::substrait::
        Expression_Cast_FailureBehavior_FAILURE_BEHAVIOR_UNSPECIFIED:
    case ::substrait::
        Expression_Cast_FailureBehavior_FAILURE_BEHAVIOR_THROW_EXCEPTION:
      return false;
    case ::substrait::
        Expression_Cast_FailureBehavior_FAILURE_BEHAVIOR_RETURN_NULL:
      return true;
    default:
      VELOX_NYI(
          "The given failure behavior is NOT supported: '{}'", failureBehavior);
  }
}

} // namespace

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
      const auto& inputNames = inputType->names();
      const int64_t inputSize = inputNames.size();
      if (colIdx <= inputSize) {
        const auto& inputTypes = inputType->children();
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
  std::vector<core::TypedExprPtr> params;
  params.reserve(substraitFunc.arguments().size());
  for (const auto& sArg : substraitFunc.arguments()) {
    params.emplace_back(toVeloxExpr(sArg.value(), inputType));
  }
  const auto& veloxFunction = substraitParser_.findVeloxFunction(
      functionMap_, substraitFunc.function_reference());
  std::string typeName =
      substraitParser_.parseType(substraitFunc.output_type())->type;
  return std::make_shared<const core::CallTypedExpr>(
      toVeloxType(typeName), std::move(params), veloxFunction);
}

std::shared_ptr<const core::ConstantTypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::Literal& substraitLit) {
  auto typeCase = substraitLit.literal_type_case();
  switch (typeCase) {
    case ::substrait::Expression_Literal::LiteralTypeCase::kBoolean:
      return std::make_shared<core::ConstantTypedExpr>(
          BOOLEAN(), variant(substraitLit.boolean()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI8:
      // SubstraitLit.i8() will return int32, so we need this type conversion.
      return std::make_shared<core::ConstantTypedExpr>(
          TINYINT(), variant(static_cast<int8_t>(substraitLit.i8())));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI16:
      // SubstraitLit.i16() will return int32, so we need this type conversion.
      return std::make_shared<core::ConstantTypedExpr>(
          SMALLINT(), variant(static_cast<int16_t>(substraitLit.i16())));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI32:
      return std::make_shared<core::ConstantTypedExpr>(
          INTEGER(), variant(substraitLit.i32()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kFp32:
      return std::make_shared<core::ConstantTypedExpr>(
          REAL(), variant(substraitLit.fp32()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI64:
      return std::make_shared<core::ConstantTypedExpr>(
          BIGINT(), variant(substraitLit.i64()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kFp64:
      return std::make_shared<core::ConstantTypedExpr>(
          DOUBLE(), variant(substraitLit.fp64()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kString:
      return std::make_shared<core::ConstantTypedExpr>(
          VARCHAR(), variant(substraitLit.string()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kNull: {
      auto veloxType =
          toVeloxType(substraitParser_.parseType(substraitLit.null())->type);
      return std::make_shared<core::ConstantTypedExpr>(
          veloxType, variant::null(veloxType->kind()));
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kVarChar:
      return std::make_shared<core::ConstantTypedExpr>(
          VARCHAR(), variant(substraitLit.var_char().value()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kList: {
      auto constantVector =
          BaseVector::wrapInConstant(1, 0, literalsToArrayVector(substraitLit));
      return std::make_shared<const core::ConstantTypedExpr>(constantVector);
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kDate:
      return std::make_shared<core::ConstantTypedExpr>(
          DATE(), variant(Date(substraitLit.date())));
    default:
      VELOX_NYI(
          "Substrait conversion not supported for type case '{}'", typeCase);
  }
}

ArrayVectorPtr SubstraitVeloxExprConverter::literalsToArrayVector(
    const ::substrait::Expression::Literal& listLiteral) {
  auto childSize = listLiteral.list().values().size();
  if (childSize == 0) {
    return makeEmptyArrayVector(pool_);
  }
  auto typeCase = listLiteral.list().values(0).literal_type_case();
  switch (typeCase) {
    case ::substrait::Expression_Literal::LiteralTypeCase::kBoolean:
      return makeArrayVector(constructFlatVector<TypeKind::BOOLEAN>(
          listLiteral, childSize, BOOLEAN(), pool_));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI8:
      return makeArrayVector(constructFlatVector<TypeKind::TINYINT>(
          listLiteral, childSize, TINYINT(), pool_));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI16:
      return makeArrayVector(constructFlatVector<TypeKind::SMALLINT>(
          listLiteral, childSize, SMALLINT(), pool_));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI32:
      return makeArrayVector(constructFlatVector<TypeKind::INTEGER>(
          listLiteral, childSize, INTEGER(), pool_));
    case ::substrait::Expression_Literal::LiteralTypeCase::kFp32:
      return makeArrayVector(constructFlatVector<TypeKind::REAL>(
          listLiteral, childSize, REAL(), pool_));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI64:
      return makeArrayVector(constructFlatVector<TypeKind::BIGINT>(
          listLiteral, childSize, BIGINT(), pool_));
    case ::substrait::Expression_Literal::LiteralTypeCase::kFp64:
      return makeArrayVector(constructFlatVector<TypeKind::DOUBLE>(
          listLiteral, childSize, DOUBLE(), pool_));
    case ::substrait::Expression_Literal::LiteralTypeCase::kString:
    case ::substrait::Expression_Literal::LiteralTypeCase::kVarChar:
      return makeArrayVector(constructFlatVector<TypeKind::VARCHAR>(
          listLiteral, childSize, VARCHAR(), pool_));
    case ::substrait::Expression_Literal::LiteralTypeCase::kNull: {
      auto veloxType =
          toVeloxType(substraitParser_.parseType(listLiteral.null())->type);
      auto kind = veloxType->kind();
      return makeArrayVector(VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          constructFlatVector, kind, listLiteral, childSize, veloxType, pool_));
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kDate:
      return makeArrayVector(constructFlatVector<TypeKind::DATE>(
          listLiteral, childSize, DATE(), pool_));
    case ::substrait::Expression_Literal::LiteralTypeCase::kTimestamp:
      return makeArrayVector(constructFlatVector<TypeKind::TIMESTAMP>(
          listLiteral, childSize, TIMESTAMP(), pool_));
    case ::substrait::Expression_Literal::LiteralTypeCase::kIntervalDayToSecond:
      return makeArrayVector(constructFlatVector<TypeKind::BIGINT>(
          listLiteral, childSize, INTERVAL_DAY_TIME(), pool_));
    case ::substrait::Expression_Literal::LiteralTypeCase::kList: {
      VectorPtr elements;
      for (auto it : listLiteral.list().values()) {
        auto v = literalsToArrayVector(it);
        if (!elements) {
          elements = v;
        } else {
          elements->append(v.get());
        }
      }
      return makeArrayVector(elements);
    }
    default:
      VELOX_NYI(
          "literalsToArrayVector not supported for type case '{}'", typeCase);
  }
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::Cast& castExpr,
    const RowTypePtr& inputType) {
  auto substraitType = substraitParser_.parseType(castExpr.type());
  auto type = toVeloxType(substraitType->type);
  bool nullOnFailure = isNullOnFailure(castExpr.failure_behavior());

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
    case ::substrait::Expression::RexTypeCase::kIfThen:
      return toVeloxExpr(substraitExpr.if_then(), inputType);
    default:
      VELOX_NYI(
          "Substrait conversion not supported for Expression '{}'", typeCase);
  }
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression_IfThen& substraitIfThen,
    const RowTypePtr& inputType) {
  std::vector<core::TypedExprPtr> inputs;
  if (substraitIfThen.has_else_()) {
    inputs.reserve(substraitIfThen.ifs_size() * 2 + 1);
  } else {
    inputs.reserve(substraitIfThen.ifs_size() * 2);
  }

  TypePtr resultType;
  for (auto& ifExpr : substraitIfThen.ifs()) {
    auto ifClauseExpr = toVeloxExpr(ifExpr.if_(), inputType);
    inputs.emplace_back(ifClauseExpr);
    auto thenClauseExpr = toVeloxExpr(ifExpr.then(), inputType);
    inputs.emplace_back(thenClauseExpr);

    if (!thenClauseExpr->type()->containsUnknown()) {
      resultType = thenClauseExpr->type();
    }
  }

  if (substraitIfThen.has_else_()) {
    auto elseClauseExpr = toVeloxExpr(substraitIfThen.else_(), inputType);
    inputs.emplace_back(elseClauseExpr);
    if (!resultType && !elseClauseExpr->type()->containsUnknown()) {
      resultType = elseClauseExpr->type();
    }
  }

  VELOX_CHECK_NOT_NULL(resultType, "Result type not found");

  return std::make_shared<const core::CallTypedExpr>(
      resultType, std::move(inputs), "if");
}

} // namespace facebook::velox::substrait
