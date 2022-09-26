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

#include "velox/substrait/VectorCreater.h"
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

template <>
IntervalDayTime getLiteralValue(
    const ::substrait::Expression::Literal& literal) {
  const auto& interval = literal.interval_day_to_second();
  int64_t milliseconds = interval.days() * kMillisInDay +
      interval.seconds() * kMillisInSecond + interval.microseconds() / 1000;
  return IntervalDayTime(milliseconds);
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
      const auto& dRef = substraitField.direct_reference();
      VELOX_CHECK(dRef.has_struct_field(), "Struct field expected.");
      int32_t colIdx = subParser_->parseReferenceSegment(dRef);
      std::optional<int32_t> childIdx;
      if (dRef.struct_field().has_child()) {
        childIdx =
            subParser_->parseReferenceSegment(dRef.struct_field().child());
      }

      const auto& inputTypes = inputType->children();
      const auto& inputNames = inputType->names();
      const int64_t inputSize = inputNames.size();

      if (colIdx >= inputSize) {
        VELOX_FAIL("Missing the column with id '{}' .", colIdx);
      }

      if (!childIdx.has_value()) {
        return std::make_shared<core::FieldAccessTypedExpr>(
            inputTypes[colIdx],
            std::make_shared<core::InputTypedExpr>(inputTypes[colIdx]),
            inputNames[colIdx]);
      } else {
        // Select a subfield in a struct by name.
        if (auto inputColumnType = asRowType(inputTypes[colIdx])) {
          if (childIdx.value() >= inputColumnType->size()) {
            VELOX_FAIL("Missing the subfield with id '{}' .", childIdx.value());
          }
          return std::make_shared<core::FieldAccessTypedExpr>(
              inputColumnType->childAt(childIdx.value()),
              std::make_shared<core::FieldAccessTypedExpr>(
                  inputTypes[colIdx], inputNames[colIdx]),
              inputColumnType->nameOf(childIdx.value()));
        } else {
          VELOX_FAIL("RowType expected.");
        }
      }
      break;
    }
    default:
      VELOX_NYI(
          "Substrait conversion not supported for Reference '{}'", typeCase);
  }
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toExtractExpr(
    const std::vector<std::shared_ptr<const core::ITypedExpr>>& params,
    const TypePtr& outputType) {
  VELOX_CHECK_EQ(params.size(), 2);
  auto functionArg =
      std::dynamic_pointer_cast<const core::ConstantTypedExpr>(params[0]);
  if (functionArg) {
    // Get the function argument.
    auto variant = functionArg->value();
    if (!variant.hasValue()) {
      VELOX_FAIL("Value expected in variant.");
    }
    // The first parameter specifies extracting from which field.
    // Only year is supported currently.
    std::string from = variant.value<std::string>();

    // The second parameter is the function parameter.
    std::vector<std::shared_ptr<const core::ITypedExpr>> exprParams;
    exprParams.reserve(1);
    exprParams.emplace_back(params[1]);
    if (from == "YEAR") {
      // Use PrestoSql year function.
      return std::make_shared<const core::CallTypedExpr>(
          outputType, std::move(exprParams), "year");
    }
    VELOX_NYI("Extract from {} not supported.", from);
  }
  VELOX_FAIL("Constant is expected to be the first parameter in extract.");
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::ScalarFunction& sFunc,
    const RowTypePtr& inputType) {
  std::vector<core::TypedExprPtr> params;
  params.reserve(sFunc.arguments().size());
  for (const auto& sArg : sFunc.arguments()) {
    params.emplace_back(toVeloxExpr(sArg.value(), inputType));
  }
  const auto& veloxFunction =
      subParser_->findVeloxFunction(functionMap_, sFunc.function_reference());
  std::string typeName = subParser_->parseType(sFunc.output_type())->type;

  if (veloxFunction == "extract") {
    return toExtractExpr(std::move(params), toVeloxType(typeName));
  }

  return std::make_shared<const core::CallTypedExpr>(
      toVeloxType(typeName), std::move(params), veloxFunction);
}

std::shared_ptr<const core::ConstantTypedExpr>
SubstraitVeloxExprConverter::literalsToConstantExpr(
    const std::vector<::substrait::Expression::Literal>& literals) {
  std::vector<variant> variants;
  variants.reserve(literals.size());
  VELOX_CHECK(literals.size() > 0, "List should have at least one item.");
  std::optional<TypePtr> literalType = std::nullopt;
  for (const auto& literal : literals) {
    auto veloxVariant = toVeloxExpr(literal)->value();
    if (!literalType.has_value()) {
      literalType = veloxVariant.inferType();
    }
    variants.emplace_back(veloxVariant);
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
  return std::make_shared<const core::ConstantTypedExpr>(constantVector);
}

core::TypedExprPtr SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::SingularOrList& singularOrList,
    const RowTypePtr& inputType) {
  VELOX_CHECK(
      singularOrList.options_size() > 0, "At least one option is expected.");
  auto options = singularOrList.options();
  std::vector<::substrait::Expression::Literal> literals;
  literals.reserve(options.size());
  for (const auto& option : options) {
    VELOX_CHECK(option.has_literal(), "Literal is expected as option.");
    literals.emplace_back(option.literal());
  }

  std::vector<std::shared_ptr<const core::ITypedExpr>> params;
  params.reserve(2);
  // First param is the value, second param is the list.
  params.emplace_back(toVeloxExpr(singularOrList.value(), inputType));
  params.emplace_back(literalsToConstantExpr(literals));
  return std::make_shared<const core::CallTypedExpr>(
      BOOLEAN(), std::move(params), "in");
}

std::shared_ptr<const core::ConstantTypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::Literal& substraitLit) {
  auto typeCase = substraitLit.literal_type_case();
  switch (typeCase) {
    case ::substrait::Expression_Literal::LiteralTypeCase::kBoolean:
      return std::make_shared<core::ConstantTypedExpr>(
          variant(substraitLit.boolean()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI8:
      // SubstraitLit.i8() will return int32, so we need this type conversion.
      return std::make_shared<core::ConstantTypedExpr>(
          variant(static_cast<int8_t>(substraitLit.i8())));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI16:
      // SubstraitLit.i16() will return int32, so we need this type conversion.
      return std::make_shared<core::ConstantTypedExpr>(
          variant(static_cast<int16_t>(substraitLit.i16())));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI32:
      return std::make_shared<core::ConstantTypedExpr>(
          variant(substraitLit.i32()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kFp32:
      return std::make_shared<core::ConstantTypedExpr>(
          variant(substraitLit.fp32()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kI64:
      return std::make_shared<core::ConstantTypedExpr>(
          variant(substraitLit.i64()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kFp64:
      return std::make_shared<core::ConstantTypedExpr>(
          variant(substraitLit.fp64()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kString:
      return std::make_shared<core::ConstantTypedExpr>(
          variant(substraitLit.string()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kNull: {
      auto veloxType =
          toVeloxType(subParser_->parseType(substraitLit.null())->type);
      return std::make_shared<core::ConstantTypedExpr>(
          veloxType, variant::null(veloxType->kind()));
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kList: {
      // Literals in List are put in a constant vector.
      std::vector<::substrait::Expression::Literal> literals;
      literals.reserve(substraitLit.list().values().size());
      for (const auto& literal : substraitLit.list().values()) {
        literals.emplace_back(literal);
      }
      return literalsToConstantExpr(literals);
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kVarChar:
      return std::make_shared<core::ConstantTypedExpr>(
          variant(substraitLit.var_char().value()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kList: {
      auto constantVector =
          BaseVector::wrapInConstant(1, 0, literalsToArrayVector(substraitLit));
      return std::make_shared<const core::ConstantTypedExpr>(constantVector);
    }
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
      return makeArrayVector(constructFlatVector<TypeKind::INTERVAL_DAY_TIME>(
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
  // TODO add flag in substrait after. now is set false.
  bool nullOnFailure = false;

  std::vector<core::TypedExprPtr> inputs{
      toVeloxExpr(castExpr.input(), inputType)};

  return std::make_shared<core::CastTypedExpr>(type, inputs, nullOnFailure);
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::IfThen& ifThenExpr,
    const RowTypePtr& inputType) {
  VELOX_CHECK(ifThenExpr.ifs().size() > 0, "If clause expected.");

  // Params are concatenated conditions and results with an optional "else" at
  // the end, e.g. {condition1, result1, condition2, result2,..else}
  std::vector<core::TypedExprPtr> params;
  // If and then expressions are in pairs.
  params.reserve(ifThenExpr.ifs().size() * 2);
  std::optional<TypePtr> outputType = std::nullopt;
  for (const auto& ifThen : ifThenExpr.ifs()) {
    params.emplace_back(toVeloxExpr(ifThen.if_(), inputType));
    const auto& thenExpr = toVeloxExpr(ifThen.then(), inputType);
    // Get output type from the first then expression.
    if (!outputType.has_value()) {
      outputType = thenExpr->type();
    }
    params.emplace_back(thenExpr);
  }

  if (ifThenExpr.has_else_()) {
    params.reserve(1);
    params.emplace_back(toVeloxExpr(ifThenExpr.else_(), inputType));
  }

  VELOX_CHECK(outputType.has_value(), "Output type should be set.");
  if (ifThenExpr.ifs().size() == 1) {
    // If there is only one if-then clause, use if expression.
    return std::make_shared<const core::CallTypedExpr>(
        outputType.value(), std::move(params), "if");
  }
  return std::make_shared<const core::CallTypedExpr>(
      outputType.value(), std::move(params), "switch");
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression& sExpr,
    const RowTypePtr& inputType) {
  std::shared_ptr<const core::ITypedExpr> veloxExpr;
  auto typeCase = sExpr.rex_type_case();
  switch (typeCase) {
    case ::substrait::Expression::RexTypeCase::kLiteral:
      return toVeloxExpr(sExpr.literal());
    case ::substrait::Expression::RexTypeCase::kScalarFunction:
      return toVeloxExpr(sExpr.scalar_function(), inputType);
    case ::substrait::Expression::RexTypeCase::kSelection:
      return toVeloxExpr(sExpr.selection(), inputType);
    case ::substrait::Expression::RexTypeCase::kCast:
      return toVeloxExpr(sExpr.cast(), inputType);
    case ::substrait::Expression::RexTypeCase::kIfThen:
      return toVeloxExpr(sExpr.if_then(), inputType);
    case ::substrait::Expression::RexTypeCase::kSingularOrList:
      return toVeloxExpr(sExpr.singular_or_list(), inputType);
    default:
      VELOX_NYI(
          "Substrait conversion not supported for Expression '{}'", typeCase);
  }
}

} // namespace facebook::velox::substrait
