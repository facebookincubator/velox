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
#include "velox/vector/VariantToVector.h"

#include "velox/type/Timestamp.h"

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
UnscaledShortDecimal getLiteralValue(
    const ::substrait::Expression::Literal& literal) {
  VELOX_CHECK(literal.decimal().scale() <= 18);
  int128_t decimalValue;
  memcpy(&decimalValue, literal.decimal().value().c_str(), 16);
  return UnscaledShortDecimal((int64_t)decimalValue);
}

template <>
UnscaledLongDecimal getLiteralValue(
    const ::substrait::Expression::Literal& literal) {
  VELOX_CHECK(literal.decimal().scale() > 18);
  int128_t decimalValue;
  memcpy(&decimalValue, literal.decimal().value().c_str(), 16);
  return UnscaledLongDecimal(decimalValue);
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

RowVectorPtr makeRowVector(const std::vector<VectorPtr>& children) {
  std::vector<std::shared_ptr<const Type>> types;
  types.resize(children.size());
  for (int i = 0; i < children.size(); i++) {
    types[i] = children[i]->type();
  }
  const size_t vectorSize = children.empty() ? 0 : children.front()->size();
  auto rowType = ROW(std::move(types));
  return std::make_shared<RowVector>(
      children[0]->pool(), rowType, BufferPtr(nullptr), vectorSize, children);
}

ArrayVectorPtr makeEmptyArrayVector(memory::MemoryPool* pool) {
  BufferPtr offsets = allocateOffsets(1, pool);
  BufferPtr sizes = allocateOffsets(1, pool);
  return std::make_shared<ArrayVector>(
      pool, ARRAY(UNKNOWN()), nullptr, 1, offsets, sizes, nullptr);
}

RowVectorPtr makeEmptyRowVector(memory::MemoryPool* pool) {
  return makeRowVector({});
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
    } else if (literal.has_binary()) {
      vector->set(index, StringView(literal.binary()));
    } else {
      VELOX_FAIL("Unexpected string or binary literal");
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

template <TypeKind kind>
VectorPtr constructFlatVectorForStruct(
    const ::substrait::Expression::Literal& child,
    const vector_size_t size,
    const TypePtr& type,
    memory::MemoryPool* pool) {
  VELOX_CHECK(type->isPrimitiveType());
  auto vector = BaseVector::create(type, size, pool);
  using T = typename TypeTraits<kind>::NativeType;
  auto flatVector = vector->as<FlatVector<T>>();
  setLiteralValue(child, flatVector, 0);
  return vector;
}

core::FieldAccessTypedExprPtr makeFieldAccessExpr(
    const std::string& name,
    const TypePtr& type,
    core::FieldAccessTypedExprPtr input) {
  if (input) {
    return std::make_shared<core::FieldAccessTypedExpr>(type, input, name);
  }

  return std::make_shared<core::FieldAccessTypedExpr>(type, name);
}
} // namespace

using facebook::velox::core::variantArrayToVector;
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
      core::FieldAccessTypedExprPtr fieldAccess{nullptr};
      const auto* tmp = &directRef.struct_field();

      auto inputColumnType = inputType;
      for (;;) {
        auto idx = tmp->field();
        fieldAccess = makeFieldAccessExpr(
            inputColumnType->nameOf(idx),
            inputColumnType->childAt(idx),
            fieldAccess);

        if (!tmp->has_child()) {
          break;
        }

        inputColumnType = asRowType(inputColumnType->childAt(idx));
        tmp = &tmp->child().struct_field();
      }
      return fieldAccess;
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
    std::string from = variant.value<std::string>();

    // The second parameter is the function parameter.
    std::vector<std::shared_ptr<const core::ITypedExpr>> exprParams;
    exprParams.reserve(1);
    exprParams.emplace_back(params[1]);
    auto iter = extractDatetimeFunctionMap_.find(from);
    if (iter != extractDatetimeFunctionMap_.end()) {
      return std::make_shared<const core::CallTypedExpr>(
          outputType, std::move(exprParams), iter->second);
    } else {
      VELOX_NYI("Extract from {} not supported.", from);
    }
  }
  VELOX_FAIL("Constant is expected to be the first parameter in extract.");
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
  const auto& veloxFunction = subParser_->findVeloxFunction(
      functionMap_, substraitFunc.function_reference());
  std::string typeName =
      subParser_->parseType(substraitFunc.output_type())->type;

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
  VELOX_CHECK_GE(literals.size(), 0, "List should have at least one item.");
  std::optional<TypePtr> literalType = std::nullopt;
  for (const auto& literal : literals) {
    auto veloxVariant = toVeloxExpr(literal)->value();
    if (!literalType.has_value()) {
      literalType = veloxVariant.inferType();
    }
    variants.emplace_back(veloxVariant);
  }
  VELOX_CHECK(literalType.has_value(), "Type expected.");
  auto varArray = variant::array(variants);
  ArrayVectorPtr arrayVector =
      variantArrayToVector(varArray.inferType(), varArray.array(), pool_);
  // Wrap the array vector into constant vector.
  auto constantVector =
      BaseVector::wrapInConstant(1 /*length*/, 0 /*index*/, arrayVector);
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
    case ::substrait::Expression_Literal::LiteralTypeCase::kDate:
      return std::make_shared<core::ConstantTypedExpr>(
          DATE(), variant(Date(substraitLit.date())));
    case ::substrait::Expression_Literal::LiteralTypeCase::kTimestamp:
      return std::make_shared<core::ConstantTypedExpr>(
          TIMESTAMP(),
          variant(Timestamp::fromMicros(substraitLit.timestamp())));
    case ::substrait::Expression_Literal::LiteralTypeCase::kVarChar:
      return std::make_shared<core::ConstantTypedExpr>(
          VARCHAR(), variant(substraitLit.var_char().value()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kList: {
      auto constantVector =
          BaseVector::wrapInConstant(1, 0, literalsToArrayVector(substraitLit));
      return std::make_shared<const core::ConstantTypedExpr>(constantVector);
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kBinary:
      return std::make_shared<core::ConstantTypedExpr>(
          VARBINARY(), variant::binary(substraitLit.binary()));
    case ::substrait::Expression_Literal::LiteralTypeCase::kStruct: {
      auto constantVector =
          BaseVector::wrapInConstant(1, 0, literalsToRowVector(substraitLit));
      return std::make_shared<const core::ConstantTypedExpr>(constantVector);
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kDecimal: {
      auto decimal = substraitLit.decimal().value();
      auto precision = substraitLit.decimal().precision();
      auto scale = substraitLit.decimal().scale();
      int128_t decimalValue;
      memcpy(&decimalValue, decimal.c_str(), 16);
      if (precision <= 18) {
        auto type = SHORT_DECIMAL(precision, scale);
        return std::make_shared<core::ConstantTypedExpr>(
            type, variant::shortDecimal((int64_t)decimalValue, type));
      } else {
        auto type = LONG_DECIMAL(precision, scale);
        return std::make_shared<core::ConstantTypedExpr>(
            type, variant::longDecimal(decimalValue, type));
      }
    }
    case ::substrait::Expression_Literal::LiteralTypeCase::kNull: {
      auto veloxType =
          toVeloxType(subParser_->parseType(substraitLit.null())->type);
      if (veloxType->isShortDecimal()) {
        return std::make_shared<core::ConstantTypedExpr>(
            veloxType, variant::shortDecimal(std::nullopt, veloxType));
      } else if (veloxType->isLongDecimal()) {
        return std::make_shared<core::ConstantTypedExpr>(
            veloxType, variant::longDecimal(std::nullopt, veloxType));
      } else {
        return std::make_shared<core::ConstantTypedExpr>(
            veloxType, variant::null(veloxType->kind()));
      }
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
          toVeloxType(subParser_->parseType(listLiteral.null())->type);
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

RowVectorPtr SubstraitVeloxExprConverter::literalsToRowVector(
    const ::substrait::Expression::Literal& structLiteral) {
  auto childSize = structLiteral.struct_().fields().size();
  if (childSize == 0) {
    return makeEmptyRowVector(pool_);
  }
  std::vector<VectorPtr> vectors;
  vectors.reserve(structLiteral.struct_().fields().size());
  for (const auto& field : structLiteral.struct_().fields()) {
    const auto& typeCase = field.literal_type_case();
    switch (typeCase) {
      case ::substrait::Expression_Literal::LiteralTypeCase::kI64: {
        vectors.emplace_back(constructFlatVectorForStruct<TypeKind::BIGINT>(
            field, 1, BIGINT(), pool_));
        break;
      }
      case ::substrait::Expression_Literal::LiteralTypeCase::kBinary: {
        vectors.emplace_back(constructFlatVectorForStruct<TypeKind::VARBINARY>(
            field, 1, VARBINARY(), pool_));
        break;
      }
      case ::substrait::Expression_Literal::LiteralTypeCase::kDecimal: {
        auto precision = field.decimal().precision();
        auto scale = field.decimal().scale();
        if (precision <= 18) {
          vectors.emplace_back(
              constructFlatVectorForStruct<TypeKind::SHORT_DECIMAL>(
                  field, 1, SHORT_DECIMAL(precision, scale), pool_));
        } else {
          vectors.emplace_back(
              constructFlatVectorForStruct<TypeKind::LONG_DECIMAL>(
                  field, 1, LONG_DECIMAL(precision, scale), pool_));
        }
        break;
      }
      default:
        VELOX_NYI(
            "literalsToRowVector not supported for type case '{}'", typeCase);
    }
  }
  return makeRowVector(vectors);
}

std::shared_ptr<const core::ITypedExpr>
SubstraitVeloxExprConverter::toVeloxExpr(
    const ::substrait::Expression::Cast& castExpr,
    const RowTypePtr& inputType) {
  auto substraitType = subParser_->parseType(castExpr.type());
  auto type = toVeloxType(substraitType->type);
  bool nullOnFailure = isNullOnFailure(castExpr.failure_behavior());

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
    case ::substrait::Expression::RexTypeCase::kSingularOrList:
      return toVeloxExpr(substraitExpr.singular_or_list(), inputType);
    default:
      VELOX_NYI(
          "Substrait conversion not supported for Expression '{}'", typeCase);
  }
}

} // namespace facebook::velox::substrait
