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
#pragma once

#include "velox/expression/ConstantExpr.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SimpleVector.h"
#include "velox/vector/VectorTypeUtils.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace facebook::velox::cudf_velox {

template <typename T>
cudf::ast::literal makeLiteralFromScalar(
    cudf::scalar& scalar,
    const TypePtr& type) {
  if constexpr (cudf::is_fixed_width<T>()) {
    if (type->isIntervalDayTime()) {
      using CudfDurationType = cudf::duration_ms;
      if constexpr (std::is_same_v<T, CudfDurationType::rep>) {
        using CudfScalarType = cudf::duration_scalar<CudfDurationType>;
        return cudf::ast::literal{*static_cast<CudfScalarType*>(&scalar)};
      }
    } else if (type->isDate()) {
      using CudfDateType = cudf::timestamp_D;
      if constexpr (std::is_same_v<T, CudfDateType::rep>) {
        using CudfScalarType = cudf::timestamp_scalar<CudfDateType>;
        return cudf::ast::literal{*static_cast<CudfScalarType*>(&scalar)};
      }
    } else {
      // Create a numeric scalar of type T, store it in the scalars vector,
      // and use its reference in the literal expression.
      using CudfScalarType = cudf::numeric_scalar<T>;
      return cudf::ast::literal{*static_cast<CudfScalarType*>(&scalar)};
    }
    VELOX_FAIL("Unsupported base type for literal");
  } else if (type->kind() == TypeKind::VARCHAR) {
    return cudf::ast::literal{*static_cast<cudf::string_scalar*>(&scalar)};
  } else {
    // TODO for non-numeric types too.
    VELOX_NYI(
        "Non-numeric types not yet implemented for type {}", type->toString());
  }
}

template <TypeKind kind>
variant getVariant(const VectorPtr& vector, size_t atIndex = 0) {
  using T = typename facebook::velox::KindToFlatVector<kind>::WrapperType;
  if constexpr (!std::is_same_v<T, ComplexType>) {
    return vector->as<SimpleVector<T>>()->valueAt(atIndex);
  } else {
    return Variant();
  }
}

template <typename T>
std::unique_ptr<cudf::scalar> makeScalarFromValue(
    const TypePtr& type,
    T value,
    bool isNull,
    std::optional<cudf::type_id> toType = std::nullopt) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  if constexpr (cudf::is_fixed_width<T>()) {
    if (type->isDecimal()) {
      VELOX_FAIL("Decimal not supported");
      /* TODO: enable after rewriting using binary ops
     using CudfDecimalType = cudf::numeric::decimal64;
     using cudfScalarType = cudf::fixed_point_scalar<CudfDecimalType>;
     auto scalar = std::make_unique<cudfScalarType>(value,
                   type->scale(),
                    true,
                    stream,
                    mr);*/
    } else if (type->isIntervalYearMonth()) {
      VELOX_FAIL("Interval year month not supported");
    } else if (type->isIntervalDayTime()) {
      using CudfDurationType = cudf::duration_ms;
      if constexpr (std::is_same_v<T, CudfDurationType::rep>) {
        return std::make_unique<cudf::duration_scalar<CudfDurationType>>(
            value, !isNull, stream, mr);
      }
    } else if (type->isDate()) {
      using CudfDateType = cudf::timestamp_D;
      if constexpr (std::is_same_v<T, CudfDateType::rep>) {
        return std::make_unique<cudf::timestamp_scalar<CudfDateType>>(
            value, !isNull, stream, mr);
      }
    } else if (toType.has_value()) {
      if (toType == cudf::type_id::DURATION_DAYS) {
        return std::make_unique<cudf::duration_scalar<cudf::duration_D>>(
            value, !isNull, stream, mr);
      }
      VELOX_FAIL(
          "Unsupported result type {}", static_cast<int32_t>(toType.value()));
    } else {
      return std::make_unique<cudf::numeric_scalar<T>>(
          value, !isNull, stream, mr);
    }
    VELOX_FAIL("Unsupported fixed-width scalar type");
  } else if constexpr (
      std::is_same_v<T, StringView> || std::is_same_v<T, std::string_view> ||
      std::is_same_v<T, std::string>) {
    return std::make_unique<cudf::string_scalar>(
        std::string_view(value.data(), value.size()), !isNull, stream, mr);
  }
  VELOX_NYI("Scalar creation not implemented for type " + type->toString());
}

template <TypeKind Kind>
static std::unique_ptr<cudf::scalar> createCudfScalar(
    const velox::VectorPtr& value,
    std::optional<cudf::type_id> toType = std::nullopt) {
  using T = typename TypeTraits<Kind>::NativeType;
  auto vector = value->as<velox::ConstantVector<T>>();
  return makeScalarFromValue<T>(
      vector->type(), vector->value(), vector->isNullAt(0), toType);
}

inline std::unique_ptr<cudf::scalar> makeScalarFromConstantExpr(
    const std::shared_ptr<velox::exec::Expr>& expr,
    std::optional<cudf::type_id> toType = std::nullopt) {
  auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr);
  VELOX_CHECK_NOT_NULL(constExpr);
  auto constValue = constExpr->value();
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      createCudfScalar, constValue->typeKind(), constValue, toType);
}

template <TypeKind kind>
cudf::ast::literal makeScalarAndLiteral(
    const TypePtr& type,
    const variant& var,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
  using T = typename TypeTraits<kind>::NativeType;
  if constexpr (cudf::is_fixed_width<T>() || kind == TypeKind::VARCHAR) {
    auto value = var.value<T>();
    auto scalar = makeScalarFromValue(type, value, false);
    scalars.emplace_back(std::move(scalar));
    return makeLiteralFromScalar<T>(*(scalars.back()), type);
  }
  VELOX_NYI("Scalar creation not implemented for type " + type->toString());
}

} // namespace facebook::velox::cudf_velox
