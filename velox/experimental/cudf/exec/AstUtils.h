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

#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SimpleVector.h"
#include "velox/vector/VectorTypeUtils.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace facebook::velox::cudf_velox {

template <TypeKind kind>
cudf::ast::literal makeScalarAndLiteral(
    const TypePtr& type,
    const variant& var,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
  using T = typename facebook::velox::KindToFlatVector<kind>::WrapperType;
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  if constexpr (cudf::is_fixed_width<T>()) {
    T value = var.value<T>();
    if (type->isShortDecimal()) {
      VELOX_FAIL("Short decimal not supported");
      /* TODO: enable after rewriting using binary ops
      using CudfDecimalType = cudf::numeric::decimal64;
      using cudfScalarType = cudf::fixed_point_scalar<CudfDecimalType>;
      auto scalar = std::make_unique<cudfScalarType>(value,
                    type->scale(),
                     true,
                     stream,
                     mr);
      scalars.emplace_back(std::move(scalar));
      return cudf::ast::literal{
          *static_cast<cudfScalarType*>(scalars.back().get())};
      */
    } else if (type->isLongDecimal()) {
      VELOX_FAIL("Long decimal not supported");
      /* TODO: enable after rewriting using binary ops
      using CudfDecimalType = cudf::numeric::decimal128;
      using cudfScalarType = cudf::fixed_point_scalar<CudfDecimalType>;
      auto scalar = std::make_unique<cudfScalarType>(value,
                    type->scale(),
                     true,
                     stream,
                     mr);
      scalars.emplace_back(std::move(scalar));
      return cudf::ast::literal{
          *static_cast<cudfScalarType*>(scalars.back().get())};
      */
    } else if (type->isIntervalYearMonth()) {
      // no support for interval year month in cudf
      VELOX_FAIL("Interval year month not supported");
    } else if (type->isIntervalDayTime()) {
      using CudfDurationType = cudf::duration_ms;
      if constexpr (std::is_same_v<T, CudfDurationType::rep>) {
        using CudfScalarType = cudf::duration_scalar<CudfDurationType>;
        auto scalar = std::make_unique<CudfScalarType>(value, true, stream, mr);
        scalars.emplace_back(std::move(scalar));
        return cudf::ast::literal{
            *static_cast<CudfScalarType*>(scalars.back().get())};
      }
    } else if (type->isDate()) {
      using CudfDateType = cudf::timestamp_D;
      if constexpr (std::is_same_v<T, CudfDateType::rep>) {
        using CudfScalarType = cudf::timestamp_scalar<CudfDateType>;
        auto scalar = std::make_unique<CudfScalarType>(value, true, stream, mr);
        scalars.emplace_back(std::move(scalar));
        return cudf::ast::literal{
            *static_cast<CudfScalarType*>(scalars.back().get())};
      }
    } else {
      // Create a numeric scalar of type T, store it in the scalars vector,
      // and use its reference in the literal expression.
      using CudfScalarType = cudf::numeric_scalar<T>;
      scalars.emplace_back(
          std::make_unique<CudfScalarType>(value, true, stream, mr));
      return cudf::ast::literal{
          *static_cast<CudfScalarType*>(scalars.back().get())};
    }
    VELOX_FAIL("Unsupported base type for literal");
  } else if (kind == TypeKind::VARCHAR) {
    auto stringValue = var.value<StringView>();
    scalars.emplace_back(
        std::make_unique<cudf::string_scalar>(stringValue, true, stream, mr));
    return cudf::ast::literal{
        *static_cast<cudf::string_scalar*>(scalars.back().get())};
  } else {
    // TODO for non-numeric types too.
    VELOX_NYI(
        "Non-numeric types not yet implemented for kind " +
        mapTypeKindToName(kind));
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

inline cudf::ast::literal createLiteral(
    const VectorPtr& vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    size_t atIndex = 0) {
  const auto kind = vector->typeKind();
  const auto& type = vector->type();
  variant value =
      VELOX_DYNAMIC_TYPE_DISPATCH(getVariant, kind, vector, atIndex);
  return VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      makeScalarAndLiteral, kind, type, value, scalars);
}

} // namespace facebook::velox::cudf_velox
