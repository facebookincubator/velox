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

#include <string_view>

namespace facebook::velox::functions::sparksql {
namespace detail {

/// Type tag for dispatching decimal native types (int64_t or int128_t).
template <typename T>
struct DecimalTypeTag {
  using type = T;
};

/// Clamps a user-supplied scale value to [-38, 38] to keep it within the
/// valid decimal precision range.
inline int32_t clampDecimalScale(int32_t s) {
  constexpr int32_t kMax = LongDecimalType::kMaxPrecision;
  if (s > kMax) {
    return kMax;
  }
  if (s < -kMax) {
    return -kMax;
  }
  return s;
}

/// Extracts a constant INTEGER value from an expression node. Used by
/// decimal special forms to read the compile-time scale argument.
/// Throws a user error if the expression is not a constant integer or is NULL.
inline int32_t extractConstantScaleArg(
    const exec::ExprPtr& expr,
    std::string_view funcName) {
  VELOX_USER_CHECK_EQ(
      expr->type()->kind(),
      TypeKind::INTEGER,
      "The second argument of {} must be INTEGER, got: {}.",
      funcName,
      expr->type()->toString());
  auto constantExpr = std::dynamic_pointer_cast<exec::ConstantExpr>(expr);
  VELOX_USER_CHECK_NOT_NULL(
      constantExpr,
      "The second argument of {} must be a constant expression.",
      funcName);
  VELOX_USER_CHECK(
      constantExpr->value()->isConstantEncoding(),
      "The second argument of {} must be a constant vector.",
      funcName);
  auto* constantVector =
      constantExpr->value()->asUnchecked<ConstantVector<int32_t>>();
  VELOX_USER_CHECK(
      !constantVector->isNullAt(0),
      "The second argument of {} must not be NULL.",
      funcName);
  return constantVector->valueAt(0);
}

/// Dispatches a factory callable over the four combinations of short/long
/// decimal for input and result types. The callable `fn` must accept two
/// DecimalTypeTag arguments and return std::shared_ptr<exec::VectorFunction>.
///
/// Usage:
///   dispatchDecimalTypes(inputType, resultType,
///       [&](auto resultTag, auto inputTag) {
///         using TResult = typename decltype(resultTag)::type;
///         using TInput = typename decltype(inputTag)::type;
///         return std::make_shared<MyFunc<TResult, TInput>>(...);
///       });
template <typename Fn>
auto dispatchDecimalTypes(
    const TypePtr& inputType,
    const TypePtr& resultType,
    Fn&& fn) {
  if (inputType->isShortDecimal()) {
    if (resultType->isShortDecimal()) {
      return fn(DecimalTypeTag<int64_t>{}, DecimalTypeTag<int64_t>{});
    }
    return fn(DecimalTypeTag<int128_t>{}, DecimalTypeTag<int64_t>{});
  }
  if (resultType->isShortDecimal()) {
    return fn(DecimalTypeTag<int64_t>{}, DecimalTypeTag<int128_t>{});
  }
  return fn(DecimalTypeTag<int128_t>{}, DecimalTypeTag<int128_t>{});
}

} // namespace detail
} // namespace facebook::velox::functions::sparksql
