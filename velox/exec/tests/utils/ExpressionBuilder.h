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

#include "velox/parse/Expressions.h"

namespace facebook::velox::expr::builder {

/// Fluent Expression Builder.
///
/// This file contains fluent methods that make it convenient to create
/// (untyped) expression trees. This provides similar functionality to a SQL
/// parser, without briging dependency on external libraries or bringing leaked
/// semantics from other systems.
///
/// The untyped expressions can then be turned into typed expressions ready for
/// execution using type binding from `core::Expressions::inferTypes()`.
///
/// The API provided is as close to the actual expression trees as possible.
/// Filters, arithmetics, function calls, literals, aliases, and more are
/// supported with this API.
///
/// For example, to create a field reference (reference to a column), one can:
///
/// > using namespace expr::builder;
/// > ExprPtr e = field("col");
///
/// Or use the "_f" literal provided:
///
/// > ExprPtr e = "col"_f;
///
/// To debug the expression generated, you can simply:
///
/// > LOG(INFO) << *e;
///
/// Filters (comparisons) can be fluently created using C++ overloaded
/// operators
///
/// > ExprPtr e = "col"_f > 10; // "col > 10"
/// > ExprPtr e = "col"_f != "bar"; // "col != 'bar'"
///
/// Null checking filters:
///
/// > ExprPtr e = isNull("col"_f) // "col is null"
/// > ExprPtr e = !isNull("col"_f) // "col is not null"
///
/// Conjuncts and "between":
///
/// > ExprPtr e = "a"_f && "b"_f || "c"_f; // "a AND b OR c"
/// > ExprPtr e = between("a"_f, 0L, 10L); // "a between 0 and 10"
///
/// C++ literals are automatically converted into ConstantExpr (expression
/// literals) when part of an expression. To explicitly create a literal:
///
/// > ExprPtr e = literal(10.3);
/// > ExprPtr e = literal("str");
/// > ExprPtr e = literalTimestamp("1970-01-01 01:30:00");
///
/// Arithmetic operators are also overloaded:
///
/// > ExprPtr e = "col"_f + 1 - 10; // "col + 1 - 10"
///
/// In any expression, as long as one of the sides is an expression node, the
/// correct expression will be created. For example, both version work as
/// expected:
///
/// > ExprPtr e = "col"_f * 100; // "col * 100"
/// > ExprPtr e = 100 * "col"_f; // "100 * col"
///
/// When building long expressions, be careful about C++ operator precedence.
/// For example:
///
/// > ExprPtr e = "col"_f + 5 * 100;
///
/// will generate the expression "col + 500" since C++ will first sum the two
/// integers. To force the expected behavior, you can explicitly spell out the
/// literal:
///
/// > ExprPtr e = "col"_f + 5 * literal(100);
/// > ExprPtr e = "col"_f + literal(5) * 100;
///
/// Both will generate "col + 5 * 100", which is "plus(col, multiply(5, 100))".
///
/// Generic function calls can be created using `call()`:
///
/// > ExprPtr e = call("func", 10L); // "func(10)"
///
/// `call()` supports arbitrary parameters, which can be other expressions or
/// (C++) literals.
///
/// Aliases can be defined using `alias()`:
///
/// > ExprPtr e = alias("col"_f, "foo"); // "col as foo"
/// > ExprPtr e = alias(10, "foo"); // "10 as foo"
///
/// All functions above can be nested and combined in arbitrary ways.
///
/// > ExprPtr e = 10L * "c1"_f > call("func", 3.4, "g"_f / "h"_f, call("j"));
///
/// is the same as "10 * c1 > func(3.4, g / h, j())".

using core::CallExpr;
using core::ConstantExpr;
using core::ExprPtr;
using core::FieldAccessExpr;
using core::IExpr;

/// Wrapper library used so we can safely overload operators.
struct ExprWrapper {
  std::shared_ptr<IExpr> ptr;

  // Enable proper comparisons.
  bool operator==(const ExprWrapper& other) const {
    return *ptr == *other.ptr;
  }

  bool operator==(const ExprPtr& other) const {
    return *ptr == *other;
  }

  // Provide better gtest failure messages.
  friend std::ostream& operator<<(std::ostream& os, const ExprWrapper& expr) {
    return os << expr.ptr->toString();
  }

  // For convenience, enable implicit conversions to ExprPtr.
  operator ExprPtr() const {
    return ptr;
  }
};

/// Field access (references to columns).
inline ExprWrapper field(const std::string& name) {
  return {std::make_shared<FieldAccessExpr>(name, std::nullopt)};
}

inline ExprWrapper operator"" _f(const char* str, size_t len) {
  return field(std::string(str, len));
}

/// Literals.
inline ExprWrapper literal(int64_t value) {
  return {std::make_shared<ConstantExpr>(BIGINT(), value, std::nullopt)};
}

inline ExprWrapper literal(int32_t value) {
  return {std::make_shared<ConstantExpr>(INTEGER(), value, std::nullopt)};
}

inline ExprWrapper literal(int16_t value) {
  return {std::make_shared<ConstantExpr>(SMALLINT(), value, std::nullopt)};
}

inline ExprWrapper literal(int8_t value) {
  return {std::make_shared<ConstantExpr>(TINYINT(), value, std::nullopt)};
}

inline ExprWrapper literal(double value) {
  return {std::make_shared<ConstantExpr>(DOUBLE(), value, std::nullopt)};
}

inline ExprWrapper literal(float value) {
  return {std::make_shared<ConstantExpr>(REAL(), value, std::nullopt)};
}

inline ExprWrapper literal(const std::string& value) {
  return {std::make_shared<ConstantExpr>(VARCHAR(), value, std::nullopt)};
}

// Timestamp literal.
inline Timestamp parseTimestamp(const std::string& timestamp) {
  return fromTimestampString(
             StringView(timestamp), util::TimestampParseMode::kPrestoCast)
      .thenOrThrow(folly::identity, [&](const Status& status) {
        VELOX_USER_FAIL("{}", status.message());
      });
}

inline ExprWrapper literalTimestamp(const std::string& value) {
  return {std::make_shared<ConstantExpr>(
      TIMESTAMP(), variant::timestamp(parseTimestamp(value)), std::nullopt)};
}

// Unpack a parameter. Either builds a ConstantExpr (literal) based on a scalar
// value, or passes through an ExprWrapper already constructed.
template <typename T>
inline std::shared_ptr<IExpr> unpack(T value) {
  return literal(value).ptr;
}

template <>
inline std::shared_ptr<IExpr> unpack<ExprWrapper>(ExprWrapper expr) {
  return expr.ptr;
}

// Unpacks a list of variadic template parameters in a std::vector<ExprPtr>. The
// elements could be ExprWrapper or C++ literals, which will get converted to
// ConstantExpr.
//
// Base of recursion.
inline std::vector<ExprPtr> unpackList() {
  return {};
}

template <typename TFirst, typename... TArgs>
inline std::vector<ExprPtr> unpackList(TFirst first, TArgs&&... args) {
  std::vector<ExprPtr> head = {unpack(first)};
  auto tail = unpackList(std::forward<TArgs>(args)...);
  head.insert(head.end(), tail.begin(), tail.end());
  return head;
}

// Macro to reduce builerplate when overloading C++ operators. The template
// magic basically means that the overload is matched if either left or right
// operands are an ExprWrapper. This is so that both "c"_f + 10 and 10 + "c"_f
// are supported, for example.
//
// If either left or right side are ExprWrapper, we either convert the other
// side as a constant/literal, or use it as-is if it is already an ExprWrapper.
#define VELOX_EXPR_BUILDER_OPERATOR(__op, __name)                         \
  template <typename T1, typename T2>                                     \
  inline std::enable_if_t<                                                \
      std::is_same_v<T1, ExprWrapper> || std::is_same_v<T2, ExprWrapper>, \
      ExprWrapper>                                                        \
  __op(T1 lhs, T2 rhs) {                                                  \
    return {std::make_shared<CallExpr>(                                   \
        __name,                                                           \
        std::vector<ExprPtr>{unpack(lhs), unpack(rhs)},                   \
        std::nullopt)};                                                   \
  }

// Define C++ operator overload for comparisons (filters).
VELOX_EXPR_BUILDER_OPERATOR(operator==, "eq");
VELOX_EXPR_BUILDER_OPERATOR(operator!=, "neq");
VELOX_EXPR_BUILDER_OPERATOR(operator<, "lt");
VELOX_EXPR_BUILDER_OPERATOR(operator<=, "lte");
VELOX_EXPR_BUILDER_OPERATOR(operator>, "gt");
VELOX_EXPR_BUILDER_OPERATOR(operator>=, "gte");

// Define C++ operator overload for arithmetics.
VELOX_EXPR_BUILDER_OPERATOR(operator+, "plus");
VELOX_EXPR_BUILDER_OPERATOR(operator-, "minus");
VELOX_EXPR_BUILDER_OPERATOR(operator*, "multiply");
VELOX_EXPR_BUILDER_OPERATOR(operator/, "divide");
VELOX_EXPR_BUILDER_OPERATOR(operator%, "mod");

VELOX_EXPR_BUILDER_OPERATOR(operator&&, "and");
VELOX_EXPR_BUILDER_OPERATOR(operator||, "or");

// "not" is an unary operator.
inline ExprWrapper operator!(ExprWrapper expr) {
  return {std::make_shared<CallExpr>(
      "not", std::vector<ExprPtr>{expr.ptr}, std::nullopt)};
}

// "is_null" is also unary.
template <typename T>
inline ExprWrapper isNull(const T& value) {
  return {std::make_shared<CallExpr>(
      "is_null", std::vector<ExprPtr>{unpack(value)}, std::nullopt)};
}

/// Regular function calls. First parameter is the function name, followed by
/// their parameters. Parameters can be other expression nodes or literals.
template <typename... TArgs>
inline ExprWrapper call(std::string name, TArgs&&... args) {
  return {std::make_shared<CallExpr>(
      std::move(name), unpackList(std::forward<TArgs>(args)...), std::nullopt)};
}

// "between" expression.
template <typename T1, typename T2>
inline ExprWrapper
between(ExprWrapper lhs, const T1& value1, const T2& value2) {
  return {std::make_shared<CallExpr>(
      "between",
      std::vector<ExprPtr>{lhs.ptr, unpack(value1), unpack(value2)},
      std::nullopt)};
}

// Add alias to an expression node (or constructs a new literal/constant with
// the alias). Accepts either:
//
// > alias(literal(1), "foo");
// > alias(1, "foo");
//
// or any other forms:
//
// > alias(field("foo"), "bar");
template <typename T>
inline ExprWrapper alias(T value, const std::string& alias) {
  auto expr = unpack(value);
  expr->alias() = alias;
  return {expr};
}

} // namespace facebook::velox::expr::builder
