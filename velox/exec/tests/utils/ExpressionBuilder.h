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

namespace facebook::velox::expr_builder {

/// Fluent Expression Builder.
///
/// This file contains fluent methods that make it convenient to create
/// (untyped) expression trees. This provides similar functionality to a SQL
/// parser, without bringing dependency on external libraries or bringing leaked
/// semantics from other systems.
///
/// The untyped expressions can then be turned into typed expressions ready for
/// execution using type binding from `core::Expressions::inferTypes()`.
///
/// The API provided is as close to the actual expression trees as possible.
/// Comparisons, arithmetics, conjuncts, function calls, literals, aliases, and
/// more are supported with this API.
///
/// For example, to create a column reference, you can:
///
/// > using namespace expr_builder;
/// > core::ExprPtr e = col("c0");
///
/// You can also use the "_c" C++ literal provided:
///
/// > core::ExprPtr e = "c0"_c;
///
/// Expressions created using ExpressionBuilder functions can be used in any
/// places that accept a ExprPtr. In practice, they create a ExprWrapper object,
/// but ExprWrappers are implicitly convertible to ExprSet.
///
/// Nested column references can be specified as either:
///
/// > col("parent", "child");
/// > col("parent").subfield("child");
///
/// To debug the expression generated, you can simply:
///
/// > LOG(INFO) << *e;
///
/// Comparisons and other expressions can be fluently created using C++
/// overloaded operators:
///
/// > col("c") > 10; // "c > 10"
/// > col("c") != "bar"; // "c != 'bar'"
/// > col("c") == nullptr; // "c = null"
///
/// C++ literals are automatically converted into ConstantExpr (expression
/// literals) when part of an expression. To explicitly create a literal you can
/// use:
///
/// > lit(10.3);
/// > lit("str");
///
/// Casts can be done using one of the two formats:
///
/// > lit(3).cast(TINYINT());
/// > cast("str", VARBINARY());
///
/// Null checking filters:
///
/// > isNull(col("c")) // "c is null"
/// > !isNull(col("c")) // "c is not null"
///
/// Conjuncts and "between":
///
/// > (col("a") && col("b")) || col("c"); // "(a AND b) OR c"
/// > between(col("a"), 0, 10); // "a between 0 and 10"
///
/// You can also use fluent version of these APIs:
///
/// > col("a").between(0, 10); // "a between 0 and 10"
///
/// Arithmetic operators are also overloaded:
///
/// > col("c") * 100 + col("b"); // "c * 100 + b"
///
/// In any expression, as long as one of the sides is an expression node, the
/// correct expression will be created. For example, both version work as
/// expected:
///
/// > col("c") * 100; // "c * 100"
/// > 100 * col("c"); // "100 * c"
///
/// When building long expressions, be careful about C++ constant folding and
/// operator precedence:
///
/// > col("c") + 5 * 100;
///
/// C++ will fold "5 * 100" and generate the expression "c + 500". To force the
/// expected behavior, you can explicitly spell out the literal:
///
/// > col("c") + 5 * lit(100);
/// > col("c") + lit(5) * 100;
///
/// Both will generate "col + 5 * 100", which is "plus(col, multiply(5, 100))".
///
/// Generic function calls can be created using `call()`:
///
/// > call("func", 10); // "func(10)"
///
/// `call()` supports arbitrary parameters, which can be other expressions or
/// (C++) literals.
///
/// Lambdas can be created using the following syntax:
///
/// > lambda({"x", "y"}, col("x") * col("y") + 1)
///
/// Where the first parameter is a vector of the lambda arguments, and the
/// second the lambda body.
///
/// All functions above can be nested and combined in arbitrary ways.
///
/// > 10L * col("c1") > call("func", 3.4, col("g") / col("h"), call("j"));
///
/// is the same as "10 * c1 > func(3.4, g / h, j())".
///
/// Comparisons, arithmetics, and other operators are mapped to function names
/// according to the table below. It is the user's responsibility to make sure
/// that there names map to their appropriate implementation:
///
///  -------------------------------
///  | C++        |  Function Name |
///  -------------------------------
///  | operator== |  eq            |
///  | operator!= |  neq           |
///  | operator<  |  lt            |
///  | operator<= |  lte           |
///  | operator>  |  gt            |
///  | operator>= |  gte           |
///  | operator!  |  not           |
///  | operator&& |  and           |
///  | operator|| |  or            |
///  | operator+  |  plus          |
///  | operator-  |  minus         |
///  | operator*  |  multiply      |
///  | operator/  |  divide        |
///  | operator%  |  mode          |
///  | operator== |  eq            |
///  -------------------------------

namespace detail {

class ExprWrapper;

/// Either builds a ConstantExpr (literal) based on a scalar value, or passes
/// through an ExprWrapper already constructed.
template <typename T>
inline ExprWrapper toExprWrapper(T value);

// Specialization for long to avoid ambiguity.
inline ExprWrapper toExprWrapper(long value);

template <>
inline ExprWrapper toExprWrapper<ExprWrapper>(ExprWrapper expr);

/// Wrapper library used so we can safely overload operators.
class ExprWrapper {
 public:
  ExprWrapper(const core::ExprPtr& expr) : expr_(expr) {}

  std::string toString() const {
    return expr_->toString();
  }

  core::ExprPtr expr() const {
    return expr_;
  }

  /// Add an alias to the current expression:
  ///
  /// > col("c0").alias("my_column");
  ExprWrapper& alias(const std::string& newAlias) {
    expr_ = expr_->withAlias(newAlias);
    return *this;
  }

  /// Add a "subfield" expression to enable access of subfields in
  /// rows/structs:
  ///
  /// > col("parent_col").subfield("child_name");
  ExprWrapper& subfield(std::string childName) {
    expr_ = std::make_shared<core::FieldAccessExpr>(
        std::move(childName), std::nullopt, std::vector<core::ExprPtr>{expr_});
    return *this;
  }

  /// Add a "cast" to the current expression:
  ///
  /// > col("c0").cast(VARBINARY());
  /// > lit(10).cast(TINYINT());
  ExprWrapper& cast(const TypePtr& castType) {
    expr_ =
        std::make_shared<core::CastExpr>(castType, expr_, false, std::nullopt);
    return *this;
  }

  /// Add a "try_cast" to the current expression:
  ///
  /// > col("c0").tryCast(VARBINARY());
  /// > lit(10).tryCast(TINYINT());
  ExprWrapper& tryCast(const TypePtr& castType) {
    expr_ =
        std::make_shared<core::CastExpr>(castType, expr_, true, std::nullopt);
    return *this;
  }

  /// Add a "is_null" to the current expression:
  ///
  /// > col("c0").isNull();
  ExprWrapper& isNull() {
    expr_ = std::make_shared<core::CallExpr>(
        "is_null", std::vector<core::ExprPtr>{expr_}, std::nullopt);
    return *this;
  }

  /// Add a "between" clause to the current expression wrapper:
  ///
  /// > col("a").between(1, 10);
  template <typename T1, typename T2>
  ExprWrapper& between(const T1& value1, const T2& value2) {
    expr_ = std::make_shared<core::CallExpr>(
        "between",
        std::vector<core::ExprPtr>{
            expr_,
            detail::toExprWrapper(value1),
            detail::toExprWrapper(value2)},
        std::nullopt);
    return *this;
  }

  /// If equality is used against an actual ExpPtr (not the wrapper), this will
  /// compare the expressions themselves.
  ///
  /// It won't assume this is generating a eq() Velox expression.
  bool operator==(const core::ExprPtr& other) const {
    return *expr_ == *other;
  }

  /// Provide better gtest failure messages.
  friend std::ostream& operator<<(std::ostream& os, const ExprWrapper& expr) {
    return os << expr.expr_->toString();
  }

  /// For convenience, enable implicit conversions to ExprPtr.
  operator core::ExprPtr() const {
    return expr_;
  }

 private:
  core::ExprPtr expr_;
};

/// Unpacks a list of variadic template parameters in a
/// std::vector<core::ExprPtr>. The elements could be ExprWrapper or C++
/// literals, which will get converted to ConstantExpr.
///
/// Base of recursion.
inline std::vector<core::ExprPtr> unpackList() {
  return {};
}

template <typename TFirst, typename... TArgs>
inline std::vector<core::ExprPtr> unpackList(TFirst first, TArgs&&... args) {
  std::vector<core::ExprPtr> head = {toExprWrapper(first)};
  auto tail = unpackList(std::forward<TArgs>(args)...);
  head.insert(head.end(), tail.begin(), tail.end());
  return head;
}

} // namespace detail

/// Column references.
inline detail::ExprWrapper col(std::string name) {
  return {std::make_shared<const core::FieldAccessExpr>(
      std::move(name), std::nullopt)};
}

/// Enable users to use a custom C++ literal to add a column reference.
/// For example: "col"_c
inline detail::ExprWrapper operator"" _c(const char* str, size_t len) {
  return col(std::string(str, len));
}

/// Nested column names. Ror rows/struct member references.
inline detail::ExprWrapper col(std::string parentName, std::string childName) {
  return col(std::move(parentName)).subfield(std::move(childName));
}

/// Literals.
inline detail::ExprWrapper lit(int64_t value) {
  return {std::make_shared<core::ConstantExpr>(BIGINT(), value, std::nullopt)};
}

inline detail::ExprWrapper lit(int32_t value) {
  return {std::make_shared<core::ConstantExpr>(INTEGER(), value, std::nullopt)};
}

inline detail::ExprWrapper lit(int16_t value) {
  return {
      std::make_shared<core::ConstantExpr>(SMALLINT(), value, std::nullopt)};
}

inline detail::ExprWrapper lit(int8_t value) {
  return {std::make_shared<core::ConstantExpr>(TINYINT(), value, std::nullopt)};
}

inline detail::ExprWrapper lit(bool value) {
  return {std::make_shared<core::ConstantExpr>(BOOLEAN(), value, std::nullopt)};
}

inline detail::ExprWrapper lit(double value) {
  return {std::make_shared<core::ConstantExpr>(DOUBLE(), value, std::nullopt)};
}

inline detail::ExprWrapper lit(float value) {
  return {std::make_shared<core::ConstantExpr>(REAL(), value, std::nullopt)};
}

/// Different string flavors.
inline detail::ExprWrapper lit(const char* value) {
  return {std::make_shared<core::ConstantExpr>(VARCHAR(), value, std::nullopt)};
}

inline detail::ExprWrapper lit(const std::string_view& value) {
  return {std::make_shared<core::ConstantExpr>(
      VARCHAR(), std::string(value), std::nullopt)};
}

inline detail::ExprWrapper lit(const std::string& value) {
  return {std::make_shared<core::ConstantExpr>(VARCHAR(), value, std::nullopt)};
}

/// lit(nullptr).
inline detail::ExprWrapper lit(std::nullptr_t) {
  return {std::make_shared<core::ConstantExpr>(
      UNKNOWN(), variant::null(TypeKind::UNKNOWN), std::nullopt)};
}

/// Macro to reduce builerplate when overloading C++ operators. The template
/// magic basically means that the overload is matched if either left or right
/// operands are an ExprWrapper. This is so that both "c"_f + 10 and 10 + "c"_f
/// are supported, for example.
///
/// If either left or right side are ExprWrapper, we either convert the other
/// side as a constant/literal, or use it as-is if it is already an ExprWrapper.
#define VELOX_EXPR_BUILDER_OPERATOR(__op, __name)                    \
  template <typename T1, typename T2>                                \
  inline std::enable_if_t<                                           \
      std::is_same_v<T1, detail::ExprWrapper> ||                     \
          std::is_same_v<T2, detail::ExprWrapper>,                   \
      detail::ExprWrapper>                                           \
  __op(T1 lhs, T2 rhs) {                                             \
    return {std::make_shared<core::CallExpr>(                        \
        __name,                                                      \
        std::vector<core::ExprPtr>{                                  \
            detail::toExprWrapper(lhs), detail::toExprWrapper(rhs)}, \
        std::nullopt)};                                              \
  }

/// Define C++ operator overload for comparisons.
VELOX_EXPR_BUILDER_OPERATOR(operator==, "eq");
VELOX_EXPR_BUILDER_OPERATOR(operator!=, "neq");
VELOX_EXPR_BUILDER_OPERATOR(operator<, "lt");
VELOX_EXPR_BUILDER_OPERATOR(operator<=, "lte");
VELOX_EXPR_BUILDER_OPERATOR(operator>, "gt");
VELOX_EXPR_BUILDER_OPERATOR(operator>=, "gte");

/// Define C++ operator overload for arithmetics.
VELOX_EXPR_BUILDER_OPERATOR(operator+, "plus");
VELOX_EXPR_BUILDER_OPERATOR(operator-, "minus");
VELOX_EXPR_BUILDER_OPERATOR(operator*, "multiply");
VELOX_EXPR_BUILDER_OPERATOR(operator/, "divide");
VELOX_EXPR_BUILDER_OPERATOR(operator%, "mod");

VELOX_EXPR_BUILDER_OPERATOR(operator&&, "and");
VELOX_EXPR_BUILDER_OPERATOR(operator||, "or");

/// "not" is an unary operator.
inline detail::ExprWrapper operator!(detail::ExprWrapper expr) {
  return {std::make_shared<core::CallExpr>(
      "not", std::vector<core::ExprPtr>{expr.expr()}, std::nullopt)};
}

/// "is_null" is also unary.
template <typename T>
inline detail::ExprWrapper isNull(const T& expr) {
  return detail::toExprWrapper(expr).isNull();
}

/// "alias" as a free function.
template <typename TInput>
inline detail::ExprWrapper alias(TInput lhs, const std::string& newAlias) {
  return detail::toExprWrapper(lhs).alias(newAlias);
}

/// "cast" as a free function.
template <typename TInput>
inline detail::ExprWrapper cast(TInput lhs, const TypePtr& castType) {
  return detail::toExprWrapper(lhs).cast(castType);
}

/// "tryCast" as a free function.
template <typename TInput>
inline detail::ExprWrapper tryCast(TInput lhs, const TypePtr& castType) {
  return detail::toExprWrapper(lhs).tryCast(castType);
}

/// "between" as a free function.
template <typename T1, typename T2>
inline detail::ExprWrapper
between(detail::ExprWrapper lhs, const T1& value1, const T2& value2) {
  return lhs.between(value1, value2);
}

/// Creates a lambda expressions, given the function parameters and an
/// expression for the function body.
template <typename TInput>
inline detail::ExprWrapper lambda(
    std::initializer_list<std::string> args,
    const TInput& body) {
  return {std::make_shared<core::LambdaExpr>(
      std::move(args), detail::toExprWrapper(body))};
}

/// Convenience lambda builder for single argument lambdas.
template <typename TInput>
inline detail::ExprWrapper lambda(std::string arg, const TInput& body) {
  return lambda({std::move(arg)}, body);
}

/// Regular function calls. First parameter is the function name, followed by
/// their parameters. Parameters can be other expression nodes or literals.
template <typename... TArgs>
inline detail::ExprWrapper call(std::string name, TArgs&&... args) {
  return {std::make_shared<core::CallExpr>(
      std::move(name),
      detail::unpackList(std::forward<TArgs>(args)...),
      std::nullopt)};
}

namespace detail {

template <typename T>
inline ExprWrapper toExprWrapper(T value) {
  return lit(value);
}

inline ExprWrapper toExprWrapper(long value) {
  return lit(static_cast<int64_t>(value));
}

template <>
inline ExprWrapper toExprWrapper<ExprWrapper>(ExprWrapper expr) {
  return expr;
}

} // namespace detail

} // namespace facebook::velox::expr_builder
