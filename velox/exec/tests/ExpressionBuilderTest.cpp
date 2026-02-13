/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/exec/tests/utils/ExpressionBuilder.h"

#include <gtest/gtest.h>
#include "velox/parse/ExpressionsParser.h"
#include "velox/type/Variant.h"

namespace facebook::velox::expr_builder::test {
namespace {

// Test convenience functions for downcasting.
template <typename T>
std::shared_ptr<const T> as(detail::ExprWrapper in) {
  return std::dynamic_pointer_cast<const T>(in.expr());
}

template <typename T>
bool is(detail::ExprWrapper in) {
  return std::dynamic_pointer_cast<const T>(in.expr()) != nullptr;
}

// Parses a SQL expression using DuckDB.
core::ExprPtr parseSql(const std::string& sql) {
  return parse::DuckSqlExpressionsParser().parseExpr(sql);
}

TEST(ExpressionBuilderTest, columnReference) {
  EXPECT_EQ(col("c0"), parseSql("c0"));
  EXPECT_EQ(parseSql("c0"), col("c0"));
  EXPECT_EQ("c0"_c, parseSql("c0"));

  EXPECT_EQ(col("parent", "child"), parseSql("parent.child"));
  EXPECT_EQ(col("parent").subfield("child"), parseSql("parent.child"));
}

TEST(ExpressionBuilderTest, literals) {
  auto validate = [](detail::ExprWrapper expr,
                     const TypePtr& expectedType,
                     variant expectedValue) {
    EXPECT_TRUE(is<core::ConstantExpr>(expr));
    auto constant = as<core::ConstantExpr>(expr);
    EXPECT_EQ(*constant->type(), *expectedType);
    EXPECT_TRUE(constant->value().equalsWithEpsilon(expectedValue));
  };

  // Integer literal types.
  validate(lit(123456L), BIGINT(), variant(123456L));
  validate(lit(123), INTEGER(), variant(123));
  validate(lit(int16_t(123)), SMALLINT(), variant(int16_t(123)));
  validate(lit(int8_t(123)), TINYINT(), variant(int8_t(123)));

  // Boolean.
  validate(lit(true), BOOLEAN(), variant(true));
  validate(lit(false), BOOLEAN(), variant(false));

  // Floating point.
  validate(lit(10.1f), REAL(), variant(10.1f));
  validate(lit(10.1), DOUBLE(), variant(10.1));

  // String.
  validate(lit("str"), VARCHAR(), variant("str"));

  // Null.
  validate(lit(nullptr), UNKNOWN(), variant::null(TypeKind::UNKNOWN));
}

TEST(ExpressionBuilderTest, comparisons) {
  // Make sure all combinations work, as long as at least one side is a
  // ExprWrapper.
  EXPECT_EQ(col("a") == lit(10L), parseSql("a = 10"));
  EXPECT_EQ(lit(10L) == col("a"), parseSql("10 = a"));

  EXPECT_EQ(col("a") == 10L, parseSql("a = 10"));
  EXPECT_EQ(10L == col("a"), parseSql("10 = a"));

  EXPECT_EQ(col("a") == col("b"), parseSql("a = b"));
  EXPECT_EQ(col("a") == nullptr, parseSql("a = null"));

  // Other comparisons.
  EXPECT_EQ(col("a") != 1.1, parseSql("a != 1.1"));
  EXPECT_EQ(col("a") != lit(1.1), parseSql("a != 1.1"));
  EXPECT_EQ(col("a") > 42L, parseSql("a > 42"));
  EXPECT_EQ(col("a") >= 42L, parseSql("a >= 42"));
  EXPECT_EQ(col("a") < 42L, parseSql("a < 42"));
  EXPECT_EQ(col("a") <= 42L, parseSql("a <= 42"));

  EXPECT_EQ(!col("a"), parseSql("not a"));
  EXPECT_EQ(isNull(col("a")), parseSql("a is null"));
  EXPECT_EQ(col("a").isNull(), parseSql("a is null"));
  EXPECT_EQ(!isNull(col("a")), parseSql("a is not null"));
  EXPECT_EQ(!col("a").isNull(), parseSql("a is not null"));

  EXPECT_EQ(isNull("a"), parseSql("\'a\' is null")); // this is "a" literal.
}

TEST(ExpressionBuilderTest, between) {
  EXPECT_EQ(between(col("a"), 0L, 10L), parseSql("a between 0 and 10"));

  EXPECT_EQ(col("a").between(0L, 10L), parseSql("a between 0 and 10"));
}

TEST(ExpressionBuilderTest, arithmetics) {
  EXPECT_EQ(col("b") + 1L, parseSql("b + 1"));
  EXPECT_EQ(1L + col("b"), parseSql("1 + b"));
  EXPECT_EQ(lit("str") + col("b"), parseSql("'str' + b"));

  EXPECT_EQ(col("b") - 1L, parseSql("b - 1"));
  EXPECT_EQ(col("b") * 1L, parseSql("b * 1"));
  EXPECT_EQ(col("b") / 1L, parseSql("b / 1"));
  EXPECT_EQ(col("b") % 1L, parseSql("b % 1"));

  EXPECT_EQ(col("b") + 1L / col("c") * 10L, parseSql("b + 1 / c * 10"));
}

TEST(ExpressionBuilderTest, conjuncts) {
  EXPECT_EQ(col("b") && 1L, parseSql("b and 1"));
  EXPECT_EQ(col("b") || 1L, parseSql("b or 1"));
  EXPECT_EQ(col("b") || false, parseSql("b or false"));

  EXPECT_EQ(col("a") && col("b") || col("c"), parseSql("a and b or c"));
}

TEST(ExpressionBuilderTest, functions) {
  EXPECT_EQ(call("func"), parseSql("func()"));
  EXPECT_EQ(
      call("func", col("a"), 100L, col("c")), parseSql("func(a, 100, c)"));

  // Nested functions.
  auto expr = call("f1", call("f2", col("a") > call("f3", col("d"))));
  EXPECT_EQ(expr, parseSql("f1(f2(a > f3(d)))"));

  expr = 10L * col("c1") > call("func", 3.4, col("g") / col("h"), call("j"));
  EXPECT_EQ(expr, parseSql("10 * c1 > func(3.4, g / h, j())"));
}

TEST(ExpressionBuilderTest, casts) {
  // Casts.
  EXPECT_EQ(lit("1").cast(TINYINT()).toString(), "cast(1 as TINYINT)");
  EXPECT_EQ(
      col("c0").cast(VARBINARY()).toString(), "cast(\"c0\" as VARBINARY)");

  EXPECT_EQ(cast(1, TINYINT()).toString(), "cast(1 as TINYINT)");
  EXPECT_EQ(
      cast(col("c0"), VARBINARY()).toString(), "cast(\"c0\" as VARBINARY)");

  // Try casts.
  EXPECT_EQ(lit("1").tryCast(TINYINT()).toString(), "try_cast(1 as TINYINT)");
  EXPECT_EQ(
      col("c0").tryCast(VARBINARY()).toString(),
      "try_cast(\"c0\" as VARBINARY)");

  EXPECT_EQ(tryCast(1, TINYINT()).toString(), "try_cast(1 as TINYINT)");
  EXPECT_EQ(
      tryCast(col("c0"), VARBINARY()).toString(),
      "try_cast(\"c0\" as VARBINARY)");
}

TEST(ExpressionBuilderTest, alias) {
  EXPECT_EQ(lit("str").alias("col"), parseSql("'str' as col"));
  EXPECT_EQ(col("c1").alias("col"), parseSql("c1 as col"));
  EXPECT_EQ((col("c1") > 1.1).alias("col"), parseSql("c1 > 1.1 as col"));

  EXPECT_EQ(
      col("c1").between(1L, 10L).alias("my_col"),
      parseSql("c1 between 1 and 10 as my_col"));

  // As a free function.
  EXPECT_EQ(alias(col("c1") == "bla", "col"), parseSql("c1 = 'bla' as col"));
}

TEST(ExpressionBuilderTest, lambdas) {
  EXPECT_EQ(lambda("x", 1L), parseSql("x -> 1"));
  EXPECT_EQ(lambda({"x"}, 1L), parseSql("x -> 1"));
  EXPECT_EQ(lambda({"x"}, col("x") + 1L), parseSql("x -> x + 1"));
  EXPECT_EQ(
      lambda({"x", "y"}, col("x") * col("y")), parseSql("(x, y) -> x * y"));
}

} // namespace
} // namespace facebook::velox::expr_builder::test
