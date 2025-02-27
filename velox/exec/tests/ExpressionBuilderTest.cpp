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

#include "velox/exec/tests/utils/ExpressionBuilder.h"

#include <gtest/gtest.h>
#include "velox/parse/ExpressionsParser.h"
#include "velox/type/Variant.h"

namespace facebook::velox::expr::builder::test {
namespace {

// Test convenience functions for downcasting.
template <typename T>
std::shared_ptr<const T> as(ExprWrapper in) {
  return std::dynamic_pointer_cast<const T>(in.ptr);
}

template <typename T>
bool is(ExprWrapper in) {
  return std::dynamic_pointer_cast<const T>(in.ptr) != nullptr;
}

// Parses a SQL expression using DuckDB.
auto parseSql(const std::string& sql) {
  return parse::parseExpr(sql, {});
}

TEST(ExpressionBuilderTest, fieldAccess) {
  ExprWrapper result;

  result = field("col");
  EXPECT_EQ(result, parseSql("col"));

  result = "col"_f;
  EXPECT_EQ(result, parseSql("col"));
}

TEST(ExpressionBuilderTest, literals) {
  auto validate = [](ExprWrapper expr, const TypePtr& type, variant value) {
    EXPECT_TRUE(is<ConstantExpr>(expr));
    auto constant = as<ConstantExpr>(expr);
    EXPECT_EQ(*constant->type(), *type);
    EXPECT_TRUE(constant->value().equalsWithEpsilon(value));
  };

  // Integer literal types.
  validate(literal(123456L), BIGINT(), variant(123456L));
  validate(literal(123), INTEGER(), variant(123));
  validate(literal(int16_t(123)), SMALLINT(), variant(int16_t(123)));
  validate(literal(int8_t(123)), TINYINT(), variant(int8_t(123)));

  validate(literal(10.1f), REAL(), variant(10.1f));
  validate(literal(10.1), DOUBLE(), variant(10.1));

  validate(literal("str"), VARCHAR(), variant("str"));
}

TEST(ExpressionBuilderTest, filters) {
  ExprWrapper result;

  result = "a"_f == 10L;
  EXPECT_EQ(result, parseSql("a = 10"));

  result = "a"_f != 10L;
  EXPECT_EQ(result, parseSql("a != 10"));

  result = "a"_f < 10L;
  EXPECT_EQ(result, parseSql("a < 10"));

  result = "a"_f <= 10L;
  EXPECT_EQ(result, parseSql("a <= 10"));

  result = "a"_f > 10L;
  EXPECT_EQ(result, parseSql("a > 10"));

  result = "a"_f >= 10L;
  EXPECT_EQ(result, parseSql("a >= 10"));

  result = isNull("a"_f);
  EXPECT_EQ(result, parseSql("a is null"));

  result = isNull("a"); // this is "a" literal.
  EXPECT_EQ(result, parseSql("\'a\' is null"));

  result = !isNull("a"_f);
  EXPECT_EQ(result, parseSql("a is not null"));

  result = between("a"_f, 0L, 10L);
  EXPECT_EQ(result, parseSql("a between 0 and 10"));

  // Reverse order. As long as one side of operators are ExprWrapper, it
  // generates expressions as expected.
  result = 10L < "a"_f;
  EXPECT_EQ(result, parseSql("10 < a"));
}

TEST(ExpressionBuilderTest, arithmetics) {
  ExprWrapper result;

  result = "b"_f + 1L;
  EXPECT_EQ(result, parseSql("b + 1"));

  result = "b"_f - 1L;
  EXPECT_EQ(result, parseSql("b - 1"));

  result = "b"_f * 1L;
  EXPECT_EQ(result, parseSql("b * 1"));

  result = "b"_f / 1L;
  EXPECT_EQ(result, parseSql("b / 1"));

  result = "b"_f % 1L;
  EXPECT_EQ(result, parseSql("b % 1"));

  result = "b"_f + 1L - 10L;
  EXPECT_EQ(result, parseSql("b + 1 - 10"));

  result = "b"_f * 1L / 10L;
  EXPECT_EQ(result, parseSql("b * 1 / 10"));
}

TEST(ExpressionBuilderTest, conjuncts) {
  ExprWrapper result;

  result = "b"_f && 1L;
  EXPECT_EQ(result, parseSql("b AND 1"));

  result = "b"_f || 1L;
  EXPECT_EQ(result, parseSql("b OR 1"));

  result = "a"_f && "b"_f || "c"_f;
  EXPECT_EQ(result, parseSql("a AND b OR c"));
}

TEST(ExpressionBuilderTest, functions) {
  ExprWrapper result;

  result = call("func");
  EXPECT_EQ(result, parseSql("func()"));

  result = call("func", field("a"));
  EXPECT_EQ(result, parseSql("func(a)"));

  result = call("func", field("a"), field("b"), field("c"));
  EXPECT_EQ(result, parseSql("func(a, b, c)"));

  result = call("func", literal(10L), literal(10.23), field("c"));
  EXPECT_EQ(result, parseSql("func(10, 10.23, c)"));

  // Nested functions.
  result = call("f1", call("f2", "a"_f > call("f3", "d"_f)));
  EXPECT_EQ(result, parseSql("f1(f2(a > f3(d)))"));
}

TEST(ExpressionBuilderTest, alias) {
  ExprWrapper result;

  result = alias(field("col"), "foo");
  EXPECT_EQ(result, parseSql("col as foo"));

  result = alias("col"_f, "foo");
  EXPECT_EQ(result, parseSql("col as foo"));

  result = alias(literal(1L), "bar");
  EXPECT_EQ(result, parseSql("1 as bar"));

  result = alias(1L, "bar");
  EXPECT_EQ(result, parseSql("1 as bar"));
}

TEST(ExpressionBuilderTest, combined) {
  ExprWrapper result;

  result = 10L * "c1"_f > call("func", 3.4, "g"_f / "h"_f, call("j"));
  EXPECT_EQ(result, parseSql("10 * c1 > func(3.4, g / h, j())"));
}

} // namespace
} // namespace facebook::velox::expr::builder::test
