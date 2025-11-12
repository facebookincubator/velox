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

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/fuzzer/SparkSql.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

using namespace facebook::velox::functions::sparksql::fuzzer;

TEST(SparkSqlTest, toTypeSql) {
  EXPECT_EQ(toTypeSql(BOOLEAN()), "BOOLEAN");
  EXPECT_EQ(toTypeSql(TINYINT()), "TINYINT");
  EXPECT_EQ(toTypeSql(SMALLINT()), "SMALLINT");
  EXPECT_EQ(toTypeSql(INTEGER()), "INT");
  EXPECT_EQ(toTypeSql(BIGINT()), "BIGINT");
  EXPECT_EQ(toTypeSql(REAL()), "FLOAT");
  EXPECT_EQ(toTypeSql(DOUBLE()), "DOUBLE");
  EXPECT_EQ(toTypeSql(VARCHAR()), "STRING");
  EXPECT_EQ(toTypeSql(VARBINARY()), "BINARY");
  EXPECT_EQ(toTypeSql(TIMESTAMP()), "TIMESTAMP");
  EXPECT_EQ(toTypeSql(DATE()), "DATE");
  EXPECT_EQ(toTypeSql(ARRAY(BOOLEAN())), "ARRAY<BOOLEAN>");
  EXPECT_EQ(toTypeSql(MAP(BOOLEAN(), INTEGER())), "MAP<BOOLEAN, INT>");
  EXPECT_EQ(
      toTypeSql(ROW({{"a", BOOLEAN()}, {"b", INTEGER()}})),
      "STRUCT<a:BOOLEAN, b:INT>");
  EXPECT_EQ(
      toTypeSql(
          ROW({{"a_", BOOLEAN()}, {"b$", INTEGER()}, {"c d", INTEGER()}})),
      "STRUCT<a_:BOOLEAN, b$:INT, c d:INT>");
  VELOX_ASSERT_THROW(toTypeSql(HUGEINT()), "Type is not supported: HUGEINT");
  VELOX_ASSERT_THROW(toTypeSql(UNKNOWN()), "Type is not supported: UNKNOWN");
  VELOX_ASSERT_THROW(
      toTypeSql(FUNCTION({INTEGER()}, INTEGER())),
      "Type is not supported: FUNCTION");
}

void toUnaryOperator(
    const std::string& operatorName,
    const std::string& expectedSql) {
  auto expression = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0")},
      operatorName);
  EXPECT_EQ(toCallSql(expression), expectedSql);
}

void toBinaryOperator(
    const std::string& operatorName,
    const std::string& expectedSql) {
  auto expression = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0"),
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1")},
      operatorName);
  EXPECT_EQ(toCallSql(expression), expectedSql);
}

void toIsNullOrIsNotNull(
    const std::string& operatorName,
    const std::string& expectedSql) {
  auto expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0")},
      operatorName);
  EXPECT_EQ(toCallSql(expression), expectedSql);
}

TEST(SparkSqlTest, toCallSql) {
  // Unary operators
  toUnaryOperator("unaryminus", "(- c0)");
  toUnaryOperator("not", "(not c0)");
  VELOX_ASSERT_THROW(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              INTEGER(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
                  std::make_shared<core::FieldAccessTypedExpr>(
                      VARCHAR(), "c1")},
              "not")),
      "Expected one argument to a unary operator");

  // Binary operators
  toBinaryOperator("add", "(c0 + c1)");
  toBinaryOperator("subtract", "(c0 - c1)");
  toBinaryOperator("multiply", "(c0 * c1)");
  toBinaryOperator("divide", "(c0 / c1)");
  toBinaryOperator("equalto", "(c0 = c1)");
  toBinaryOperator("lessthan", "(c0 < c1)");
  toBinaryOperator("greaterthan", "(c0 > c1)");
  toBinaryOperator("lessthanorequal", "(c0 <= c1)");
  toBinaryOperator("greaterthanorequal", "(c0 >= c1)");
  VELOX_ASSERT_THROW(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              INTEGER(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0"),
                  std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1"),
                  std::make_shared<core::FieldAccessTypedExpr>(
                      INTEGER(), "c2")},
              "add")),
      "Expected two arguments to a binary operator");

  // Functions IS NULL and NOT NULL
  toIsNullOrIsNotNull("isnull", "(c0 is null)");
  toIsNullOrIsNotNull("isnotnull", "(c0 is not null)");
  VELOX_ASSERT_THROW(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0"),
                  std::make_shared<core::FieldAccessTypedExpr>(
                      INTEGER(), "c1")},
              "isnull")),
      "Expected one argument to function 'isnull' or 'isnotnull'");

  // Function IN
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "b")},
              "in")),
      "'a' in ('b')");
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "b"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "c"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "d")},
              "in")),
      "'a' in ('b', 'c', 'd')");
  VELOX_ASSERT_THROW(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a")},
              "in")),
      "Expected at least two arguments to function 'in'");

  // Function LIKE
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a")},
              "like")),
      "(c0 like 'a')");
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "b")},
              "like")),
      "(c0 like 'a' escape 'b')");
  VELOX_ASSERT_THROW(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a")},
              "like")),
      "Expected at least two arguments to function 'like'");
  VELOX_ASSERT_THROW(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "b"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "c"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "d")},
              "like")),
      "Expected at most three arguments to function 'like'");

  // Functions OR and AND
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true),
                  std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), false)},
              "or")),
      "(BOOLEAN('true') or BOOLEAN('false'))");
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true),
                  std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), false)},
              "and")),
      "(BOOLEAN('true') and BOOLEAN('false'))");
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true),
                  std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), false),
                  std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true),
                  std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), false)},
              "or")),
      "(BOOLEAN('true') or BOOLEAN('false') or BOOLEAN('true') or BOOLEAN('false'))");
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true),
                  std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), false),
                  std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true),
                  std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), false)},
              "and")),
      "(BOOLEAN('true') and BOOLEAN('false') and BOOLEAN('true') and BOOLEAN('false'))");
  VELOX_ASSERT_THROW(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(), std::vector<core::TypedExprPtr>{}, "or")),
      "Expected at least two arguments to function 'or' or 'and'");

  // Functions ARRAY_CONSTRUCTOR and ROW_CONSTRUCTOR
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              ARRAY(INTEGER()),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "b"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "c")},
              "array")),
      "ARRAY('a', 'b', 'c')");
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              ARRAY(INTEGER()), std::vector<core::TypedExprPtr>{}, "array")),
      "ARRAY()");

  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "b"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "c"),
                  std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "d")},
              "row_constructor")),
      "STRUCT('a', 'b', 'c', 'd')");
  VELOX_ASSERT_THROW(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(), std::vector<core::TypedExprPtr>{}, "row_constructor")),
      "Expected at least one argument to function 'row_constructor'");

  // Function BETWEEN
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0"),
                  std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1"),
                  std::make_shared<core::FieldAccessTypedExpr>(
                      INTEGER(), "c2")},
              "between")),
      "(c0 between c1 and c2)");
  // Edge case check for ambiguous parantheses processing, query will fail
  // without the parantheses wrapping the left-hand side.
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::CallTypedExpr>(
                      BOOLEAN(),
                      std::vector<core::TypedExprPtr>{
                          std::make_shared<core::FieldAccessTypedExpr>(
                              INTEGER(), "c0"),
                          std::make_shared<core::FieldAccessTypedExpr>(
                              INTEGER(), "c0"),
                          std::make_shared<core::ConstantTypedExpr>(
                              INTEGER(), variant::null(TypeKind::INTEGER))},
                      "between"),
                  std::make_shared<core::FieldAccessTypedExpr>(
                      INTEGER(), "c0")},
              "lessthan")),
      "((c0 between c0 and cast(null as INT)) < c0)");
  VELOX_ASSERT_THROW(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(), std::vector<core::TypedExprPtr>{}, "between")),
      "Expected three arguments to function 'between'");

  // Function SWITCH, builds 'CASE WHEN ... THEN ... ELSE ... END' SQL
  // SWITCH cases with no ELSE.
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              INTEGER(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c0"),
                  std::make_shared<core::FieldAccessTypedExpr>(
                      VARCHAR(), "c1")},
              "switch")),
      "case when c0 then c1 end");
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              INTEGER(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c0"),
                  std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1"),
                  std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c2"),
                  std::make_shared<core::FieldAccessTypedExpr>(
                      INTEGER(), "c3")},
              "switch")),
      "case when c0 then c1 when c2 then c3 end");
  // SWITCH case with ELSE.
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              INTEGER(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c0"),
                  std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1"),
                  std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c2"),
                  std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c3"),
                  std::make_shared<core::FieldAccessTypedExpr>(
                      INTEGER(), "c4")},
              "switch")),
      "case when c0 then c1 when c2 then c3 else c4 end");
  VELOX_ASSERT_THROW(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              INTEGER(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::FieldAccessTypedExpr>(
                      INTEGER(), "c0")},
              "switch")),
      "Expected at least two arguments to function 'switch'");

  // Generic functions
  EXPECT_EQ(
      toCallSql(
          std::make_shared<core::CallTypedExpr>(
              REAL(),
              std::vector<core::TypedExprPtr>{},
              "function_with_no_args")),
      "function_with_no_args()");
}

TEST(SparkSqlTest, toConcatSql) {
  auto expression = core::ConcatTypedExpr(
      {"field0", "field1"},
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "b")});

  EXPECT_EQ(
      toConcatSql(expression),
      "cast(struct('a', 'b') as STRUCT<field0:STRING, field1:STRING>)");
}

TEST(SparkSqlTest, toCallInputsSql) {
  std::stringstream sql;
  auto expression = std::make_shared<core::FieldAccessTypedExpr>(
      VARCHAR(),
      std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
      "field0");

  toCallInputsSql({expression}, sql);
  EXPECT_EQ(sql.str(), "c0.field0");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
