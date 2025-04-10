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
#include "velox/exec/fuzzer/PrestoSql.h"
#include "velox/functions/prestosql/types/JsonType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

namespace facebook::velox::exec::test {
namespace {

TEST(PrestoSqlTest, toTypeSql) {
  EXPECT_EQ(toTypeSql(BOOLEAN()), "BOOLEAN");
  EXPECT_EQ(toTypeSql(TINYINT()), "TINYINT");
  EXPECT_EQ(toTypeSql(SMALLINT()), "SMALLINT");
  EXPECT_EQ(toTypeSql(INTEGER()), "INTEGER");
  EXPECT_EQ(toTypeSql(BIGINT()), "BIGINT");
  EXPECT_EQ(toTypeSql(REAL()), "REAL");
  EXPECT_EQ(toTypeSql(DOUBLE()), "DOUBLE");
  EXPECT_EQ(toTypeSql(VARCHAR()), "VARCHAR");
  EXPECT_EQ(toTypeSql(VARBINARY()), "VARBINARY");
  EXPECT_EQ(toTypeSql(TIMESTAMP()), "TIMESTAMP");
  EXPECT_EQ(toTypeSql(DATE()), "DATE");
  EXPECT_EQ(toTypeSql(TIMESTAMP_WITH_TIME_ZONE()), "TIMESTAMP WITH TIME ZONE");
  EXPECT_EQ(toTypeSql(ARRAY(BOOLEAN())), "ARRAY(BOOLEAN)");
  EXPECT_EQ(toTypeSql(MAP(BOOLEAN(), INTEGER())), "MAP(BOOLEAN, INTEGER)");
  EXPECT_EQ(
      toTypeSql(ROW({{"a", BOOLEAN()}, {"b", INTEGER()}})),
      "ROW(a BOOLEAN, b INTEGER)");
  EXPECT_EQ(
      toTypeSql(
          ROW({{"a_", BOOLEAN()}, {"b$", INTEGER()}, {"c d", INTEGER()}})),
      "ROW(a_ BOOLEAN, b$ INTEGER, c d INTEGER)");
  EXPECT_EQ(toTypeSql(JSON()), "JSON");
  EXPECT_EQ(toTypeSql(UNKNOWN()), "UNKNOWN");
  VELOX_ASSERT_THROW(
      toTypeSql(FUNCTION({INTEGER()}, INTEGER())),
      "Type is not supported: FUNCTION");
}

TEST(PrestoSqlTest, toCallSql) {
  // Unary operators
  auto expression = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0")},
      "negate");
  EXPECT_EQ(toCallSql(expression), "(- c0)");
  expression = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0")},
      "not");
  EXPECT_EQ(toCallSql(expression), "(not c0)");
  expression = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0")},
      "not");
  VELOX_ASSERT_THROW(toCallSql(expression), "(2 vs. 1)");

  // Binary operators
  expression = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0"),
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1")},
      "plus");
  EXPECT_EQ(toCallSql(expression), "(c0 + c1)");
  expression = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0"),
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1"),
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")},
      "plus");
  VELOX_ASSERT_THROW(toCallSql(expression), "(3 vs. 2)");

  // IS NULL or NOT NULL
  expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0")},
      "is_null");
  EXPECT_EQ(toCallSql(expression), "(c0 is null)");
  expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0")},
      "not_null");
  EXPECT_EQ(toCallSql(expression), "(c0 is not null)");
  expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0"),
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1")},
      "is_null");
  VELOX_ASSERT_THROW(toCallSql(expression), "(2 vs. 1)");

  // IN
  expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "b"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "c"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "d")},
      "in");
  EXPECT_EQ(toCallSql(expression), "'a' in ('b', 'c', 'd')");
  expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a")},
      "in");
  VELOX_ASSERT_THROW(toCallSql(expression), "(1 vs. 2)");

  // LIKE
  expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a")},
      "like");
  EXPECT_EQ(toCallSql(expression), "(c0 like 'a')");
  expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "b")},
      "like");
  EXPECT_EQ(toCallSql(expression), "(c0 like 'a' escape 'b')");

  // OR or AND
  expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c1")},
      "or");
  EXPECT_EQ(toCallSql(expression), "(c0 or c1)");
  expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c1")},
      "and");
  EXPECT_EQ(toCallSql(expression), "(c0 and c1)");

  // ARRAY_CONSTRUCTOR
  expression = std::make_shared<core::CallTypedExpr>(
      ARRAY(INTEGER()),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "b"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "c")},
      "array_constructor");
  EXPECT_EQ(toCallSql(expression), "ARRAY['a', 'b', 'c']");

  // BETWEEN
  expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0"),
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1"),
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")},
      "between");
  EXPECT_EQ(toCallSql(expression), "c0 between c1 and c2");

  // ROW_CONSTRUCTOR
  expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "a"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "b"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "c"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), "d")},
      "row_constructor");
  EXPECT_EQ(toCallSql(expression), "row('a', 'b', 'c', 'd')");

  // Builds subscript SQL expression
  expression = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(
              ARRAY(INTEGER()), "array"),
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0")},
      "subscript");
  EXPECT_EQ(toCallSql(expression), "array[c0]");
  expression = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(
              ARRAY(INTEGER()), "array"),
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c0"),
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1")},
      "subscript");
  VELOX_ASSERT_THROW(toCallSql(expression), "(3 vs. 2)");

  // Generic functions
  expression = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(ARRAY(INTEGER()), "c0"),
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1")},
      "array_top_n");
  EXPECT_EQ(toCallSql(expression), "array_top_n(c0, c1)");
}

} // namespace
} // namespace facebook::velox::exec::test
