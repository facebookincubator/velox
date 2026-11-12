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
#include <folly/container/F14Map.h>
#include <gtest/gtest.h>
#include "velox/duckdb/conversion/DuckParser.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/IExpr.h"

namespace facebook::velox::core {
namespace {

class ExprTest : public testing::Test {};

ExprPtr parse(const std::string& sql) {
  return duckdb::parseExpr(sql, {});
}

TEST_F(ExprTest, placeholder) {
  const ExprPtr placeholder =
      std::make_shared<PlaceholderExpr>(BIGINT(), 1, std::nullopt);
  EXPECT_EQ(placeholder->kind(), IExpr::Kind::kPlaceholder);
  EXPECT_EQ(placeholder->toString(), "?1");

  const ExprPtr same =
      std::make_shared<PlaceholderExpr>(BIGINT(), 1, std::nullopt);
  EXPECT_EQ(*placeholder, *same);
  EXPECT_EQ(placeholder->hash(), same->hash());
  EXPECT_NE(
      *placeholder,
      *std::make_shared<PlaceholderExpr>(BIGINT(), 2, std::nullopt));
  EXPECT_NE(
      *placeholder,
      *std::make_shared<PlaceholderExpr>(VARCHAR(), 1, std::nullopt));

  const auto aliased = placeholder->withAlias("a");
  EXPECT_EQ(aliased->toString(), "?1 AS a");
  EXPECT_EQ(*aliased->dropAlias(), *placeholder);
}

TEST_F(ExprTest, hashMap) {
  folly::F14FastMap<ExprPtr, int, IExprHash, IExprEqual> exprs;

  {
    auto expr1 = parse("a + 1");
    auto expr2 = parse("a + b");
    auto expr3 = parse("12");
    auto expr4 = parse("a + b");

    exprs.emplace(expr1, 1);
    exprs.emplace(expr2, 2);
    exprs.emplace(expr3, 3);
    exprs.emplace(expr4, 4);

    ASSERT_EQ(3, exprs.size());

    EXPECT_EQ(exprs.at(expr1), 1);
    EXPECT_EQ(exprs.at(expr2), 2);
    EXPECT_EQ(exprs.at(expr3), 3);
    EXPECT_EQ(exprs.at(expr4), 2);
  }

  {
    auto expr1 = parse("cast(a as bigint) + 1");
    auto expr2 = parse("cast(a as int) + 1");
    auto expr3 = parse("cast(a as bigint) + 1");

    exprs.emplace(expr1, 5);
    exprs.emplace(expr2, 6);
    exprs.emplace(expr3, 7);

    ASSERT_EQ(5, exprs.size());

    EXPECT_EQ(exprs.at(expr1), 5);
    EXPECT_EQ(exprs.at(expr2), 6);
    EXPECT_EQ(exprs.at(expr3), 5);
  }
}

TEST_F(ExprTest, atTimeZone) {
  const auto value = parse("a");
  const auto timeZone = parse("'America/Los_Angeles'");
  const AtTimeZoneExpr::OperandTypes operandTypes{
      .timestamp = TIMESTAMP(),
      .timestampWithTimeZone = TIMESTAMP_WITH_TIME_ZONE(),
      .time = TIME(),
      .timeWithTimeZone = TIME_WITH_TIME_ZONE(),
  };
  const ExprPtr expr = std::make_shared<AtTimeZoneExpr>(
      value, timeZone, operandTypes, std::nullopt);
  const auto* atTimeZone = expr->as<AtTimeZoneExpr>();

  ASSERT_NE(atTimeZone, nullptr);
  EXPECT_EQ(expr->kind(), IExpr::Kind::kAtTimeZone);
  EXPECT_EQ(expr->toString(), R"($at_time_zone("a",America/Los_Angeles))");
  EXPECT_EQ(*atTimeZone->operandTypes().timestamp, *operandTypes.timestamp);
  EXPECT_EQ(
      *atTimeZone->operandTypes().timestampWithTimeZone,
      *operandTypes.timestampWithTimeZone);
  EXPECT_EQ(*atTimeZone->operandTypes().time, *operandTypes.time);
  EXPECT_EQ(
      *atTimeZone->operandTypes().timeWithTimeZone,
      *operandTypes.timeWithTimeZone);

  const ExprPtr same = std::make_shared<AtTimeZoneExpr>(
      value, timeZone, operandTypes, std::nullopt);
  EXPECT_EQ(*expr, *same);
  EXPECT_EQ(expr->hash(), same->hash());
  EXPECT_NE(
      *expr,
      *std::make_shared<AtTimeZoneExpr>(
          parse("b"), timeZone, operandTypes, std::nullopt));
  EXPECT_NE(
      *expr,
      *std::make_shared<AtTimeZoneExpr>(
          value, parse("'UTC'"), operandTypes, std::nullopt));
  for (const auto member : {
           &AtTimeZoneExpr::OperandTypes::timestamp,
           &AtTimeZoneExpr::OperandTypes::timestampWithTimeZone,
           &AtTimeZoneExpr::OperandTypes::time,
           &AtTimeZoneExpr::OperandTypes::timeWithTimeZone,
       }) {
    auto differentOperandTypes = operandTypes;
    differentOperandTypes.*member = VARCHAR();
    EXPECT_NE(
        *expr,
        *std::make_shared<AtTimeZoneExpr>(
            value, timeZone, std::move(differentOperandTypes), std::nullopt));
  }

  const auto aliased = expr->withAlias("zoned");
  EXPECT_EQ(
      aliased->toString(),
      R"($at_time_zone("a",America/Los_Angeles) AS zoned)");
  EXPECT_EQ(*aliased->dropAlias(), *expr);

  const auto replaced = expr->replaceInputs({parse("b"), parse("'UTC'")});
  EXPECT_EQ(
      *replaced,
      *std::make_shared<AtTimeZoneExpr>(
          parse("b"), parse("'UTC'"), operandTypes, std::nullopt));
}

TEST_F(ExprTest, tryAsRootColumn) {
  // Top-level field access on an input column.
  {
    auto expr = parse("a");
    const auto* field = FieldAccessExpr::tryAsRootColumn(expr);
    ASSERT_NE(field, nullptr);
    EXPECT_EQ(field->name(), "a");
  }

  // Dereference (a.b) is a FieldAccess whose input is not an InputExpr.
  {
    auto expr = parse("a.b");
    EXPECT_EQ(FieldAccessExpr::tryAsRootColumn(expr), nullptr);
  }

  // Non-FieldAccess expressions are not root columns.
  {
    EXPECT_EQ(FieldAccessExpr::tryAsRootColumn(parse("a + 1")), nullptr);
    EXPECT_EQ(FieldAccessExpr::tryAsRootColumn(parse("42")), nullptr);
    EXPECT_EQ(
        FieldAccessExpr::tryAsRootColumn(parse("cast(a as bigint)")), nullptr);
  }
}

TEST_F(ExprTest, dropAlias) {
  EXPECT_EQ(parse("a + b")->dropAlias()->toString(), R"(plus("a","b"))");
  EXPECT_EQ(parse("a + b as c")->dropAlias()->toString(), R"(plus("a","b"))");

  EXPECT_EQ(
      parse("cast(a as date)")->dropAlias()->toString(),
      R"(cast("a" as DATE))");
  EXPECT_EQ(
      parse("cast(a as date) as b")->dropAlias()->toString(),
      R"(cast("a" as DATE))");

  EXPECT_EQ(parse("a + 10")->dropAlias()->toString(), R"(plus("a",10))");
  EXPECT_EQ(parse("a + 10 as b")->dropAlias()->toString(), R"(plus("a",10))");
}

} // namespace
} // namespace facebook::velox::core
