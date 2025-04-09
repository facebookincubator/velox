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
#include "velox/expression/ExpressionOptimizer.h"
#include "velox/expression/ExprCompiler.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

namespace facebook::velox::expression::test {

class ExpressionOptimizerTest : public testing::Test,
                                public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    functions::prestosql::registerAllScalarFunctions("");
    expression::registerExpressionOptimizations();
    parse::registerTypeResolver();
  }

  void TearDown() override {
    exec::unregisterExpressionRewrites();
  }

  core::TypedExprPtr makeTypedExpr(
      const std::string& expression,
      const RowTypePtr& type) {
    auto untyped = parse::parseExpr(expression, {});
    return core::Expressions::inferTypes(untyped, type, execCtx_->pool());
  }

  core::TypedExprPtr optimizeExpression(
      const core::TypedExprPtr& expr,
      const core::QueryConfig& config,
      memory::MemoryPool* pool) {
    auto optimized = exec::rewriteExpression(expr, config, pool);
    return expression::constantFold(optimized, config, pool);
  }

  void testExpression(
      const core::TypedExprPtr& input,
      const core::TypedExprPtr& expected) {
    auto optimizedInput =
        optimizeExpression(input, queryCtx_->queryConfig(), pool());
    auto optimizedExpected =
        optimizeExpression(expected, queryCtx_->queryConfig(), pool());
    // The constant value in ConstantTypedExpr can either be in valueVector_ or
    // in the variant value_. String comparison is used to compare the optimized
    // expressions, since ITypedExpr comparison with equality operator will fail
    // for `ConstantTypedExpr`s if the underlying constant values do not have
    // the same internal representation.
    ASSERT_EQ(optimizedInput->toString(), optimizedExpected->toString());
  }

  void testExpression(
      const std::string& input,
      const std::string& expected,
      const RowTypePtr& inputType = ROW({}),
      const RowTypePtr& expectedType = ROW({})) {
    testExpression(
        makeTypedExpr(input, inputType), makeTypedExpr(expected, expectedType));
  }

  void setQueryTimeZone(const std::string& timeZone) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kSessionTimezone, timeZone},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
    });
  }

  std::shared_ptr<core::QueryCtx> queryCtx_{core::QueryCtx::create()};
  std::unique_ptr<core::ExecCtx> execCtx_{
      std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get())};
};

TEST_F(ExpressionOptimizerTest, coalesce) {
  auto type = ROW({"a"}, {BIGINT()});
  testExpression(
      "coalesce(2 * 3 * a, 1 - 1, cast(null as bigint))",
      "coalesce(6 * a, 0)",
      type,
      type);
  testExpression("coalesce(null, null)", "coalesce(null)");
  testExpression(
      "coalesce(2 * 3 * a, 1 - 1, cast(null as bigint))",
      "coalesce(6 * a, 0)",
      type,
      type);
  testExpression("coalesce(a, a)", "a", type, type);
  testExpression("coalesce(2 * a, 2 * a)", "BIGINT '2' * a", type, type);
  testExpression(
      "coalesce(a, coalesce(a, 1))", "coalesce(a, BIGINT '1')", type, type);
  testExpression(
      "coalesce(a, 2, coalesce(a, 1))", "coalesce(a, BIGINT '2')", type, type);
  type = ROW({"a"}, {DOUBLE()});
  testExpression(
      "coalesce(2.0E0 * a, 1.0E0/2.0E0, cast(null as double))",
      "coalesce(2.0E0 * a, 0.5E0)",
      type,
      type);
  testExpression(
      "coalesce(a, 1.0E0/2.0E0, 12.34E0, cast(null as double))",
      "coalesce(a, 0.5E0, 12.34E0)",
      type,
      type);

  type = ROW({"a", "b"}, {BIGINT(), BIGINT()});
  testExpression("coalesce(a, b, a)", "coalesce(a, b)", type, type);
  testExpression(
      "coalesce(coalesce(a, coalesce(a, 1)), b)",
      "coalesce(a, BIGINT '1')",
      type,
      ROW({"a"}, {BIGINT()}));
  type = ROW({"a", "b", "c"}, {BIGINT(), BIGINT(), BIGINT()});
  testExpression("coalesce(a, b, a, c)", "coalesce(a, b, c)", type, type);
  testExpression("coalesce(6, b, a, c)", "6", type);
  testExpression("coalesce(2 * 3, b, a, c)", "6", type);
  testExpression(
      "coalesce(coalesce(a, coalesce(b, c)), 1)",
      "coalesce(a, b, c, BIGINT '1')",
      type,
      type);
}

TEST_F(ExpressionOptimizerTest, conditionals) {
  // IF
  testExpression("IF(2 = 2, 3, 4)", "3");
  testExpression("IF(1 = 2, 3, 4)", "4");
  testExpression("IF(1 = 2, BIGINT '3', 4)", "4");
  testExpression("IF(1 = 2, 3000000000, 4)", "4");
  testExpression("IF(true, 3, 4)", "3");
  testExpression("IF(false, 3, 4)", "4");
  testExpression("IF(true, 3, cast(null as bigint))", "3");
  testExpression("IF(false, 3, cast(null as bigint))", "cast(null as bigint)");
  testExpression("IF(true, cast(null as bigint), 4)", "cast(null as bigint)");
  testExpression("IF(false, cast(null as bigint), 4)", "4");
  testExpression(
      "IF(true, cast(null as bigint), cast(null as bigint))",
      "cast(null as bigint)");
  testExpression(
      "IF(false, cast(null as bigint), cast(null as bigint))",
      "cast(null as bigint)");
  testExpression("IF(true, 3.5E0, 4.2E0)", "3.5E0");
  testExpression("IF(false, 3.5E0, 4.2E0)", "4.2E0");
  testExpression("IF(true, 'foo', 'bar')", "'foo'");
  testExpression("IF(false, 'foo', 'bar')", "'bar'");
  testExpression("IF(true, 1.01, 1.02)", "1.01");
  testExpression("IF(false, 1.01, 1.02)", "1.02");
  testExpression("IF(true, 1234567890.123, 1.02)", "1234567890.123");
  testExpression("IF(false, 1.01, 1234567890.123)", "1234567890.123");

  // CASE
  testExpression("case 1 when 1 then 32 + 1 when 1 then 34 end", "33");
  testExpression("case when true then 30 + 3 else 30 - 3 end", "33");
  testExpression("case when false then 1 else 27 + 6 end", "33");
  testExpression(
      "case when false then 10000000 * 1000 else abs(-33) end", "33");
  testExpression(
      "case when 1200 + 10 * 3 + 4 = 1200 + 34 then 11 * 3 else -1000 * 2 end",
      "33");
  testExpression(
      "case when 100 * 10 = 1200 + 10 * 3 + 4 then 1200 else -1000 end",
      "-1000");
  testExpression("case when true then 1238 - 4 else 100 end", "1234");
  testExpression("case when false then 1 else 1230 + 4 end", "1234");
}

TEST_F(ExpressionOptimizerTest, conjunct) {
  // AND
  testExpression("true and false", "false");
  testExpression("false and true", "false");
  testExpression("false and false", "false");
  testExpression("true and null", "null");
  testExpression("false and null", "false");
  testExpression("null and true", "null");
  testExpression("null and false", "false");
  testExpression("null and null", "null");

  auto type = ROW({"a"}, {VARCHAR()});
  testExpression("a='z' and true", "a='z'", type, type);
  testExpression("a='z' and false", "false", type);
  testExpression("true and a='z'", "a='z'", type, type);
  testExpression("false and a='z'", "false", type);
  type = ROW({"a", "b"}, {VARCHAR(), BIGINT()});
  testExpression("a='z' and b=1+1", "a='z' and b=2", type, type);

  // AND with flatten
  testExpression("true and (false and true)", "false");
  testExpression("true and (true and (true and false))", "false");
  type = ROW({"a"}, {VARCHAR()});
  testExpression("a='z' and (true and (true and true))", "a='z'", type, type);
  testExpression("a='z' and (true and (true and false))", "false", type, type);

  // OR
  testExpression("true or true", "true");
  testExpression("true or false", "true");
  testExpression("false or true", "true");
  testExpression("false or false", "false");
  testExpression("true or null", "true");
  testExpression("null or true", "true");
  testExpression("null or null", "null");
  testExpression("false or null", "null");
  testExpression("null or false", "null");

  type = ROW({"a"}, {VARCHAR()});
  testExpression("a='z' or true", "true", type);
  testExpression("a='z' or false", "a='z'", type, type);
  testExpression("true or a='z'", "true", type);
  testExpression("false or a='z'", "a='z'", type, type);
  type = ROW({"a", "b"}, {VARCHAR(), BIGINT()});
  testExpression("a='z' or b=1+1", "a='z' or b=2", type, type);

  // OR with flatten
  testExpression("true or (true or true)", "true");
  testExpression("false or (false or (false or true))", "true");

  type = ROW({"a"}, {VARCHAR()});
  testExpression("a='z' or (false or (false or true))", "true", type);
  testExpression("a='z' or (false or (false or false))", "a='z'", type, type);
}

TEST_F(ExpressionOptimizerTest, constantFolding) {
  // BETWEEN.
  testExpression("3 between 2 and 4", "true");
  testExpression("2 between 3 and 4", "false");
  testExpression("'cc' between 'b' and 'd'", "true");
  testExpression("'b' between 'cc' and 'd'", "false");

  // IN.
  testExpression("3 in (2, 4, 3, 5)", "true");
  testExpression("3 in (2, 4, 9, 5)", "false");
  testExpression("'foo' in ('bar', 'baz', 'foo', 'blah')", "true");
  testExpression("'foo' in ('bar', 'baz', 'buz', 'blah')", "false");
  testExpression(
      "'foo' in ('bar', cast(null as varchar), 'foo', 'blah')", "true");

  // CAST.
  testExpression("cast(BIGINT '123' as VARCHAR)", "123");
  testExpression("cast(12300000000 as VARCHAR)", "12300000000");
  testExpression("cast(-12300000000 as VARCHAR)", "-12300000000");
  testExpression("cast(123.456E0 as VARCHAR)", "123.456");
  testExpression("cast(-123.456E0 as VARCHAR)", "-123.456");

  // Ensure session timezone from queryConfig is used.
  setQueryTimeZone("Pacific/Apia");
  testExpression("hour(from_unixtime(9.98489045321E8))", "3");

  setQueryTimeZone("America/Los_Angeles");
  testExpression("hour(from_unixtime(9.98489045321E8))", "7");
}

} // namespace facebook::velox::expression::test
