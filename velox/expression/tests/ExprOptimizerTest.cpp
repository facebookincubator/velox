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
#include <fmt/format.h>
#include <gtest/gtest.h>

#include "velox/expression/ExprConstants.h"
#include "velox/expression/ExprOptimizer.h"
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::expression::test {
namespace {

class ExprOptimizerTest : public testing::Test,
                          public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    functions::prestosql::registerAllScalarFunctions("");
    parse::registerTypeResolver();
  }

  void TearDown() override {
    expression::ExprRewriteRegistry::instance().clear();
  }

  core::TypedExprPtr makeTypedExpr(
      const std::string& expression,
      const RowTypePtr& type) {
    auto untyped = parse::parseExpr(expression, {});
    return core::Expressions::inferTypes(untyped, type, execCtx_->pool());
  }

  core::TypedExprPtr optimize(
      const std::string& expression,
      const RowTypePtr& type = ROW({}),
      bool replaceEvalErrorWithFailExpr = false) {
    const auto expr = makeTypedExpr(expression, type);
    return expression::optimize(
        expr, queryCtx_.get(), pool(), replaceEvalErrorWithFailExpr);
  }

  void testExpression(
      const std::string& input,
      const std::string& expected,
      const RowTypePtr& inputType = ROW({}),
      const RowTypePtr& expectedType = ROW({})) {
    auto optimized = optimize(input, inputType);
    auto expectedExpr = makeTypedExpr(expected, expectedType);
    SCOPED_TRACE(fmt::format(
        "Input: {}\nOptimized: {}\nExpected: {}",
        input,
        optimized->toString(),
        expected));

    // TODO: https://github.com/facebookincubator/velox/issues/15215.
    // The constant value in ConstantTypedExpr can either be in valueVector_ or
    // in the variant value_. String comparison is used to compare the optimized
    // expressions, since ITypedExpr comparison with equality operator will fail
    // for `ConstantTypedExpr`s if the underlying constant values do not have
    // the same internal representation.
    ASSERT_EQ(optimized->toString(), expectedExpr->toString());
  }

  void assertFailCall(
      const std::string& expression,
      const std::string& expected) {
    auto optimized = optimize(expression, ROW({}), true);
    ASSERT_EQ(optimized->toString(), expected);
  }

  void setQueryTimeZone(const std::string& timeZone) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kSessionTimezone, timeZone},
         {core::QueryConfig::kAdjustTimestampToTimezone, "true"}});
  }

  std::shared_ptr<core::QueryCtx> queryCtx_{core::QueryCtx::create()};
  std::unique_ptr<core::ExecCtx> execCtx_{
      std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get())};
};

TEST_F(ExprOptimizerTest, constantFolding) {
  testExpression("3 between 2 and 4", "true");
  testExpression("2 between 3 and 4", "false");
  testExpression("'cc' between 'b' and 'd'", "true");
  testExpression("'b' between 'cc' and 'd'", "false");

  testExpression("cast(BIGINT '123' as VARCHAR)", "123");
  testExpression("cast(12300000000 as VARCHAR)", "12300000000");
  testExpression("cast(-12300000000 as VARCHAR)", "-12300000000");
  testExpression("cast(123.456E0 as VARCHAR)", "123.456");
  testExpression("cast(-123.456E0 as VARCHAR)", "-123.456");

  auto type = ROW({"a", "b"}, {VARCHAR(), BIGINT()});
  testExpression("a = 'z' and b = (1 + 1)", "a = 'z' and b = 2", type, type);
  type = ROW({"a", "b"}, {VARCHAR(), VARCHAR()});
  testExpression(
      "concat(substr(a, 1 + 5 - 2), substr(b, 2 * 3))",
      "concat(substr(a, 4), substr(b, 6))",
      type,
      type);
  type = ROW({"a", "b", "c"}, {VARCHAR(), BIGINT(), INTEGER()});
  testExpression(
      "a = 'z' and b = (1 + 1) or c = (5 - 1) * 3",
      "a = 'z' and b = 2 or c = 12",
      type,
      type);
  type = ROW({"a", "b", "c"}, {VARCHAR(), VARCHAR(), INTEGER()});
  testExpression(
      "strpos(substr(a, (1 + 5 - 2)), substr(b, 2 * 3)) + c - (4 - 1)",
      "strpos(substr(a, 4), substr(b, 6)) + c - 3",
      type,
      type);

  // Ensure session timezone from queryConfig is used.
  setQueryTimeZone("Pacific/Apia");
  testExpression("hour(from_unixtime(9.98489045321E8))", "3");
  setQueryTimeZone("America/Los_Angeles");
  testExpression("hour(from_unixtime(9.98489045321E8))", "7");
}

TEST_F(ExprOptimizerTest, failExpression) {
  assertFailCall("0 / 0", "fail(division by zero)");
  assertFailCall(
      "CAST(4000000000 as INTEGER)",
      "fail(Cannot cast BIGINT '4000000000' to INTEGER. Overflow during arithmetic conversion: )");
  assertFailCall(
      "CAST(-4000000000 as INTEGER)",
      "fail(Cannot cast BIGINT '-4000000000' to INTEGER. Negative overflow during arithmetic conversion: )");
  assertFailCall(
      "(INTEGER '400000' * INTEGER '200000')",
      "fail(integer overflow: 400000 * 200000)");

  // Verify only the failing subexpression is replaced by fail expression.
  const auto optimized = optimize("a + (0 / 0)", ROW({"a"}, {BIGINT()}), true);
  const auto inputs = optimized->inputs();
  ASSERT_EQ(inputs.size(), 2);
  ASSERT_TRUE(inputs.at(0)->isFieldAccessKind());
  ASSERT_EQ(optimized->toString(), "plus(ROW[\"a\"],fail(division by zero))");
}

} // namespace
} // namespace facebook::velox::expression::test
