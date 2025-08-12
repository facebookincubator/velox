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

#include "velox/expression/ExprOptimizer.h"
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::expression::test {

class ExprOptimizerTest : public testing::Test,
                          public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    functions::prestosql::registerAllScalarFunctions("");
    parse::registerTypeResolver();
    const auto configData = std::unordered_map<std::string, std::string>{
        {core::QueryConfig::kExprApplySpecialFormRewrites, "true"}};
    queryCtx_ = core::QueryCtx::create(
        nullptr, core::QueryConfig{std::move(configData)});
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
      const std::shared_ptr<core::QueryCtx>& queryCtx,
      memory::MemoryPool* pool) {
    return optimizeExpressions({expr}, queryCtx, pool).front();
  }

  void testExpression(
      const core::TypedExprPtr& input,
      const core::TypedExprPtr& expected) {
    auto optimizedInput = optimizeExpression(input, queryCtx_, pool());
    auto optimizedExpected = optimizeExpression(expected, queryCtx_, pool());
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
    queryCtx_->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kSessionTimezone, timeZone},
         {core::QueryConfig::kAdjustTimestampToTimezone, "true"}});
  }

  std::shared_ptr<core::QueryCtx> queryCtx_{core::QueryCtx::create()};
  std::unique_ptr<core::ExecCtx> execCtx_{
      std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get())};
};

TEST_F(ExprOptimizerTest, conjunct) {
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

  testExpression("true and (false and true)", "false");
  testExpression("true and (true and (true and false))", "false");
  type = ROW({"a"}, {VARCHAR()});
  testExpression("a='z' and (true and (true and true))", "a='z'", type, type);
  testExpression("a='z' and (true and (true and false))", "false", type, type);

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

  testExpression("true or (true or true)", "true");
  testExpression("false or (false or (false or true))", "true");

  type = ROW({"a"}, {VARCHAR()});
  testExpression("a='z' or (false or (false or true))", "true", type);
  testExpression("a='z' or (false or (false or false))", "a='z'", type, type);
}

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

  // Ensure session timezone from queryConfig is used.
  setQueryTimeZone("Pacific/Apia");
  testExpression("hour(from_unixtime(9.98489045321E8))", "3");

  setQueryTimeZone("America/Los_Angeles");
  testExpression("hour(from_unixtime(9.98489045321E8))", "7");
}

} // namespace facebook::velox::expression::test
