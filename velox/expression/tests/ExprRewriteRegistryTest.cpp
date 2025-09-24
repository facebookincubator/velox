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
#include "velox/expression/ExprRewriteRegistry.h"
#include "gtest/gtest.h"
#include "velox/expression/ExprConstants.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::expression {

namespace {

class ExprRewriteRegistryTest : public testing::Test,
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

  void testExpression(
      const std::string& inputExpr,
      const std::string& expectedExpr,
      bool returnFailFunction = false) {
    const auto input = makeTypedExpr(inputExpr, ROW({}));
    const auto expected = makeTypedExpr(expectedExpr, ROW({}));
    auto optimizedInput = expression::ExprRewriteRegistry::instance().rewrite(
        input, queryCtx_.get(), pool(), true, returnFailFunction);
    auto optimizedExpected =
        expression::ExprRewriteRegistry::instance().rewrite(
            expected, queryCtx_.get(), pool(), true, returnFailFunction);
    // The constant value in ConstantTypedExpr can either be in valueVector_ or
    // in the variant value_. String comparison is used to compare the optimized
    // expressions, since ITypedExpr comparison with equality operator will fail
    // for `ConstantTypedExpr`s if the underlying constant values do not have
    // the same internal representation.
    ASSERT_EQ(optimizedInput->toString(), optimizedExpected->toString());
  }

  void testExpressionFailFunction(
      const std::string& inputExpr,
      const std::string& expectedErrorMessage) {
    const auto input = makeTypedExpr(inputExpr, ROW({}));
    auto optimizedInput = expression::ExprRewriteRegistry::instance().rewrite(
        input, queryCtx_.get(), pool(), true, true);
    ASSERT_TRUE(optimizedInput->isCallKind());
    const auto call = optimizedInput->asUnchecked<core::CallTypedExpr>();
    ASSERT_EQ(call->name(), kFail);

    ASSERT_TRUE(call->inputs().at(0)->isConstantKind());
    const auto error =
        call->inputs().at(0)->asUnchecked<core::ConstantTypedExpr>();
    ASSERT_TRUE(error->type()->isVarchar());
    ASSERT_TRUE(
        error->toString().find(expectedErrorMessage) != std::string::npos);
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

TEST_F(ExprRewriteRegistryTest, basic) {
  expression::ExprRewriteRegistry registry;
  expression::ExpressionRewrite testRewrite =
      [&](const core::TypedExprPtr& input) {
        return std::make_shared<core::CallTypedExpr>(
            input->type(), "test_expr", input, input);
      };
  registry.registerRewrite(testRewrite);

  auto input =
      std::make_shared<core::ConstantTypedExpr>(BIGINT(), int64_t(123));
  const auto rewritten = registry.rewrite(input, queryCtx_.get(), pool_.get());
  ASSERT_TRUE(rewritten->isCallKind());
  ASSERT_TRUE(rewritten->type()->isBigint());
  const auto rewrittenCall = rewritten->asUnchecked<core::CallTypedExpr>();
  ASSERT_EQ(rewrittenCall->inputs().size(), 2);
  ASSERT_EQ(rewrittenCall->name(), "test_expr");

  registry.clear();
  const auto rewriteAfterClear =
      registry.rewrite(input, queryCtx_.get(), pool_.get());
  ASSERT_TRUE(*rewriteAfterClear == *input);
}

TEST_F(ExprRewriteRegistryTest, constantFolding) {
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

TEST_F(ExprRewriteRegistryTest, failFunction) {
  testExpressionFailFunction("0 / 0", "division by zero");
}

} // namespace
} // namespace facebook::velox::expression
