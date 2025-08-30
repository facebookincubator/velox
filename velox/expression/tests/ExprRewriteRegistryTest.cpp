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

namespace facebook::velox::expression::test {

class ExprRewriteRegistryTest : public testing::Test {};

TEST_F(ExprRewriteRegistryTest, basic) {
  expression::ExpressionRewrite testRewrite =
      [&](const core::TypedExprPtr& input) {
        return std::make_shared<core::CallTypedExpr>(
            input->type(),
            std::vector<core::TypedExprPtr>{input, input},
            "test_expr");
      };
  auto rewrite =
      expression::expressionRewriteRegistry().getExpressionRewrite("test");
  ASSERT_EQ(rewrite, nullptr);
  expression::registerExpressionRewrite(
      "test", std::make_unique<expression::ExpressionRewrite>(testRewrite));
  rewrite =
      expression::expressionRewriteRegistry().getExpressionRewrite("test");
  ASSERT_NE(rewrite, nullptr);

  auto input =
      std::make_shared<core::ConstantTypedExpr>(BIGINT(), variant(123));
  const auto rewritten = rewrite->operator()(input);
  ASSERT_TRUE(rewritten->isCallKind());
  ASSERT_TRUE(rewritten->type()->isBigint());
  const auto rewrittenCall =
      dynamic_cast<const core::CallTypedExpr*>(rewritten.get());
  ASSERT_EQ(rewrittenCall->inputs().size(), 2);
  ASSERT_EQ(rewrittenCall->name(), "test_expr");

  expression::unregisterExpressionRewrites();
  ASSERT_EQ(
      expression::expressionRewriteRegistry()
          .getExpressionRewriteNames()
          .size(),
      0);
}

} // namespace facebook::velox::expression::test
