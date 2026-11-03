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

#include "velox/expression/ExprConstants.h"
#include "velox/expression/SwitchRewrite.h"
#include "velox/expression/tests/SpecialFormRewriteTestBase.h"

namespace facebook::velox::expression {
namespace {

class SwitchRewriteTest : public expression::test::SpecialFormRewriteTestBase {
};

TEST_F(SwitchRewriteTest, basic) {
  const auto type = ROW({"a"}, {BIGINT()});
  testRewrite("case when false then 1234 when true then a end", "a", type);
  testRewrite(
      "case when false then 1234 when false then 3456 else a end", "a", type);
  testRewrite(
      "case when false then 100 when a > 2 then 200 when a > 4 then 300 else 5678 end",
      "case when a > 2 then 200 when a > 4 then 300 else 5678 end",
      type);
  testRewrite(
      "case when a < 5 then 1234 when a > 10 then 3456 when true then 6789 else 0 end",
      "case when a < 5 then 1234 when a > 10 then 3456 else 6789 end",
      type);
}

TEST_F(SwitchRewriteTest, valuesEqualToElse) {
  const auto type = ROW("a", BIGINT());
  testRewrite("if(a > 1, a, a)", "a", type);
  testRewrite("case when a > 1 then 5 when a > 2 then 5 else 5 end", "5", type);
  testRewrite(
      "case when a > 1 then 1 when a > 2 then 2 when a > 3 then 5 else 5 end",
      "case when a > 1 then 1 when a > 2 then 2 else 5 end",
      type);
  // Only trailing values equal to the else value are removed.
  testRewrite(
      "case when a > 1 then 5 when a > 2 then 1 else 5 end",
      "case when a > 1 then 5 when a > 2 then 1 else 5 end",
      type);
  // A true condition makes its value the new else value.
  testRewrite("case when a > 1 then a when true then a end", "a", type);

  testRewrite("case a when 1 then 5 when 2 then 5 else 5 end", "5", type);
  testRewrite(
      "case a when 1 then 1 when 2 then 2 when 3 then 5 else 5 end",
      "case a when 1 then 1 when 2 then 2 else 5 end",
      type);

  // 'a' is compared to the WHEN values as DOUBLE here, but would be compared
  // as BIGINT without 'when 0.5'. The first case matches a = 2^53 + 1 only as
  // DOUBLE. Cases after 'when 0.5' are still removed.
  testRewrite(
      "case a when 9007199254740992 then 'x' when 0.5 then 'y' when 2 then 'y' else 'y' end",
      "case a when 9007199254740992 then 'x' when 0.5 then 'y' else 'y' end",
      type);
}

TEST_F(SwitchRewriteTest, valuesNotComparable) {
  // Lambdas can't be compared when the expression runs, so a failing condition
  // raises its error there. The conditions are kept to raise it here too.
  const auto lambda = std::make_shared<core::LambdaTypedExpr>(
      ROW("x", BIGINT()),
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "x"));
  const auto ifExpr = std::make_shared<core::CallTypedExpr>(
      lambda->type(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "a"),
          lambda,
          lambda},
      kIf);
  EXPECT_EQ(SwitchRewrite::rewrite(ifExpr), nullptr);

  const auto caseExpr = std::make_shared<core::CallTypedExpr>(
      lambda->type(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "b"),
          std::make_shared<core::ConstantTypedExpr>(
              BIGINT(), variant(int64_t{1})),
          lambda,
          lambda},
      kCase);
  EXPECT_EQ(SwitchRewrite::rewrite(caseExpr), nullptr);
}

} // namespace
} // namespace facebook::velox::expression
