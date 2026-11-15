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
#include "velox/core/Expressions.h"

namespace facebook::velox::core::test {
namespace {

class TypedExprWithInputsTest : public testing::Test {
 protected:
  const TypedExprPtr a_ = std::make_shared<FieldAccessTypedExpr>(BIGINT(), "a");
  const TypedExprPtr b_ = std::make_shared<FieldAccessTypedExpr>(BIGINT(), "b");
};

TypedExprPtr makeRow(const TypedExprPtr& input) {
  return std::make_shared<ConcatTypedExpr>(
      std::vector<std::string>{"x"}, std::vector<TypedExprPtr>{input});
}

// Each kind of expression rebuilt over new inputs equals the same kind of
// expression built directly over them.
TEST_F(TypedExprWithInputsTest, inputs) {
  // Pairs of an expression over 'a' and the same expression over 'b'.
  const std::vector<std::pair<TypedExprPtr, TypedExprPtr>> cases = {
      {std::make_shared<CallTypedExpr>(BIGINT(), "plus", a_, a_),
       std::make_shared<CallTypedExpr>(BIGINT(), "plus", b_, b_)},
      {std::make_shared<CastTypedExpr>(VARCHAR(), a_, true),
       std::make_shared<CastTypedExpr>(VARCHAR(), b_, true)},
      {std::make_shared<NullIfTypedExpr>(a_, a_, BIGINT()),
       std::make_shared<NullIfTypedExpr>(b_, b_, BIGINT())},
      {makeRow(a_), makeRow(b_)},
      {std::make_shared<DereferenceTypedExpr>(BIGINT(), makeRow(a_), 0),
       std::make_shared<DereferenceTypedExpr>(BIGINT(), makeRow(b_), 0)},
      {std::make_shared<FieldAccessTypedExpr>(BIGINT(), makeRow(a_), "x"),
       std::make_shared<FieldAccessTypedExpr>(BIGINT(), makeRow(b_), "x")},
  };

  for (const auto& [expr, expected] : cases) {
    SCOPED_TRACE(expr->toString());
    const auto rebuilt = expr->withInputs(expected->inputs());
    EXPECT_TRUE(*rebuilt == *expected) << rebuilt->toString();
  }
}

// Expressions without inputs come back as equal copies. A lambda's body is not
// one of its inputs.
TEST_F(TypedExprWithInputsTest, noInputs) {
  const std::vector<TypedExprPtr> exprs = {
      std::make_shared<InputTypedExpr>(ROW("a", BIGINT())),
      std::make_shared<ConstantTypedExpr>(BIGINT(), Variant(int64_t{1})),
      a_,
      std::make_shared<LambdaTypedExpr>(
          ROW("x", BIGINT()),
          std::make_shared<CallTypedExpr>(
              BIGINT(),
              "plus",
              std::make_shared<FieldAccessTypedExpr>(BIGINT(), "x"),
              a_)),
  };

  for (const auto& expr : exprs) {
    SCOPED_TRACE(expr->toString());
    EXPECT_TRUE(*expr->withInputs({}) == *expr);
  }
}

TEST_F(TypedExprWithInputsTest, wrongNumberOfInputs) {
  const auto call = std::make_shared<CallTypedExpr>(BIGINT(), "plus", a_, a_);
  VELOX_ASSERT_THROW(call->withInputs({b_}), "(1 vs. 2)");
  VELOX_ASSERT_THROW(a_->withInputs({b_}), "(1 vs. 0)");
}

} // namespace
} // namespace facebook::velox::core::test
