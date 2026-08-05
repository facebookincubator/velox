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

#include "velox/core/Expressions.h"

#include <gtest/gtest.h>

namespace facebook::velox::core {
namespace {

FieldAccessTypedExprPtr field(const std::string& name) {
  return std::make_shared<FieldAccessTypedExpr>(BIGINT(), name);
}

CallTypedExprPtr call(std::vector<TypedExprPtr> inputs) {
  return std::make_shared<CallTypedExpr>(
      BIGINT(), std::move(inputs), "multiply");
}

// An expression with no shared subexpressions prints exactly as before: no
// labels are introduced.
TEST(TypedExprToStringTest, noSharing) {
  auto expr = call({field("a"), field("b")});
  EXPECT_EQ(expr->toString(), "multiply(\"a\",\"b\")");
}

// A subexpression reached from more than one parent is printed once, tagged
// with "as #N", and referenced by "#N" thereafter.
TEST(TypedExprToStringTest, sharedSubexpression) {
  auto shared = call({field("x"), field("x")});
  auto expr = call({shared, shared});
  EXPECT_EQ(expr->toString(), "multiply(multiply(\"x\",\"x\") as #1,#1)");
}

// A leaf reached from more than one parent is cheap to reprint, so it is not
// labeled.
TEST(TypedExprToStringTest, sharedLeafNotLabeled) {
  auto x = field("x");
  auto expr = call({x, x});
  EXPECT_EQ(expr->toString(), "multiply(\"x\",\"x\")");
}

// A deeply shared DAG (each level squares the one below) would expand to a
// tree of 2^depth nodes. Sharing-aware printing keeps it linear, so this
// returns quickly and stays small.
TEST(TypedExprToStringTest, deeplySharedDagIsLinear) {
  constexpr int kDepth = 40;
  TypedExprPtr expr = field("x");
  for (int i = 0; i < kDepth; ++i) {
    expr = call({expr, expr});
  }

  const auto text = expr->toString();
  // Each level contributes one full "multiply(...)" plus one "#N"
  // back-reference, so the output is O(depth), nowhere near 2^depth.
  EXPECT_LT(text.size(), 2000);
  EXPECT_NE(text.find(" as #"), std::string::npos);
}

} // namespace
} // namespace facebook::velox::core
