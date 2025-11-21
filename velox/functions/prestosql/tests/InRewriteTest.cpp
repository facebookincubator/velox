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
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox::functions {
namespace {

class InRewriteTest : public functions::test::FunctionBaseTest {
 protected:
  static core::TypedExprPtr constant(
      const TypePtr& type,
      const Variant& value) {
    return std::make_shared<core::ConstantTypedExpr>(type, value);
  }

  static core::TypedExprPtr field(
      const TypePtr& type,
      const std::string& name) {
    return std::make_shared<core::FieldAccessTypedExpr>(type, name);
  }

  static core::TypedExprPtr makeIn(
      std::initializer_list<core::TypedExprPtr> inputs) {
    return std::make_shared<core::CallTypedExpr>(BOOLEAN(), "in", inputs);
  }

  bool testInRewrite(
      const core::TypedExprPtr& expr,
      const core::TypedExprPtr& expected) {
    const auto rewritten =
        expression::ExprRewriteRegistry::instance().rewrite(expr);
    return *rewritten == *expected;
  }
};

TEST_F(InRewriteTest, basic) {
  // 1234 in (2, a, b, 1234) -> true.
  auto in = makeIn(
      {constant(INTEGER(), 1234),
       constant(INTEGER(), 2),
       field(INTEGER(), "a"),
       field(INTEGER(), "b"),
       constant(INTEGER(), 1234)});
  auto expected = constant(BOOLEAN(), true);
  ASSERT_TRUE(testInRewrite(in, expected));

  // 81 in (2, 4, a, b, 9) -> 81 in (a, b)
  in = makeIn(
      {constant(INTEGER(), 81),
       constant(INTEGER(), 2),
       field(INTEGER(), "a"),
       constant(INTEGER(), 4),
       field(INTEGER(), "b"),
       constant(INTEGER(), 1234)});
  expected = makeIn(
      {constant(INTEGER(), 81), field(INTEGER(), "a"), field(INTEGER(), "b")});
  ASSERT_TRUE(testInRewrite(in, expected));

  // 5 in (NULL, a, 5) -> true
  in = makeIn(
      {constant(BIGINT(), 5LL),
       constant(BIGINT(), variant::null(TypeKind::BIGINT)),
       field(BIGINT(), "a"),
       constant(BIGINT(), 5LL)});
  expected = constant(BOOLEAN(), true);
  ASSERT_TRUE(testInRewrite(in, expected));

  // NULL in (5, NULL, a) -> NULL
  in = makeIn(
      {constant(BIGINT(), variant::null(TypeKind::BIGINT)),
       constant(BIGINT(), 5LL),
       constant(BIGINT(), variant::null(TypeKind::BIGINT)),
       field(BIGINT(), "a")});
  expected = constant(BOOLEAN(), variant::null(TypeKind::BOOLEAN));
  ASSERT_TRUE(testInRewrite(in, expected));

  // NULL in ('foo', 'bar', a) -> NULL
  in = makeIn(
      {constant(VARCHAR(), variant::null(TypeKind::VARCHAR)),
       constant(VARCHAR(), "foo"),
       constant(VARCHAR(), "bar"),
       field(VARCHAR(), "a")});
  expected = constant(BOOLEAN(), variant::null(TypeKind::BOOLEAN));
  ASSERT_TRUE(testInRewrite(in, expected));

  // 'hello' in ('foo', NULL, 'bar', a, NULL) -> 'hello' in (NULL, a)
  in = makeIn(
      {constant(VARCHAR(), "hello"),
       constant(VARCHAR(), "foo"),
       constant(VARCHAR(), variant::null(TypeKind::VARCHAR)),
       constant(VARCHAR(), "bar"),
       field(VARCHAR(), "a"),
       constant(VARCHAR(), variant::null(TypeKind::VARCHAR))});
  expected = makeIn(
      {constant(VARCHAR(), "hello"),
       constant(VARCHAR(), variant::null(TypeKind::VARCHAR)),
       field(VARCHAR(), "a")});
  ASSERT_TRUE(testInRewrite(in, expected));
}

} // namespace
} // namespace facebook::velox::functions
