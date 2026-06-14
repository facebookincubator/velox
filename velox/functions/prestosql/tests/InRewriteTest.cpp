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
        expression::ExprRewriteRegistry::instance().rewrite(expr, pool());
    return *rewritten == *expected;
  }
};

TEST_F(InRewriteTest, basic) {
  // 1234 in (2, a, b, 1234) -> true
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
       constant(INTEGER(), 9)});
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

  // 'hello' in ('foo', NULL, 'bar', a, b) -> 'hello' in (NULL, a, b)
  in = makeIn(
      {constant(VARCHAR(), "hello"),
       constant(VARCHAR(), "foo"),
       constant(VARCHAR(), variant::null(TypeKind::VARCHAR)),
       constant(VARCHAR(), "bar"),
       field(VARCHAR(), "a"),
       field(VARCHAR(), "b")});
  expected = makeIn(
      {constant(VARCHAR(), "hello"),
       constant(VARCHAR(), variant::null(TypeKind::VARCHAR)),
       field(VARCHAR(), "a"),
       field(VARCHAR(), "b")});
  ASSERT_TRUE(testInRewrite(in, expected));
}

TEST_F(InRewriteTest, nullAsIndeterminate) {
  // NULL in (5, NULL, a) -> NULL in (5, NULL, a)
  auto in = makeIn(
      {constant(BIGINT(), variant::null(TypeKind::BIGINT)),
       constant(BIGINT(), 5LL),
       constant(BIGINT(), variant::null(TypeKind::BIGINT)),
       field(BIGINT(), "a")});
  ASSERT_TRUE(testInRewrite(in, in));

  // NULL in ('foo', a, 'bar', NULL) -> NULL in ('foo', a, 'bar', NULL)
  in = makeIn(
      {constant(VARCHAR(), variant::null(TypeKind::VARCHAR)),
       constant(VARCHAR(), "foo"),
       field(VARCHAR(), "a"),
       constant(VARCHAR(), "bar"),
       constant(VARCHAR(), variant::null(TypeKind::VARCHAR))});
  ASSERT_TRUE(testInRewrite(in, in));

  // {0: null} IN ({0: 1}, c2) -> {0: null} IN ({0: 1}, c2)
  auto c0 = BaseVector::wrapInConstant(
      1, 0, makeMapVectorFromJson<int32_t, int32_t>({"{0: null}"}));
  auto c1 = BaseVector::wrapInConstant(
      1, 0, makeMapVectorFromJson<int32_t, int32_t>({"{0: 1}"}));
  auto c2 = makeMapVectorFromJson<int32_t, int32_t>({"{0: 1, 10: 2}"});
  in = makeIn(
      {std::make_shared<core::ConstantTypedExpr>(c0),
       std::make_shared<core::ConstantTypedExpr>(c1),
       std::make_shared<core::FieldAccessTypedExpr>(c2->type(), "c2")});
  ASSERT_TRUE(testInRewrite(in, in));

  // Result of {0: null} == {0: 1} is indeterminate; {0: null} IN ({0: 1}, c2)
  // should not be simplified by rewrite and should evaluate to null.
  auto result = evaluate(in, makeRowVector({"c0", "c1", "c2"}, {c0, c1, c2}));
  velox::test::assertEqualVectors(
      makeNullConstant(TypeKind::BOOLEAN, 1), result);
}

} // namespace
} // namespace facebook::velox::functions
