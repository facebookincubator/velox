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
#include "velox/expression/tests/SpecialFormRewriteTestBase.h"

namespace facebook::velox::expression {
namespace {

class InRewriteTest : public expression::test::SpecialFormRewriteTestBase {
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
};

TEST_F(InRewriteTest, basic) {
  // 1234 in (2, a, b, 1234) -> true.
  const auto type = ROW({"a", "b"}, {BIGINT(), BIGINT()});
  auto in = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      "in",
      constant(INTEGER(), 1234),
      constant(INTEGER(), 2),
      field(INTEGER(), "a"),
      field(INTEGER(), "b"),
      constant(INTEGER(), 1234));
  auto expected = constant(BOOLEAN(), true);
  testRewrite(in, expected, type);

  // 81 in (2, 4, a, b, 9) -> 81 in (a, b)
  in = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      "in",
      constant(INTEGER(), 81),
      constant(INTEGER(), 2),
      field(INTEGER(), "a"),
      constant(INTEGER(), 4),
      field(INTEGER(), "b"),
      constant(INTEGER(), 1234));
  expected = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      "in",
      constant(INTEGER(), 81),
      field(INTEGER(), "a"),
      field(INTEGER(), "b"));
  testRewrite(in, expected, type);

  // 'hello' in ('foo', 'bar', a, 'hello') -> true
  in = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      "in",
      constant(VARCHAR(), "hello"),
      constant(VARCHAR(), "foo"),
      constant(VARCHAR(), "bar"),
      field(VARCHAR(), "a"),
      constant(VARCHAR(), "hello"));
  expected = constant(BOOLEAN(), true);
  testRewrite(in, expected, ROW({"a"}, {VARCHAR()}));
}

} // namespace
} // namespace facebook::velox::expression
