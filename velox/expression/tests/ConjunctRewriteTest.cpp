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

#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox::expression {
namespace {

class ConjunctRewriteTest : public functions::test::FunctionBaseTest {
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

  /// Validates special form expression rewrites that can short circuit
  /// expression evaluation.
  /// @param expr Input SQL expression to be rewritten.
  /// @param expected Expected SQL expression after rewriting `expr`.
  /// @param type Row type containing input fields referenced by `expr`.
  void testRewrite(
      const std::string& expr,
      const std::string& expected,
      const RowTypePtr& type = ROW({})) {
    const auto typedExpr = makeTypedExpr(expr, type);
    const auto rewritten =
        expression::ExprRewriteRegistry::instance().rewrite(typedExpr);
    const auto expectedExpr = makeTypedExpr(expected, type);
    SCOPED_TRACE(fmt::format("Input: {}", typedExpr->toString()));
    SCOPED_TRACE(fmt::format("Rewritten: {}", rewritten->toString()));
    SCOPED_TRACE(fmt::format("Expected: {}", expectedExpr->toString()));

    ASSERT_TRUE(*rewritten == *expectedExpr);
  }
};

TEST_F(ConjunctRewriteTest, basic) {
  testRewrite("true and true", "true");
  testRewrite("false or false", "false");
  testRewrite("null::boolean and false", "false");
  testRewrite("true or null::boolean", "true");
  testRewrite(
      "null::boolean and null::boolean and null::boolean", "null::boolean");
  testRewrite(
      "null::boolean or null::boolean or null::boolean", "null::boolean");
  testRewrite("null::boolean and true", "null::boolean");
  testRewrite("false or null::boolean", "null::boolean");

  const auto type = ROW({"a", "b"}, {VARCHAR(), BIGINT()});
  testRewrite(
      "null::boolean and a = 'z' and true", "null::boolean and a = 'z'", type);
  testRewrite(
      "a = 'z' or null::boolean or false", "a = 'z' or null::boolean", type);
  testRewrite("true and a = 'z' and true", "a = 'z'", type);
  testRewrite("false or a = 'z' or false", "a = 'z'", type);
  testRewrite("a = 'z' and b = 2 and false", "false", type);
  testRewrite("((a = 'z' and b = 2) and false)", "false", type);
  testRewrite("a = 'z' or b = 2 or true", "true", type);
  testRewrite("(a = 'z' or (b = 2 or true))", "true", type);
}

} // namespace
} // namespace facebook::velox::expression
