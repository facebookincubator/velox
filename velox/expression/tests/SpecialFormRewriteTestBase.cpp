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

#include "velox/expression/tests/SpecialFormRewriteTestBase.h"
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

namespace facebook::velox::expression::test {

void SpecialFormRewriteTestBase::SetUpTestCase() {
  memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
}

void SpecialFormRewriteTestBase::SetUp() {
  functions::prestosql::registerAllScalarFunctions("");
  parse::registerTypeResolver();
}

void SpecialFormRewriteTestBase::TearDown() {
  expression::ExprRewriteRegistry::instance().clear();
}

void SpecialFormRewriteTestBase::testRewrite(
    const core::TypedExprPtr& expr,
    const core::TypedExprPtr& expected,
    const RowTypePtr& type) {
  const auto rewritten =
      expression::ExprRewriteRegistry::instance().rewrite(expr);
  if (*rewritten != *expected) {
    SCOPED_TRACE(fmt::format("Input: {}", expr->toString()));
    SCOPED_TRACE(fmt::format("Rewritten: {}", rewritten->toString()));
    SCOPED_TRACE(fmt::format("Expected: {}", expected->toString()));
  }

  ASSERT_TRUE(*rewritten == *expected);
}

void SpecialFormRewriteTestBase::testRewrite(
    const std::string& expr,
    const std::string& expected,
    const RowTypePtr& type) {
  const auto typedExpr = makeTypedExpr(expr, type);
  const auto expectedExpr = makeTypedExpr(expected, type);

  testRewrite(typedExpr, expectedExpr, type);
}

} // namespace facebook::velox::expression::test
