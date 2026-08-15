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

#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox::expression::test {

class SpecialFormRewriteTestBase : public functions::test::FunctionBaseTest {
 protected:
  static void SetUpTestCase();

  void SetUp() override;

  void TearDown() override;

  /// Validates special form expression rewrites that can short circuit
  /// expression evaluation.
  /// @param expr Input SQL expression to be rewritten.
  /// @param expected Expected SQL expression after rewriting `expr`.
  /// @param type Row type containing input fields referenced by `expr`.
  void testRewrite(
      const std::string& expr,
      const std::string& expected,
      const RowTypePtr& type = ROW({}));
};

} // namespace facebook::velox::expression::test
