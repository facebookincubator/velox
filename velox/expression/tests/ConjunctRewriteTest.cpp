/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/expression/tests/SpecialFormRewriteTestBase.h"

namespace facebook::velox::expression {
namespace {

class ConjunctRewriteTest
    : public expression::test::SpecialFormRewriteTestBase {};

TEST_F(ConjunctRewriteTest, basic) {
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
