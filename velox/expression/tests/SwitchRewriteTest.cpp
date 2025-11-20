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

#include "velox/expression/tests/SpecialFormRewriteTestBase.h"

namespace facebook::velox::expression {
namespace {

class SwitchRewriteTest : public expression::test::SpecialFormRewriteTestBase {
};

TEST_F(SwitchRewriteTest, basic) {
  testRewrite("if(true, 'hello', 'world')", "if(true, 'hello', 'world')");
  testRewrite(
      "case when false then 1 when true then 3 end",
      "case when false then 1 when true then 3 end");
  testRewrite(
      "case when false then 1 when false then 3 end",
      "case when false then 1 when false then 3 end");
  testRewrite(
      "case when false then 1 when false then 3 else 2 end",
      "case when false then 1 when false then 3 else 2 end");
  testRewrite(
      "case when false then 'hello' when false then 'world' when true then 'foo' else 'bar' end",
      "case when false then 'hello' when false then 'world' when true then 'foo' else 'bar' end");

  const auto type = ROW({"a"}, {BIGINT()});
  testRewrite(
      "case when false then 1234 when true then a end",
      "case when false then 1234 when true then a end",
      type);
  testRewrite(
      "case when false then 1234 when false then 3456 else a end",
      "case when false then 1234 when false then 3456 else a end",
      type);
  testRewrite(
      "case when false then 100 when a > 2 then 200 when a > 4 then 300 else 5678 end",
      "case when false then 100 when a > 2 then 200 when a > 4 then 300 else 5678 end",
      type);
  testRewrite(
      "case when a < 5 then 1234 when a > 10 then 3456 when true then 6789 else 0 end",
      "case when a < 5 then 1234 when a > 10 then 3456 when true then 6789 else 0 end",
      type);
}

} // namespace
} // namespace facebook::velox::expression
