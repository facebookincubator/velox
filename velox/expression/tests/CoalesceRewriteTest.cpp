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

class CoalesceRewriteTest
    : public expression::test::SpecialFormRewriteTestBase {};

TEST_F(CoalesceRewriteTest, basic) {
  auto type = ROW({"a", "b"}, {BIGINT(), BIGINT()});
  testRewrite("coalesce(null::bigint, 1, a)", "1", type);
  testRewrite(
      "coalesce(null::bigint, 6 * a, 5 * b, 1, null::bigint)",
      "coalesce(6 * a, 5 * b, 1)",
      type);
  testRewrite(
      "coalesce(null::bigint, coalesce(null::bigint, a, coalesce(1, b)))",
      "coalesce(a, 1)",
      type);
  testRewrite("coalesce(a, null::bigint, a)", "a", type);
  testRewrite(
      "coalesce(a, b, a, coalesce(coalesce(b, a), b))", "coalesce(a, b)", type);
  testRewrite(
      "coalesce(a, coalesce(2, coalesce(b, a)), b)", "coalesce(a, 2)", type);
}

} // namespace
} // namespace facebook::velox::expression
