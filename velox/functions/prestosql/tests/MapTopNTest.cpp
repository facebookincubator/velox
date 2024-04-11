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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class MapTopNTest : public test::FunctionBaseTest {};

TEST_F(MapTopNTest, basic) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, int64_t>({
          "{1:3, 2:5, 3:1, 4:4, 5:2}",
          "{1:3, 2:5, 3:null, 4:4, 5:2}",
          "{1:null, 2:null, 3:1, 4:4, 5:null}",
          "{1:10, 2:7, 3:11, 5:4}",
          "{1:10, 2:7, 3:0}",
          "{1:null, 2:10}",
          "{}",
          "{1:null, 2:null, 3:null}",
      }),
  });

  auto result = evaluate("map_top_n(c0, 3)", data);

  auto expected = makeMapVectorFromJson<int32_t, int64_t>({
      "{2:5, 4:4, 1:3}",
      "{2:5, 4:4, 1:3}",
      "{4:4, 3:1, 5:null}",
      "{3:11, 1:10, 2:7}",
      "{1:10, 2:7, 3:0}",
      "{2:10, 1:null}",
      "{}",
      "{1:null, 2:null, 3:null}",
  });

  assertEqualVectors(expected, result);

  // n = 0. Expect empty maps.
  result = evaluate("map_top_n(c0, 0)", data);

  expected = makeMapVectorFromJson<int32_t, int64_t>(
      {"{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}"});

  assertEqualVectors(expected, result);

  // n is negative. Expect an error.
  VELOX_ASSERT_THROW(
      evaluate("map_top_n(c0, -1)", data),
      "n must be greater than or equal to 0");
}

} // namespace
} // namespace facebook::velox::functions
