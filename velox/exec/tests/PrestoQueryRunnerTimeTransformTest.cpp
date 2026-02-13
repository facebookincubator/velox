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

#include "velox/exec/tests/PrestoQueryRunnerIntermediateTypeTransformTestBase.h"

namespace facebook::velox::exec::test {
namespace {

class PrestoQueryRunnerTimeTransformTest
    : public PrestoQueryRunnerIntermediateTypeTransformTestBase {};

// Test that TIME is recognized as an intermediate type that needs
// transformation
TEST_F(PrestoQueryRunnerTimeTransformTest, isIntermediateOnlyType) {
  // Core test: TIME should be an intermediate type
  ASSERT_TRUE(isIntermediateOnlyType(TIME()));

  // Complex types containing TIME should also be intermediate types
  ASSERT_TRUE(isIntermediateOnlyType(ARRAY(TIME())));
  ASSERT_TRUE(isIntermediateOnlyType(MAP(VARCHAR(), TIME())));
  ASSERT_TRUE(isIntermediateOnlyType(MAP(TIME(), VARCHAR())));
  ASSERT_TRUE(isIntermediateOnlyType(ROW({TIME(), BIGINT()})));
}

TEST_F(PrestoQueryRunnerTimeTransformTest, roundTrip) {
  // Test basic TIME values (no nulls, some nulls, all nulls)
  std::vector<std::optional<int64_t>> no_nulls{0, 3661000, 43200000, 86399999};
  test(makeNullableFlatVector(no_nulls, TIME()));

  std::vector<std::optional<int64_t>> some_nulls{
      0, 3661000, std::nullopt, 86399999};
  test(makeNullableFlatVector(some_nulls, TIME()));

  std::vector<std::optional<int64_t>> all_nulls{
      std::nullopt, std::nullopt, std::nullopt};
  test(makeNullableFlatVector(all_nulls, TIME()));
}

TEST_F(PrestoQueryRunnerTimeTransformTest, transformArray) {
  auto input = makeNullableFlatVector(
      std::vector<std::optional<int64_t>>{
          0, // 00:00:00.000
          1000, // 00:00:01.000
          3661000, // 01:01:01.000
          43200000, // 12:00:00.000 (noon)
          86399999, // 23:59:59.999
          3723456, // 01:02:03.456
          45678901, // 12:41:18.901
          std::nullopt,
          72000000, // 20:00:00.000
          36000000 // 10:00:00.000
      },
      TIME());
  testArray(input);
}

TEST_F(PrestoQueryRunnerTimeTransformTest, transformMap) {
  // keys can't be null for maps
  auto keys = makeNullableFlatVector(
      std::vector<std::optional<int64_t>>{
          0, // 00:00:00.000
          3661000, // 01:01:01.000
          43200000, // 12:00:00.000
          86399999, // 23:59:59.999
          36000000, // 10:00:00.000
          72000000, // 20:00:00.000
          1800000, // 00:30:00.000
          7200000, // 02:00:00.000
          64800000, // 18:00:00.000
          32400000 // 09:00:00.000
      },
      TIME());

  auto values = makeNullableFlatVector<int64_t>(
      {100, 200, std::nullopt, 400, 500, std::nullopt, 700, 800, 900, 1000},
      BIGINT());

  testMap(keys, values);
}

TEST_F(PrestoQueryRunnerTimeTransformTest, transformRow) {
  auto input = makeNullableFlatVector(
      std::vector<std::optional<int64_t>>{
          0, // 00:00:00.000
          3661000, // 01:01:01.000
          43200000, // 12:00:00.000
          86399999, // 23:59:59.999
          std::nullopt,
          36000000, // 10:00:00.000
          72000000, // 20:00:00.000
          1800000, // 00:30:00.000
          7200000, // 02:00:00.000
          64800000 // 18:00:00.000
      },
      TIME());
  testRow({input}, {"time_col"});
}

} // namespace
} // namespace facebook::velox::exec::test
