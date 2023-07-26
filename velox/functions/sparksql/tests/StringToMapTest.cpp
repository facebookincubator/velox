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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions::sparksql::test {
using namespace facebook::velox::test;
namespace {
class StringToMapTest : public SparkFunctionBaseTest {
 protected:
  void testStringToMap(
      const std::vector<StringView>& inputs,
      const std::vector<std::pair<StringView, std::optional<StringView>>>&
          expect) {
    std::vector<VectorPtr> row;
    row.emplace_back(makeFlatVector<StringView>({inputs[0]}));
    std::string expr =
        fmt::format("str_to_map(c0, '{}', '{}')", inputs[1], inputs[2]);
    auto result = evaluate<MapVector>(expr, makeRowVector(row));
    auto expected = makeMapVector<StringView, StringView>({expect});
    assertEqualVectors(result, expected);
  }
};

TEST_F(StringToMapTest, Basics) {
  testStringToMap(
      {"a:1,b:2,c:3", ",", ":"}, {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap({"a: ,b:2", ",", ":"}, {{"a", " "}, {"b", "2"}});
  testStringToMap({"", ",", ":"}, {{"", std::nullopt}});
  testStringToMap({"a", ",", ":"}, {{"a", std::nullopt}});
  testStringToMap(
      {"a=1,b=2,c=3", ",", "="}, {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap({"", ",", "="}, {{"", std::nullopt}});
  testStringToMap(
      {"a::1,b::2,c::3", ",", "c"},
      {{"", "::3"}, {"a::1", std::nullopt}, {"b::2", std::nullopt}});
  testStringToMap(
      {"a:1_b:2_c:3", "_", ":"}, {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  // Same delimiters.
  testStringToMap(
      {"a:1,b:2,c:3", ",", ","},
      {{"a:1", std::nullopt}, {"b:2", std::nullopt}, {"c:3", std::nullopt}});
  testStringToMap(
      {"a:1_b:2_c:3", "_", "_"},
      {{"a:1", std::nullopt}, {"b:2", std::nullopt}, {"c:3", std::nullopt}});
  // Exception for duplicated keys.
  VELOX_ASSERT_THROW(
      testStringToMap({"a:1,b:2,a:3", ",", ":"}, {{"a", "3"}, {"b", "2"}}),
      "Duplicate keys are not allowed: ('a').");
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
