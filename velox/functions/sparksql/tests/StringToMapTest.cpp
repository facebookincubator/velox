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

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {
class StringToMapTest : public SparkFunctionBaseTest {
 protected:
  VectorPtr evaluateStringToMap(const std::vector<StringView>& inputs) {
    const std::string expr =
        fmt::format("str_to_map(c0, '{}', '{}')", inputs[1], inputs[2]);
    return evaluate<MapVector>(
        expr, makeRowVector({makeFlatVector<StringView>({inputs[0]})}));
  }

  void testStringToMap(
      const std::vector<StringView>& inputs,
      const std::vector<std::pair<StringView, std::optional<StringView>>>&
          expect) {
    auto result = evaluateStringToMap(inputs);
    auto expectVector = makeMapVector<StringView, StringView>({expect});
    assertEqualVectors(expectVector, result);
  }

  VectorPtr evaluateStringToMapLastWin(const std::vector<StringView>& inputs) {
    const std::string expr = fmt::format(
        "str_to_map_last_win(c0, '{}', '{}')", inputs[1], inputs[2]);
    return evaluate<MapVector>(
        expr, makeRowVector({makeFlatVector<StringView>({inputs[0]})}));
  }

  void testStringToMapLastWin(
      const std::vector<StringView>& inputs,
      const std::vector<std::pair<StringView, std::optional<StringView>>>&
          expect) {
    auto result = evaluateStringToMapLastWin(inputs);
    auto expectVector = makeMapVector<StringView, StringView>({expect});
    assertEqualVectors(expectVector, result);
  }
};

TEST_F(StringToMapTest, basic) {
  testStringToMap(
      {"a:1,b:2,c:3", ",", ":"}, {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap({"a: ,b:2", ",", ":"}, {{"a", " "}, {"b", "2"}});
  testStringToMap({"a:,b:2", ",", ":"}, {{"a", ""}, {"b", "2"}});
  testStringToMap({"", ",", ":"}, {{"", std::nullopt}});
  testStringToMap({"a", ",", ":"}, {{"a", std::nullopt}});
  testStringToMap(
      {"a=1,b=2,c=3", ",", "="}, {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap({"", ",", "="}, {{"", std::nullopt}});
  testStringToMap(
      {"a::1,b::2,c::3", ",", "c"},
      {{"a::1", std::nullopt}, {"b::2", std::nullopt}, {"", "::3"}});
  testStringToMap(
      {"a:1_b:2_c:3", "_", ":"}, {{"a", "1"}, {"b", "2"}, {"c", "3"}});

  // Same delimiters.
  testStringToMap(
      {"a:1,b:2,c:3", ",", ","},
      {{"a:1", std::nullopt}, {"b:2", std::nullopt}, {"c:3", std::nullopt}});
  testStringToMap(
      {"a:1_b:2_c:3", "_", "_"},
      {{"a:1", std::nullopt}, {"b:2", std::nullopt}, {"c:3", std::nullopt}});

  // Exception for illegal delimiters.
  // Empty string is used.
  VELOX_ASSERT_THROW(
      evaluateStringToMap({"a:1,b:2", "", ":"}),
      "entryDelimiter's size should be 1.");
  VELOX_ASSERT_THROW(
      evaluateStringToMap({"a:1,b:2", ",", ""}),
      "keyValueDelimiter's size should be 1.");
  // Delimiter's length > 1.
  VELOX_ASSERT_THROW(
      evaluateStringToMap({"a:1,b:2", ";;", ":"}),
      "entryDelimiter's size should be 1.");
  VELOX_ASSERT_THROW(
      evaluateStringToMap({"a:1,b:2", ",", "::"}),
      "keyValueDelimiter's size should be 1.");
  // Unicode character is used.
  VELOX_ASSERT_THROW(
      evaluateStringToMap({"a:1,b:2", "å", ":"}),
      "entryDelimiter's size should be 1.");
  VELOX_ASSERT_THROW(
      evaluateStringToMap({"a:1,b:2", ",", "æ"}),
      "keyValueDelimiter's size should be 1.");

  // Exception for duplicated keys.
  VELOX_ASSERT_THROW(
      evaluateStringToMap({"a:1,b:2,a:3", ",", ":"}),
      "Duplicate keys are not allowed: 'a'.");
  VELOX_ASSERT_THROW(
      evaluateStringToMap({":1,:2", ",", ":"}),
      "Duplicate keys are not allowed: ''.");
}

TEST_F(StringToMapTest, basicLastWinWithoutDuplicateKey) {
  testStringToMapLastWin(
      {"a:1,b:2,c:3", ",", ":"}, {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMapLastWin({"a: ,b:2", ",", ":"}, {{"a", " "}, {"b", "2"}});
  testStringToMapLastWin({"a:,b:2", ",", ":"}, {{"a", ""}, {"b", "2"}});
  testStringToMapLastWin({"", ",", ":"}, {{"", std::nullopt}});
  testStringToMapLastWin({"a", ",", ":"}, {{"a", std::nullopt}});
  testStringToMapLastWin(
      {"a=1,b=2,c=3", ",", "="}, {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMapLastWin({"", ",", "="}, {{"", std::nullopt}});
  testStringToMapLastWin(
      {"a::1,b::2,c::3", ",", "c"},
      {{"a::1", std::nullopt}, {"b::2", std::nullopt}, {"", "::3"}});
  testStringToMapLastWin(
      {"a:1_b:2_c:3", "_", ":"}, {{"a", "1"}, {"b", "2"}, {"c", "3"}});

  // Same delimiters.
  testStringToMapLastWin(
      {"a:1,b:2,c:3", ",", ","},
      {{"a:1", std::nullopt}, {"b:2", std::nullopt}, {"c:3", std::nullopt}});
  testStringToMapLastWin(
      {"a:1_b:2_c:3", "_", "_"},
      {{"a:1", std::nullopt}, {"b:2", std::nullopt}, {"c:3", std::nullopt}});

  testStringToMapLastWin(
      {"a:1;b:2;c:333;jack:1000;b:10;c:20", ";", ":"},
      {{"a", "1"}, {"b", "10"}, {"c", "20"}, {"jack", "1000"}});

  testStringToMapLastWin(
      {"a:1;b:2;c:333;jack:1000;b:10;c:20;", ";", ":"},
      {{"", std::nullopt},
       {"a", "1"},
       {"b", "10"},
       {"c", "20"},
       {"jack", "1000"}});

  testStringToMapLastWin(
      {"a:::1;b:2;c:333;jack:10:00;b:10;c:20;:;;", ";", ":"},
      {{"", std::nullopt},
       {"a", "::1"},
       {"b", "10"},
       {"c", "20"},
       {"jack", "10:00"}});

  testStringToMapLastWin(
      {"a:::1;b:2;c:333;jack:10:00;b:10;c:20;:;;", ";", ":"},
      {{"", std::nullopt},
       {"a", "::1"},
       {"b", "10"},
       {"c", "20"},
       {"jack", "10:00"}});

  testStringToMapLastWin(
      {"a:::1;b:2;c:333;jack:10:00;b:10;c:20;:", ";", ":"},
      {{"", ""}, {"a", "::1"}, {"b", "10"}, {"c", "20"}, {"jack", "10:00"}});

  testStringToMapLastWin({"", ";", ":"}, {{"", std::nullopt}});

  testStringToMapLastWin({";", ";", ":"}, {{"", std::nullopt}});

  testStringToMapLastWin({":", ";", ":"}, {{"", ""}});

  testStringToMapLastWin({"::::", ";", ":"}, {{"", ":::"}});

  testStringToMapLastWin(
      {"jack;rose", ";", ":"},
      {{"jack", std::nullopt}, {"rose", std::nullopt}});
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
