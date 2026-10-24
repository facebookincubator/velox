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
    assertEqualVectors(result, expectVector);
  }

  // Sets the Spark 'spark.sql.mapKeyDedupPolicy' equivalent query config.
  // true corresponds to EXCEPTION, false to LAST_WIN.
  void setThrowExceptionOnDuplicateMapKeys(bool value) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kThrowExceptionOnDuplicateMapKeys,
         value ? "true" : "false"},
    });
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
}

TEST_F(StringToMapTest, duplicateKeysThrow) {
  setThrowExceptionOnDuplicateMapKeys(true);

  VELOX_ASSERT_USER_THROW(
      evaluateStringToMap({"a:1,b:2,a:3", ",", ":"}),
      "Duplicate keys are not allowed: 'a'.");
  VELOX_ASSERT_USER_THROW(
      evaluateStringToMap({":1,:2", ",", ":"}),
      "Duplicate keys are not allowed: ''.");
  // Entries without a key/value delimiter still carry a key, so they take part
  // in deduplication like in Spark.
  VELOX_ASSERT_USER_THROW(
      evaluateStringToMap({"a,a", ",", ":"}),
      "Duplicate keys are not allowed: 'a'.");
  VELOX_ASSERT_USER_THROW(
      evaluateStringToMap({"a:1,a", ",", ":"}),
      "Duplicate keys are not allowed: 'a'.");
}

TEST_F(StringToMapTest, duplicateKeysLastWin) {
  setThrowExceptionOnDuplicateMapKeys(false);

  // The repeated key keeps the position of its first occurrence and takes the
  // last value, matching Spark's ArrayBasedMapBuilder.
  testStringToMap({"a:1,b:2,a:3", ",", ":"}, {{"a", "3"}, {"b", "2"}});
  testStringToMap({"b:1,a:2,b:3,a:4", ",", ":"}, {{"b", "3"}, {"a", "4"}});
  testStringToMap({"a:1,a:2,a:3", ",", ":"}, {{"a", "3"}});
  testStringToMap({":1,:2", ",", ":"}, {{"", "2"}});

  // A later entry without a key/value delimiter overwrites the value with
  // NULL, and a later entry with a value overwrites a NULL.
  testStringToMap({"a:1,a", ",", ":"}, {{"a", std::nullopt}});
  testStringToMap({"a,a:1", ",", ":"}, {{"a", "1"}});
  testStringToMap({"a,a", ",", ":"}, {{"a", std::nullopt}});

  // Distinct keys are unaffected.
  testStringToMap(
      {"a:1,b:2,c:3", ",", ":"}, {{"a", "1"}, {"b", "2"}, {"c", "3"}});
}

TEST_F(StringToMapTest, lastWinPreservesFirstKeyPosition) {
  setThrowExceptionOnDuplicateMapKeys(false);

  // Map equality ignores entry order, so pin the position through map_keys.
  auto result = evaluate<ArrayVector>(
      "map_keys(str_to_map(c0, ',', ':'))",
      makeRowVector({makeFlatVector<StringView>({"b:1,a:2,b:3"})}));
  assertEqualVectors(makeArrayVector<StringView>({{"b", "a"}}), result);
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
