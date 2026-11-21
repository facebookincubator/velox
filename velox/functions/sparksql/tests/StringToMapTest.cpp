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
#include "folly/ScopeGuard.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {
class StringToMapTest : public SparkFunctionBaseTest {
 protected:
  VectorPtr evaluateStringToMap(const std::vector<StringView>& inputs) {
    std::string expr;
    if (inputs.size() == 1) {
      expr = "str_to_map(c0)";
    } else if (inputs.size() == 2) {
      expr = fmt::format("str_to_map(c0, '{}')", inputs[1]);
    } else {
      expr = fmt::format("str_to_map(c0, '{}', '{}')", inputs[1], inputs[2]);
    }
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
  testStringToMap(
      {"a:1,b:2,", ",", ":"},
      {{"a", "1"}, {"b", "2"}, {"", std::nullopt}});
}

TEST_F(StringToMapTest, defaultDelimiters) {
  // 1 argument (defaults to ',' and ':')
  testStringToMap({"a:1,b:2,c:3"}, {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap({"a: ,b:2"}, {{"a", " "}, {"b", "2"}});
  testStringToMap({"a:,b:2"}, {{"a", ""}, {"b", "2"}});
  testStringToMap({""}, {{"", std::nullopt}});
  testStringToMap({"a"}, {{"a", std::nullopt}});

  // 2 arguments (keyValueDelimiter defaults to ':')
  testStringToMap(
      {"a:1_b:2_c:3", "_"}, {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap({"a:1,b:2", "_"}, {{"a", "1,b:2"}});
  testStringToMap({"a=1,b=2", "_"}, {{"a=1,b=2", std::nullopt}});
  testStringToMap({"", "_"}, {{"", std::nullopt}});
}

TEST_F(StringToMapTest, multiCharDelimiters) {
  testStringToMap(
      {"a::1__b::2__c::3", "__", "::"},
      {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap(
      {"a::1__b:::2__c::::3", "__", "::"},
      {{"a", "1"}, {"b", ":2"}, {"c", "::3"}});
  testStringToMap(
      {"a->1;b->2;c->3", ";", "->"},
      {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap(
      {"a==1&&b==2&&c==3", "&&", "=="},
      {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap(
      {"a::1,b::2,c::3", ",", "::"},
      {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap(
      {"a:1,,b:2,,c:3", ",,", ":"},
      {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap(
      {"a:1,b:2,c:3", ",,", "::"},
      {{"a:1,b:2,c:3", std::nullopt}});

  // Multi-byte Unicode delimiters.
  testStringToMap(
      {"a::1\xe2\x98\xba\x62::2\xe2\x98\xba\x63::3", "\xe2\x98\xba", "::"},
      {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap(
      {"a==1\xe2\x9e\x94\x62==2\xe2\x9e\x94\x63==3", "\xe2\x9e\x94", "=="},
      {{"a", "1"}, {"b", "2"}, {"c", "3"}});
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

  // Duplicate keys with default and multi-character delimiters.
  VELOX_ASSERT_USER_THROW(
      evaluateStringToMap({"a:1,b:2,a:3"}),
      "Duplicate keys are not allowed: 'a'.");
  VELOX_ASSERT_USER_THROW(
      evaluateStringToMap({"a::1__b::2__a::3", "__", "::"}),
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

  // Multi-character delimiters with last win.
  testStringToMap(
      {"a::1__b::2__a::3", "__", "::"}, {{"a", "3"}, {"b", "2"}});
  testStringToMap(
      {"a:1,b:2,a:3"}, {{"a", "3"}, {"b", "2"}});
}

TEST_F(StringToMapTest, lastWinPreservesFirstKeyPosition) {
  setThrowExceptionOnDuplicateMapKeys(false);

  // Map equality ignores entry order, so pin the position through map_keys.
  auto result = evaluate<ArrayVector>(
      "map_keys(str_to_map(c0, ',', ':'))",
      makeRowVector({makeFlatVector<StringView>({"b:1,a:2,b:3"})}));
  assertEqualVectors(makeArrayVector<StringView>({{"b", "a"}}), result);
}

TEST_F(StringToMapTest, regexDelimiters) {
  // Escaped regex metacharacters.
  testStringToMap({"a=1|b=2", "\\|", "="}, {{"a", "1"}, {"b", "2"}});
  testStringToMap({"a.1,b.2", ",", "\\."}, {{"a", "1"}, {"b", "2"}});
  testStringToMap({"a$1|b$2", "\\|", "\\$"}, {{"a", "1"}, {"b", "2"}});
  testStringToMap({"a+1?b+2", "\\?", "\\+"}, {{"a", "1"}, {"b", "2"}});

  // Octal ASCII escape delimiter.
  testStringToMap({"a$1,b$2", ",", "\\044"}, {{"a", "1"}, {"b", "2"}});

  // Regular expression patterns (character classes, quantifiers, alternation).
  testStringToMap(
      {"a:1; b:2, c:3", "[;,]\\s*", ":"},
      {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap(
      {"a:=1|b==2|c:3", "\\|", "[:=]+"},
      {{"a", "1"}, {"b", "2"}, {"c", "3"}});
  testStringToMap(
      {"k1:v1__99__k2:v2", "__\\d+__", ":"},
      {{"k1", "v1"}, {"k2", "v2"}});

  // keyValueDelimiter only splits at the first match (limit = 2).
  testStringToMap(
      {"a::1::2,b:3", ",", ":+"}, {{"a", "1::2"}, {"b", "3"}});

  // Zero-width regex match.
  testStringToMap({"a", ",", "z|"}, {{"a", ""}});

  // Invalid regex patterns throw user errors.
  VELOX_ASSERT_USER_THROW(
      evaluateStringToMap({"a:1,b:2", "[", ":"}),
      "invalid regular expression");
  VELOX_ASSERT_USER_THROW(
      evaluateStringToMap({"a:1,b:2", ",", "["}),
      "invalid regular expression");
  VELOX_ASSERT_USER_THROW(
      evaluateStringToMap({"a:1,b:2", "\\", ":"}),
      "invalid regular expression");
}

TEST_F(StringToMapTest, emptyDelimiters) {
  // Aligns with Spark 3.4+/3.5 UTF8String.split when delimiters are empty.
  testStringToMap(
      {"a:1,b", "", ":"},
      {{"a", std::nullopt},
       {"", ""},
       {"1", std::nullopt},
       {",", std::nullopt},
       {"b", std::nullopt}});
  testStringToMap({"", "", ":"}, {{"", std::nullopt}});
  testStringToMap({"ab,c", ",", ""}, {{"a", "b"}, {"c", std::nullopt}});
  testStringToMap({"abc,d", ",", ""}, {{"a", "b"}, {"d", std::nullopt}});
}

TEST_F(StringToMapTest, nonConstantDelimiters) {
  auto input = makeFlatVector<StringView>(
      {"a:1,b:2", "a::1__b::2", "a=1|b==2", "a=1_b=2"});
  auto entryDelimiters =
      makeFlatVector<StringView>({",", "__", "\\|", "_"});
  auto kvDelimiters = makeFlatVector<StringView>({":", "::", "=+", "="});

  auto result = evaluate<MapVector>(
      "str_to_map(c0, c1, c2)",
      makeRowVector({input, entryDelimiters, kvDelimiters}));
  auto expected = makeMapVector<StringView, StringView>(
      {{{"a", "1"}, {"b", "2"}},
       {{"a", "1"}, {"b", "2"}},
       {{"a", "1"}, {"b", "2"}},
       {{"a", "1"}, {"b", "2"}}});
  assertEqualVectors(expected, result);

  auto twoArgResult = evaluate<MapVector>(
      "str_to_map(c0, c1)",
      makeRowVector(
          {makeFlatVector<StringView>({"a:1,b:2", "a:1_b:2"}),
           makeFlatVector<StringView>({",", "_"})}));
  auto twoArgExpected = makeMapVector<StringView, StringView>(
      {{{"a", "1"}, {"b", "2"}}, {{"a", "1"}, {"b", "2"}}});
  assertEqualVectors(twoArgExpected, twoArgResult);
}

TEST_F(StringToMapTest, bypassRegexCacheLimit) {
  auto guard =
      folly::makeGuard([&]() { queryCtx_->testingOverrideConfigUnsafe({}); });
  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kExprMaxCompiledRegexes, "5"}});

  auto input = makeFlatVector<StringView>(
      {"a--1@@b--2",
       "c==3##d==4",
       "e__5~~f__6",
       "g::7%%h::8",
       "i;;9&&j;;10",
       "k,,11!!l,,12"});
  auto entryDelimiters =
      makeFlatVector<StringView>({"@@", "##", "~~", "%%", "&&", "!!"});
  auto kvDelimiters =
      makeFlatVector<StringView>({"--", "==", "__", "::", ";;", ",,"});
  auto expected = makeMapVector<StringView, StringView>(
      {{{"a", "1"}, {"b", "2"}},
       {{"c", "3"}, {"d", "4"}},
       {{"e", "5"}, {"f", "6"}},
       {{"g", "7"}, {"h", "8"}},
       {{"i", "9"}, {"j", "10"}},
       {{"k", "11"}, {"l", "12"}}});

  // Plain-string fast paths skip regex compilation and bypass the cache limit.
  VectorPtr result;
  EXPECT_NO_THROW(
      result = evaluate<MapVector>(
          "str_to_map(c0, c1, c2)",
          makeRowVector({input, entryDelimiters, kvDelimiters})));
  assertEqualVectors(expected, result);

  // Distinct regex patterns exceeding the limit still throw.
  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kExprMaxCompiledRegexes, "2"}});
  VELOX_ASSERT_USER_THROW(
      evaluate<MapVector>(
          "str_to_map(c0, c1, c2)",
          makeRowVector(
              {makeFlatVector<StringView>({"a:1,b:2", "c:3;d:4", "e:5|f:6"}),
               makeFlatVector<StringView>({"[,]+", "[;]+", "[|]+"}),
               makeFlatVector<StringView>({":", ":", ":"})})),
      "Max number of regex reached");
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
