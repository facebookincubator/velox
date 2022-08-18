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

#include "velox/functions/prestosql/tests/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/JsonType.h"

namespace facebook::velox::functions::prestosql {

namespace {

class JsonFunctionsTest : public functions::test::FunctionBaseTest {
 public:
  std::optional<bool> is_json_scalar(std::optional<std::string> json) {
    return evaluateOnce<bool>("is_json_scalar(c0)", json);
  }

  std::optional<int64_t> json_array_length(std::optional<std::string> json) {
    return evaluateOnce<int64_t>("json_array_length(c0)", json);
  }

  template <typename T>
  std::optional<bool> json_array_contains(
      std::optional<std::string> json,
      std::optional<T> value) {
    return evaluateOnce<bool>("json_array_contains(c0, c1)", json, value);
  }

  std::optional<Json> json_extract(
      std::optional<std::string> json,
      std::optional<std::string> value) {
    return evaluateOnce<Json>("json_extract(c0, c1)", json, value);
  }

  std::optional<std::string> json_extract_scalar(
      std::optional<std::string> json,
      std::optional<std::string> path) {
    return evaluateOnce<std::string>("json_extract_scalar(c0, c1)", json, path);
  }

  void evaluateWithJsonType(
      const std::vector<std::optional<StringView>>& json,
      const std::vector<std::optional<StringView>>& path,
      const std::vector<std::optional<StringView>>& expected) {
    auto jsonVector = makeNullableFlatVector<Json>(json, JSON());
    auto pathVector = makeNullableFlatVector<StringView>(path);
    auto expectedVector = makeNullableFlatVector<StringView>(expected);

    ::facebook::velox::test::assertEqualVectors(
        expectedVector,
        evaluate<SimpleVector<StringView>>(
            "json_extract_scalar(c0, c1)",
            makeRowVector({jsonVector, pathVector})));
  }

  void evaluateWithJsonTypeReturnJson(
      const std::vector<std::optional<StringView>>& json,
      const std::vector<std::optional<StringView>>& path,
      const std::vector<std::optional<Json>>& expected) {
    auto jsonVector = makeNullableFlatVector<Json>(json, JSON());
    auto pathVector = makeNullableFlatVector<StringView>(path);
    auto expectedVector = makeNullableFlatVector<Json>(expected);

    ::facebook::velox::test::assertEqualVectors(
        expectedVector,
        evaluate<SimpleVector<Json>>(
            "json_extract_scalar(c0, c1)",
            makeRowVector({jsonVector, pathVector})));
  }
};

TEST_F(JsonFunctionsTest, isJsonScalar) {
  // Scalars.
  EXPECT_EQ(is_json_scalar(R"(1)"), true);
  EXPECT_EQ(is_json_scalar(R"(123456)"), true);
  EXPECT_EQ(is_json_scalar(R"("hello")"), true);
  EXPECT_EQ(is_json_scalar(R"("thefoxjumpedoverthefence")"), true);
  EXPECT_EQ(is_json_scalar(R"(1.1)"), true);
  EXPECT_EQ(is_json_scalar(R"("")"), true);
  EXPECT_EQ(is_json_scalar(R"(true)"), true);

  // Lists and maps
  EXPECT_EQ(is_json_scalar(R"([1,2])"), false);
  EXPECT_EQ(is_json_scalar(R"({"k1":"v1"})"), false);
  EXPECT_EQ(is_json_scalar(R"({"k1":[0,1,2]})"), false);
  EXPECT_EQ(is_json_scalar(R"({"k1":""})"), false);
};

TEST_F(JsonFunctionsTest, jsonExtractScalarSimple) {
  // Scalars.
  EXPECT_EQ(json_extract_scalar(R"(1)", "$"), "1");
  EXPECT_EQ(json_extract_scalar(R"(123456)", "$"), "123456");
  EXPECT_EQ(json_extract_scalar(R"("hello")", "$"), "hello");
  EXPECT_EQ(json_extract_scalar(R"(1.1)", "$"), "1.1");
  EXPECT_EQ(json_extract_scalar(R"("")", "$"), "");
  EXPECT_EQ(json_extract_scalar(R"(true)", "$"), "true");

  // Simple lists.
  EXPECT_EQ(json_extract_scalar(R"([1,2])", "$[0]"), "1");
  EXPECT_EQ(json_extract_scalar(R"([1,2])", "$[1]"), "2");
  EXPECT_EQ(json_extract_scalar(R"([1,2])", "$[2]"), std::nullopt);
  EXPECT_EQ(json_extract_scalar(R"([1,2])", "$[999]"), std::nullopt);

  // Simple maps.
  EXPECT_EQ(json_extract_scalar(R"({"k1":"v1"})", "$.k1"), "v1");
  EXPECT_EQ(json_extract_scalar(R"({"k1":"v1"})", "$.k2"), std::nullopt);
  EXPECT_EQ(json_extract_scalar(R"({"k1":"v1"})", "$.k1.k3"), std::nullopt);
  EXPECT_EQ(json_extract_scalar(R"({"k1":[0,1,2]})", "$.k1"), std::nullopt);
  EXPECT_EQ(json_extract_scalar(R"({"k1":""})", "$.k1"), "");

  // Nested
  EXPECT_EQ(json_extract_scalar(R"({"k1":{"k2": 999}})", "$.k1.k2"), "999");
  EXPECT_EQ(json_extract_scalar(R"({"k1":[1,2,3]})", "$.k1[0]"), "1");
  EXPECT_EQ(json_extract_scalar(R"({"k1":[1,2,3]})", "$.k1[2]"), "3");
  EXPECT_EQ(json_extract_scalar(R"([{"k1":"v1"}, 2])", "$[0].k1"), "v1");
  EXPECT_EQ(json_extract_scalar(R"([{"k1":"v1"}, 2])", "$[1]"), "2");
  EXPECT_EQ(
      json_extract_scalar(
          R"([{"k1":[{"k2": ["v1", "v2"]}]}])", "$[0].k1[0].k2[1]"),
      "v2");
}

TEST_F(JsonFunctionsTest, jsonExtractScalarJsonType) {
  // Scalars.
  evaluateWithJsonType(
      {R"(1)"_sv,
       R"(123456)"_sv,
       R"("hello")"_sv,
       R"(1.1)"_sv,
       R"("")"_sv,
       R"(true)"},
      {"$"_sv, "$"_sv, "$"_sv, "$"_sv, "$"_sv, "$"_sv},
      {"1"_sv, "123456"_sv, "hello"_sv, "1.1"_sv, ""_sv, "true"_sv});

  // Simple lists.
  evaluateWithJsonType(
      {R"([1,2])"_sv, R"([1,2])"_sv, R"([1,2])"_sv, R"([1,2])"_sv},
      {"$[0]"_sv, "$[1]"_sv, "$[2]"_sv, "$[999]"_sv},
      {"1"_sv, "2"_sv, std::nullopt, std::nullopt});

  // Simple maps.
  evaluateWithJsonType(
      {R"({"k1":"v1"})"_sv,
       R"({"k1":"v1"})"_sv,
       R"({"k1":"v1"})"_sv,
       R"({"k1":[0,1,2]})"_sv,
       R"({"k1":""})"_sv},
      {"$.k1"_sv, "$.k2"_sv, "$.k1.k3"_sv, "$.k1"_sv, "$.k1"_sv},
      {"v1"_sv, std::nullopt, std::nullopt, std::nullopt, ""_sv});

  // Nested
  evaluateWithJsonType(
      {R"({"k1":{"k2": 999}})"_sv,
       R"({"k1":[1,2,3]})"_sv,
       R"({"k1":[1,2,3]})"_sv,
       R"([{"k1":"v1"}, 2])"_sv,
       R"([{"k1":"v1"}, 2])"_sv,
       R"([{"k1":[{"k2": ["v1", "v2"]}]}])"_sv},
      {"$.k1.k2"_sv,
       "$.k1[0]"_sv,
       "$.k1[2]"_sv,
       "$[0].k1"_sv,
       "$[1]"_sv,
       "$[0].k1[0].k2[1]"_sv},
      {"999"_sv, "1"_sv, "3"_sv, "v1"_sv, "2"_sv, "v2"_sv});
}

TEST_F(JsonFunctionsTest, jsonExtractScalarUtf8) {
  EXPECT_EQ(
      json_extract_scalar(R"({"k1":"I \u2665 UTF-8"})", "$.k1"),
      "I \u2665 UTF-8");
  EXPECT_EQ(
      json_extract_scalar("{\"k1\":\"I \u2665 UTF-8\"}", "$.k1"),
      "I \u2665 UTF-8");

  EXPECT_EQ(
      json_extract_scalar(
          "{\"k1\":\"I \U0001D11E playing in G-clef\"}", "$.k1"),
      "I \U0001D11E playing in G-clef");
}

TEST_F(JsonFunctionsTest, jsonExtractScalarInvalidPath) {
  EXPECT_THROW(json_extract_scalar(R"([0,1,2])", ""), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"([0,1,2])", "$[]"), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"([0,1,2])", "$[-1]"), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"({"k1":"v1"})", "$k1"), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"({"k1":"v1"})", "$.k1."), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"({"k1":"v1"})", "$.k1]"), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"({"k1":"v1)", "$.k1]"), VeloxUserError);
}

// TODO: Folly tries to convert scalar integers, and in case they are large
// enough it overflows and throws conversion error. In this case, we do out best
// and return NULL, but in Presto java the large integer is returned as-is as a
// string.
TEST_F(JsonFunctionsTest, jsonExtractScalarOverflow) {
  EXPECT_EQ(
      json_extract_scalar(
          R"(184467440737095516151844674407370955161518446744073709551615)",
          "$"),
      std::nullopt);
}

TEST_F(JsonFunctionsTest, jsonExtractSimple) {
  EXPECT_EQ(json_extract(R"(1)", "$"), R"(1)");
  EXPECT_EQ(json_extract(R"(123456)", "$"), R"(123456)");
  EXPECT_EQ(json_extract(R"("hello")", "$"), R"("hello")");
  EXPECT_EQ(json_extract(R"(1.1)", "$"), R"(1.1)");
  EXPECT_EQ(json_extract(R"("")", "$"), R"("")");
  EXPECT_EQ(json_extract(R"(true)", "$"), R"(true)");
  EXPECT_EQ(json_extract(R"({"k1":"v1"})", "$.k1"), R"("v1")");
  EXPECT_EQ(json_extract(R"([1,2])", "$[0]"), R"(1)");
  EXPECT_EQ(json_extract(R"({"k1":{"k2": 999}})", "$.k1.k2"), R"(999)");
  EXPECT_EQ(json_extract(R"({"k1":[1,2,3]})", "$.k1[0]"), R"(1)");
  EXPECT_EQ(json_extract(R"([{"k1":"v1"}, 2])", "$[0].k1"), R"("v1")");
  EXPECT_EQ(
      json_extract(R"([{"k1":[{"k2": ["v1", "v2"]}]}])", "$[0].k1[0].k2[1]"),
      R"("v2")");

  EXPECT_EQ(json_extract(R"([1,2])", "$[2]"), std::nullopt);
  EXPECT_EQ(json_extract(R"({"k1":"v1"})", "$.k2"), std::nullopt);
  EXPECT_EQ(json_extract(R"({"k1":"v1"})", "$.k1.k3"), std::nullopt);
  EXPECT_EQ(json_extract(R"([1,2,[1,2,3]])", "$[2]"), R"([1,2,3])");
  EXPECT_EQ(json_extract(R"({"k1":[0,1,2]})", "$.k1"), R"([0,1,2])");
  EXPECT_EQ(
      json_extract(R"({"k1":[0,1,2], "k2":{"k3":"v3"}})", "$.k2"),
      R"({"k3":"v3"})");
  EXPECT_EQ(json_extract(R"({"k1":""})", "$.k1"), R"("")");
  EXPECT_EQ(json_extract(R"([{"k1":"v1"}, [1,2]])", "$[1]"), R"([1,2])");
  EXPECT_EQ(
      json_extract(R"([{"k1":{"k2": ["v1", "v2"]}}])", "$[0].k1"),
      R"({"k2":["v1","v2"]})");
  EXPECT_EQ(
      json_extract(R"({"k1": ["k3", {"k2": "v1"}]})", "$.k1[1]"),
      R"({"k2":"v1"})");
}

TEST_F(JsonFunctionsTest, jsonExtractInvalidPath) {
  EXPECT_THROW(json_extract(R"([[0,1,2],1,2])", ""), VeloxUserError);
  EXPECT_THROW(json_extract(R"([[0,1,2],1,2])", "$[]"), VeloxUserError);
  EXPECT_THROW(json_extract(R"([[0,1,2],1,2])", "$[-1]"), VeloxUserError);
  EXPECT_THROW(
      json_extract(R"({"k1":"v1", "k2":[1,2,3]})", "$k2"), VeloxUserError);
  EXPECT_THROW(
      json_extract(R"({"k1":"v1", "k2":[1,2,3]})", "$.k2."), VeloxUserError);
  EXPECT_THROW(
      json_extract(R"({"k1":"v1", "k2":[1,2,3]})", "$.k2]"), VeloxUserError);
  EXPECT_THROW(
      json_extract(R"({"k1":"v1",  "k2":[1,2,3)", "$.k2]"), VeloxUserError);
}

TEST_F(JsonFunctionsTest, jsonArrayLength) {
  EXPECT_EQ(json_array_length(R"([])"), 0);
  EXPECT_EQ(json_array_length(R"([1])"), 1);
  EXPECT_EQ(json_array_length(R"([1, 2, 3])"), 3);
  EXPECT_EQ(
      json_array_length(
          R"([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])"),
      20);

  EXPECT_EQ(json_array_length(R"(1)"), std::nullopt);
  EXPECT_EQ(json_array_length(R"("hello")"), std::nullopt);
  EXPECT_EQ(json_array_length(R"("")"), std::nullopt);
  EXPECT_EQ(json_array_length(R"(true)"), std::nullopt);
  EXPECT_EQ(json_array_length(R"({"k1":"v1"})"), std::nullopt);
  EXPECT_EQ(json_array_length(R"({"k1":[0,1,2]})"), std::nullopt);
  EXPECT_EQ(json_array_length(R"({"k1":[0,1,2], "k2":"v1"})"), std::nullopt);
}

TEST_F(JsonFunctionsTest, jsonArrayContainsBool) {
  EXPECT_EQ(json_array_contains<bool>(R"([])", true), false);
  EXPECT_EQ(json_array_contains<bool>(R"([1, 2, 3])", false), false);
  EXPECT_EQ(json_array_contains<bool>(R"([1.2, 2.3, 3.4])", true), false);
  EXPECT_EQ(
      json_array_contains<bool>(R"(["hello", "presto", "world"])", false),
      false);
  EXPECT_EQ(json_array_contains<bool>(R"(1)", true), std::nullopt);
  EXPECT_EQ(
      json_array_contains<bool>(R"("thefoxjumpedoverthefence")", false),
      std::nullopt);
  EXPECT_EQ(json_array_contains<bool>(R"("")", false), std::nullopt);
  EXPECT_EQ(json_array_contains<bool>(R"(true)", true), std::nullopt);
  EXPECT_EQ(
      json_array_contains<bool>(R"({"k1":[0,1,2], "k2":"v1"})", true),
      std::nullopt);

  EXPECT_EQ(json_array_contains<bool>(R"([true, false])", true), true);
  EXPECT_EQ(json_array_contains<bool>(R"([true, true])", false), false);
  EXPECT_EQ(
      json_array_contains<bool>(R"([123, 123.456, true, "abc"])", true), true);
  EXPECT_EQ(
      json_array_contains<bool>(R"([123, 123.456, true, "abc"])", false),
      false);
  EXPECT_EQ(
      json_array_contains<bool>(
          R"([false, false, false, false, false, false, false,
false, false, false, false, false, false, true, false, false, false, false])",
          true),
      true);
  EXPECT_EQ(
      json_array_contains<bool>(
          R"([true, true, true, true, true, true, true,
true, true, true, true, true, true, true, true, true, true, true])",
          false),
      false);
}

TEST_F(JsonFunctionsTest, jsonArrayContainsInt) {
  EXPECT_EQ(json_array_contains<int64_t>(R"([])", 0), false);
  EXPECT_EQ(json_array_contains<int64_t>(R"([1.2, 2.3, 3.4])", 2), false);
  EXPECT_EQ(json_array_contains<int64_t>(R"([1.2, 2.0, 3.4])", 2), false);
  EXPECT_EQ(
      json_array_contains<int64_t>(R"(["hello", "presto", "world"])", 2),
      false);
  EXPECT_EQ(
      json_array_contains<int64_t>(R"([false, false, false])", 17), false);
  EXPECT_EQ(json_array_contains<int64_t>(R"(1)", 1), std::nullopt);
  EXPECT_EQ(
      json_array_contains<int64_t>(R"("thefoxjumpedoverthefence")", 1),
      std::nullopt);
  EXPECT_EQ(json_array_contains<int64_t>(R"("")", 1), std::nullopt);
  EXPECT_EQ(json_array_contains<int64_t>(R"(true)", 1), std::nullopt);
  EXPECT_EQ(
      json_array_contains<int64_t>(R"({"k1":[0,1,2], "k2":"v1"})", 1),
      std::nullopt);

  EXPECT_EQ(json_array_contains<int64_t>(R"([1, 2, 3])", 1), true);
  EXPECT_EQ(json_array_contains<int64_t>(R"([1, 2, 3])", 4), false);
  EXPECT_EQ(
      json_array_contains<int64_t>(R"([123, 123.456, true, "abc"])", 123),
      true);
  EXPECT_EQ(
      json_array_contains<int64_t>(R"([123, 123.456, true, "abc"])", 456),
      false);
  EXPECT_EQ(
      json_array_contains<int64_t>(
          R"([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])",
          17),
      true);
  EXPECT_EQ(
      json_array_contains<int64_t>(
          R"([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])",
          23),
      false);
}

TEST_F(JsonFunctionsTest, jsonArrayContainsDouble) {
  EXPECT_EQ(json_array_contains<double>(R"([])", 2.3), false);
  EXPECT_EQ(json_array_contains<double>(R"([1, 2, 3])", 2.3), false);
  EXPECT_EQ(json_array_contains<double>(R"([1, 2, 3])", 2.0), false);
  EXPECT_EQ(
      json_array_contains<double>(R"(["hello", "presto", "world"])", 2.3),
      false);
  EXPECT_EQ(
      json_array_contains<double>(R"([false, false, false])", 2.3), false);
  EXPECT_EQ(json_array_contains<double>(R"(1)", 2.3), std::nullopt);
  EXPECT_EQ(
      json_array_contains<double>(R"("thefoxjumpedoverthefence")", 2.3),
      std::nullopt);
  EXPECT_EQ(json_array_contains<double>(R"("")", 2.3), std::nullopt);
  EXPECT_EQ(json_array_contains<double>(R"(true)", 2.3), std::nullopt);
  EXPECT_EQ(
      json_array_contains<double>(R"({"k1":[0,1,2], "k2":"v1"})", 2.3),
      std::nullopt);

  EXPECT_EQ(json_array_contains<double>(R"([1.2, 2.3, 3.4])", 2.3), true);
  EXPECT_EQ(json_array_contains<double>(R"([1.2, 2.3, 3.4])", 2.4), false);
  EXPECT_EQ(
      json_array_contains<double>(R"([123, 123.456, true, "abc"])", 123.456),
      true);
  EXPECT_EQ(
      json_array_contains<double>(R"([123, 123.456, true, "abc"])", 456.789),
      false);
  EXPECT_EQ(
      json_array_contains<double>(
          R"([1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5])",
          4.5),
      true);
  EXPECT_EQ(
      json_array_contains<double>(
          R"([1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5])",
          4.3),
      false);
}

TEST_F(JsonFunctionsTest, jsonArrayContainsString) {
  EXPECT_EQ(json_array_contains<std::string>(R"([])", ""), false);
  EXPECT_EQ(json_array_contains<std::string>(R"([1, 2, 3])", "1"), false);
  EXPECT_EQ(
      json_array_contains<std::string>(R"([1.2, 2.3, 3.4])", "2.3"), false);
  EXPECT_EQ(
      json_array_contains<std::string>(R"([true, false])", R"("true")"), false);
  EXPECT_EQ(json_array_contains<std::string>(R"(1)", "1"), std::nullopt);
  EXPECT_EQ(
      json_array_contains<std::string>(R"("thefoxjumpedoverthefence")", "1"),
      std::nullopt);
  EXPECT_EQ(json_array_contains<std::string>(R"("")", "1"), std::nullopt);
  EXPECT_EQ(json_array_contains<std::string>(R"(true)", "1"), std::nullopt);
  EXPECT_EQ(
      json_array_contains<std::string>(R"({"k1":[0,1,2], "k2":"v1"})", "1"),
      std::nullopt);

  EXPECT_EQ(
      json_array_contains<std::string>(
          R"(["hello", "presto", "world"])", "presto"),
      true);
  EXPECT_EQ(
      json_array_contains<std::string>(
          R"(["hello", "presto", "world"])", "nation"),
      false);
  EXPECT_EQ(
      json_array_contains<std::string>(R"([123, 123.456, true, "abc"])", "abc"),
      true);
  EXPECT_EQ(
      json_array_contains<std::string>(R"([123, 123.456, true, "abc"])", "def"),
      false);
  EXPECT_EQ(
      json_array_contains<std::string>(
          R"(["hello", "presto", "world", "hello", "presto", "world", "hello", "presto", "world", "hello",
"presto", "world", "hello", "presto", "world", "hello", "presto", "world"])",
          "hello"),
      true);
  EXPECT_EQ(
      json_array_contains<std::string>(
          R"(["hello", "presto", "world", "hello", "presto", "world", "hello", "presto", "world", "hello",
"presto", "world", "hello", "presto", "world", "hello", "presto", "world"])",
          "hola"),
      false);
  EXPECT_EQ(
      json_array_contains<std::string>(
          R"(["hello", "presto", "world", 1, 2, 3, true, false, 1.2, 2.3, {"k1":[0,1,2], "k2":"v1"}])",
          "world"),
      true);
  EXPECT_EQ(
      json_array_contains<std::string>(
          R"(["the fox jumped over the fence", "hello presto world"])",
          "hello velox world"),
      false);
  EXPECT_EQ(
      json_array_contains<std::string>(
          R"(["the fox jumped over the fence", "hello presto world"])",
          "the fox jumped over the fence"),
      true);
}

} // namespace

} // namespace facebook::velox::functions::prestosql
