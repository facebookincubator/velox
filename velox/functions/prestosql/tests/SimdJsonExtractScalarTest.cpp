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

#include "simdjson.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/JsonType.h"

namespace facebook::velox::functions::prestosql {

namespace {

const std::string kTestJson = std::string("{\n") + "    \"store\": {\n" +
    "        \"book\": [\n" + "            {\n" +
    "                \"category\": \"reference\",\n" +
    "                \"author\": \"Nigel Rees\",\n" +
    "                \"title\": \"Sayings of the Century\",\n" +
    "                \"price\": 8.95\n" + "            },\n" +
    "            {\n" + "                \"category\": \"fiction\",\n" +
    "                \"author\": \"Evelyn Waugh\",\n" +
    "                \"title\": \"Sword of Honour\",\n" +
    "                \"price\": 12.99\n" + "            },\n" +
    "            {\n" + "                \"category\": \"fiction\",\n" +
    "                \"author\": \"Herman Melville\",\n" +
    "                \"title\": \"Moby Dick\",\n" +
    "                \"isbn\": \"0-553-21311-3\",\n" +
    "                \"price\": 8.99\n" + "            },\n" +
    "            {\n" + "                \"category\": \"fiction\",\n" +
    "                \"author\": \"J. R. R. Tolkien\",\n" +
    "                \"title\": \"The Lord of the Rings\",\n" +
    "                \"isbn\": \"0-395-19395-8\",\n" +
    "                \"price\": 22.99\n" + "            }\n" + "        ],\n" +
    "        \"bicycle\": {\n" + "            \"color\": \"red\",\n" +
    "            \"price\": 19.95\n" + "        }\n" + "    },\n" +
    "    \"expensive\": 10\n" + "}";

class SIMDJsonExtractScalarTest : public functions::test::FunctionBaseTest {
 public:
  std::optional<std::string> json_extract_scalar(
      std::optional<std::string> json,
      std::optional<std::string> path) {
    return evaluateOnce<std::string>(
        "simd_json_extract_scalar(c0, c1)", json, path);
  }

  void evaluateWithJsonType(
      const std::vector<std::optional<StringView>>& json,
      const std::vector<std::optional<StringView>>& path,
      const std::vector<std::optional<StringView>>& expected) {
    auto jsonVector = makeNullableFlatVector<StringView>(json, JSON());
    auto pathVector = makeNullableFlatVector<StringView>(path);
    auto expectedVector = makeNullableFlatVector<StringView>(expected);

    ::facebook::velox::test::assertEqualVectors(
        expectedVector,
        evaluate<SimpleVector<StringView>>(
            "json_extract_scalar(c0, c1)",
            makeRowVector({jsonVector, pathVector})));
  }
};

TEST_F(SIMDJsonExtractScalarTest, simple) {
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

TEST_F(SIMDJsonExtractScalarTest, jsonType) {
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

TEST_F(SIMDJsonExtractScalarTest, utf8) {
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

TEST_F(SIMDJsonExtractScalarTest, invalidPath) {
  EXPECT_THROW(json_extract_scalar(R"([0,1,2])", ""), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"([0,1,2])", "$[]"), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"([0,1,2])", "$[-1]"), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"({"k1":"v1"})", "$k1"), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"({"k1":"v1"})", "$.k1."), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"({"k1":"v1"})", "$.k1]"), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"({"k1":"v1)", "$.k1]"), VeloxUserError);
}

TEST_F(SIMDJsonExtractScalarTest, overflow) {
  // simdjson-difference: simdjson process the input as a string
  EXPECT_EQ(
      json_extract_scalar(
          R"(184467440737095516151844674407370955161518446744073709551615)",
          "$"),
      "184467440737095516151844674407370955161518446744073709551615");
}

TEST_F(SIMDJsonExtractScalarTest, jsonExtractScalar) {
  // simple json path
  EXPECT_EQ(
      json_extract_scalar(R"({"x": {"a" : 1, "b" : 2} })", "$"), std::nullopt);
  EXPECT_EQ(
      json_extract_scalar(R"({"x": {"a" : 1, "b" : 2} })", "$.x"),
      std::nullopt);
  EXPECT_EQ(json_extract_scalar(R"({"x": {"a" : 1, "b" : 2} })", "$.x.a"), "1");
  EXPECT_EQ(
      json_extract_scalar(R"({"x": {"a" : 1, "b" : 2} })", "$.x.c"),
      std::nullopt);
  EXPECT_EQ(
      json_extract_scalar(R"({"x": {"a" : 1, "b" : [2, 3]}})", "$.x.b[1]"),
      "3");
  EXPECT_EQ(json_extract_scalar(R"([1,2,3])", "$[1]"), "2");
  EXPECT_EQ(json_extract_scalar(R"([1,null,3])", "$[1]"), "null");

  // complex json path
  EXPECT_EQ(
      json_extract_scalar(kTestJson, "$.store.book[*].isbn"), std::nullopt);
  EXPECT_EQ(
      json_extract_scalar(kTestJson, "$.store.book[1].author"), "Evelyn Waugh");

  // invalid json
  EXPECT_EQ(json_extract_scalar(R"(INVALID_JSON)", "$"), std::nullopt);
  // invalid JsonPath
  assertUserError(
      [&]() { json_extract_scalar(R"({"":""})", ""); }, "Invalid JSON path: ");
  assertUserError(
      [&]() { json_extract_scalar(R"([1,2,3])", "$...invalid"); },
      "Invalid JSON path: $...invalid");
}

} // namespace

} // namespace facebook::velox::functions::prestosql
