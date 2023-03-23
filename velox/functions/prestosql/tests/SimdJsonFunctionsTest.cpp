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
#include "velox/functions/FunctionRegistry.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/JsonType.h"

namespace facebook::velox::functions::prestosql {
namespace {

class SIMDJsonFunctionsTest : public functions::test::FunctionBaseTest {
 protected:
  VectorPtr makeJsonVector(std::optional<std::string> json) {
    std::optional<StringView> s = json.has_value()
        ? std::make_optional(StringView(json.value()))
        : std::nullopt;
    return makeNullableFlatVector<StringView>({s}, JSON());
  }

  std::optional<bool> isJsonScalar(std::optional<std::string> json) {
    return evaluateOnce<bool>(
        "is_json_scalar(c0)", makeRowVector({makeJsonVector(json)}));
  }
  template <typename T>
  std::optional<bool> jsonArrayContains(
      std::optional<std::string> json,
      std::optional<T> value) {
    return evaluateOnce<bool>(
        "json_array_contains(c0, c1)",
        makeRowVector(
            {makeJsonVector(json), makeNullableFlatVector<T>({value})}));
  }

  std::optional<std::string> jsonParse(std::optional<std::string> json) {
    return evaluateOnce<std::string>("json_parse(c0)", json);
  }

  std::optional<int64_t> jsonArrayLength(std::optional<std::string> json) {
    return evaluateOnce<int64_t>(
        "json_array_length(c0)", makeRowVector({makeJsonVector(json)}));
  }

  std::optional<int64_t> jsonSize(
      std::optional<std::string> json,
      const std::string& path) {
    return evaluateOnce<int64_t>(
        "json_size(c0, c1)",
        makeRowVector(
            {makeJsonVector(json), makeFlatVector<std::string>({path})}));
  }
};

TEST_F(SIMDJsonFunctionsTest, jsonArrayContainsBool) {
  // Velox test case
  EXPECT_EQ(jsonArrayContains<bool>(R"([])", true), false);
  EXPECT_EQ(jsonArrayContains<bool>(R"([1, 2, 3])", false), false);
  EXPECT_EQ(jsonArrayContains<bool>(R"([1.2, 2.3, 3.4])", true), false);
  EXPECT_EQ(
      jsonArrayContains<bool>(R"(["hello", "presto", "world"])", false), false);
  EXPECT_EQ(jsonArrayContains<bool>(R"(1)", true), std::nullopt);
  EXPECT_EQ(
      jsonArrayContains<bool>(R"("thefoxjumpedoverthefence")", false),
      std::nullopt);
  EXPECT_EQ(jsonArrayContains<bool>(R"("")", false), std::nullopt);
  EXPECT_EQ(jsonArrayContains<bool>(R"(true)", true), std::nullopt);
  EXPECT_EQ(
      jsonArrayContains<bool>(R"({"k1":[0,1,2], "k2":"v1"})", true),
      std::nullopt);

  EXPECT_EQ(jsonArrayContains<bool>(R"([true, false])", true), true);
  EXPECT_EQ(jsonArrayContains<bool>(R"([true, true])", false), false);
  EXPECT_EQ(
      jsonArrayContains<bool>(R"([123, 123.456, true, "abc"])", true), true);
  EXPECT_EQ(
      jsonArrayContains<bool>(R"([123, 123.456, true, "abc"])", false), false);
  EXPECT_EQ(
      jsonArrayContains<bool>(
          R"([false, false, false, false, false, false, false,
false, false, false, false, false, false, true, false, false, false, false])",
          true),
      true);
  EXPECT_EQ(
      jsonArrayContains<bool>(
          R"([true, true, true, true, true, true, true,
true, true, true, true, true, true, true, true, true, true, true])",
          false),
      false);

  // Presto test case
  EXPECT_EQ(jsonArrayContains<bool>(R"([])", true), false);
  EXPECT_EQ(jsonArrayContains<bool>(R"([true])", true), true);
  EXPECT_EQ(jsonArrayContains<bool>(R"([false])", false), true);
  EXPECT_EQ(jsonArrayContains<bool>(R"([true, false])", false), true);
  EXPECT_EQ(jsonArrayContains<bool>(R"([false, true])", true), true);
  EXPECT_EQ(jsonArrayContains<bool>(R"([1])", true), false);
  EXPECT_EQ(jsonArrayContains<bool>(R"([[true]])", true), false);
  EXPECT_EQ(
      jsonArrayContains<bool>(R"([1, "foo", null, "true"])", true), false);
  EXPECT_EQ(
      jsonArrayContains<bool>(
          R"([2, 4, {"a": [8, 9]}, [], [5], false])", false),
      true);
  EXPECT_EQ(jsonArrayContains<bool>(std::nullopt, true), std::nullopt);
  EXPECT_EQ(jsonArrayContains<bool>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(jsonArrayContains<bool>(R"([])", std::nullopt), std::nullopt);
}

TEST_F(SIMDJsonFunctionsTest, jsonArrayContainsLong) {
  // Velox test case
  EXPECT_EQ(jsonArrayContains<int64_t>(R"([])", 0), false);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"([1.2, 2.3, 3.4])", 2), false);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"([1.2, 2.0, 3.4])", 2), false);
  EXPECT_EQ(
      jsonArrayContains<int64_t>(R"(["hello", "presto", "world"])", 2), false);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"([false, false, false])", 17), false);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"(1)", 1), std::nullopt);
  EXPECT_EQ(
      jsonArrayContains<int64_t>(R"("thefoxjumpedoverthefence")", 1),
      std::nullopt);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"("")", 1), std::nullopt);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"(true)", 1), std::nullopt);
  EXPECT_EQ(
      jsonArrayContains<int64_t>(R"({"k1":[0,1,2], "k2":"v1"})", 1),
      std::nullopt);

  EXPECT_EQ(jsonArrayContains<int64_t>(R"([1, 2, 3])", 1), true);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"([1, 2, 3])", 4), false);
  EXPECT_EQ(
      jsonArrayContains<int64_t>(R"([123, 123.456, true, "abc"])", 123), true);
  EXPECT_EQ(
      jsonArrayContains<int64_t>(R"([123, 123.456, true, "abc"])", 456), false);
  EXPECT_EQ(
      jsonArrayContains<int64_t>(
          R"([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])",
          17),
      true);
  EXPECT_EQ(
      jsonArrayContains<int64_t>(
          R"([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])",
          23),
      false);

  // Presto test case
  EXPECT_EQ(jsonArrayContains<int64_t>(R"([])", 1), false);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"([3])", 3), true);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"([-4])", -4), true);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"([1.0])", 1), false);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"([[2]])", 2), false);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"([1, "foo", null, "8"])", 8), false);
  EXPECT_EQ(
      jsonArrayContains<int64_t>(R"([2, 4, {"a": [8, 9]}, [], [5], 6])", 6),
      true);

  // simdjson-difference:
  // Presto return false in this case, but simdjson throw an error because
  // "92233720368547758071" exceeds the range of long type, so return
  // std::nullopt
  EXPECT_EQ(
      jsonArrayContains<int64_t>(R"([92233720368547758071])", -9),
      std::nullopt);
  EXPECT_EQ(jsonArrayContains<int64_t>(std::nullopt, 1), std::nullopt);
  EXPECT_EQ(
      jsonArrayContains<int64_t>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(jsonArrayContains<int64_t>(R"([3])", std::nullopt), std::nullopt);
}

TEST_F(SIMDJsonFunctionsTest, jsonArrayContainsDouble) {
  // Velox test case
  EXPECT_EQ(jsonArrayContains<double>(R"([])", 2.3), false);
  EXPECT_EQ(jsonArrayContains<double>(R"([1, 2, 3])", 2.3), false);
  EXPECT_EQ(jsonArrayContains<double>(R"([1, 2, 3])", 2.0), false);
  EXPECT_EQ(
      jsonArrayContains<double>(R"(["hello", "presto", "world"])", 2.3), false);
  EXPECT_EQ(jsonArrayContains<double>(R"([false, false, false])", 2.3), false);
  EXPECT_EQ(jsonArrayContains<double>(R"(1)", 2.3), std::nullopt);
  EXPECT_EQ(
      jsonArrayContains<double>(R"("thefoxjumpedoverthefence")", 2.3),
      std::nullopt);
  EXPECT_EQ(jsonArrayContains<double>(R"("")", 2.3), std::nullopt);
  EXPECT_EQ(jsonArrayContains<double>(R"(true)", 2.3), std::nullopt);
  EXPECT_EQ(
      jsonArrayContains<double>(R"({"k1":[0,1,2], "k2":"v1"})", 2.3),
      std::nullopt);

  EXPECT_EQ(jsonArrayContains<double>(R"([1.2, 2.3, 3.4])", 2.3), true);
  EXPECT_EQ(jsonArrayContains<double>(R"([1.2, 2.3, 3.4])", 2.4), false);
  EXPECT_EQ(
      jsonArrayContains<double>(R"([123, 123.456, true, "abc"])", 123.456),
      true);
  EXPECT_EQ(
      jsonArrayContains<double>(R"([123, 123.456, true, "abc"])", 456.789),
      false);
  EXPECT_EQ(
      jsonArrayContains<double>(
          R"([1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5])",
          4.5),
      true);
  EXPECT_EQ(
      jsonArrayContains<double>(
          R"([1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5])",
          4.3),
      false);

  // Presto test case
  EXPECT_EQ(jsonArrayContains<double>(R"([])", 1), false);
  EXPECT_EQ(jsonArrayContains<double>(R"([1.5])", 1.5), true);
  EXPECT_EQ(jsonArrayContains<double>(R"([-9.5])", -9.5), true);
  EXPECT_EQ(jsonArrayContains<double>(R"([1])", 1.0), false);
  EXPECT_EQ(jsonArrayContains<double>(R"([[2.5]])", 2.5), false);
  EXPECT_EQ(
      jsonArrayContains<double>(R"([1, "foo", null, "8.2"])", 8.2), false);
  EXPECT_EQ(
      jsonArrayContains<double>(R"([2, 4, {"a": [8, 9]}, [], [5], 6.1])", 6.1),
      true);
  // simdjson-difference:
  // Presto return false in this case, but simdjson throw an error because
  // "9.6E400" can't be parsed, so return std::nullopt
  EXPECT_EQ(jsonArrayContains<double>(R"([9.6E400])", 4.2), std::nullopt);
  EXPECT_EQ(jsonArrayContains<double>(std::nullopt, 1.5), std::nullopt);
  EXPECT_EQ(
      jsonArrayContains<double>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(jsonArrayContains<double>(R"([3.5])", std::nullopt), std::nullopt);
}

TEST_F(SIMDJsonFunctionsTest, jsonArrayContainsString) {
  // Velox test case
  EXPECT_EQ(jsonArrayContains<std::string>(R"([])", ""), false);
  EXPECT_EQ(jsonArrayContains<std::string>(R"([1, 2, 3])", "1"), false);
  EXPECT_EQ(jsonArrayContains<std::string>(R"([1.2, 2.3, 3.4])", "2.3"), false);
  EXPECT_EQ(
      jsonArrayContains<std::string>(R"([true, false])", R"("true")"), false);
  EXPECT_EQ(jsonArrayContains<std::string>(R"(1)", "1"), std::nullopt);
  EXPECT_EQ(
      jsonArrayContains<std::string>(R"("thefoxjumpedoverthefence")", "1"),
      std::nullopt);
  EXPECT_EQ(jsonArrayContains<std::string>(R"("")", "1"), std::nullopt);
  EXPECT_EQ(jsonArrayContains<std::string>(R"(true)", "1"), std::nullopt);
  EXPECT_EQ(
      jsonArrayContains<std::string>(R"({"k1":[0,1,2], "k2":"v1"})", "1"),
      std::nullopt);

  EXPECT_EQ(
      jsonArrayContains<std::string>(
          R"(["hello", "presto", "world"])", "presto"),
      true);
  EXPECT_EQ(
      jsonArrayContains<std::string>(
          R"(["hello", "presto", "world"])", "nation"),
      false);
  EXPECT_EQ(
      jsonArrayContains<std::string>(R"([123, 123.456, true, "abc"])", "abc"),
      true);
  EXPECT_EQ(
      jsonArrayContains<std::string>(R"([123, 123.456, true, "abc"])", "def"),
      false);
  EXPECT_EQ(
      jsonArrayContains<std::string>(
          R"(["hello", "presto", "world", "hello", "presto", "world", "hello", "presto", "world", "hello",
"presto", "world", "hello", "presto", "world", "hello", "presto", "world"])",
          "hello"),
      true);
  EXPECT_EQ(
      jsonArrayContains<std::string>(
          R"(["hello", "presto", "world", "hello", "presto", "world", "hello", "presto", "world", "hello",
"presto", "world", "hello", "presto", "world", "hello", "presto", "world"])",
          "hola"),
      false);
  EXPECT_EQ(
      jsonArrayContains<std::string>(
          R"(["hello", "presto", "world", 1, 2, 3, true, false, 1.2, 2.3, {"k1":[0,1,2], "k2":"v1"}])",
          "world"),
      true);
  EXPECT_EQ(
      jsonArrayContains<std::string>(
          R"(["the fox jumped over the fence", "hello presto world"])",
          "hello velox world"),
      false);
  EXPECT_EQ(
      jsonArrayContains<std::string>(
          R"(["the fox jumped over the fence", "hello presto world"])",
          "the fox jumped over the fence"),
      true);

  // Presto test case
  EXPECT_EQ(jsonArrayContains<std::string>(R"([])", "x"), false);
  EXPECT_EQ(jsonArrayContains<std::string>(R"(["foo"])", "foo"), true);
  EXPECT_EQ(jsonArrayContains<std::string>(R"(["8"])", "8"), true);
  EXPECT_EQ(jsonArrayContains<std::string>(R"([1, "foo", null])", "foo"), true);
  EXPECT_EQ(jsonArrayContains<std::string>(R"([1, 5])", "5"), false);
  EXPECT_EQ(
      jsonArrayContains<std::string>(
          R"([2, 4, {"a": [8, 9]}, [], [5], "6"])", "6"),
      true);
  EXPECT_EQ(jsonArrayContains<std::string>(R"([""])", ""), true);
  EXPECT_EQ(jsonArrayContains<std::string>(R"([""])", "x"), false);
  EXPECT_EQ(jsonArrayContains<std::string>(std::nullopt, "x"), std::nullopt);
  EXPECT_EQ(jsonArrayContains<std::string>(std::nullopt, ""), std::nullopt);
  EXPECT_EQ(
      jsonArrayContains<std::string>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(
      jsonArrayContains<std::string>(R"([""])", std::nullopt), std::nullopt);
  EXPECT_EQ(jsonArrayContains<std::string>("", "x"), std::nullopt);
  EXPECT_EQ(jsonArrayContains<std::string>(R"(123)", "2.5"), std::nullopt);
  EXPECT_EQ(jsonArrayContains<std::string>(R"([)", "8"), std::nullopt);
  EXPECT_EQ(jsonArrayContains<std::string>(R"([1,0,])", "true"), std::nullopt);
  EXPECT_EQ(jsonArrayContains<std::string>(R"([1,,0])", "true"), std::nullopt);
}

TEST_F(SIMDJsonFunctionsTest, isJsonScalar) {
  // Scalars.
  EXPECT_EQ(isJsonScalar(R"(1)"), true);
  EXPECT_EQ(isJsonScalar(R"(123456)"), true);
  EXPECT_EQ(isJsonScalar(R"("hello")"), true);
  EXPECT_EQ(isJsonScalar(R"("thefoxjumpedoverthefence")"), true);
  EXPECT_EQ(isJsonScalar(R"(1.1)"), true);
  EXPECT_EQ(isJsonScalar(R"("")"), true);
  EXPECT_EQ(isJsonScalar(R"(true)"), true);

  // Lists and maps
  EXPECT_EQ(isJsonScalar(R"([1,2])"), false);
  EXPECT_EQ(isJsonScalar(R"({"k1":"v1"})"), false);
  EXPECT_EQ(isJsonScalar(R"({"k1":[0,1,2]})"), false);
  EXPECT_EQ(isJsonScalar(R"({"k1":""})"), false);
}

TEST_F(SIMDJsonFunctionsTest, jsonParse) {
  // JsonFunctionsTest->jsonParse test case
  EXPECT_EQ(jsonParse(std::nullopt), std::nullopt);
  EXPECT_EQ(jsonParse(R"(true)"), "true");
  EXPECT_EQ(jsonParse(R"(null)"), "null");
  EXPECT_EQ(jsonParse(R"(42)"), "42");
  EXPECT_EQ(jsonParse(R"("abc")"), R"("abc")");
  // simdjson-difference:
  // simdjson will remove the spaces between every element. "[1, 2, 3]" ->
  // "[1,2,3]"
  EXPECT_EQ(jsonParse(R"([1, 2, 3])"), "[1,2,3]");
  EXPECT_EQ(jsonParse(R"({"k1":"v1"})"), R"({"k1":"v1"})");
  // simdjson-difference:
  // simdjson will remove the spaces between every element. "["k1", "v1"]" ->
  // "["k1","v1"]"
  EXPECT_EQ(jsonParse(R"(["k1", "v1"])"), R"(["k1","v1"])");
  assertUserError(
      [&]() { jsonParse(R"({"k1":})"); },
      "The JSON document has an improper structure: missing or superfluous commas, braces, missing keys, etc.");
  assertUserError(
      [&]() { jsonParse(R"({:"k1"})"); },
      "The JSON document has an improper structure: missing or superfluous commas, braces, missing keys, etc.");
  assertUserError(
      [&]() { jsonParse(R"(not_json)"); },
      "Problem while parsing an atom starting with the letter 'n'");

  // Presto test case
  assertUserError(
      [&]() { jsonParse(R"(INVALID)"); },
      "The JSON document has an improper structure: missing or superfluous commas, braces, missing keys, etc.");
  assertUserError(
      [&]() { jsonParse(R"("x": 1)"); },
      "The JSON document has an improper structure: missing or superfluous commas, braces, missing keys, etc.");
  assertUserError(
      [&]() { jsonParse(R"(["a": 1, "b": 2])"); },
      "The JSON document has an improper structure: missing or superfluous commas, braces, missing keys, etc.");
}

TEST_F(SIMDJsonFunctionsTest, jsonArrayLength) {
  // Velox test case
  EXPECT_EQ(jsonArrayLength(R"([])"), 0);
  EXPECT_EQ(jsonArrayLength(R"([1])"), 1);
  EXPECT_EQ(jsonArrayLength(R"([1, 2, 3])"), 3);
  EXPECT_EQ(
      jsonArrayLength(
          R"([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])"),
      20);

  EXPECT_EQ(jsonArrayLength(R"(1)"), std::nullopt);
  EXPECT_EQ(jsonArrayLength(R"("hello")"), std::nullopt);
  EXPECT_EQ(jsonArrayLength(R"("")"), std::nullopt);
  EXPECT_EQ(jsonArrayLength(R"(true)"), std::nullopt);
  EXPECT_EQ(jsonArrayLength(R"({"k1":"v1"})"), std::nullopt);
  EXPECT_EQ(jsonArrayLength(R"({"k1":[0,1,2]})"), std::nullopt);
  EXPECT_EQ(jsonArrayLength(R"({"k1":[0,1,2], "k2":"v1"})"), std::nullopt);

  // Presto test case
  EXPECT_EQ(jsonArrayLength(R"([1, "foo", null])"), 3);
  EXPECT_EQ(jsonArrayLength(R"([2, 4, {"a": [8, 9]}, [], [5], 4])"), 6);
  EXPECT_EQ(jsonArrayLength(std::nullopt), std::nullopt);
}

TEST_F(SIMDJsonFunctionsTest, jsonSize) {
  EXPECT_EQ(jsonSize(R"({"k1":{"k2": 999}, "k3": 1})", "$.k1.k2"), 0);
  EXPECT_EQ(jsonSize(R"({"k1":{"k2": 999}, "k3": 1})", "$.k1"), 1);
  EXPECT_EQ(jsonSize(R"({"k1":{"k2": 999}, "k3": 1})", "$"), 2);
  EXPECT_EQ(jsonSize(R"({"k1":{"k2": 999}, "k3": 1})", "$.k3"), 0);
  EXPECT_EQ(jsonSize(R"({"k1":{"k2": 999}, "k3": [1, 2, 3, 4]})", "$.k3"), 4);
  EXPECT_EQ(jsonSize(R"({"k1":{"k2": 999}, "k3": 1})", "$.k4"), std::nullopt);
  EXPECT_EQ(jsonSize(R"({"k1":{"k2": 999}, "k3"})", "$.k4"), std::nullopt);
  EXPECT_EQ(jsonSize(R"({"k1":{"k2": 999}, "k3": true})", "$.k3"), 0);
  EXPECT_EQ(jsonSize(R"({"k1":{"k2": 999}, "k3": null})", "$.k3"), 0);
  EXPECT_EQ(
      jsonSize(
          R"({"k1":{"k2": 999, "k3": [{"k4": [1, 2, 3]}]}})", "$.k1.k3[0].k4"),
      3);
}

TEST_F(SIMDJsonFunctionsTest, invalidPath) {
  VELOX_ASSERT_THROW(jsonSize(R"([0,1,2])", ""), "Invalid JSON path");
  VELOX_ASSERT_THROW(jsonSize(R"([0,1,2])", "$[]"), "Invalid JSON path");
  VELOX_ASSERT_THROW(jsonSize(R"([0,1,2])", "$[-1]"), "Invalid JSON path");
  VELOX_ASSERT_THROW(jsonSize(R"({"k1":"v1"})", "$k1"), "Invalid JSON path");
  VELOX_ASSERT_THROW(jsonSize(R"({"k1":"v1"})", "$.k1."), "Invalid JSON path");
  VELOX_ASSERT_THROW(jsonSize(R"({"k1":"v1"})", "$.k1]"), "Invalid JSON path");
}

} // namespace

} // namespace facebook::velox::functions::prestosql
