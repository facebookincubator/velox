#include "velox/functions/FunctionRegistry.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/JsonType.h"

namespace facebook::velox::functions::prestosql {
namespace {

const std::string kTestJson = std::string("{\n") +
"    \"store\": {\n" +
"        \"book\": [\n" +
"            {\n" +
"                \"category\": \"reference\",\n" +
"                \"author\": \"Nigel Rees\",\n" +
"                \"title\": \"Sayings of the Century\",\n" +
"                \"price\": 8.95\n" +
"            },\n" +
"            {\n" +
"                \"category\": \"fiction\",\n" +
"                \"author\": \"Evelyn Waugh\",\n" +
"                \"title\": \"Sword of Honour\",\n" +
"                \"price\": 12.99\n" +
"            },\n" +
"            {\n" +
"                \"category\": \"fiction\",\n" +
"                \"author\": \"Herman Melville\",\n" +
"                \"title\": \"Moby Dick\",\n" +
"                \"isbn\": \"0-553-21311-3\",\n" +
"                \"price\": 8.99\n" +
"            },\n" +
"            {\n" +
"                \"category\": \"fiction\",\n" +
"                \"author\": \"J. R. R. Tolkien\",\n" +
"                \"title\": \"The Lord of the Rings\",\n" +
"                \"isbn\": \"0-395-19395-8\",\n" +
"                \"price\": 22.99\n" +
"            }\n" +
"        ],\n" +
"        \"bicycle\": {\n" +
"            \"color\": \"red\",\n" +
"            \"price\": 19.95\n" +
"        }\n" +
"    },\n" +
"    \"expensive\": 10\n" +
"}";

}

class SIMDJsonFunctionsTest : public functions::test::FunctionBaseTest {
 public:
  template <typename T>
  std::optional<bool> json_array_contains(
      std::optional<std::string> json,
      std::optional<T> value) {
    return evaluateOnce<bool>("simd_json_array_contains(c0, c1)", json, value);
  }

  std::optional<std::string> json_parse(std::optional<std::string> json) {
    return evaluateOnce<std::string>("simd_json_parse(c0)", json);
  }

  std::optional<std::string> json_extract_scalar(
      std::optional<std::string> json,
      std::optional<std::string> path) {
    return evaluateOnce<std::string>(
        "simd_json_extract_scalar(c0, c1)", json, path);
  }

  std::optional<int64_t> json_valid(std::optional<std::string> json) {
    return evaluateOnce<int64_t>("simd_json_valid(c0)", json);
  }

  std::optional<int64_t> json_array_length(std::optional<std::string> json) {
    return evaluateOnce<int64_t>("simd_json_array_length(c0)", json);
  }

  std::optional<std::string> json_array_get(
      std::optional<std::string> json,
      std::optional<int64_t> index) {
    return evaluateOnce<std::string>("simd_json_array_get(c0, c1)", json, index);
  }

  std::optional<std::string> json_format(std::optional<std::string> json) {
    return evaluateOnce<std::string>("simd_json_format(c0)", json);
  }

  std::optional<std::string> json_keys(std::optional<std::string> json) {
    return evaluateOnce<std::string>("simd_json_keys(c0)", json);
  }

  std::optional<std::string> json_unquote(std::optional<std::string> json) {
    return evaluateOnce<std::string>("simd_json_unquote(c0)", json);
  }
};

TEST_F(SIMDJsonFunctionsTest, jsonArrayContainsBool) {
  // Velox test case
  EXPECT_EQ(json_array_contains<bool>(R"([])", true), false);
  EXPECT_EQ(json_array_contains<bool>(R"([1, 2, 3])", false), false);
  EXPECT_EQ(json_array_contains<bool>(R"([1.2, 2.3, 3.4])", true), false);
  EXPECT_EQ(
      json_array_contains<bool>(R"(["hello", "presto", "world"])", false),
      false);
  EXPECT_EQ(json_array_contains<bool>(R"(1)", true), false);
  EXPECT_EQ(
      json_array_contains<bool>(R"("thefoxjumpedoverthefence")", false),
      false);
  EXPECT_EQ(json_array_contains<bool>(R"("")", false), false);
  EXPECT_EQ(json_array_contains<bool>(R"(true)", true), false);
  EXPECT_EQ(
      json_array_contains<bool>(R"({"k1":[0,1,2], "k2":"v1"})", true),
      false);

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

  // Presto test case
  EXPECT_EQ(json_array_contains<bool>(R"([])", true), false);
  EXPECT_EQ(json_array_contains<bool>(R"([true])", true), true);
  EXPECT_EQ(json_array_contains<bool>(R"([false])", false), true);
  EXPECT_EQ(json_array_contains<bool>(R"([true, false])", false), true);
  EXPECT_EQ(json_array_contains<bool>(R"([false, true])", true), true);
  EXPECT_EQ(json_array_contains<bool>(R"([1])", true), false);
  EXPECT_EQ(json_array_contains<bool>(R"([[true]])", true), false);
  EXPECT_EQ(
      json_array_contains<bool>(R"([1, "foo", null, "true"])", true), false);
  EXPECT_EQ(
      json_array_contains<bool>(
          R"([2, 4, {"a": [8, 9]}, [], [5], false])", false),
      true);
  EXPECT_EQ(json_array_contains<bool>(std::nullopt, true), std::nullopt);
  EXPECT_EQ(
      json_array_contains<bool>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(json_array_contains<bool>(R"([])", std::nullopt), std::nullopt);
}

TEST_F(SIMDJsonFunctionsTest, jsonArrayContainsLong) {
  // Velox test case
  EXPECT_EQ(json_array_contains<int64_t>(R"([])", 0), false);
  EXPECT_EQ(json_array_contains<int64_t>(R"([1.2, 2.3, 3.4])", 2), false);
  EXPECT_EQ(json_array_contains<int64_t>(R"([1.2, 2.0, 3.4])", 2), false);
  EXPECT_EQ(
      json_array_contains<int64_t>(R"(["hello", "presto", "world"])", 2),
      false);
  EXPECT_EQ(
      json_array_contains<int64_t>(R"([false, false, false])", 17), false);
  EXPECT_EQ(json_array_contains<int64_t>(R"(1)", 1), false);
  EXPECT_EQ(
      json_array_contains<int64_t>(R"("thefoxjumpedoverthefence")", 1),
      false);
  EXPECT_EQ(json_array_contains<int64_t>(R"("")", 1), false);
  EXPECT_EQ(json_array_contains<int64_t>(R"(true)", 1), false);
  EXPECT_EQ(
      json_array_contains<int64_t>(R"({"k1":[0,1,2], "k2":"v1"})", 1),
      false);

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

  // Presto test case
  EXPECT_EQ(json_array_contains<int64_t>(R"([])", 1), false);
  EXPECT_EQ(json_array_contains<int64_t>(R"([3])", 3), true);
  EXPECT_EQ(json_array_contains<int64_t>(R"([-4])", -4), true);
  EXPECT_EQ(json_array_contains<int64_t>(R"([1.0])", 1), false);
  EXPECT_EQ(json_array_contains<int64_t>(R"([[2]])", 2), false);
  EXPECT_EQ(json_array_contains<int64_t>(R"([1, "foo", null, "8"])", 8), false);
  EXPECT_EQ(
      json_array_contains<int64_t>(R"([2, 4, {"a": [8, 9]}, [], [5], 6])", 6),
      true);
  // throw en error and result=false but return std::nullopt
  EXPECT_EQ(
      json_array_contains<int64_t>(R"([92233720368547758071])", -9), false);
  EXPECT_EQ(json_array_contains<int64_t>(std::nullopt, 1), std::nullopt);
  EXPECT_EQ(
      json_array_contains<int64_t>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(json_array_contains<int64_t>(R"([3])", std::nullopt), std::nullopt);
}

TEST_F(SIMDJsonFunctionsTest, jsonArrayContainsDouble) {
  // Velox test case
  EXPECT_EQ(json_array_contains<double>(R"([])", 2.3), false);
  EXPECT_EQ(json_array_contains<double>(R"([1, 2, 3])", 2.3), false);
  EXPECT_EQ(json_array_contains<double>(R"([1, 2, 3])", 2.0), false);
  EXPECT_EQ(
      json_array_contains<double>(R"(["hello", "presto", "world"])", 2.3),
      false);
  EXPECT_EQ(
      json_array_contains<double>(R"([false, false, false])", 2.3), false);
  EXPECT_EQ(json_array_contains<double>(R"(1)", 2.3), false);
  EXPECT_EQ(
      json_array_contains<double>(R"("thefoxjumpedoverthefence")", 2.3),
      false);
  EXPECT_EQ(json_array_contains<double>(R"("")", 2.3), false);
  EXPECT_EQ(json_array_contains<double>(R"(true)", 2.3), false);
  EXPECT_EQ(
      json_array_contains<double>(R"({"k1":[0,1,2], "k2":"v1"})", 2.3),
      false);

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

  // Presto test case
  EXPECT_EQ(json_array_contains<double>(R"([])", 1), false);
  EXPECT_EQ(json_array_contains<double>(R"([1.5])", 1.5), true);
  EXPECT_EQ(json_array_contains<double>(R"([-9.5])", -9.5), true);
  EXPECT_EQ(json_array_contains<double>(R"([1])", 1.0), false);
  EXPECT_EQ(json_array_contains<double>(R"([[2.5]])", 2.5), false);
  EXPECT_EQ(
      json_array_contains<double>(R"([1, "foo", null, "8.2"])", 8.2), false);
  EXPECT_EQ(
      json_array_contains<double>(
          R"([2, 4, {"a": [8, 9]}, [], [5], 6.1])", 6.1),
      true);
  EXPECT_EQ(json_array_contains<double>(R"([9.6E400])", 4.2), false);
  EXPECT_EQ(json_array_contains<double>(std::nullopt, 1.5), std::nullopt);
  EXPECT_EQ(
      json_array_contains<double>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(
      json_array_contains<double>(R"([3.5])", std::nullopt), std::nullopt);
}

TEST_F(SIMDJsonFunctionsTest, jsonArrayContainsString) {
  // Velox test case
  EXPECT_EQ(json_array_contains<std::string>(R"([])", ""), false);
  EXPECT_EQ(json_array_contains<std::string>(R"([1, 2, 3])", "1"), false);
  EXPECT_EQ(
      json_array_contains<std::string>(R"([1.2, 2.3, 3.4])", "2.3"), false);
  EXPECT_EQ(
      json_array_contains<std::string>(R"([true, false])", R"("true")"), false);
  EXPECT_EQ(json_array_contains<std::string>(R"(1)", "1"), false);
  EXPECT_EQ(
      json_array_contains<std::string>(R"("thefoxjumpedoverthefence")", "1"),
      false);
  EXPECT_EQ(json_array_contains<std::string>(R"("")", "1"), false);
  EXPECT_EQ(json_array_contains<std::string>(R"(true)", "1"), false);
  EXPECT_EQ(
      json_array_contains<std::string>(R"({"k1":[0,1,2], "k2":"v1"})", "1"),
      false);

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

  // Presto test case
  EXPECT_EQ(json_array_contains<std::string>(R"([])", "x"), false);
  EXPECT_EQ(json_array_contains<std::string>(R"(["foo"])", "foo"), true);
  EXPECT_EQ(json_array_contains<std::string>(R"(["8"])", "8"), true);
  EXPECT_EQ(
      json_array_contains<std::string>(R"([1, "foo", null])", "foo"), true);
  EXPECT_EQ(json_array_contains<std::string>(R"([1, 5])", "5"), false);
  EXPECT_EQ(
      json_array_contains<std::string>(
          R"([2, 4, {"a": [8, 9]}, [], [5], "6"])", "6"),
      true);
  EXPECT_EQ(json_array_contains<std::string>(R"([""])", ""), true);
  EXPECT_EQ(json_array_contains<std::string>(R"([""])", "x"), false);
  EXPECT_EQ(json_array_contains<std::string>(std::nullopt, "x"), std::nullopt);
  EXPECT_EQ(json_array_contains<std::string>(std::nullopt, ""), std::nullopt);
  EXPECT_EQ(
      json_array_contains<std::string>(std::nullopt, std::nullopt),
      std::nullopt);
  EXPECT_EQ(
      json_array_contains<std::string>(R"([""])", std::nullopt), std::nullopt);
  EXPECT_EQ(json_array_contains<std::string>("", "x"), false);
  EXPECT_EQ(json_array_contains<std::string>(R"(123)", "2.5"), false);
  EXPECT_EQ(json_array_contains<std::string>(R"([)", "8"), false);
  EXPECT_EQ(
      json_array_contains<std::string>(R"([1,0,])", "true"), false);
  EXPECT_EQ(
      json_array_contains<std::string>(R"([1,,0])", "true"), false);
}

TEST_F(SIMDJsonFunctionsTest, jsonParse) {
  // Presto test case
  EXPECT_EQ(json_parse(R"([1, 2, 3])"), "[1,2,3]");
  EXPECT_EQ(json_parse(R"("abc")"), "\"abc\"");
  EXPECT_EQ(json_parse(R"(42)"), "42");
  EXPECT_EQ(json_parse(R"(true)"), "true");
  EXPECT_EQ(json_parse(R"(null)"), "null");

  // invalid json parse
  assertUserError(
      [&]() { json_parse(R"(not_json)"); },
      "Cannot convert 'not_json' to JSON");
  assertUserError(
      [&]() { json_parse(R"(INVALID)"); }, "Cannot convert 'INVALID' to JSON");
  assertUserError(
      [&]() { json_parse(R"("x": 1)"); }, "Cannot convert '\"x\": 1' to JSON");
  assertUserError(
      [&]() { json_parse(R"(["a": 1, "b": 2])"); },
      R"(Cannot convert '["a": 1, "b": 2]' to JSON)");
}

TEST_F(SIMDJsonFunctionsTest, jsonExtractScalar) {
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
  /*
  EXPECT_EQ(json_extract_scalar(kTestJson, "$..price"), std::nullopt);
  EXPECT_EQ(
      json_extract_scalar(kTestJson, "$.store.book[?(@.price < 10)].title"),
      std::nullopt);
  EXPECT_EQ(json_extract_scalar(kTestJson, "max($..price)"), "22.99");
  EXPECT_EQ(
      json_extract_scalar(kTestJson, "concat($..category)"),
      "referencefictionfictionfiction");
  EXPECT_EQ(json_extract_scalar(kTestJson, "$.store.keys()"), std::nullopt);
  */
  EXPECT_EQ(
      json_extract_scalar(kTestJson, "$.store.book[1].author"), "Evelyn Waugh");

  // invalid json
  EXPECT_EQ(json_extract_scalar(R"(INVALID_JSON)", "$"), std::nullopt);

  // invalid JsonPath
  EXPECT_THROW(json_extract_scalar(R"({"":""})", ""), VeloxUserError);
  EXPECT_THROW(json_extract_scalar(R"([1,2,3])", "$...invalid"), VeloxUserError);
}

TEST_F(SIMDJsonFunctionsTest, jsonValid) {
  EXPECT_EQ(json_valid(R"({"[3, 3]": [4, 4]})"), 1);
  EXPECT_EQ(json_valid(R"({"a":null,"b":"[7,8]"})"), 1);
  EXPECT_EQ(json_valid(R"([1, 2, 3])"), 1);
  EXPECT_EQ(json_valid(R"({"a": 5, "b": {"c": "2022-11-22 00:00:00"}})"), 1);
  EXPECT_EQ(json_valid(R"(hello)"), 0);
  EXPECT_EQ(json_valid(R"("hello")"), 1);
  EXPECT_EQ(json_valid(R"(1)"), 1);
  EXPECT_EQ(json_valid(R"({"true":})"), 0);
  EXPECT_EQ(json_valid(R"(true)"), 1);
  EXPECT_EQ(json_valid(R"(null)"), 1);
  EXPECT_EQ(json_valid(std::nullopt), std::nullopt);
}

TEST_F(SIMDJsonFunctionsTest, jsonArrayLength) {
  // Velox test case
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

  // Presto test case
  EXPECT_EQ(json_array_length(R"([1, "foo", null])"), 3);
  EXPECT_EQ(json_array_length(R"([2, 4, {"a": [8, 9]}, [], [5], 4])"), 6);
  EXPECT_EQ(json_array_length(std::nullopt), std::nullopt);
}

TEST_F(SIMDJsonFunctionsTest, jsonKeys) {
  EXPECT_EQ(json_keys(R"({"[3, 3]": [4, 4]})"), "[\"[3, 3]\"]");
  EXPECT_EQ(json_keys(R"({"a":null,"[7,8]":"b"})"), "[\"a\",\"[7,8]\"]");
  EXPECT_EQ(json_keys(R"([1, 2, 3])"), std::nullopt);
  EXPECT_EQ(
      json_keys(R"({"a": 5, "b": {"c": "2022-11-22 00:00:00"}})"),
      "[\"a\",\"b\"]");
  EXPECT_EQ(json_keys(R"({})"), "[]");
  EXPECT_EQ(json_keys(std::nullopt), std::nullopt);
}

} // namespace facebook::velox::functions::prestosql
