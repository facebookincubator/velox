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
#include <stdint.h>
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class GetJsonObjectTest : public SparkFunctionBaseTest {
 protected:
  std::optional<std::string> getJsonObject(
      const std::string& json,
      const std::string& jsonPath) {
    return evaluateOnce<std::string>(
        "get_json_object(c0, c1)",
        std::optional<std::string>(json),
        std::optional<std::string>(jsonPath));
  }
};

TEST_F(GetJsonObjectTest, basic) {
  EXPECT_EQ(getJsonObject(R"({"hello": "3.5"})", "$.hello"), "3.5");
  EXPECT_EQ(getJsonObject(R"({"hello": 3.5})", "$.hello"), "3.5");
  EXPECT_EQ(getJsonObject(R"({"hello": 292222730})", "$.hello"), "292222730");
  EXPECT_EQ(getJsonObject(R"({"hello": -292222730})", "$.hello"), "-292222730");
  EXPECT_EQ(getJsonObject(R"({"my": {"hello": 3.5}})", "$.my.hello"), "3.5");
  EXPECT_EQ(getJsonObject(R"({"my": {"hello": true}})", "$.my.hello"), "true");
  EXPECT_EQ(getJsonObject(R"({"hello": ""})", "$.hello"), "");
  EXPECT_EQ(
      "0.0215434648799772",
      getJsonObject(R"({"score":0.0215434648799772})", "$.score"));
  // Returns input json if json path is "$".
  EXPECT_EQ(
      getJsonObject(R"({"name": "Alice", "age": 5, "id": "001"})", "$"),
      R"({"name":"Alice","age":5,"id":"001"})");
  EXPECT_EQ(
      getJsonObject(R"({"name": "Alice", "age": 5, "id": "001"})", "$  "),
      R"({"name":"Alice","age":5,"id":"001"})");
  EXPECT_EQ(
      getJsonObject(R"({"name": "Alice", "age": 5, "id": "001"})", "$.age"),
      "5");
  EXPECT_EQ(
      getJsonObject(R"({"name": "Alice", "age": 5, "id": "001"})", "$.id"),
      "001");
  EXPECT_EQ(
      getJsonObject(
          R"([{"my": {"info": {"name": "Alice", "age": "5", "id": "001"}}}, {"other": "v1"}])",
          "$[0]['my']['info']['age']"),
      "5");
  EXPECT_EQ(
      getJsonObject(
          R"([{"my": {"info": {"name": "Alice", "age": "5", "id": "001"}}}, {"other": "v1"}])",
          "$[0].my.info.age"),
      "5");

  // Json object with space in key.
  EXPECT_EQ(getJsonObject(R"({"a b": "1"})", "$.a b"), "1");
  EXPECT_EQ(getJsonObject(R"({"a": "1"})", "$. a"), "1");
  EXPECT_EQ(getJsonObject(R"({"a b": "1"})", "$. a b"), "1");
  EXPECT_EQ(getJsonObject(R"({"two spaces": "1"})", "$.  two spaces"), "1");
  EXPECT_EQ(getJsonObject(R"({"a": "1"})", "$.a "), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a ": "1"})", "$.a "), "1");
  EXPECT_EQ(getJsonObject(R"({"a b": "1"})", "$.a b "), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({ "a b": "1" })", "$['a b']"), "1");
  EXPECT_EQ(getJsonObject(R"({ "a b ": "1" })", "$['a b ']"), "1");
  EXPECT_EQ(getJsonObject(R"({ " a b": "1" })", "$[' a b']"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({ "a b": "1" })", "$[ 'a b']"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({ "a b": "1" })", "$['a b' ]"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a": {"a b": 1}})", "$.a['a b']"), "1");
  EXPECT_EQ(
      getJsonObject(R"({"a": {" a b": 1}})", "$.a[' a b']"), std::nullopt);
  EXPECT_EQ(
      getJsonObject(R"({"two spaces": "1"})", "$.  two spaces "), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a": "1"})", "$ .a"), std::nullopt);
  EXPECT_EQ(
      getJsonObject(R"({"my": {"hello": true}})", "$.  my.  hello"), "true");
  EXPECT_EQ(
      getJsonObject(R"({"my": {"hello": true}})", "$.my.  hello"), "true");
  EXPECT_EQ(getJsonObject(R"({"a$ ": {"b": 10}})", "$.a$ .b"), "10");
  EXPECT_EQ(getJsonObject(R"({"a$. b": {"c": 10}})", "$.a$. b"), std::nullopt);
  // Json object as result.
  EXPECT_EQ(
      getJsonObject(
          R"({"my": {"info": {"name": "Alice", "age": "5", "id": "001"}}})",
          "$.my.info"),
      R"({"name":"Alice","age":"5","id":"001"})");
  EXPECT_EQ(
      getJsonObject(
          R"({"my": {"info": {"name": "Alice", "age": "5", "id": "001"}}})",
          "$['my']['info']"),
      R"({"name":"Alice","age":"5","id":"001"})");

  // Array as result.
  EXPECT_EQ(
      getJsonObject(
          R"([{"my": {"info": {"name": "Alice"}}}, {"other": ["v1", "v2"]}])",
          "$[1].other"),
      R"(["v1","v2"])");
  // Array element as result.
  EXPECT_EQ(
      getJsonObject(
          R"([{"my": {"info": {"name": "Alice"}}}, {"other": ["v1", "v2"]}])",
          "$[1].other[0]"),
      "v1");
  EXPECT_EQ(
      getJsonObject(
          R"([{"my": {"info": {"name": "Alice"}}}, {"other": ["v1", "v2"]}])",
          "$[1].other[1]"),
      "v2");

  // Valid escape sequences: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX.
  EXPECT_EQ(getJsonObject(R"({"hello": "a\"b"})", "$.hello"), R"(a"b)");
  EXPECT_EQ(getJsonObject(R"({"hello": "a\\b"})", "$.hello"), R"(a\b)");
  EXPECT_EQ(getJsonObject(R"({"hello": "a\/b"})", "$.hello"), "a/b");
  EXPECT_EQ(getJsonObject(R"({"hello": "a\bb"})", "$.hello"), "a\bb");
  EXPECT_EQ(getJsonObject(R"({"hello": "a\fb"})", "$.hello"), "a\fb");
  EXPECT_EQ(getJsonObject(R"({"hello": "a\nb"})", "$.hello"), "a\nb");
  EXPECT_EQ(getJsonObject(R"({"hello": "a\rb"})", "$.hello"), "a\rb");
  EXPECT_EQ(getJsonObject(R"({"hello": "a\tb"})", "$.hello"), "a\tb");
  EXPECT_EQ(getJsonObject(R"({"hello": "\u0041"})", "$.hello"), "A");
  EXPECT_EQ(getJsonObject(R"({"hello": "\u000A"})", "$.hello"), "\n");
}

TEST_F(GetJsonObjectTest, supplementaryPlaneInObject) {
  // See appendWithSupplementaryEscapes for the behavior under test.

  // Literal emoji U+1F9E7 inside an object value, alongside a CJK string
  // that must stay literal.
  EXPECT_EQ(
      getJsonObject(R"({"a": {"name": "🧧会员"}})", "$.a"),
      R"({"name":"\uD83E\uDDE7会员"})");

  // Literal emoji U+1F947 inside an array.
  EXPECT_EQ(getJsonObject(R"({"a": ["🥇"]})", "$.a"), R"(["\uD83E\uDD47"])");

  // Already-escaped '\u' input arrives as ASCII bytes and is preserved
  // verbatim (no double escaping). Inside a raw string literal the backslash
  // is literal, so the input/expected below are the literal bytes
  // \ u D 8 3 E \ u D D E 7 g i f t.
  EXPECT_EQ(
      getJsonObject(R"({"a": {"name": "\uD83E\uDDE7gift"}})", "$.a"),
      R"({"name":"\uD83E\uDDE7gift"})");

  // A supplementary-plane character extracted as a scalar string leaf is
  // decoded to its literal form, unaffected by this change.
  EXPECT_EQ(getJsonObject(R"({"a": "🥇"})", "$.a"), "🥇");
}

TEST_F(GetJsonObjectTest, compactResult) {
  EXPECT_EQ(
      getJsonObject(R"({"a": {"b" : [1, "x y"]}})", "$.a"),
      R"({"b":[1,"x y"]})");
  EXPECT_EQ(getJsonObject(R"( [ 1 , 2 ] )", "$"), "[1,2]");
  EXPECT_EQ(getJsonObject(R"({"a":"unterminated})", "$"), std::nullopt);
}

TEST_F(GetJsonObjectTest, escapedResult) {
  EXPECT_EQ(
      getJsonObject(R"({"a":["x\/y","\u0041","\u00e9","\u2028"]})", "$.a"),
      "[\"x/y\",\"A\",\"\u00e9\",\"\u2028\"]");
  EXPECT_EQ(
      getJsonObject(R"({"a":["\b\f\n\r\t","q\"z","\\"]})", "$.a"),
      R"(["\b\f\n\r\t","q\"z","\\"])");
  EXPECT_EQ(
      getJsonObject(R"({"a":["\u0001","\u001f","\u000a","\u0022"]})", "$.a"),
      R"(["\u0001","\u001F","\n","\""])");
  EXPECT_EQ(
      getJsonObject(R"({"a":["\ud83d\ude00","\ud83d","\uD800\u0041"]})", "$.a"),
      R"(["\uD83D\uDE00","\uD83D","\uD800A"])");
  EXPECT_EQ(
      getJsonObject(R"({"k\u00e9\/":{"\n":1}})", "$"),
      "{\"k\u00e9/\":{\"\\n\":1}}");
  EXPECT_EQ(
      getJsonObject(R"({"a":[1.0e2, 1e-7, -0, 0.10]})", "$.a"),
      "[1.0e2,1e-7,-0,0.10]");
}

TEST_F(GetJsonObjectTest, nonAsciiResultOfAsciiInput) {
  const auto input = makeRowVector({
      makeFlatVector<std::string>({
          R"({"a":["\u00e9"]})",
          R"({"a":"\u00e9"})",
      }),
      makeFlatVector<std::string>({"$.a", "$.a"}),
  });
  const SelectivityVector rows(input->size());
  for (const auto& child : input->children()) {
    auto* strings = child->asFlatVector<StringView>();
    strings->computeAndSetIsAscii(rows);
    ASSERT_TRUE(strings->isAscii(rows).value());
  }

  velox::test::assertEqualVectors(
      makeFlatVector<int32_t>({5, 1}),
      evaluate("length(get_json_object(c0, c1))", input));
}

TEST_F(GetJsonObjectTest, nullResult) {
  // Field not found.
  EXPECT_EQ(getJsonObject(R"({"hello": "3.5"})", "$.hi"), std::nullopt);

  // Illegal json.
  EXPECT_EQ(getJsonObject(R"({"hello"-3.5})", "$.hello"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a": bad, "b": string})", "$.a"), std::nullopt);

  // Illegal json path.
  EXPECT_EQ(getJsonObject(R"({"hello": "3.5"})", "$hello"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"hello": "3.5"})", "$."), std::nullopt);
  // The first char is not '$'.
  EXPECT_EQ(getJsonObject(R"({"hello": "3.5"})", ".hello"), std::nullopt);
  // Contains '$' not in the first position.
  EXPECT_EQ(getJsonObject(R"({"hello": "3.5"})", "$.$hello"), std::nullopt);

  // Invalid ending character.
  EXPECT_EQ(
      getJsonObject(
          R"([{"my": {"info": {"name": "Alice"quoted""}}}, {"other": ["v1", "v2"]}])",
          "$[0].my.info.name"),
      std::nullopt);

  // Invalid escape sequence.
  EXPECT_EQ(getJsonObject(R"({"hello": "\x"})", "$.hello"), std::nullopt);
  EXPECT_EQ(
      getJsonObject(R"({"hello": "test", "invalid": "\@"})", "$.hello"),
      std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"hello": "\๑"})", "$.hello"), std::nullopt);
  // Invalid unicode escape sequences.
  EXPECT_EQ(getJsonObject(R"({"hello": "\u"})", "$.hello"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"hello": "\u123"})", "$.hello"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"hello": "\uGHIJ"})", "$.hello"), std::nullopt);
}

TEST_F(GetJsonObjectTest, incompleteJson) {
  EXPECT_EQ(getJsonObject(R"({"hello": "3.5"},)", "$.hello"), "3.5");
  EXPECT_EQ(getJsonObject(R"({"hello": "3.5",,,,})", "$.hello"), std::nullopt);
  EXPECT_EQ(
      getJsonObject(R"({"hello": "3.5",,,,"taskSort":"2"})", "$.hello"),
      std::nullopt);
  EXPECT_EQ(
      getJsonObject(
          R"({"hello": "3.5","taskSort":"2",,,,,"taskSort",})", "$.hello"),
      std::nullopt);
  EXPECT_EQ(
      getJsonObject(R"({"hello": "3.5","taskSort":"2",,,,,,})", "$.hello"),
      std::nullopt);
  EXPECT_EQ(
      getJsonObject(R"({"hello": 3.5,"taskSort":"2",,,,,,})", "$.hello"),
      std::nullopt);
  EXPECT_EQ(
      getJsonObject(R"({"hello": "boy","taskSort":"2"},,,,,)", "$.hello"),
      "boy");
  EXPECT_EQ(getJsonObject(R"({"hello": "boy\n"},)", "$.hello"), "boy\n");
  EXPECT_EQ(getJsonObject(R"({"hello": "boy\n\t"},)", "$.hello"), "boy\n\t");
  EXPECT_EQ(
      getJsonObject(
          R"([{"my": {"info": {"name": "Alice"}}}, {"other": ["v1", "v2"]}],)",
          "$[1].other[1]"),
      "v2");
  EXPECT_EQ(
      getJsonObject(
          R"({"my": {"info": {"name": "Alice", "age": "5", "id": "001"}}},)",
          "$['my']['info']"),
      R"({"name":"Alice","age":"5","id":"001"})");
}

TEST_F(GetJsonObjectTest, malformedRootValue) {
  EXPECT_EQ(getJsonObject(R"({"a":1,"b":[1,2)", "$.a"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":1,})", "$.a"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":[1,]})", "$.a[0]"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":1,"b":01})", "$.a"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":"x","b":1 "c":2})", "$.a"), std::nullopt);
  EXPECT_EQ(
      getJsonObject(R"({"a":[{"b":1}, {"b":2} x]})", "$.a[0].b"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":1,"b":{"c":[tru]}})", "$.a"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":[1,]})", "$"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":1,"b":[1e400]})", "$.a"), "1");
}

TEST_F(GetJsonObjectTest, rootValueOnly) {
  EXPECT_EQ(getJsonObject(R"({"a":"x"} trailing)", "$"), R"({"a":"x"})");
  EXPECT_EQ(getJsonObject(R"({"a":1} {"a":2})", "$"), R"({"a":1})");
  EXPECT_EQ(getJsonObject(R"({"a":"x"}})", "$.a"), "x");
  EXPECT_EQ(getJsonObject("123 x", "$"), "123");
  EXPECT_EQ(getJsonObject("true]", "$"), "true");
  EXPECT_EQ(getJsonObject(R"("s"x)", "$"), "s");
  EXPECT_EQ(getJsonObject(R"( "a\nb" )", "$"), "a\nb");
  EXPECT_EQ(getJsonObject("null", "$"), "null");
  EXPECT_EQ(getJsonObject("123x", "$"), std::nullopt);
  EXPECT_EQ(getJsonObject("123,", "$"), std::nullopt);
  EXPECT_EQ(getJsonObject("   ", "$"), std::nullopt);
}

TEST_F(GetJsonObjectTest, singleQuotes) {
  EXPECT_EQ(getJsonObject(R"({"a":1,"b":'x'})", "$.a"), "1");
  EXPECT_EQ(getJsonObject(R"({"a":1,'b':2})", "$.a"), "1");
  EXPECT_EQ(getJsonObject(R"(['x',1])", "$[1]"), "1");
  EXPECT_EQ(getJsonObject(R"({"a":'x"y'})", "$.a"), R"(x"y)");
  EXPECT_EQ(getJsonObject(R"({"a":"\'"})", "$.a"), "'");
  EXPECT_EQ(getJsonObject(R"({"a":{"b":'s'}})", "$.a"), R"({"b":"s"})");
  EXPECT_EQ(getJsonObject(R"([ true, '', ''])", "$"), R"([true,"",""])");
  EXPECT_EQ(getJsonObject(R"({"a":'x})", "$.a"), std::nullopt);
}

TEST_F(GetJsonObjectTest, controlCharacters) {
  EXPECT_EQ(getJsonObject("{\"a\":1,\"b\":\"x\x01y\"}", "$.a"), "1");
  EXPECT_EQ(getJsonObject("{\"a\":1,\"b\":\"x\x01y\"}", "$.b"), "x\x01y");
  EXPECT_EQ(getJsonObject("{\"a\":\"x\x01\ty\"}", "$.a"), "x\x01\ty");
  EXPECT_EQ(
      getJsonObject("{\"a\":{\"b\":\"\x01\t\"}}", "$.a"),
      R"({"b":"\u0001\t"})");
  EXPECT_EQ(getJsonObject("\"\x01s\"", "$"), "\x01s");
  EXPECT_EQ(getJsonObject("{\"a\":1,\"b\":'x\x01'}", "$.a"), "1");
  EXPECT_EQ(getJsonObject("{\"a\":1,\"b\":'x\x01'}", "$.b"), "x\x01");
  EXPECT_EQ(getJsonObject("{\"a\":'x\x01',\"b\":1}", "$.a"), "x\x01");
  EXPECT_EQ(getJsonObject("{'a\x01':1,\"b\":2}", "$.a"), std::nullopt);
  EXPECT_EQ(getJsonObject("{'a\x01':1,\"b\":2}", "$.b"), "2");
}

TEST_F(GetJsonObjectTest, mismatchedBrackets) {
  EXPECT_EQ(getJsonObject(R"({"a":[1,"x","b":0},"b":2})", "$.b"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":1,"b":[1})", "$.a"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"([[1,"x","b":0}],1])", "$[0]"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":"[}"})", "$.a"), "[}");
  EXPECT_EQ(getJsonObject(R"({"a":1}])", "$.a"), "1");
}

TEST_F(GetJsonObjectTest, number) {
  EXPECT_EQ(getJsonObject(R"({"f": +INF})", "$.f"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"f": -INF})", "$.f"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"f": NaN})", "$.f"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"f": Infinity})", "$.f"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"f": +21.00})", "$.f"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"f": +0.00})", "$.f"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"f": -0.00})", "$.f"), "-0.0");
  EXPECT_EQ(getJsonObject(R"({"f": 0.00})", "$.f"), "0.0");
  EXPECT_EQ(getJsonObject(R"({"f": -21.00})", "$.f"), "-21.0");
  EXPECT_EQ(getJsonObject(R"({"f": -21.010})", "$.f"), "-21.01");
  EXPECT_EQ(getJsonObject(R"({"f": 21e3})", "$.f"), "21000.0");
  EXPECT_EQ(getJsonObject(R"({"f": -21E-3})", "$.f"), "-0.021");
  EXPECT_EQ(getJsonObject(R"({"f": 1e400})", "$.f"), R"("Infinity")");
  EXPECT_EQ(getJsonObject(R"({"f": [-1e400]})", "$.f[0]"), R"("-Infinity")");
  EXPECT_EQ(getJsonObject(R"({"f":1.0e400})", "$.f"), R"("Infinity")");
  EXPECT_EQ(getJsonObject(R"({"f":-1e400})", "$.f"), R"("-Infinity")");
  EXPECT_EQ(getJsonObject(R"({"f":1e309})", "$.f"), R"("Infinity")");
  EXPECT_EQ(getJsonObject(R"({"f":01e400})", "$.f"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"f":-01e400})", "$.f"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"f":1.e400})", "$.f"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"f":.1e400})", "$.f"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"f":1e})", "$.f"), std::nullopt);
  EXPECT_EQ(getJsonObject("01e400", "$"), std::nullopt);
  EXPECT_EQ(getJsonObject("-01e400", "$"), std::nullopt);
  EXPECT_EQ(getJsonObject("1.e400", "$"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"i": +0})", "$.i"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"i": -00})", "$.i"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"i": 00})", "$.i"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"i": 001})", "$.i"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"i": -0})", "$.i"), "0");
  EXPECT_EQ(getJsonObject(R"({"i": 0})", "$.i"), "0");
  EXPECT_EQ(
      getJsonObject(
          R"({"big": 98765432109876543210987654321098765432 })", "$.big"),
      "98765432109876543210987654321098765432");
  EXPECT_EQ(
      getJsonObject(
          R"({"big":  -98765432109876543210987654321098765432})", "$.big"),
      "-98765432109876543210987654321098765432");
  EXPECT_EQ(
      getJsonObject(
          R"({"nested": {"num": -1234567890123456789012345678901234567890 }})",
          "$.nested.num"),
      "-1234567890123456789012345678901234567890");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
