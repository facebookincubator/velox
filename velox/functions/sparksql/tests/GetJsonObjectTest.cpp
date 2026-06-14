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
      R"({"name": "Alice", "age": 5, "id": "001"})");
  EXPECT_EQ(
      getJsonObject(R"({"name": "Alice", "age": 5, "id": "001"})", "$  "),
      R"({"name": "Alice", "age": 5, "id": "001"})");
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
      R"({"name": "Alice", "age": "5", "id": "001"})");
  EXPECT_EQ(
      getJsonObject(
          R"({"my": {"info": {"name": "Alice", "age": "5", "id": "001"}}})",
          "$['my']['info']"),
      R"({"name": "Alice", "age": "5", "id": "001"})");

  // Array as result.
  EXPECT_EQ(
      getJsonObject(
          R"([{"my": {"info": {"name": "Alice"}}}, {"other": ["v1", "v2"]}])",
          "$[1].other"),
      R"(["v1", "v2"])");
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
  // Constains '$' not in the first position.
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
  EXPECT_EQ(getJsonObject(R"({"hello": "3.5",,,,})", "$.hello"), "3.5");
  EXPECT_EQ(
      getJsonObject(R"({"hello": "3.5",,,,"taskSort":"2"})", "$.hello"), "3.5");
  EXPECT_EQ(
      getJsonObject(
          R"({"hello": "3.5","taskSort":"2",,,,,"taskSort",})", "$.hello"),
      "3.5");
  EXPECT_EQ(
      getJsonObject(R"({"hello": "3.5","taskSort":"2",,,,,,})", "$.hello"),
      "3.5");
  EXPECT_EQ(
      getJsonObject(R"({"hello": 3.5,"taskSort":"2",,,,,,})", "$.hello"),
      "3.5");
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
      R"({"name": "Alice", "age": "5", "id": "001"})");

  // Incomplete json with wildcard.
  EXPECT_EQ(getJsonObject(R"({"a":[1,2]},)", "$.a[*]"), "[1,2]");
  EXPECT_EQ(getJsonObject(R"({"a":[1,2],})", "$.a[*]"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":[1,2,]})", "$.a[*]"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":[1,2,,3]})", "$.a[*]"), std::nullopt);
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

TEST_F(GetJsonObjectTest, wildcard) {
  const std::string json =
      R"({"store":{"fruit":[{"weight":8,"type":"apple"},{"weight":9,"type":"pear"}],)"
      R"("basket":[[1,2,{"b":"y","a":"x"}],[3,4],[5,6]],)"
      R"("book":[{"author":"Nigel Rees","title":"Sayings of the Century","category":"reference","price":8.95},)"
      R"({"author":"Herman Melville","title":"Moby Dick","category":"fiction","price":8.99,"isbn":"0-553-21311-3"},)"
      R"({"author":"J. R. R. Tolkien","title":"The Lord of the Rings","category":"fiction",)"
      R"("reader":[{"age":25,"name":"bob"},{"age":26,"name":"jack"}],"price":22.99,"isbn":"0-395-19395-8"}],)"
      R"("bicycle":{"price":19.95,"color":"red"}},"email":"amy@only_for_json_udf_test.net","owner":"amy","zip code":"94025","fb:testid":"1234"})";

  EXPECT_EQ(
      getJsonObject(json, "$.store.book[*]"),
      R"([{"author":"Nigel Rees","title":"Sayings of the Century","category":"reference","price":8.95},)"
      R"({"author":"Herman Melville","title":"Moby Dick","category":"fiction","price":8.99,"isbn":"0-553-21311-3"},)"
      R"({"author":"J. R. R. Tolkien","title":"The Lord of the Rings","category":"fiction",)"
      R"("reader":[{"age":25,"name":"bob"},{"age":26,"name":"jack"}],"price":22.99,"isbn":"0-395-19395-8"}])");
  EXPECT_EQ(
      getJsonObject(json, "$.store.book[*].category"),
      R"(["reference","fiction","fiction"])");
  EXPECT_EQ(
      getJsonObject(json, "$.store.book[*].isbn"),
      R"(["0-553-21311-3","0-395-19395-8"])");
  EXPECT_EQ(
      getJsonObject(json, "$.store.book[*].reader"),
      R"([{"age":25,"name":"bob"},{"age":26,"name":"jack"}])");

  EXPECT_EQ(getJsonObject(json, "$.store.book[*].reader[0].age"), "25");
  EXPECT_EQ(getJsonObject(json, "$.store.book[*].reader[*].age"), "[25,26]");
  EXPECT_EQ(
      getJsonObject(json, "$.store.basket[*]"),
      R"([[1,2,{"b":"y","a":"x"}],[3,4],[5,6]])");
  EXPECT_EQ(
      getJsonObject(json, "$.store.basket[0][*]"),
      R"([1,2,{"b":"y","a":"x"}])");
  EXPECT_EQ(
      getJsonObject(json, "$.store.basket[*][*]"),
      R"([1,2,{"b":"y","a":"x"},3,4,5,6])");
  EXPECT_EQ(getJsonObject(json, "$.store.basket[0][*].b"), R"(["y"])");

  // Number result for wildcard.
  EXPECT_EQ(
      getJsonObject(R"([{"a":1}, {"b":2}, {"a":3e-2}])", "$[*].a"), "[1,0.03]");
  EXPECT_EQ(getJsonObject(R"({"a":[{"f":21e3}]})", "$.a[*].f"), "21000.0");
  EXPECT_EQ(
      getJsonObject(R"({"a":[{"f":21e3},{"f":-21E-3}]})", "$.a[*].f"),
      "[21000.0,-0.021]");
  EXPECT_EQ(getJsonObject(R"({"a":[{"i":-0}]})", "$.a[*].i"), "0");
  EXPECT_EQ(getJsonObject(R"([21e3,-0])", "$[*]"), "[21000.0,0]");
}

TEST_F(GetJsonObjectTest, wildcardNested) {
  EXPECT_EQ(getJsonObject(R"([1,null,2])", "$[*]"), "[1,null,2]");
  EXPECT_EQ(getJsonObject(R"({"a":[1,2,3]})", "$.a[*]"), "[1,2,3]");
  EXPECT_EQ(getJsonObject(R"({"a":[{"12": 1}]})", "$.a[*].12"), "1");
  EXPECT_EQ(getJsonObject(R"({"a":[{"12": 1}]})", "$.a[*]['12']"), "1");
  EXPECT_EQ(getJsonObject(R"({"a":[1]})", "$.a[*]"), "1");
  EXPECT_EQ(getJsonObject(R"({"a":[]})", "$.a[*]"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":"hello"})", "$.a[*]"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":123})", "$.a[*]"), std::nullopt);

  EXPECT_EQ(
      getJsonObject(R"({"a":[{"x":1},{"y":2}]})", "$.a[*]"),
      R"([{"x":1},{"y":2}])");
  EXPECT_EQ(
      getJsonObject(R"({"a":[{"b":1},{"c":2},{"b":3}]})", "$.a[*].b"), "[1,3]");
  EXPECT_EQ(getJsonObject(R"({"a":[{"b":1},{"c":2}]})", "$.a[*].b"), "1");
  EXPECT_EQ(
      getJsonObject(R"({"a":[{"b":"x"},{"b":"y"}]})", "$.a[*].b"),
      R"(["x","y"])");
  EXPECT_EQ(
      getJsonObject(R"({"a":[{"b":[1,2]},{"b":[3]}]})", "$.a[*].b[*]"),
      "[[1,2],[3]]");
  EXPECT_EQ(getJsonObject(R"([1,2,3])", "$[*]"), "[1,2,3]");
  EXPECT_EQ(
      getJsonObject(R"({"a":[1,"str",true]})", "$.a[*]"), R"([1,"str",true])");
  EXPECT_EQ(
      getJsonObject(R"({"a":[{"b":1},{"b":2}]})", "$['a'][*].b"), "[1,2]");
  EXPECT_EQ(
      getJsonObject(
          R"({"a":{"b":[{"c":[10,20]},{"c":[30]}]}})", "$.a.b[*].c[*]"),
      "[[10,20],[30]]");
  EXPECT_EQ(
      getJsonObject(R"({"a":[{"b":[1]},{"b":[]}]})", "$.a[*].b[*]"), "[1],[]");
  EXPECT_EQ(getJsonObject(R"({"a":[[],[]]})", "$.a[0][*]"), std::nullopt);
  EXPECT_EQ(getJsonObject(R"({"a":[[1,2],[3,4]]})", "$.a[*][*]"), "[1,2,3,4]");
  EXPECT_EQ(
      getJsonObject(R"({"a":[[1,2],3,[4,5]]})", "$.a[*][*]"), "[1,2,3,4,5]");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
