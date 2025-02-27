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
#include <functions/sparksql/tests/SparkFunctionBaseTest.h>

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {
constexpr float kNaNFloat = std::numeric_limits<float>::quiet_NaN();
constexpr float kInfFloat = std::numeric_limits<float>::infinity();
constexpr double kNaNDouble = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfDouble = std::numeric_limits<double>::infinity();

class ToJsonTest : public SparkFunctionBaseTest {
 protected:
  core::CallTypedExprPtr createToJson(const TypePtr& inputType) {
    std::vector<core::TypedExprPtr> inputs = {
      std::make_shared<core::FieldAccessTypedExpr>(inputType, "c0")};
    return std::make_shared<const core::CallTypedExpr>(
        VARCHAR(), std::move(inputs), "to_json");
  }

  void testToJson(const VectorPtr& input, const VectorPtr& expected) {
    auto expr = createToJson(input->type());
    testEncodings(expr, {input}, expected);
  }
};

TEST_F(ToJsonTest, basicStruct) {
  auto input = makeRowVector({"a"}, {makeFlatVector<int64_t>({1, 2, 3})});
  auto expected = makeFlatVector<std::string>(
    {R"({"a":1})", R"({"a":2})", R"({"a":3})"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, basicArray) {
  auto input = makeArrayVector<int64_t>({{1}, {2, 3}, {}});
  auto expected = makeFlatVector<std::string>({R"([1])", R"([2,3])", R"([])"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, basicMap) {
  auto input = makeMapVector<std::string, int64_t>(
      {{{"a", 1}}, {{"b", 2}}, {{"c", 3}}});
  auto expected = makeFlatVector<std::string>(
      {R"({"a":1})", R"({"b":2})", R"({"c":3})"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, basicBool) {
  auto data = makeNullableFlatVector<bool>(
      {true, false, std::nullopt});
  auto input = makeRowVector({"a"}, {data});
  auto expected = makeFlatVector<std::string>(
      {R"({"a":true})", R"({"a":false})", R"({"a":null})"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, basicString) {
  auto data = makeNullableFlatVector<std::string>({"str1", "str2", std::nullopt, "str\"3\"", std::nullopt});
  auto input = makeRowVector({"a"}, {data});
  auto expected = makeFlatVector<std::string>(
      {R"({"a":"str1"})",
       R"({"a":"str2"})",
       R"({"a":null})",
       R"({"a":"str\"3\""})",
       R"({"a":null})"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, basicTinyInt) {
  auto data = makeNullableFlatVector<int8_t>({0, 127, 128, -128, -129, std::nullopt});
  auto input = makeRowVector({"a"}, {data});
  auto expected = makeFlatVector<std::string>(
      {R"({"a":0})",
       R"({"a":127})",
       R"({"a":-128})",
       R"({"a":-128})",
       R"({"a":127})",
       R"({"a":null})"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, basicSmallInt) {
  auto data = makeNullableFlatVector<int16_t>({0, 32768, -32769, std::nullopt});
  auto input = makeRowVector({"a"}, {data});
  auto expected = makeFlatVector<std::string>(
      {R"({"a":0})",
       R"({"a":-32768})",
       R"({"a":32767})",
       R"({"a":null})"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, basicInt) {
  auto data = makeNullableFlatVector<int32_t>({0, 2147483648, -2147483649, std::nullopt});
  auto input = makeRowVector({"a"}, {data});
  auto expected = makeFlatVector<std::string>(
      {R"({"a":0})",
       R"({"a":-2147483648})",
       R"({"a":2147483647})",
       R"({"a":null})"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, basicFloat) {
  auto data = makeNullableFlatVector<float>(
      {1.0, kNaNFloat, kInfFloat, -kInfFloat, std::nullopt});
  auto input = makeRowVector({"a"}, {data});
  auto expected = makeFlatVector<std::string>(
      {R"({"a":1.0})",
       R"({"a":"NaN"})",
       R"({"a":"Infinity"})",
       R"({"a":"-Infinity"})",
       R"({"a":null})"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, basicDouble) {
  auto data = makeNullableFlatVector<double>(
      {1.0, kNaNDouble, kInfDouble, -kInfDouble, std::nullopt});
  auto input = makeRowVector({"a"}, {data});
  auto expected = makeFlatVector<std::string>(
      {R"({"a":1.0})",
       R"({"a":"NaN"})",
       R"({"a":"Infinity"})",
       R"({"a":"-Infinity"})",
       R"({"a":null})"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, basicDecimal) {
  auto data = makeNullableFlatVector<int64_t>(
      {12345, 0, -67890, std::nullopt}, DECIMAL(10, 2));
  auto input = makeRowVector({"a"}, {data});
  auto expected = makeFlatVector<std::string>(
      {R"({"a":123.45})",
       R"({"a":0.00})",
       R"({"a":-678.90})",
       R"({"a":null})"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, basicTimestamp) {
  auto data = makeNullableFlatVector<Timestamp>(
      {Timestamp(0, 0),
       Timestamp(1582934400, 0),
       Timestamp(-2208988800, 0),
       std::nullopt});
  auto input = makeRowVector({"a"}, {data});
  // UTC time zone.
  auto expected = makeFlatVector<std::string>(
      {R"({"a":"1970-01-01T00:00:00.000Z"})",
       R"({"a":"2020-02-29T00:00:00.000Z"})",
       R"({"a":"1900-01-01T00:00:00.000Z"})",
       R"({"a":null})"});
  testToJson(input, expected);
  // Los_Angeles time zone.
  setTimezone("America/Los_Angeles");
  expected = makeFlatVector<std::string>(
      {R"({"a":"1969-12-31T16:00:00.000-08:00"})",
       R"({"a":"2020-02-28T16:00:00.000-08:00"})",
       R"({"a":"1899-12-31T16:00:00.000-08:00"})",
       R"({"a":null})"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, basicDate) {
  auto data = makeNullableFlatVector<int32_t>(
      {0, 18321, -25567, 2932896, std::nullopt}, DateType::get());
  auto input = makeRowVector({"a"}, {data});
  auto expected = makeFlatVector<std::string>(
      {R"({"a":"1970-01-01"})",
       R"({"a":"2020-02-29"})",
       R"({"a":"1900-01-01"})",
       R"({"a":"9999-12-31"})",
       R"({"a":null})"});
  testToJson(input, expected);
}

TEST_F(ToJsonTest, nestedComplexType) {
  auto data1 = makeNullableFlatVector<std::string>({"str1", "str2", "str3"});
  auto data2 = makeNullableArrayVector<int64_t>({
      {1, 2, 3},
      {},
      {std::nullopt}});
  auto data3 = makeMapVector<std::string, int64_t>(
      {{{"key1", 1}}, {{"key2", 2}}, {{"key3", 3}}});
  auto input = makeRowVector({"a", "b", "c"}, {data1, data2, data3});
  auto expected = makeFlatVector<std::string>(
      {R"({"a":"str1","b":[1,2,3],"c":{"key1":1}})",
       R"({"a":"str2","b":[],"c":{"key2":2}})",
       R"({"a":"str3","b":[null],"c":{"key3":3}})"});
  testToJson(input, expected);
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
