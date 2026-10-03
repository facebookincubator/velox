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
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/tests/JsonTestUtil.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {
class FromJsonTest : public SparkFunctionBaseTest {
 protected:
  static core::CallTypedExprPtr createFromJsonWithConfig(
      const TypePtr& outputType,
      const std::vector<std::string>& configArguments) {
    std::vector<core::TypedExprPtr> inputs = {
        std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0")};
    for (const auto& argument : configArguments) {
      inputs.emplace_back(
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), argument));
    }
    return std::make_shared<const core::CallTypedExpr>(
        outputType, std::move(inputs), "from_json");
  }

  void testFromJson(const VectorPtr& input, const VectorPtr& expected) {
    auto expr = createFromJson(expected->type());
    testEncodings(expr, {input}, expected);
  }

  // Evaluates from_json with config arguments and checks the result.
  void testFromJsonWithConfig(
      const VectorPtr& input,
      const VectorPtr& expected,
      const std::vector<std::string>& configArguments) {
    auto expr = createFromJsonWithConfig(expected->type(), configArguments);
    auto result = evaluate(expr, makeRowVector({input}));
    assertEqualVectors(expected, result);
  }

  // Evaluates from_json with config arguments and checks the user error.
  void testFromJsonThrows(
      const VectorPtr& input,
      const TypePtr& outputType,
      const std::vector<std::string>& configArguments,
      const std::string& expectedMessage) {
    auto expr = createFromJsonWithConfig(outputType, configArguments);
    VELOX_ASSERT_USER_THROW(
        evaluate(expr, makeRowVector({input})), expectedMessage);
  }

  static std::string failfastMessage(std::string_view record) {
    return "[MALFORMED_RECORD_IN_PARSING.WITHOUT_SUGGESTION] "
           "Malformed records are detected in record parsing. "
           "Parse Mode: FAILFAST. Record: " +
        std::string(record);
  }

  // Enables Spark's legacy date formatter policy.
  void enableLegacyDateFormatter() {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kLegacyDateFormatter),
          "true"}});
  }
};

TEST_F(FromJsonTest, basicStruct) {
  auto expected = makeFlatVector<int64_t>({1, 2, 3});
  auto input = makeFlatVector<std::string>(
      {R"({"Id": 1})", R"({"Id": 2})", R"({"Id": 3})"});
  testFromJson(input, makeRowVector({"Id"}, {expected}));
}

TEST_F(FromJsonTest, basicArray) {
  auto expected = makeArrayVector<int64_t>({{1}, {2}, {}});
  auto input = makeFlatVector<std::string>({R"([1])", R"([2])", R"([])"});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, arrayOfRowWithScalarInput) {
  auto input = makeFlatVector<std::string>(
      {R"(746314092223)", R"(true)", R"("invalid")"});
  const auto outputType = ARRAY(ROW({"text", "type"}, {VARCHAR(), VARCHAR()}));
  auto expected =
      BaseVector::createNullConstant(outputType, input->size(), pool());
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, basicMap) {
  auto expected = makeMapVector<std::string, int64_t>(
      {{{"a", 1}}, {{"b", 2}}, {{"c", 3}}, {{"3", 3}}});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})", R"({"b": 2})", R"({"c": 3})", R"({"3": 3})"});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, basicBool) {
  auto expected = makeNullableFlatVector<bool>(
      {true, false, std::nullopt, std::nullopt, std::nullopt});
  auto input = makeFlatVector<std::string>(
      {R"({"a": true})",
       R"({"a": false})",
       R"({"a": 1})",
       R"({"a": 0.0})",
       R"({"a": "true"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

// Integer tests pin both inclusive bounds and the first out-of-range value on
// each side so an off-by-one range check or a narrowing cast is caught.
TEST_F(FromJsonTest, basicTinyInt) {
  auto expected = makeNullableFlatVector<int8_t>(
      {1,
       std::numeric_limits<int8_t>::max(),
       std::numeric_limits<int8_t>::min(),
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": 127})",
       R"({"a": -128})",
       R"({"a": -129})",
       R"({"a": 128})",
       R"({"a": 1.0})",
       R"({"a": "1"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicSmallInt) {
  auto expected = makeNullableFlatVector<int16_t>(
      {1,
       std::numeric_limits<int16_t>::max(),
       std::numeric_limits<int16_t>::min(),
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": 32767})",
       R"({"a": -32768})",
       R"({"a": -32769})",
       R"({"a": 32768})",
       R"({"a": 1.0})",
       R"({"a": "1"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicInt) {
  auto expected = makeNullableFlatVector<int32_t>(
      {1,
       std::numeric_limits<int32_t>::max(),
       std::numeric_limits<int32_t>::min(),
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": 2147483647})",
       R"({"a": -2147483648})",
       R"({"a": -2147483649})",
       R"({"a": 2147483648})",
       R"({"a": 2.0})",
       R"({"a": "3"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

// 9223372036854775808 is parsed by simdjson as an unsigned integer and must
// be rejected instead of wrapping to INT64_MIN.
TEST_F(FromJsonTest, basicBigInt) {
  auto expected = makeNullableFlatVector<int64_t>(
      {1,
       std::numeric_limits<int64_t>::max(),
       std::numeric_limits<int64_t>::min(),
       std::nullopt,
       std::nullopt,
       std::nullopt});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": 9223372036854775807})",
       R"({"a": -9223372036854775808})",
       R"({"a": 9223372036854775808})",
       R"({"a": 2.0})",
       R"({"a": "3"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicFloat) {
  auto expected = makeNullableFlatVector<float>(
      {1.0,          2.0,          -3.4028235E38, 3.4028235E38, -kInfFloat,
       kInfFloat,    0.0,          0.0,           std::nullopt, std::nullopt,
       std::nullopt, std::nullopt, std::nullopt,  std::nullopt, std::nullopt,
       std::nullopt, kNaNFloat,    kNaNFloat,     -kInfFloat,   -kInfFloat,
       -kInfFloat,   -kInfFloat,   kInfFloat,     kInfFloat,    kInfFloat,
       kInfFloat,    kInfFloat,    kInfFloat});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": 2.0})",
       R"({"a": -3.4028235E38})", // Min float value
       R"({"a": 3.4028235E38})", // Max float value
       R"({"a": -3.4028235E39})",
       R"({"a": 3.4028235E39})",
       R"({"a": 0})",
       R"({"a": 1.0e-200})",
       R"({"a": "3"})",
       R"({"a": 1.1.0})", // Multiple decimal points.
       R"({"a": 1.})", // Missing fraction digits after a decimal point.
       R"({"a": 01})", // Leading zero.
       R"({"a": 1e})", // Missing exponent digits after ‘e’ or ‘E’.
       R"({"a": 1e+})", // Missing exponent digits after ‘e’ or ‘E’.
       R"({"a": .e10})", // Missing digits.
       R"({"a": -.})", // Missing digits entirely.
       R"({"a": "NaN"})",
       R"({"a": NaN})",
       R"({"a": "-Infinity"})",
       R"({"a": -Infinity})",
       R"({"a": "-INF"})",
       R"({"a": -INF})",
       R"({"a": "+Infinity"})",
       R"({"a": +Infinity})",
       R"({"a": "Infinity"})",
       R"({"a": Infinity})",
       R"({"a": "+INF"})",
       R"({"a": +INF})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicDouble) {
  auto expected = makeNullableFlatVector<double>(
      {1.0,
       2.0,
       -1.7976931348623158e+308,
       1.7976931348623158e+308,
       -kInfDouble,
       kInfDouble,
       0.0,
       0.0,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       kNaNDouble,
       kNaNDouble,
       -kInfDouble,
       -kInfDouble,
       -kInfDouble,
       -kInfDouble,
       kInfDouble,
       kInfDouble,
       kInfDouble,
       kInfDouble,
       kInfDouble,
       kInfDouble});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": 2.0})",
       R"({"a": -1.7976931348623158e+308})", // Min double value
       R"({"a": 1.7976931348623158e+308})", // Max double value
       R"({"a": -1.7976931348623158e+309})",
       R"({"a": 1.7976931348623158e+309})",
       R"({"a": 0})",
       R"({"a": 1.0e-2000})",
       R"({"a": "3"})",
       R"({"a": 1.1.0})", // Multiple decimal points.
       R"({"a": 1.})", // Missing fraction digits after a decimal point.
       R"({"a": 01})", // Leading zero.
       R"({"a": 1e})", // Missing exponent digits after ‘e’ or ‘E’.
       R"({"a": 1e+})", // Missing exponent digits after ‘e’ or ‘E’.
       R"({"a": .e10})", // Missing digits.
       R"({"a": -.})", // Missing digits entirely.
       R"({"a": "NaN"})",
       R"({"a": NaN})",
       R"({"a": "-Infinity"})",
       R"({"a": -Infinity})",
       R"({"a": "-INF"})",
       R"({"a": -INF})",
       R"({"a": "+Infinity"})",
       R"({"a": +Infinity})",
       R"({"a": "Infinity"})",
       R"({"a": Infinity})",
       R"({"a": "+INF"})",
       R"({"a": +INF})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicDate) {
  auto expected = makeNullableFlatVector<int32_t>(
      {18809,
       18809,
       18809,
       0,
       18809,
       18809,
       -713975,
       15,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt},
      DATE());
  auto input = makeFlatVector<std::string>(
      {R"({"a": "2021-07-01T"})",
       R"({"a": "2021-07-01"})",
       R"({"a": "2021-7"})",
       R"({"a": "1970"})",
       R"({"a": "2021-07-01T00:00GMT+01:00"})",
       R"({"a": "2021-7-GMTGMT1"})",
       R"({"a": "0015-03-16T123123"})",
       R"({"a": "015"})",
       R"({"a": "15.0"})",
       R"({"a": "AA"})",
       R"({"a": "1999-08 01"})",
       R"({"a": "2020/12/1"})",
       R"({"a": ""})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicShortDecimal) {
  auto expected = makeNullableFlatVector<int64_t>(
      {53210, -100, std::nullopt, std::nullopt}, DECIMAL(7, 2));
  auto input = makeFlatVector<std::string>(
      {R"({"a": "5.321E2"})",
       R"({"a": -1})",
       R"({"a": 55555555555555555555.5555})",
       R"({"a": "+1BD"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicLongDecimal) {
  auto expected = makeNullableFlatVector<int128_t>(
      {53210000,
       -100000,
       HugeInt::build(0xffff, 0xffffffffffffffff),
       std::nullopt},
      DECIMAL(38, 5));
  auto input = makeFlatVector<std::string>(
      {R"({"a": "5.321E2"})",
       R"({"a": -1})",
       R"({"a": 12089258196146291747.06175})",
       R"({"a": "+1BD"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicString) {
  auto expected = makeNullableFlatVector<StringView>(
      {"1", "2.0", "true", "{\"b\": \"test\"}", "[1, 2]"});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": 2.0})",
       R"({"a": "true"})",
       R"({"a": {"b": "test"}})",
       R"({"a": [1, 2]})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, nestedComplexType) {
  // ARRAY(ROW(BIGINT))
  std::vector<vector_size_t> offsets;
  offsets.push_back(0);
  offsets.push_back(1);
  offsets.push_back(2);
  auto arrayVector = makeArrayVector(
      offsets, makeRowVector({"a"}, {makeFlatVector<int64_t>({1, 2, 2})}));
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})", R"([{"a": 2}])", R"([{"a": 2}])"});
  testFromJson(input, arrayVector);

  // MAP(ARRAY(ROW(BIGINT, INTEGER)))
  auto keyVector = makeFlatVector<StringView>({"a", "b", "c"});
  auto valueVector = makeArrayVector(
      offsets,
      makeRowVector(
          {"d", "e"},
          {makeFlatVector<int64_t>({1, 2, 3}),
           makeNullableFlatVector<int32_t>({3, 4, std::nullopt})}));
  auto mapVector = makeMapVector(offsets, keyVector, valueVector);
  auto mapInput = makeFlatVector<std::string>(
      {R"({"a": [{"d": 1, "e": 3}]})",
       R"({"b": [{"d": 2, "e": 4}]})",
       R"({"c": [{"d": 3}]})"});
  testFromJson(mapInput, mapVector);

  // ROW(ROW(ROW(BIGINT, INTEGER)))
  auto rowVector = makeRowVector(
      {"a"},
      {makeRowVector(
          {"b"},
          {makeRowVector(
              {"d1", "e1"},
              {makeNullableFlatVector<int64_t>({1, 2, 3}),
               makeNullableFlatVector<int32_t>({3, 4, std::nullopt})})})});
  auto rowInput = makeFlatVector<std::string>(
      {R"({"a": {"b": {"d1": 1, "e1": 3, "e1": 4}}})", // Legacy
                                                       // case-insensitive
                                                       // matching is
                                                       // first-wins.
       R"({"a": {"b": {"D1": 2, "e1": 4}}})", // Case-insensitive match
                                              // (default).
       R"({"a": {"b": {"d1": 3, "f3": 3}}})"}); // Key not in schema.
  testFromJson(rowInput, rowVector);

  // ROW(ARRAY[BIGINT], BIGINT)
  std::vector<vector_size_t> offsets1;
  offsets1.push_back(0);
  offsets1.push_back(1);
  offsets1.push_back(3);
  auto arrayVector1 =
      makeArrayVector(offsets1, makeFlatVector<int64_t>({1, 2, 2, 3, 3, 3}));
  auto rowVector1 = makeRowVector(
      {"a", "b"}, {arrayVector1, makeFlatVector<int64_t>({1, 2, 3})});
  auto rowInput1 = makeFlatVector<std::string>(
      {R"({"a": [1], "b": 1})",
       R"({"a": [2, 2], "b": 2})",
       R"({"a": [3, 3, 3], "b": 3})"});
  testFromJson(rowInput1, rowVector1);

  // ROW(ROW(BIGINT, ARRAY[BIGINT]), ARRAY[BIGINT])
  auto rowVector2 = makeRowVector(
      {"a", "b"},
      {makeRowVector(
           {"c", "d"}, {makeFlatVector<int64_t>({1, 3, 4}), arrayVector1}),
       arrayVector1});
  auto rowInput2 = makeFlatVector<std::string>(
      {R"({"a": {"c": 1, "d": [1]}, "b": [1]})",
       R"({"a": {"c": 3, "d": [2, 2]}, "b": [2, 2]})",
       R"({"a": {"c": 4, "d": [3, 3, 3]}, "b": [3, 3, 3]})"});
  testFromJson(rowInput2, rowVector2);

  // ROW(ROW(ROW(BIGINT)), ROW(ROW(BIGINT, BIGINT)))
  auto rowVector3 = makeRowVector(
      {"a", "d"},
      {makeRowVector(
           {"b"}, {makeRowVector({"c"}, {makeFlatVector<int64_t>({1, 2, 3})})}),
       makeRowVector(
           {"e"},
           {makeRowVector(
               {"f", "g"},
               {makeFlatVector<int64_t>({1, 2, 3}),
                makeFlatVector<int64_t>({4, 5, 6})})})});
  auto rowInput3 = makeFlatVector<std::string>(
      {R"({"a": {"b": {"c": 1}}, "d": {"e": {"f": 1, "g": 4}}})",
       R"({"a": {"b": {"c": 2}}, "d": {"e": {"f": 2, "g": 5}}})",
       R"({"a": {"b": {"c": 3}}, "d": {"e": {"f": 3, "g": 6}}})"});
  testFromJson(rowInput3, rowVector3);

  // ROW(ARRAY[ROW(BIGINT)], ARRAY[ROW(BIGINT, BIGINT)])
  std::vector<vector_size_t> offsets2;
  offsets2.push_back(0);
  offsets2.push_back(1);
  offsets2.push_back(2);
  auto arrayVector3 = makeArrayVector(
      offsets2,
      makeRowVector({"c"}, {makeFlatVector<int64_t>({1, 2, 3, 4, 5})}));
  std::vector<vector_size_t> offsets3;
  offsets3.push_back(0);
  offsets3.push_back(1);
  offsets3.push_back(3);
  auto arrayVector4 = makeArrayVector(
      offsets3,
      makeRowVector(
          {"d", "e"},
          {makeFlatVector<int64_t>({3, 2, 2, 1}),
           makeFlatVector<int64_t>({7, 4, 8, 9})}));
  auto rowVector4 = makeRowVector({"a", "b"}, {arrayVector3, arrayVector4});
  auto rowInput4 = makeFlatVector<std::string>(
      {R"({"a": [{"c": 1}], "b": [{"d": 3, "e": 7}]})",
       R"({"a": [{"c": 2}], "b": [{"d": 2, "e": 4}, {"d": 2, "e": 8}]})",
       R"({"a": [{"c": 3}, {"c": 4}, {"c": 5}], "b": [{"d": 1, "e": 9}]})"});
  testFromJson(rowInput4, rowVector4);
}

TEST_F(FromJsonTest, structEmptyArray) {
  auto expected = makeNullableFlatVector<int64_t>({std::nullopt, 2, 3});
  auto input =
      makeFlatVector<std::string>({R"([])", R"({"a": 2})", R"({"a": 3})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, structEmptyStruct) {
  auto expected = makeNullableFlatVector<int64_t>({std::nullopt, 2, 3});
  auto input =
      makeFlatVector<std::string>({R"({ })", R"({"a": 2})", R"({"a": 3})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, structWrongSchema) {
  auto expected = makeNullableFlatVector<int64_t>({std::nullopt, 2, 3});
  auto input = makeFlatVector<std::string>(
      {R"({"b": 2})", R"({"a": 2})", R"({"a": 3})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, structWrongData) {
  auto expected = makeNullableFlatVector<int64_t>({std::nullopt, 2, 3});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 2.1})", R"({"a": 2})", R"({"a": 3})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, invalidType) {
  auto primitiveTypeOutput = makeFlatVector<int64_t>({2, 2, 3});
  auto mapOutput =
      makeMapVector<int64_t, int64_t>({{{1, 1}}, {{2, 2}}, {{3, 3}}});
  auto input = makeFlatVector<std::string>({R"(2)", R"({2)", R"({3)"});
  VELOX_ASSERT_USER_THROW(
      testFromJson(input, primitiveTypeOutput), "Unsupported type BIGINT.");
  VELOX_ASSERT_USER_THROW(
      testFromJson(input, mapOutput), "Unsupported type MAP<BIGINT,BIGINT>.");
}

// Spark only supports MAP<STRING, _> in from_json.
TEST_F(FromJsonTest, mapNonVarcharKeyRejected) {
  auto input = makeFlatVector<std::string>({R"({"1":"a"})"});
  auto schema = ROW({"m"}, {MAP(INTEGER(), VARCHAR())});
  auto expr = createFromJson(schema);
  VELOX_ASSERT_USER_THROW(
      evaluate(expr, makeRowVector({input})),
      "Unsupported type ROW<m:MAP<INTEGER,VARCHAR>>.");
}

TEST_F(FromJsonTest, invalidJson) {
  auto expected = makeNullableFlatVector<int32_t>(
      {std::nullopt, std::nullopt, std::nullopt});
  auto input =
      makeFlatVector<std::string>({R"("a": 1})", R"({a: 1})", R"({"a" 1})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, objectIterationErrorsFollowParseMode) {
  auto input = makeFlatVector<std::string>(
      {R"({"a":[1] "b":2})", R"({"a":[2],"b":3,})", R"({"a":[4],"b":5})"});
  auto expected = makeRowVector(
      {"a", "b", "_corrupt_record"},
      {makeNullableArrayVector<int32_t>({std::nullopt, std::nullopt, {{4}}}),
       makeNullableFlatVector<int32_t>({std::nullopt, std::nullopt, 5}),
       makeNullableFlatVector<StringView>(
           {StringView(R"({"a":[1] "b":2})"),
            StringView(R"({"a":[2],"b":3,})"),
            std::nullopt})});
  auto fromJson = createFromJsonWithConfig(
      expected->type(), {"columnNameOfCorruptRecord=_corrupt_record"});
  testEncodings(fromJson, {input}, expected);

  auto failfast = createFromJsonWithConfig(expected->type(), {"mode=FAILFAST"});
  auto tryExpr =
      std::make_shared<core::CallTypedExpr>(expected->type(), "try", failfast);
  expected->setNull(0, true);
  expected->setNull(1, true);
  testEncodings(tryExpr, {input}, expected);
  VELOX_ASSERT_USER_THROW(
      evaluate(failfast, makeRowVector({input})),
      failfastMessage(R"({"a":[1] "b":2})"));
}

TEST_F(FromJsonTest, skippedValuesStillValidateJson) {
  const auto deeplyNested = std::string(R"({"ignored":)") +
      std::string(1100, '[') + "0" + std::string(1100, ']') + R"(,"a":5})";
  auto input = makeFlatVector<std::string>(
      {R"({"ignored":nul,"a":1})",
       R"({"ignored":{"x":1 "y":2},"a":2})",
       R"({"a":3,"a":"\q"})",
       R"({"ignored":[1e400,18446744073709551616,NaN],"a":4})",
       deeplyNested});
  auto expected = makeRowVector(
      {"a"},
      {makeNullableFlatVector<int32_t>(
          {std::nullopt, std::nullopt, std::nullopt, 4, std::nullopt})});
  testEncodings(createFromJson(expected->type()), {input}, expected);
  testFromJsonThrows(
      makeFlatVector<std::string>({R"({"ignored":NaN,"a":1})"}),
      expected->type(),
      {"mode=FAILFAST", "allowNonNumericNumbers=false"},
      failfastMessage(R"({"ignored":NaN,"a":1})"));
}

TEST_F(FromJsonTest, jsonNullUsesMalformedRecordPolicy) {
  auto input = makeNullableFlatVector<std::string>(
      {"null", " null ", "\nnull\t", std::nullopt});
  for (const auto& type : std::vector<TypePtr>{
           ROW({"a"}, {INTEGER()}),
           ARRAY(INTEGER()),
           MAP(VARCHAR(), INTEGER())}) {
    auto nulls = BaseVector::createNullConstant(type, input->size(), pool());
    VectorPtr expected = nulls;
    if (type->kind() == TypeKind::ROW) {
      expected = makeRowVector(
          {"a"},
          {makeNullableFlatVector<int32_t>(
              {std::nullopt, std::nullopt, std::nullopt, std::nullopt})});
      expected->setNull(3, true);
    }
    testEncodings(createFromJson(type), {input}, expected);
    auto failfast = createFromJsonWithConfig(type, {"mode=FAILFAST"});
    auto tryExpr = std::make_shared<core::CallTypedExpr>(type, "try", failfast);
    testEncodings(tryExpr, {input}, nulls);
    testFromJsonThrows(input, type, {"mode=FAILFAST"}, failfastMessage("null"));
  }

  auto expected = makeRowVector(
      {"a", "_corrupt_record"},
      {makeNullableFlatVector<int32_t>(
           {std::nullopt, std::nullopt, std::nullopt, std::nullopt}),
       makeNullableFlatVector<StringView>(
           {StringView("null"),
            StringView(" null "),
            StringView("\nnull\t"),
            std::nullopt})});
  expected->setNull(3, true);
  testEncodings(
      createFromJsonWithConfig(
          expected->type(), {"columnNameOfCorruptRecord=_corrupt_record"}),
      {input},
      expected);
}

TEST_F(FromJsonTest, malformedPayloadsCollapseToNullPermissive) {
  {
    SCOPED_TRACE("ROW root");
    auto input = makeNullableFlatVector<std::string>(
        {R"({"a":1})",
         R"({"a":2})",
         std::nullopt,
         std::string("not json"),
         std::string(""),
         std::string("nul"),
         std::string("tru"),
         std::string("123abc"),
         R"({})"});
    auto expected = makeRowVector(
        {"a"},
        {makeNullableFlatVector<int32_t>(
            {1,
             2,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt})});
    expected->setNull(2, true);
    expected->setNull(4, true);
    testFromJson(input, expected);
  }

  {
    SCOPED_TRACE("ARRAY root");
    auto input = makeNullableFlatVector<std::string>(
        {R"([1,2,3])",
         std::nullopt,
         std::string("not json"),
         std::string(""),
         std::string("nul"),
         std::string("tru"),
         std::string("123abc"),
         R"([4])"});
    auto expected = makeNullableArrayVector<int32_t>(
        {{{{1, 2, 3}}},
         std::nullopt,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         {{{4}}}});
    testFromJson(input, expected);
  }

  {
    SCOPED_TRACE("MAP root");
    auto input = makeNullableFlatVector<std::string>(
        {R"({"a":1})",
         std::nullopt,
         std::string("not json"),
         std::string(""),
         std::string("nul"),
         std::string("tru"),
         std::string("123abc"),
         R"({"b":2})"});
    using MapEntry = std::pair<std::string, std::optional<int32_t>>;
    std::vector<std::optional<std::vector<MapEntry>>> data{
        std::vector<MapEntry>{{"a", 1}},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::vector<MapEntry>{{"b", 2}}};
    testFromJson(input, makeNullableMapVector<std::string, int32_t>(data));
  }
}

TEST_F(FromJsonTest, schemaDepthGuard) {
  const auto makeNestedRow = [](int wrappers) {
    TypePtr type = INTEGER();
    for (int i = 0; i < wrappers; ++i) {
      type = ROW({"a"}, {type});
    }
    return type;
  };
  std::string json;
  for (int i = 0; i < 1000; ++i) {
    json += R"({"a":)";
  }
  json += "1";
  json.append(1000, '}');
  auto input = makeRowVector({makeFlatVector<std::string>({json})});

  VectorPtr result = evaluate(createFromJson(makeNestedRow(1000)), input);
  for (int i = 0; i < 1000; ++i) {
    ASSERT_FALSE(result->isNullAt(0)) << "depth " << i;
    result = result->as<RowVector>()->childAt(0);
  }
  ASSERT_FALSE(result->isNullAt(0));
  EXPECT_EQ(result->as<SimpleVector<int32_t>>()->valueAt(0), 1);

  const auto expectedMessage =
      "from_json: schema nesting depth 1001 exceeds maximum 1000 -- deeply "
      "nested ARRAY/MAP/ROW schemas can stack-overflow the worker.";
  VELOX_ASSERT_USER_THROW(
      evaluate(createFromJson(makeNestedRow(1001)), input), expectedMessage);

  TypePtr nestedArray = INTEGER();
  for (int i = 0; i < 1001; ++i) {
    nestedArray = ARRAY(nestedArray);
  }
  VELOX_ASSERT_USER_THROW(
      evaluate(
          createFromJson(ROW({"a"}, {nestedArray})),
          makeRowVector({makeFlatVector<std::string>({R"({"a":[1]})"})})),
      expectedMessage);

  TypePtr nestedMap = INTEGER();
  for (int i = 0; i < 1001; ++i) {
    nestedMap = MAP(VARCHAR(), nestedMap);
  }
  VELOX_ASSERT_USER_THROW(
      evaluate(
          createFromJson(ROW({"m"}, {nestedMap})),
          makeRowVector({makeFlatVector<std::string>({R"({"m":{}})"})})),
      expectedMessage);
}

// Recognized Spark allow* options are accepted while simdjson remains strict.
TEST_F(FromJsonTest, recognizedButUnsupportedAllowOptionsAccepted) {
  auto input = makeFlatVector<std::string>({R"({"a":1})"});
  auto expected = makeFlatVector<int32_t>({1});
  for (const auto& option :
       {"allowSingleQuotes",
        "allowComments",
        "allowUnquotedFieldNames",
        "allowBackslashEscapingAnyCharacter",
        "allowUnquotedControlChars",
        "allowNumericLeadingZeros"}) {
    testFromJsonWithConfig(
        input,
        makeRowVector({"a"}, {expected}),
        {std::string(option) + "=true"});
  }
}

TEST_F(FromJsonTest, boolOptionCaseInsensitive) {
  auto input = makeFlatVector<std::string>({R"({"a":NaN})"});
  auto schemaFloat = ROW({"a"}, {REAL()});
  auto expectedNaN = makeRowVector({"a"}, {makeFlatVector<float>({kNaNFloat})});
  auto expectedNull =
      makeRowVector({"a"}, {makeNullableFlatVector<float>({std::nullopt})});

  for (const auto& spelling : {"true", "TRUE", "True", "trUe"}) {
    auto expr = createFromJsonWithConfig(
        schemaFloat, {std::string("allowNonNumericNumbers=") + spelling});
    auto result = evaluate(expr, makeRowVector({input}));
    assertEqualVectors(expectedNaN, result);
  }

  for (const auto& spelling : {"false", "FALSE", "False"}) {
    auto expr = createFromJsonWithConfig(
        schemaFloat, {std::string("allowNonNumericNumbers=") + spelling});
    auto result = evaluate(expr, makeRowVector({input}));
    assertEqualVectors(expectedNull, result);
  }
}

TEST_F(FromJsonTest, invalidBooleanOptionsRejected) {
  auto input = makeFlatVector<std::string>({R"({"a":1})"});
  for (const auto& option :
       {"allowNonNumericNumbers",
        "enablePartialResults",
        "caseSensitiveFieldMatch",
        "allowSingleQuotes",
        "multiLine"}) {
    for (const auto& spelling : {"no", "0", "", "yes", " true "}) {
      testFromJsonThrows(
          input,
          ROW({"a"}, {INTEGER()}),
          {std::string(option) + "=" + spelling},
          "from_json: boolean option must be true or false, got '" +
              std::string(spelling) + "'.");
    }
  }
}

TEST_F(FromJsonTest, optionNamesAreCaseInsensitive) {
  auto input = makeFlatVector<std::string>(
      {R"({"A":1})", R"({"a":2})", R"({"a":"bad"})"});
  auto expected = makeRowVector(
      {"a", "BadRecord"},
      {makeNullableFlatVector<int32_t>({std::nullopt, 2, std::nullopt}),
       makeNullableFlatVector<StringView>(
           {std::nullopt, std::nullopt, StringView(R"({"a":"bad"})")})});
  auto fromJson = createFromJsonWithConfig(
      expected->type(),
      {"CASESENSITIVEFIELDMATCH=true",
       "ColumnNameOfCorruptRecord=BadRecord",
       "MODE=PERMISSIVE"});
  testEncodings(fromJson, {input}, expected);
  testFromJsonThrows(
      input,
      expected->type(),
      {"MoDe=FAILFAST"},
      failfastMessage(R"({"a":"bad"})"));
}

// Without an option, local timestamps use the query session timezone. Spark's
// `timeZone` option, the `sessionTimezone` alias, and any option-name casing
// override it, including its DST offset. An explicit offset in the value still
// takes precedence.
TEST_F(FromJsonTest, timeZoneAliasMatchesSessionTimezone) {
  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kSessionTimezone, "Asia/Tokyo"}});
  auto input = makeFlatVector<std::string>(
      {R"({"t":"2015-08-26 12:00:00"})",
       R"({"t":"2015-01-15 12:00:00"})",
       R"({"t":"2015-08-26T12:00:00.000+05:30"})"});
  // 12:00 JST (UTC+9) on both dates, and 12:00 at UTC+05:30.
  auto sessionExpected = makeRowVector(
      {"t"},
      {makeFlatVector<Timestamp>(
          {Timestamp(1'440'558'000, 0),
           Timestamp(1'421'290'800, 0),
           Timestamp(1'440'570'600, 0)})});
  testEncodings(
      createFromJson(sessionExpected->type()), {input}, sessionExpected);

  // 12:00 EDT (UTC-4), 12:00 EST (UTC-5), and 12:00 at UTC+05:30.
  auto expected = makeRowVector(
      {"t"},
      {makeFlatVector<Timestamp>(
          {Timestamp(1'440'604'800, 0),
           Timestamp(1'421'341'200, 0),
           Timestamp(1'440'570'600, 0)})});
  for (const auto& option :
       {"timeZone=America/New_York",
        "sessionTimezone=America/New_York",
        "TIMEZONE=America/New_York"}) {
    testEncodings(
        createFromJsonWithConfig(expected->type(), {option}),
        {input},
        expected);
  }
}

// Options that do not affect from_json string parsing are accepted as no-ops.
TEST_F(FromJsonTest, inertSparkJsonOptionsAccepted) {
  auto input = makeFlatVector<std::string>({R"({"a":1})"});
  auto expected = makeFlatVector<int32_t>({1});
  for (const auto& option :
       {"multiLine",
        "prefersDecimal",
        "dropFieldIfAllNull",
        "ignoreNullFields"}) {
    testFromJsonWithConfig(
        input,
        makeRowVector({"a"}, {expected}),
        {std::string(option) + "=true"});
  }
  for (const auto& option :
       {"samplingRatio", "lineSep", "encoding", "charset", "locale"}) {
    testFromJsonWithConfig(
        input,
        makeRowVector({"a"}, {expected}),
        {std::string(option) + "=irrelevant"});
  }
}

// Case-insensitive matching remains the default for backward compatibility.
TEST_F(FromJsonTest, caseInsensitiveMixedCase) {
  auto input = makeFlatVector<std::string>(
      {R"({"Name":"Alice","AGE":30})", R"({"name":"Carol","Name":"Dave"})"});
  auto expected = makeRowVector(
      {"name", "age"},
      {makeNullableFlatVector<StringView>(
           {StringView("Alice"), StringView("Carol")}),
       makeNullableFlatVector<int32_t>({30, std::nullopt})});
  testFromJsonWithConfig(input, expected, {});
}

// Keys are matched after JSON unescaping in both matching modes. A non-ASCII
// schema uses the Unicode lowercase path for legacy matching, which still
// folds ASCII key case; case-sensitive matching does not fold.
TEST_F(FromJsonTest, fieldMatchingUsesUnescapedKeys) {
  auto input = makeFlatVector<std::string>(
      {R"({"\u0061":1,"\u00e9":2})", R"({"A":3,"é":4})", R"({"a":5,"é":6})"});
  auto legacyExpected = makeRowVector(
      {"a", "é"},
      {makeFlatVector<int32_t>({1, 3, 5}), makeFlatVector<int32_t>({2, 4, 6})});
  testEncodings(
      createFromJson(legacyExpected->type()), {input}, legacyExpected);

  auto caseSensitiveExpected = makeRowVector(
      {"a", "é"},
      {makeNullableFlatVector<int32_t>({1, std::nullopt, 5}),
       makeFlatVector<int32_t>({2, 4, 6})});
  testEncodings(
      createFromJsonWithConfig(
          caseSensitiveExpected->type(), {"caseSensitiveFieldMatch=true"}),
      {input},
      caseSensitiveExpected);
}

// Case-sensitive matching uses Spark's last-value-wins behavior.
TEST_F(FromJsonTest, duplicateKeysLastWins) {
  auto input = makeFlatVector<std::string>(
      {R"({"a":1,"a":2,"a":3})", R"({"Id":2,"id":1})", R"({"Id":2})"});
  auto expected = makeRowVector(
      {"a", "id"},
      {makeNullableFlatVector<int32_t>({3, std::nullopt, std::nullopt}),
       makeNullableFlatVector<int32_t>({std::nullopt, 1, std::nullopt})});
  testFromJsonWithConfig(input, expected, {"caseSensitiveFieldMatch=true"});
}

TEST_F(FromJsonTest, duplicateKeysMultiField) {
  auto input =
      makeFlatVector<std::string>({R"({"a":"first","b":1,"a":"second"})"});
  auto expected = makeRowVector(
      {"a", "b"},
      {makeNullableFlatVector<StringView>({StringView("second")}),
       makeNullableFlatVector<int32_t>({1})});
  testFromJsonWithConfig(input, expected, {"caseSensitiveFieldMatch=true"});
}

TEST_F(FromJsonTest, duplicateKeysNested) {
  auto input = makeFlatVector<std::string>({R"({"o":{"y":1,"y":2}})"});
  auto inner = makeRowVector({"y"}, {makeNullableFlatVector<int32_t>({2})});
  auto expected = makeRowVector({"o"}, {inner});
  testFromJsonWithConfig(input, expected, {"caseSensitiveFieldMatch=true"});
}

TEST_F(FromJsonTest, duplicateKeysComplexValuesLastWins) {
  const std::vector<std::string> options{"caseSensitiveFieldMatch=true"};
  {
    SCOPED_TRACE("ARRAY");
    auto input = makeFlatVector<std::string>(
        {R"({"a":[1],"a":[2]})",
         R"({"a":[1,2,3],"a":[9]})",
         R"({"a":[1],"a":[]})"});
    auto expected =
        makeRowVector({"a"}, {makeArrayVector<int32_t>({{2}, {9}, {}})});
    testFromJsonWithConfig(input, expected, options);
  }

  {
    SCOPED_TRACE("MAP");
    auto input = makeFlatVector<std::string>(
        {R"({"m":{"k":1},"m":{"k":2}})",
         R"({"m":{"a":1,"b":2},"m":{"c":3}})",
         R"({"m":{"x":1},"m":{}})"});
    auto expected = makeRowVector(
        {"m"},
        {makeMapVector<std::string, int32_t>({{{"k", 2}}, {{"c", 3}}, {}})});
    testFromJsonWithConfig(input, expected, options);
  }

  {
    SCOPED_TRACE("nested ROW");
    auto input =
        makeFlatVector<std::string>({R"({"r":{"a":[1]},"r":{"a":[2]}})"});
    auto expected = makeRowVector(
        {"r"}, {makeRowVector({"a"}, {makeArrayVector<int32_t>({{2}})})});
    testFromJsonWithConfig(input, expected, options);
  }
}

TEST_F(FromJsonTest, duplicateKeysNullHandling) {
  {
    SCOPED_TRACE("case-sensitive");
    auto input = makeFlatVector<std::string>(
        {R"({"a":1,"a":null})",
         R"({"a":42,"a":null})",
         R"({"a":7,"b":"x","a":null})"});
    auto expected = makeRowVector(
        {"a", "b"},
        {makeNullableFlatVector<int32_t>(
             {std::nullopt, std::nullopt, std::nullopt}),
         makeNullableFlatVector<StringView>(
             {std::nullopt, std::nullopt, StringView("x")})});
    testFromJsonWithConfig(input, expected, {"caseSensitiveFieldMatch=true"});
  }

  {
    SCOPED_TRACE("case-insensitive");
    auto input = makeFlatVector<std::string>(
        {R"({"a":null,"a":2})",
         R"({"a":null,"a":42})",
         R"({"a":null,"a":null})"});
    auto expected = makeRowVector(
        {"a"}, {makeNullableFlatVector<int32_t>({2, 42, std::nullopt})});
    testFromJsonWithConfig(input, expected, {});
  }
}

TEST_F(FromJsonTest, duplicateKeysCaseInsensitiveFirstWins) {
  auto input = makeFlatVector<std::string>(
      {R"({"a":1,"a":2})",
       R"({"a":7,"a":8,"a":9})",
       R"({"a":null,"a":1})",
       R"({"a":2,"a":null})",
       R"({"id":1,"Id":2})",
       R"({"Id":2,"id":1})"});
  auto expected = makeRowVector(
      {"a", "id"},
      {makeNullableFlatVector<int32_t>(
           {1, 7, 1, 2, std::nullopt, std::nullopt}),
       makeNullableFlatVector<int32_t>(
           {std::nullopt, std::nullopt, std::nullopt, std::nullopt, 1, 2})});
  testFromJsonWithConfig(input, expected, {});
}

// simdjson remains RFC-8259 strict when allowSingleQuotes is provided.
TEST_F(FromJsonTest, allowSingleQuotesIgnoredDivergesFromSpark) {
  auto input = makeFlatVector<std::string>({R"({'a':1})"});
  auto expected =
      makeRowVector({"a"}, {makeNullableFlatVector<int32_t>({std::nullopt})});
  testFromJsonWithConfig(input, expected, {"allowSingleQuotes=true"});
}

TEST_F(FromJsonTest, failfastValidJson) {
  auto input = makeFlatVector<std::string>({R"({"a":1})"});
  auto expected = makeRowVector({"a"}, {makeNullableFlatVector<int32_t>({1})});
  testFromJsonWithConfig(input, expected, {"mode=FAILFAST"});
}

TEST_F(FromJsonTest, failfastTypeMismatch) {
  auto input = makeFlatVector<std::string>({R"({"a":"not_int"})"});
  auto outputType = ROW({"a"}, {INTEGER()});
  testFromJsonThrows(
      input,
      outputType,
      {"mode=FAILFAST"},
      failfastMessage(R"({"a":"not_int"})"));
}

TEST_F(FromJsonTest, failfastInvalidJson) {
  auto input = makeFlatVector<std::string>({R"(bad json content here)"});
  auto outputType = ROW({"a"}, {INTEGER()});
  testFromJsonThrows(
      input,
      outputType,
      {"mode=FAILFAST"},
      failfastMessage("bad json content here"));
}

TEST_F(FromJsonTest, failfastSuppressedErrorDoesNotLeakWriterState) {
  auto input = makeFlatVector<std::string>({R"([1,2,"bad"])", R"([5])"});
  auto outputType = ARRAY(INTEGER());
  auto fromJson = createFromJsonWithConfig(outputType, {"mode=FAILFAST"});
  auto tryExpr =
      std::make_shared<core::CallTypedExpr>(outputType, "try", fromJson);
  auto result = evaluate(tryExpr, makeRowVector({input}));
  auto expected = makeNullableArrayVector<int32_t>({std::nullopt, {{{5}}}});
  assertEqualVectors(expected, result);
}

TEST_F(FromJsonTest, constantFailfastFinishesWriterBeforeThrow) {
  auto input = makeConstant(StringView(R"([1,2,"bad"])"), 2);
  auto outputType = ARRAY(INTEGER());
  auto fromJson = createFromJsonWithConfig(outputType, {"mode=FAILFAST"});
  auto tryExpr =
      std::make_shared<core::CallTypedExpr>(outputType, "try", fromJson);
  auto result = evaluate(tryExpr, makeRowVector({input}));
  auto expected =
      makeNullableArrayVector<int32_t>({std::nullopt, std::nullopt});
  assertEqualVectors(expected, result);
}

// Constant-input FAILFAST errors are reported through EvalCtx for every row,
// so TRY suppresses them as it does for flat input.
TEST_F(FromJsonTest, constantFailfastPartialResultSuppressedByTry) {
  auto input = makeConstant(StringView(R"({"a":"bad","b":1})"), 3);
  auto outputType = ROW({"a", "b"}, {INTEGER(), INTEGER()});
  auto fromJson = createFromJsonWithConfig(outputType, {"mode=FAILFAST"});
  auto tryExpr =
      std::make_shared<core::CallTypedExpr>(outputType, "try", fromJson);
  auto result = evaluate(tryExpr, makeRowVector({input}));
  assertEqualVectors(
      BaseVector::createNullConstant(outputType, 3, pool()), result);
  VELOX_ASSERT_USER_THROW(
      evaluate(fromJson, makeRowVector({input})),
      failfastMessage(R"({"a":"bad","b":1})"));
}

TEST_F(FromJsonTest, constantFunctionPreservesUnselectedRows) {
  auto outputType = ARRAY(INTEGER());
  exec::ExprSet exprSet(
      {createFromJsonWithConfig(outputType, {"mode=FAILFAST"})}, &execCtx_);
  auto function = exprSet.expr(0)->vectorFunction();
  SelectivityVector rows(6, false);
  rows.setValid(2, true);
  rows.setValid(4, true);
  rows.updateBounds();

  for (const auto& json : {R"([1,2,"bad"])", R"([1,2])"}) {
    std::vector<VectorPtr> args{makeConstant(StringView(json), 6)};
    auto input = makeRowVector(args);
    exec::EvalCtx context(&execCtx_, &exprSet, input.get());
    context.mutableThrowOnError() = false;
    VectorPtr result = makeArrayVector<int32_t>({{0}, {1}, {2}, {3}, {4}, {5}});
    function->apply(rows, args, outputType, context, result);
    if (std::string_view(json) == R"([1,2,"bad"])") {
      ASSERT_NE(context.errors(), nullptr);
      for (vector_size_t row = 0; row < 6; ++row) {
        EXPECT_EQ(context.errors()->hasErrorAt(row), rows.isValid(row));
      }
      assertEqualVectors(
          makeNullableArrayVector<int32_t>(
              {{{0}}, {{1}}, std::nullopt, {{3}}, std::nullopt, {{5}}}),
          result);
    } else {
      assertEqualVectors(
          makeArrayVector<int32_t>({{0}, {1}, {1, 2}, {3}, {1, 2}, {5}}),
          result);
    }

    auto saved = BaseVector::copy(*result, pool());
    SelectivityVector emptyRows(6, false);
    function->apply(emptyRows, args, outputType, context, result);
    assertEqualVectors(saved, result);
  }
}

TEST_F(FromJsonTest, preservesResultsInConditionalExpressions) {
  auto outputType = ARRAY(INTEGER());
  const auto fromJson = [&](const std::string& fieldName) {
    std::vector<core::TypedExprPtr> inputs{
        std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), fieldName)};
    return std::make_shared<const core::CallTypedExpr>(
        outputType, std::move(inputs), "from_json");
  };
  const auto condition = [](const std::string& fieldName) {
    return std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), fieldName);
  };

  auto input = makeRowVector(
      {makeFlatVector<std::string>({R"([1])", R"([2])", R"([3])", R"([4])"}),
       makeFlatVector<bool>({true, false, false, true}),
       makeFlatVector<std::string>(
           {R"([10])", R"([20])", R"([30])", R"([40])"}),
       makeFlatVector<bool>({false, true, false, false}),
       makeFlatVector<std::string>(
           {R"([100])", R"([200])", R"([300])", R"([400])"})});

  auto ifExpression = std::make_shared<core::CallTypedExpr>(
      outputType, "if", condition("c1"), fromJson("c0"), fromJson("c2"));
  assertEqualVectors(
      makeArrayVector<int32_t>({{1}, {20}, {30}, {4}}),
      evaluate(ifExpression, input));

  auto caseExpression = std::make_shared<core::CallTypedExpr>(
      outputType,
      "switch",
      condition("c1"),
      fromJson("c0"),
      condition("c3"),
      fromJson("c2"),
      fromJson("c4"));
  assertEqualVectors(
      makeArrayVector<int32_t>({{1}, {20}, {300}, {4}}),
      evaluate(caseExpression, input));
}

TEST_F(FromJsonTest, suppressedConversionErrorDoesNotLeakNestedState) {
  auto input = makeFlatVector<std::string>(
      {R"({"a":[1,2],"ts":"2024-01-01 UTC"})", R"({"a":[3]})"});
  auto outputType = ROW({"a", "ts"}, {ARRAY(INTEGER()), TIMESTAMP()});
  // Parsing time zone long names throws a user error for every value.
  auto fromJson =
      createFromJsonWithConfig(outputType, {"timestampFormat=yyyy-MM-dd zzzz"});
  auto tryExpr =
      std::make_shared<core::CallTypedExpr>(outputType, "try", fromJson);
  auto result = evaluate(tryExpr, makeRowVector({input}));
  auto expected = makeRowVector(
      {"a", "ts"},
      {makeArrayVector<int32_t>({{}, {3}}),
       makeNullableFlatVector<Timestamp>({std::nullopt, std::nullopt})});
  expected->setNull(0, true);
  assertEqualVectors(expected, result);
}

// Spark's parser finds no first token in empty or JSON-whitespace-only input
// and returns a top-level null in every parse mode. The corrupt-record column
// is not populated, and FAILFAST does not throw.
TEST_F(FromJsonTest, blankInputReturnsNullRowInEveryMode) {
  auto input = makeFlatVector<std::string>({"", " \n\t", "\r\n"});
  auto rowType = ROW({"a", "_corrupt_record"}, {INTEGER(), VARCHAR()});
  for (const auto& options : std::vector<std::vector<std::string>>{
           {},
           {"mode=FAILFAST"},
           {"columnNameOfCorruptRecord=_corrupt_record"},
           {"mode=FAILFAST", "columnNameOfCorruptRecord=_corrupt_record"}}) {
    testEncodings(
        createFromJsonWithConfig(rowType, options),
        {input},
        BaseVector::createNullConstant(rowType, input->size(), pool()));
  }
  auto arrayType = ARRAY(INTEGER());
  testEncodings(
      createFromJsonWithConfig(arrayType, {"mode=FAILFAST"}),
      {input},
      BaseVector::createNullConstant(arrayType, input->size(), pool()));
}

// Only JSON whitespace is blank. Other whitespace characters are invalid JSON
// tokens, so they follow the malformed-record policy as in Spark.
TEST_F(FromJsonTest, nonJsonWhitespaceIsMalformed) {
  auto input = makeFlatVector<std::string>(
      {std::string("\v"), std::string("\f"), std::string("\xE3\x80\x80")});
  auto outputType = ROW({"a"}, {INTEGER()});
  auto expected = makeRowVector(
      {"a"},
      {makeNullableFlatVector<int32_t>(
          {std::nullopt, std::nullopt, std::nullopt})});
  testFromJson(input, expected);
  testFromJsonThrows(
      input,
      outputType,
      {"mode=FAILFAST"},
      failfastMessage(std::string_view("\v", 1)));
}

TEST_F(FromJsonTest, failfastPartialFields) {
  auto input = makeFlatVector<std::string>({R"({"a":1})"});
  auto expected = makeRowVector(
      {"a", "b"},
      {makeNullableFlatVector<int32_t>({1}),
       makeNullableFlatVector<StringView>({std::nullopt})});
  testFromJsonWithConfig(input, expected, {"mode=FAILFAST"});
}

// Spark's from_json does not support DROPMALFORMED mode in any casing.
TEST_F(FromJsonTest, dropmalformedRejected) {
  auto input = makeFlatVector<std::string>({R"({"a":1})"});
  auto outputType = ROW({"a"}, {INTEGER()});
  for (const auto& mode :
       {"mode=DROPMALFORMED", "mode=dropMalformed", "mode=dropmalformed"}) {
    const auto value = std::string_view(mode).substr(5);
    testFromJsonThrows(
        input,
        outputType,
        {mode},
        "[PARSE_MODE_UNSUPPORTED] The function `from_json` doesn't support "
        "the " +
            std::string(value) +
            " mode. Acceptable modes are PERMISSIVE and FAILFAST.");
  }
}

// Spark's ParseMode.fromString falls back to PERMISSIVE for unknown modes,
// including an empty value and near-miss spellings of FAILFAST.
TEST_F(FromJsonTest, unknownModeUsesPermissive) {
  auto input =
      makeFlatVector<std::string>({R"({"a":1})", "bad json", R"({"a":"x"})"});
  auto expected = makeRowVector(
      {"a", "_corrupt_record"},
      {makeNullableFlatVector<int32_t>({1, std::nullopt, std::nullopt}),
       makeNullableFlatVector<StringView>(
           {std::nullopt,
            StringView("bad json"),
            StringView(R"({"a":"x"})")})});
  for (const auto& mode : {"mode=BOGUS", "mode=", "mode=FAIL_FAST"}) {
    testEncodings(
        createFromJsonWithConfig(
            expected->type(),
            {mode, "columnNameOfCorruptRecord=_corrupt_record"}),
        {input},
        expected);
  }
}

TEST_F(FromJsonTest, modeIsCaseInsensitive) {
  auto input = makeFlatVector<std::string>({"bad json"});
  auto outputType = ROW({"a"}, {INTEGER()});
  testFromJsonThrows(
      input, outputType, {"mode=Failfast"}, failfastMessage("bad json"));
}

TEST_F(FromJsonTest, invalidSessionTimezoneRejected) {
  auto input = makeFlatVector<std::string>({R"({"a":1})"});
  auto outputType = ROW({"a"}, {INTEGER()});
  testFromJsonThrows(
      input,
      outputType,
      {"sessionTimezone=Mars/Olympus_Mons"},
      "Invalid sessionTimezone 'Mars/Olympus_Mons'.");
  testFromJsonThrows(
      input, outputType, {"timeZone="}, "Invalid sessionTimezone ''.");
}

TEST_F(FromJsonTest, unknownOptionRejected) {
  auto input = makeFlatVector<std::string>({R"({"a":1})"});
  auto outputType = ROW({"a"}, {INTEGER()});
  testFromJsonThrows(
      input,
      outputType,
      {"caseSensetive=true"},
      "from_json: unrecognized option 'caseSensetive'. Supported options: "
      "allowNonNumericNumbers, mode, columnNameOfCorruptRecord, "
      "timestampFormat, dateFormat, enablePartialResults, sessionTimezone "
      "(alias: timeZone), caseSensitiveFieldMatch. Spark allow* flags "
      "(allowSingleQuotes, allowComments, allowUnquotedFieldNames, "
      "allowBackslashEscapingAnyCharacter, allowUnquotedControlChars, "
      "allowNumericLeadingZeros) and inert Spark JSONOptions keys "
      "(multiLine, prefersDecimal, dropFieldIfAllNull, samplingRatio, "
      "lineSep, encoding, charset, ignoreNullFields, locale) are "
      "accepted-but-ignored with a warning.");
}

TEST_F(FromJsonTest, malformedOptionRejected) {
  auto input = makeFlatVector<std::string>({R"({"a":1})"});
  auto outputType = ROW({"a"}, {INTEGER()});
  testFromJsonThrows(
      input,
      outputType,
      {"enablePartialResults"},
      "from_json: optional argument 1 must be of the form \"key=value\", got "
      "'enablePartialResults'.");

  std::vector<core::TypedExprPtr> nullInputs{
      std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
      std::make_shared<core::ConstantTypedExpr>(
          VARCHAR(), variant::null(TypeKind::VARCHAR))};
  auto nullExpression = std::make_shared<const core::CallTypedExpr>(
      outputType, std::move(nullInputs), "from_json");
  VELOX_ASSERT_USER_THROW(
      evaluate(nullExpression, makeRowVector({input})),
      "from_json: optional argument 1 must not be null.");

  std::vector<core::TypedExprPtr> nonConstantInputs{
      std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
      std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c1")};
  auto nonConstantExpression = std::make_shared<const core::CallTypedExpr>(
      outputType, std::move(nonConstantInputs), "from_json");
  VELOX_ASSERT_USER_THROW(
      evaluate(
          nonConstantExpression,
          makeRowVector(
              {input,
               makeFlatVector<std::string>({"enablePartialResults=true"})})),
      "from_json: optional argument 1 must be a constant string of the form "
      "\"key=value\".");

  std::vector<core::TypedExprPtr> nonVarcharInputs{
      std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
      std::make_shared<core::ConstantTypedExpr>(INTEGER(), 1)};
  auto nonVarcharExpression = std::make_shared<const core::CallTypedExpr>(
      outputType, std::move(nonVarcharInputs), "from_json");
  VELOX_ASSERT_USER_THROW(
      evaluate(nonVarcharExpression, makeRowVector({input})),
      "from_json: optional argument 1 must be VARCHAR, got INTEGER.");
}

// ARRAY and MAP roots have no corrupt-record column, so malformed input and
// conversion failures still return a null root.
TEST_F(FromJsonTest, corruptRecordIgnoredForNonRow) {
  const std::vector<std::string> options{
      "columnNameOfCorruptRecord=_corrupt_record"};
  auto arrayInput =
      makeFlatVector<std::string>({R"([1,2,3])", "bad json", R"([1,"x"])"});
  auto expectedArray = makeNullableArrayVector<int32_t>(
      {{{1, 2, 3}}, std::nullopt, std::nullopt});
  testEncodings(
      createFromJsonWithConfig(expectedArray->type(), options),
      {arrayInput},
      expectedArray);

  auto mapInput =
      makeFlatVector<std::string>({R"({"k":1})", "bad json", R"({"k":"x"})"});
  using MapEntry = std::pair<std::string, std::optional<int32_t>>;
  auto expectedMap = makeNullableMapVector<std::string, int32_t>(
      std::vector<std::optional<std::vector<MapEntry>>>{
          std::vector<MapEntry>{{"k", 1}}, std::nullopt, std::nullopt});
  testEncodings(
      createFromJsonWithConfig(expectedMap->type(), options),
      {mapInput},
      expectedMap);
}

// Without a matching corrupt-record column, malformed records keep Spark's
// non-null row with null fields. The column lookup is case-sensitive, so a
// differently cased column is an ordinary data field.
TEST_F(FromJsonTest, corruptRecordIgnoredWhenColumnIsAbsent) {
  const std::vector<std::string> options{
      "columnNameOfCorruptRecord=_corrupt_record"};
  auto input =
      makeFlatVector<std::string>({R"({"a":1})", "bad json", R"({"a":"x"})"});
  auto expected = makeRowVector(
      {"a"},
      {makeNullableFlatVector<int32_t>({1, std::nullopt, std::nullopt})});
  testEncodings(
      createFromJsonWithConfig(expected->type(), options), {input}, expected);

  auto casedInput = makeFlatVector<std::string>(
      {R"({"a":1,"_Corrupt_Record":"data"})", "bad json", R"({"a":"x"})"});
  auto casedExpected = makeRowVector(
      {"a", "_Corrupt_Record"},
      {makeNullableFlatVector<int32_t>({1, std::nullopt, std::nullopt}),
       makeNullableFlatVector<StringView>(
           {StringView("data"), std::nullopt, std::nullopt})});
  testEncodings(
      createFromJsonWithConfig(casedExpected->type(), options),
      {casedInput},
      casedExpected);
}

// Spark requires the corrupt-record column to be a string.
TEST_F(FromJsonTest, corruptRecordColumnMustBeVarchar) {
  auto input = makeFlatVector<std::string>({"bad json"});
  testFromJsonThrows(
      input,
      ROW({"a", "_corrupt_record"}, {INTEGER(), INTEGER()}),
      {"columnNameOfCorruptRecord=_corrupt_record"},
      "The corrupt record column '_corrupt_record' must be of VARCHAR type, "
      "got INTEGER.");
}

TEST_F(FromJsonTest, nonNumericNumbersDisallowed) {
  auto input = makeFlatVector<std::string>(
      {R"({"f":NaN,"d":"NaN","i":1})",
       R"({"f":Infinity,"d":"Infinity","i":2})",
       R"({"f":"+Infinity","d":+Infinity,"i":3})",
       R"({"f":"+INF","d":+INF,"i":4})",
       R"({"f":-Infinity,"d":"-Infinity","i":5})",
       R"({"f":"-INF","d":-INF,"i":6})",
       R"({"f":1.5,"d":-2.5,"i":7})"});
  std::vector<std::optional<StringView>> corrupt;
  for (vector_size_t row = 0; row < input->size() - 1; ++row) {
    corrupt.emplace_back(input->valueAt(row));
  }
  corrupt.emplace_back(std::nullopt);
  auto expected = makeRowVector(
      {"f", "d", "i", "_corrupt_record"},
      {makeNullableFlatVector<float>(
           {std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            1.5}),
       makeNullableFlatVector<double>(
           {std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            -2.5}),
       makeFlatVector<int32_t>({1, 2, 3, 4, 5, 6, 7}),
       makeNullableFlatVector<StringView>(corrupt)});
  testEncodings(
      createFromJsonWithConfig(
          expected->type(),
          {"allowNonNumericNumbers=false",
           "columnNameOfCorruptRecord=_corrupt_record"}),
      {input},
      expected);
  testFromJsonThrows(
      input,
      expected->type(),
      {"allowNonNumericNumbers=false", "mode=FAILFAST"},
      failfastMessage(R"({"f":NaN,"d":"NaN","i":1})"));
}

TEST_F(FromJsonTest, numericOverflowIgnoresNonNumericOption) {
  auto input = makeFlatVector<std::string>(
      {R"({"f":1e39,"d":1e400})",
       R"({"f":-1e39,"d":-1e400})",
       R"({"f":1e-400,"d":1e-400})"});
  auto expected = makeRowVector(
      {"f", "d"},
      {makeFlatVector<float>({kInfFloat, -kInfFloat, 0}),
       makeFlatVector<double>({kInfDouble, -kInfDouble, 0})});
  auto fromJson = createFromJsonWithConfig(
      expected->type(), {"allowNonNumericNumbers=false", "mode=FAILFAST"});
  testEncodings(fromJson, {input}, expected);
}

TEST_F(FromJsonTest, floatingPointTokensAreDecodedAndValidated) {
  auto input = makeFlatVector<std::string>(
      {R"({"a":"N\u0061N"})",
       R"({"a":"\u002bInfinity" })",
       R"({"a":-INF })",
       R"({"a":nan})",
       R"({"a":+1e400})",
       R"({"a":-01e400})",
       R"({"a":1.e400})",
       R"({"a":"\"NaN\""})"});
  for (const auto& type : std::vector<TypePtr>{REAL(), DOUBLE()}) {
    VectorPtr values;
    if (type->isReal()) {
      values = makeNullableFlatVector<float>(
          {kNaNFloat,
           kInfFloat,
           -kInfFloat,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt});
    } else {
      values = makeNullableFlatVector<double>(
          {kNaNDouble,
           kInfDouble,
           -kInfDouble,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt});
    }
    auto expected = makeRowVector({"a"}, {values});
    testEncodings(createFromJson(expected->type()), {input}, expected);
    auto disallowed = createFromJsonWithConfig(
        expected->type(), {"allowNonNumericNumbers=false"});
    auto nullValues =
        BaseVector::createNullConstant(type, input->size(), pool());
    testEncodings(disallowed, {input}, makeRowVector({"a"}, {nullValues}));
  }
}

TEST_F(FromJsonTest, corruptRecordCapturesFailedRows) {
  auto input = makeFlatVector<std::string>(
      {R"({"a":1,"b":2})",
       R"(bad json)",
       R"({"a":"not_int","b":2})",
       R"({"a":1,"b":"x"})",
       R"({"a":3})"});
  auto expected = makeRowVector(
      {"a", "b", "_corrupt_record"},
      {makeNullableFlatVector<int32_t>({1, std::nullopt, std::nullopt, 1, 3}),
       makeNullableFlatVector<int32_t>(
           {2, std::nullopt, 2, std::nullopt, std::nullopt}),
       makeNullableFlatVector<StringView>(
           {std::nullopt,
            input->valueAt(1),
            input->valueAt(2),
            input->valueAt(3),
            std::nullopt})});
  testEncodings(
      createFromJsonWithConfig(
          expected->type(), {"columnNameOfCorruptRecord=_corrupt_record"}),
      {input},
      expected);
}

TEST_F(FromJsonTest, corruptRecordColumnIsNotParsedFromJson) {
  auto input = makeFlatVector<std::string>(
      {R"({"a":1,"_corrupt_record":"ignored"})",
       R"(malformed record that is longer than inline storage)",
       R"({"a":3,"_corrupt_record":"also ignored"})"});
  auto expected = makeRowVector(
      {"a", "_corrupt_record"},
      {makeNullableFlatVector<int32_t>({1, std::nullopt, 3}),
       makeNullableFlatVector<StringView>(
           {std::nullopt,
            StringView("malformed record that is longer than inline storage"),
            std::nullopt})});
  testFromJsonWithConfig(
      input, expected, {"columnNameOfCorruptRecord=_corrupt_record"});
}

TEST_F(FromJsonTest, emptyCorruptRecordColumnName) {
  auto input = makeFlatVector<std::string>(
      {R"({"a":1,"":"ignored"})", "bad json", R"({"a":3})"});
  auto expected = makeRowVector(
      {"a", ""},
      {makeNullableFlatVector<int32_t>({1, std::nullopt, 3}),
       makeNullableFlatVector<StringView>(
           {std::nullopt, StringView("bad json"), std::nullopt})});
  testEncodings(
      createFromJsonWithConfig(
          expected->type(), {"columnNameOfCorruptRecord="}),
      {input},
      expected);
}

TEST_F(FromJsonTest, customDateFormat) {
  auto input = makeFlatVector<std::string>({R"({"d":"08/26/2015"})"});
  auto expr =
      createFromJsonWithConfig(ROW({"d"}, {DATE()}), {"dateFormat=MM/dd/yyyy"});
  auto result = evaluate(expr, makeRowVector({input}));
  auto expectedRow = makeRowVector(
      {"d"},
      {makeFlatVector<int32_t>(
          1, [](auto) { return 16673; }, nullptr, DATE())});
  assertEqualVectors(expectedRow, result);
}

TEST_F(FromJsonTest, dateFormatMismatch) {
  auto input = makeFlatVector<std::string>({R"({"d":"2015-08-26"})"});
  auto expr =
      createFromJsonWithConfig(ROW({"d"}, {DATE()}), {"dateFormat=MM/dd/yyyy"});
  auto result = evaluate(expr, makeRowVector({input}));
  auto expectedRow = makeRowVector(
      {"d"},
      {makeFlatVector<int32_t>(
          1, [](auto) { return 0; }, [](auto) { return true; }, DATE())});
  assertEqualVectors(expectedRow, result);
}

TEST_F(FromJsonTest, emptyFormatsDoNotUseDefaultParser) {
  auto input = makeFlatVector<std::string>(
      {R"({"d":"2024-01-01","ts":"2024-01-01 00:00:00"})",
       R"({"d":"1970-01-01","ts":1})",
       R"({"d":"","ts":""})"});
  auto expected = makeRowVector(
      {"d", "ts"},
      {makeNullableFlatVector<int32_t>(
           {std::nullopt, std::nullopt, std::nullopt}, DATE()),
       makeNullableFlatVector<Timestamp>(
           {std::nullopt, Timestamp(1, 0), std::nullopt})});
  testEncodings(
      createFromJsonWithConfig(
          expected->type(), {"dateFormat=", "timestampFormat="}),
      {input},
      expected);
}

TEST_F(FromJsonTest, dateFormatIgnoresSessionTimezone) {
  auto input = makeFlatVector<std::string>({R"({"d":"08/26/2015"})"});
  auto expr = createFromJsonWithConfig(
      ROW({"d"}, {DATE()}),
      {"dateFormat=MM/dd/yyyy", "sessionTimezone=America/Los_Angeles"});
  auto result = evaluate(expr, makeRowVector({input}));
  auto expectedRow = makeRowVector(
      {"d"},
      {makeFlatVector<int32_t>(
          1, [](auto) { return 16673; }, nullptr, DATE())});
  assertEqualVectors(expectedRow, result);
}

// The default CORRECTED policy uses the Joda formatter, which requires the
// whole input to match. spark.sql.legacy.timeParserPolicy=LEGACY selects
// SimpleDateFormat-compatible parsing, which ignores trailing text.
TEST_F(FromJsonTest, dateFormatFollowsLegacyTimeParserPolicy) {
  auto input = makeFlatVector<std::string>(
      {R"({"d":"2015-08-26"})",
       R"({"d":"2015-08-26junk"})",
       R"({"d":"08/26/2015"})"});
  auto rowType = ROW({"d"}, {DATE()});
  testEncodings(
      createFromJsonWithConfig(rowType, {"dateFormat=yyyy-MM-dd"}),
      {input},
      makeRowVector(
          {"d"},
          {makeNullableFlatVector<int32_t>(
              {16'673, std::nullopt, std::nullopt}, DATE())}));

  enableLegacyDateFormatter();
  testEncodings(
      createFromJsonWithConfig(rowType, {"dateFormat=yyyy-MM-dd"}),
      {input},
      makeRowVector(
          {"d"},
          {makeNullableFlatVector<int32_t>(
              {16'673, 16'673, std::nullopt}, DATE())}));
}

TEST_F(FromJsonTest, partialResultsAtRoot) {
  auto input = makeFlatVector<std::string>({R"({"a":1,"b":"x"})"});
  auto expected = makeRowVector(
      {"a", "b"},
      {makeNullableFlatVector<int32_t>({1}),
       makeNullableFlatVector<int32_t>({std::nullopt})});
  for (const auto enabled : {true, false}) {
    SCOPED_TRACE(enabled);
    testFromJsonWithConfig(
        input,
        expected,
        {std::string("enablePartialResults=") + (enabled ? "true" : "false")});
  }
}

TEST_F(FromJsonTest, partialResultsInNestedRow) {
  {
    SCOPED_TRACE("disabled");
    auto input = makeFlatVector<std::string>(
        {R"({"a":1,"nested":{"x":"bad","y":2}})",
         R"({"a":2,"nested":{"x":3,"y":4}})",
         R"({"a":5,"nested":{"x":6,"y":"bad"}})"});
    auto expected = makeRowVector(
        {"a", "nested", "_corrupt_record"},
        {makeFlatVector<int32_t>({1, 2, 5}),
         makeRowVector(
             {"x", "y"},
             {makeFlatVector<int32_t>({0, 3, 0}),
              makeFlatVector<int32_t>({0, 4, 0})},
             [](auto row) { return row != 1; }),
         makeNullableFlatVector<StringView>(
             {input->valueAt(0), std::nullopt, input->valueAt(2)})});
    testEncodings(
        createFromJsonWithConfig(
            expected->type(),
            {"enablePartialResults=false",
             "columnNameOfCorruptRecord=_corrupt_record"}),
        {input},
        expected);
    testFromJsonThrows(
        input,
        expected->type(),
        {"enablePartialResults=false", "mode=FAILFAST"},
        failfastMessage(R"({"a":1,"nested":{"x":"bad","y":2}})"));
  }

  {
    SCOPED_TRACE("enabled");
    auto input =
        makeFlatVector<std::string>({R"({"a":1,"nested":{"x":"bad","y":2}})"});
    auto expected = makeRowVector(
        {"a", "nested"},
        {makeFlatVector<int32_t>({1}),
         makeRowVector(
             {"x", "y"},
             {makeNullableFlatVector<int32_t>({std::nullopt}),
              makeFlatVector<int32_t>({2})})});
    testFromJsonWithConfig(input, expected, {"enablePartialResults=true"});
  }
}

TEST_F(FromJsonTest, failfastOverridesCorruptRecord) {
  auto input = makeFlatVector<std::string>({R"(bad json)"});
  auto outputType = ROW({"a", "_corrupt_record"}, {INTEGER(), VARCHAR()});
  testFromJsonThrows(
      input,
      outputType,
      {"mode=FAILFAST", "columnNameOfCorruptRecord=_corrupt_record"},
      failfastMessage("bad json"));
}

TEST_F(FromJsonTest, multiRowBatch) {
  auto input = makeFlatVector<std::string>(
      {R"({"a":1})",
       R"({"a":2})",
       R"(bad json)",
       R"({"a":4})",
       R"({"a":"not_int"})",
       R"({"a":6,"extra":"ignored"})"});
  auto expected = makeRowVector(
      {"a", "b"},
      {makeNullableFlatVector<int32_t>(
           {1, 2, std::nullopt, 4, std::nullopt, 6}),
       makeNullableFlatVector<StringView>(
           {std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt})});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, corruptRecordWithAllNestedSiblings) {
  auto input = makeFlatVector<std::string>(
      {R"({"a":1,"b":[1,2],"c":{"k":1},"d":{"x":1,"y":2}})",
       R"(bad json)",
       R"({"a":2,"b":[3,4,5],"c":{"k":2,"k2":3},"d":{"x":3,"y":4}})",
       R"(also bad)",
       R"({"a":3,"b":[],"c":{},"d":{"x":5,"y":6}})"});
  auto outputType =
      ROW({"a", "b", "c", "d", "_corrupt_record"},
          {INTEGER(),
           ARRAY(INTEGER()),
           MAP(VARCHAR(), INTEGER()),
           ROW({"x", "y"}, {INTEGER(), INTEGER()}),
           VARCHAR()});
  auto expr = createFromJsonWithConfig(
      outputType, {"columnNameOfCorruptRecord=_corrupt_record"});
  using MapEntry = std::pair<std::string, std::optional<int32_t>>;
  auto expected = makeRowVector(
      {"a", "b", "c", "d", "_corrupt_record"},
      {makeNullableFlatVector<int32_t>({1, std::nullopt, 2, std::nullopt, 3}),
       makeNullableArrayVector<int32_t>(
           {{{1, 2}}, std::nullopt, {{3, 4, 5}}, std::nullopt, {{}}}),
       makeNullableMapVector<std::string, int32_t>(
           std::vector<std::optional<std::vector<MapEntry>>>{
               std::vector<MapEntry>{{"k", 1}},
               std::nullopt,
               std::vector<MapEntry>{{"k", 2}, {"k2", 3}},
               std::nullopt,
               std::vector<MapEntry>{}}),
       makeRowVector(
           {"x", "y"},
           {makeFlatVector<int32_t>({1, 0, 3, 0, 5}),
            makeFlatVector<int32_t>({2, 0, 4, 0, 6})},
           [](auto row) { return row == 1 || row == 3; }),
       makeNullableFlatVector<StringView>(
           {std::nullopt,
            StringView("bad json"),
            std::nullopt,
            StringView("also bad"),
            std::nullopt})});
  testEncodings(expr, {input}, expected);

  auto result = evaluate(expr, makeRowVector({input}));
  auto* rowResult = result->as<RowVector>();
  auto* arrayChild = rowResult->childAt(1)->as<ArrayVector>();
  auto* mapChild = rowResult->childAt(2)->as<MapVector>();
  auto* nestedRow = rowResult->childAt(3)->as<RowVector>();
  ASSERT_TRUE(arrayChild != nullptr);
  ASSERT_TRUE(mapChild != nullptr);
  ASSERT_TRUE(nestedRow != nullptr);
  // Rows 1 and 3 are corrupt: every nested child must be null and have clean
  // offsets/sizes / grandchild nulls.
  for (vector_size_t row : {1, 3}) {
    EXPECT_TRUE(arrayChild->isNullAt(row)) << "row " << row;
    EXPECT_EQ(arrayChild->sizeAt(row), 0) << "row " << row;
    EXPECT_EQ(arrayChild->offsetAt(row), 0) << "row " << row;
    EXPECT_TRUE(mapChild->isNullAt(row)) << "row " << row;
    EXPECT_EQ(mapChild->sizeAt(row), 0) << "row " << row;
    EXPECT_EQ(mapChild->offsetAt(row), 0) << "row " << row;
    EXPECT_TRUE(nestedRow->isNullAt(row)) << "row " << row;
    EXPECT_TRUE(nestedRow->childAt(0)->isNullAt(row)) << "row " << row;
    EXPECT_TRUE(nestedRow->childAt(1)->isNullAt(row)) << "row " << row;
  }
}

TEST_F(FromJsonTest, corruptRecordPartialNestedValues) {
  {
    SCOPED_TRACE("ARRAY");
    auto input = makeFlatVector<std::string>(
        {R"({"a":1,"b":[10,20]})",
         R"({"a":2,"b":[5,"not_an_integer"]})",
         R"({"a":3,"b":[7,8,9]})"});
    auto outputType =
        ROW({"a", "b", "_corrupt_record"},
            {INTEGER(), ARRAY(INTEGER()), VARCHAR()});
    auto expr = createFromJsonWithConfig(
        outputType, {"columnNameOfCorruptRecord=_corrupt_record"});
    auto expected = makeRowVector(
        {"a", "b", "_corrupt_record"},
        {makeFlatVector<int32_t>({1, 2, 3}),
         makeNullableArrayVector<int32_t>(
             {{{10, 20}}, std::nullopt, {{7, 8, 9}}}),
         makeNullableFlatVector<StringView>(
             {std::nullopt, input->valueAt(1), std::nullopt})});
    testEncodings(expr, {input}, expected);
    auto result = evaluate(expr, makeRowVector({input}));
    auto* arrayChild = result->as<RowVector>()->childAt(1)->as<ArrayVector>();
    ASSERT_TRUE(arrayChild != nullptr);
    EXPECT_TRUE(arrayChild->isNullAt(1));
    EXPECT_EQ(arrayChild->sizeAt(1), 0);
    EXPECT_EQ(arrayChild->offsetAt(1), 0);
  }

  {
    SCOPED_TRACE("MAP");
    auto input = makeFlatVector<std::string>(
        {R"({"a":1,"c":{"k":1}})",
         R"({"a":2,"c":{"k":2,"bad":"not_an_integer"}})",
         R"({"a":3,"c":{"k":3}})"});
    auto outputType =
        ROW({"a", "c", "_corrupt_record"},
            {INTEGER(), MAP(VARCHAR(), INTEGER()), VARCHAR()});
    auto expr = createFromJsonWithConfig(
        outputType, {"columnNameOfCorruptRecord=_corrupt_record"});
    using MapEntry = std::pair<std::string, std::optional<int32_t>>;
    auto expected = makeRowVector(
        {"a", "c", "_corrupt_record"},
        {makeFlatVector<int32_t>({1, 2, 3}),
         makeNullableMapVector<std::string, int32_t>(
             std::vector<std::optional<std::vector<MapEntry>>>{
                 std::vector<MapEntry>{{"k", 1}},
                 std::nullopt,
                 std::vector<MapEntry>{{"k", 3}}}),
         makeNullableFlatVector<StringView>(
             {std::nullopt, input->valueAt(1), std::nullopt})});
    testEncodings(expr, {input}, expected);
    auto result = evaluate(expr, makeRowVector({input}));
    auto* mapChild = result->as<RowVector>()->childAt(1)->as<MapVector>();
    ASSERT_TRUE(mapChild != nullptr);
    EXPECT_TRUE(mapChild->isNullAt(1));
    EXPECT_EQ(mapChild->sizeAt(1), 0);
    EXPECT_EQ(mapChild->offsetAt(1), 0);
  }

  {
    SCOPED_TRACE("ROW");
    auto input = makeFlatVector<std::string>(
        {R"({"a":1,"d":{"x":1,"y":2}})",
         R"({"a":2,"d":{"x":4,"y":"not_an_integer"}})",
         R"({"a":3,"d":{"x":5,"y":6}})"});
    auto outputType =
        ROW({"a", "d", "_corrupt_record"},
            {INTEGER(), ROW({"x", "y"}, {INTEGER(), INTEGER()}), VARCHAR()});
    auto expected = makeRowVector(
        {"a", "d", "_corrupt_record"},
        {makeFlatVector<int32_t>({1, 2, 3}),
         makeRowVector(
             {"x", "y"},
             {makeFlatVector<int32_t>({1, 4, 5}),
              makeNullableFlatVector<int32_t>({2, std::nullopt, 6})}),
         makeNullableFlatVector<StringView>(
             {std::nullopt, input->valueAt(1), std::nullopt})});
    testEncodings(
        createFromJsonWithConfig(
            outputType, {"columnNameOfCorruptRecord=_corrupt_record"}),
        {input},
        expected);
  }
}

TEST_F(FromJsonTest, timestampDefaultFormat) {
  auto input = makeFlatVector<std::string>(
      {R"({"ts":"2024-01-15T10:30:00"})",
       R"({"ts":"2024-01-15T10:30:00.123"})",
       R"({"ts":"2024-01-15T10:30:00.123456"})",
       R"({"ts":"2024-01-15 10:30:00.1"})"});
  auto expected = makeRowVector(
      {"ts"},
      {makeFlatVector<Timestamp>(
          {Timestamp(1'705'314'600, 0),
           Timestamp(1'705'314'600, 123'000'000),
           Timestamp(1'705'314'600, 123'456'000),
           Timestamp(1'705'314'600, 100'000'000)})});
  testEncodings(
      createFromJsonWithConfig(expected->type(), {"sessionTimezone=UTC"}),
      {input},
      expected);
}

// Spark converts epoch seconds with `getLongValue * 1000000L`, so microsecond
// overflow wraps. Integers outside the signed 64-bit range fail the field.
TEST_F(FromJsonTest, timestampFromEpochSecondsWrapsLikeSpark) {
  auto input = makeFlatVector<std::string>(
      {R"({"ts":9223372036854})",
       R"({"ts":9223372036855})",
       R"({"ts":-9223372036855})",
       R"({"ts":9223372036854775807})",
       R"({"ts":-9223372036854775808})",
       R"({"ts":9223372036854775808})",
       R"({"ts":1})",
       R"({"ts":-1})"});
  auto expected = makeRowVector(
      {"ts"},
      {makeNullableFlatVector<Timestamp>(
          {Timestamp(9'223'372'036'854, 0),
           Timestamp(-9'223'372'036'855, 448'384'000),
           Timestamp(9'223'372'036'854, 551'616'000),
           Timestamp(-1, 0),
           Timestamp(0, 0),
           std::nullopt,
           Timestamp(1, 0),
           Timestamp(-1, 0)})});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, customTimestampPreservesMicroseconds) {
  auto input = makeFlatVector<std::string>(
      {R"({"ts":"2024-01-15 10:30:00.123456789"})",
       R"({"ts":"1969-12-31 23:59:59.000001999"})",
       R"({"ts":"1970-01-01 00:00:00.1"})"});
  auto expected = makeRowVector(
      {"ts"},
      {makeFlatVector<Timestamp>(
          {Timestamp(1'705'314'600, 123'456'000),
           Timestamp(-1, 1'000),
           Timestamp(0, 100'000'000)})});
  auto fromJson = createFromJsonWithConfig(
      expected->type(),
      {"timestampFormat=yyyy-MM-dd HH:mm:ss.SSSSSSSSS", "timeZone=UTC"});
  testEncodings(fromJson, {input}, expected);

  auto legacyInput = makeFlatVector<std::string>(
      {R"({"ts":"1970-01-01 00:00:00.1"})",
       R"({"ts":"1970-01-01 00:00:00.12"})",
       R"({"ts":"1970-01-01 00:00:00.123"})"});
  enableLegacyDateFormatter();
  testFromJsonWithConfig(
      legacyInput,
      makeRowVector(
          {"ts"},
          {makeFlatVector<Timestamp>(
              {Timestamp(0, 1'000'000),
               Timestamp(0, 12'000'000),
               Timestamp(0, 123'000'000)})}),
      {"timestampFormat=yyyy-MM-dd HH:mm:ss.SSS", "timeZone=UTC"});
}

TEST_F(FromJsonTest, customTimestampNumericFieldsDoNotOverflow) {
  auto input = makeFlatVector<std::string>(
      {R"({"ts":"18446744073709553586-01-01"})",
       R"({"ts":"-18446744073709553586-01-01"})",
       R"({"ts":"00000000000000001970-01-01"})"});
  auto expected = makeRowVector(
      {"ts"},
      {makeNullableFlatVector<Timestamp>(
          {std::nullopt, std::nullopt, Timestamp(0, 0)})});
  const auto format = "timestampFormat=" + std::string(20, 'y') + "-MM-dd";
  testEncodings(
      createFromJsonWithConfig(expected->type(), {format, "timeZone=UTC"}),
      {input},
      expected);

  enableLegacyDateFormatter();
  auto legacyInput = makeFlatVector<std::string>(
      {R"({"ts":"1970-01-01 00:00:00.18446744073709551616"})",
       R"({"ts":"1970-01-01 00:00:00.2147484"})",
       R"({"ts":"1970-01-01 00:00:00.1000"})",
       R"({"ts":"1970-01-01 00:00:00.123"})"});
  auto legacyExpected = makeRowVector(
      {"ts"},
      {makeNullableFlatVector<Timestamp>(
          {std::nullopt,
           std::nullopt,
           std::nullopt,
           Timestamp(0, 123'000'000)})});
  testEncodings(
      createFromJsonWithConfig(
          legacyExpected->type(),
          {"timestampFormat=yyyy-MM-dd HH:mm:ss." + std::string(20, 'S'),
           "timeZone=UTC"}),
      {legacyInput},
      legacyExpected);
}

// An explicit offset in the value takes precedence over the session timezone
// for both the default parser and a custom timestampFormat.
TEST_F(FromJsonTest, timestampWithTimezone) {
  // 10:30 at +05:30, -08:00, and +00:00.
  auto expected = makeRowVector(
      {"ts"},
      {makeFlatVector<Timestamp>(
          {Timestamp(1'705'294'800, 0),
           Timestamp(1'705'343'400, 0),
           Timestamp(1'705'314'600, 0)})});
  auto input = makeFlatVector<std::string>(
      {R"({"ts":"2024-01-15T10:30:00.000+05:30"})",
       R"({"ts":"2024-01-15T10:30:00-08:00"})",
       R"({"ts":"2024-01-15T10:30:00+00:00"})"});
  auto customInput = makeFlatVector<std::string>(
      {R"({"ts":"2024-01-15 10:30:00+0530"})",
       R"({"ts":"2024-01-15 10:30:00-0800"})",
       R"({"ts":"2024-01-15 10:30:00+0000"})"});
  for (const auto& timeZone : {"timeZone=UTC", "timeZone=America/New_York"}) {
    testEncodings(
        createFromJsonWithConfig(expected->type(), {timeZone}),
        {input},
        expected);
    testEncodings(
        createFromJsonWithConfig(
            expected->type(),
            {"timestampFormat=yyyy-MM-dd HH:mm:ssZ", timeZone}),
        {customInput},
        expected);
  }
}

// Spark accepts only strings and integral epoch seconds for TIMESTAMP. Invalid
// and empty strings, fractional or exponent numbers, booleans, and containers
// fail only the TIMESTAMP field in PERMISSIVE mode.
TEST_F(FromJsonTest, timestampInvalidString) {
  auto input = makeFlatVector<std::string>(
      {R"({"ts":"not-a-timestamp","a":1})",
       R"({"ts":"","a":2})",
       R"({"ts":1.5,"a":3})",
       R"({"ts":1e3,"a":4})",
       R"({"ts":true,"a":5})",
       R"({"ts":[1],"a":6})"});
  auto expected = makeRowVector(
      {"ts", "a"},
      {makeNullableFlatVector<Timestamp>(
           std::vector<std::optional<Timestamp>>(6, std::nullopt)),
       makeFlatVector<int32_t>({1, 2, 3, 4, 5, 6})});
  testEncodings(
      createFromJsonWithConfig(expected->type(), {"sessionTimezone=UTC"}),
      {input},
      expected);
}

TEST_F(FromJsonTest, timestampResolvesDaylightSavingTransitions) {
  {
    SCOPED_TRACE("default format");
    auto input = makeFlatVector<std::string>(
        {R"({"ts":"2024-03-10 02:30:00"})",
         R"({"ts":"2024-11-03 01:30:00"})",
         R"({"ts":"2024-03-10 02:30:00 America/New_York"})"});
    auto expected = makeRowVector(
        {"ts"},
        {makeFlatVector<Timestamp>(
            {Timestamp(1'710'055'800, 0),
             Timestamp(1'730'611'800, 0),
             Timestamp(1'710'055'800, 0)})});
    testFromJsonWithConfig(
        input, expected, {"sessionTimezone=America/New_York"});
  }

  {
    SCOPED_TRACE("custom format");
    auto input = makeFlatVector<std::string>(
        {R"({"ts":"2024-03-10 02:30"})",
         R"({"ts":"2024-11-03 01:30"})",
         R"({"ts":"2024-07-01 12:00"})"});
    auto expected = makeRowVector(
        {"ts"},
        {makeFlatVector<Timestamp>(
            {Timestamp(1'710'055'800, 0),
             Timestamp(1'730'611'800, 0),
             Timestamp(1'719'849'600, 0)})});
    testEncodings(
        createFromJsonWithConfig(
            expected->type(),
            {"timestampFormat=yyyy-MM-dd HH:mm",
             "sessionTimezone=America/New_York"}),
        {input},
        expected);
  }
}

TEST_F(FromJsonTest, timestampSupportsExtendedYears) {
  auto input = makeFlatVector<std::string>(
      {R"({"ts":"40000-01-01 00:00:00","a":1})",
       R"({"ts":"2024-01-15T10:30:00","a":2})"});
  auto expected = makeRowVector(
      {"ts", "a"},
      {makeFlatVector<Timestamp>(
           {Timestamp(1'200'110'860'800, 0), Timestamp(1'705'314'600, 0)}),
       makeFlatVector<int32_t>({1, 2})});
  testFromJsonWithConfig(
      input, expected, {"mode=FAILFAST", "sessionTimezone=UTC"});
}

TEST_F(FromJsonTest, timestampStringsRespectMicrosecondRange) {
  auto input = makeFlatVector<std::string>(
      {R"({"ts":"+294247-01-10 04:00:54.775807"})",
       R"({"ts":"+294247-01-10 04:00:54.775808"})",
       R"({"ts":"-290308-12-21 19:59:05.224192"})",
       R"({"ts":"-290308-12-21 19:59:05.224191"})",
       R"({"ts":"+300000-01-01 00:00:00.000000"})"});
  auto expected = makeRowVector(
      {"ts"},
      {makeNullableFlatVector<Timestamp>(
          {Timestamp(9'223'372'036'854, 775'807'000),
           std::nullopt,
           Timestamp(-9'223'372'036'855, 224'192'000),
           std::nullopt,
           std::nullopt})});
  testEncodings(
      createFromJsonWithConfig(expected->type(), {"timeZone=UTC"}),
      {input},
      expected);

  // Spark's custom formatter checks the whole-second multiplication before
  // adding the fraction, unlike the default parser's minimum-second handling.
  expected->childAt(0)->setNull(2, true);
  testEncodings(
      createFromJsonWithConfig(
          expected->type(),
          {"timestampFormat=yyyy-MM-dd HH:mm:ss.SSSSSS", "timeZone=UTC"}),
      {input},
      expected);
  testFromJsonThrows(
      input,
      expected->type(),
      {"mode=FAILFAST", "sessionTimezone=UTC"},
      failfastMessage(R"({"ts":"+294247-01-10 04:00:54.775808"})"));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
