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

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox::functions {
namespace {

class GooglePolylineFunctionsTest : public test::FunctionBaseTest {
 protected:
  std::optional<std::string> evaluateEncode(
      const std::optional<std::vector<std::optional<std::string>>>& points,
      std::optional<int32_t> precision = std::nullopt) {
    auto arrayVec = makeNullableArrayVector<std::string>({{points}});

    if (precision.has_value()) {
      auto precisionVec = makeFlatVector<int32_t>({precision.value()});
      auto input = makeRowVector({arrayVec, precisionVec});
      return evaluateOnce<std::string>(
          "google_polyline_encode(transform(c0, x -> ST_GeometryFromText(x)), c1)",
          input);
    } else {
      auto input = makeRowVector({arrayVec});
      return evaluateOnce<std::string>(
          "google_polyline_encode(transform(c0, x -> ST_GeometryFromText(x)))",
          input);
    }
  }

  facebook::velox::VectorPtr evaluateDecode(
      const std::string& encoded,
      std::optional<int32_t> precision = std::nullopt) {
    auto encodeVec = makeFlatVector<std::string>({encoded});

    if (precision.has_value()) {
      auto precisionVec = makeFlatVector<int32_t>({precision.value()});
      auto input = makeRowVector({encodeVec, precisionVec});
      return evaluate(
          "transform(google_polyline_decode(c0, c1), x -> ST_AsText(x))",
          input);
    } else {
      auto input = makeRowVector({encodeVec});
      return evaluate(
          "transform(google_polyline_decode(c0), x -> ST_AsText(x))", input);
    }
  }

  std::optional<std::string> evaluateRoundTrip(
      const std::optional<std::vector<std::optional<std::string>>>& points,
      std::optional<int32_t> precision = std::nullopt) {
    auto arrayVec = makeNullableArrayVector<std::string>({{points}});

    if (precision.has_value()) {
      auto precisionVec = makeFlatVector<int32_t>({precision.value()});
      auto input = makeRowVector({arrayVec, precisionVec});

      return evaluateOnce<std::string>(
          "google_polyline_encode(google_polyline_decode("
          "google_polyline_encode(transform(c0, x -> ST_GeometryFromText(x)), c1), c1), c1)",
          input);
    } else {
      auto input = makeRowVector({arrayVec});

      return evaluateOnce<std::string>(
          "google_polyline_encode(google_polyline_decode("
          "google_polyline_encode(transform(c0, x -> ST_GeometryFromText(x)))))",
          input);
    }
  }

  void validateEncodedResult(
      const std::optional<std::vector<std::optional<std::string>>>& points,
      const std::string& expected,
      std::optional<int32_t> precision = std::nullopt) {
    auto result = evaluateEncode(points, precision);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(expected, result.value());
  }

  void validateDecodedResult(
      const std::string& encoded,
      const std::vector<std::optional<std::string>>& expectedPoints,
      std::optional<int32_t> precision = std::nullopt) {
    auto output = evaluateDecode(encoded, precision);
    auto expected = makeNullableArrayVector<std::string>(
        std::vector<std::vector<std::optional<std::string>>>{expectedPoints});
    facebook::velox::test::assertEqualVectors(expected, output);
  }
};

TEST_F(GooglePolylineFunctionsTest, encodeWithDefaultPrecision) {
  validateEncodedResult({{"POINT (38.5 -120.2)"}}, "_p~iF~ps|U");

  validateEncodedResult(
      {{"POINT (38.5 -120.2)", "POINT (40.7 -120.95)"}}, "_p~iF~ps|U_ulLnnqC");

  validateEncodedResult(
      {{"POINT (38.5 -120.2)",
        "POINT (40.7 -120.95)",
        "POINT (43.252 -126.453)"}},
      "_p~iF~ps|U_ulLnnqC_mqNvxq`@");

  validateEncodedResult(
      {{"POINT (37.78327 -122.43877)",
        "POINT (37.75885 -122.43533)",
        "POINT (37.76373 -122.41027)",
        "POINT (37.76781 -122.42538)",
        "POINT (37.76781 -122.42538)",
        "POINT (37.76835 -122.45422)",
        "POINT (37.78327 -122.43877)"}},
      "mpreFhyhjVrwCoTo]s{CoXl}A??kBfsDg|Aq_B");

  validateEncodedResult(std::vector<std::optional<std::string>>{}, "");
}

TEST_F(GooglePolylineFunctionsTest, encodeCustomPrecision) {
  validateEncodedResult(
      {{"POINT (38.5 -120.2)",
        "POINT (40.7 -120.95)",
        "POINT (43.252 -126.453)"}},
      "_izlhA~rlgdF_{geC~ywl@_kwzCn`{nI",
      6);

  validateEncodedResult({{"POINT (38.5 -120.2)"}}, "aWbjA", 1);

  validateEncodedResult(
      {{"POINT (38.5 -120.2)",
        "POINT (40.7 -120.95)",
        "POINT (43.252 -126.453)"}},
      "_p~iF~ps|U_ulLnnqC_mqNvxq`@",
      5);

  validateEncodedResult(
      {{"POINT (38.5 -120.2)"}}, "___yomm|u{jT~~~r_klgxdvaA", 16);
}

TEST_F(GooglePolylineFunctionsTest, encodeErrorCases) {
  VELOX_ASSERT_THROW(
      evaluateEncode({{"POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"}}),
      "Non-point geometry in google_polyline_encode input at index 0.");

  VELOX_ASSERT_THROW(
      evaluateEncode(
          {{"POINT (38.5 -120.2)", "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"}}),
      "Non-point geometry in google_polyline_encode input at index 1.");

  VELOX_ASSERT_THROW(
      evaluateEncode({{"POINT (37.78327 -122.43877)"}}, 0),
      "Polyline precision must be greater or equal to 1");

  VELOX_ASSERT_THROW(
      evaluateEncode({{"POINT (37.78327 -122.43877)"}}, -5),
      "Polyline precision must be greater or equal to 1");

  VELOX_ASSERT_THROW(
      evaluateEncode({{"POINT (37.78327 -122.43877)"}}, 17),
      "Polyline precision exponent must not exceed 16");

  VELOX_ASSERT_THROW(
      evaluateEncode({{"POINT (37.78327 -122.43877)"}}, 100),
      "Polyline precision exponent must not exceed 16");
}

TEST_F(GooglePolylineFunctionsTest, decodeWithDefaultPrecision) {
  validateDecodedResult("_p~iF~ps|U", {{"POINT (38.5 -120.2)"}});

  validateDecodedResult(
      "_p~iF~ps|U_ulLnnqC_mqNvxq`@",
      {{"POINT (38.5 -120.2)",
        "POINT (40.7 -120.95)",
        "POINT (43.252 -126.453)"}});

  validateDecodedResult(
      "_p~iF~ps|U_ulLnnqC", {{"POINT (38.5 -120.2)", "POINT (40.7 -120.95)"}});

  validateDecodedResult("", std::vector<std::optional<std::string>>{});

  validateDecodedResult(
      "mpreFhyhjVrwCoTo]s{CoXl}A??kBfsDg|Aq_B",
      {{"POINT (37.78327 -122.43877)",
        "POINT (37.75885 -122.43533)",
        "POINT (37.76373 -122.41027)",
        "POINT (37.76781 -122.42538)",
        "POINT (37.76781 -122.42538)",
        "POINT (37.76835 -122.45422)",
        "POINT (37.78327 -122.43877)"}});
}

TEST_F(GooglePolylineFunctionsTest, decodeCustomPrecision) {
  validateDecodedResult(
      "_izlhA~rlgdF_{geC~ywl@_kwzCn`{nI",
      {{"POINT (38.5 -120.2)",
        "POINT (40.7 -120.95)",
        "POINT (43.252 -126.453)"}},
      6);

  validateDecodedResult("aWbjA", {{"POINT (38.5 -120.2)"}}, 1);

  validateDecodedResult(
      "_p~iF~ps|U_ulLnnqC_mqNvxq`@",
      {{"POINT (38.5 -120.2)",
        "POINT (40.7 -120.95)",
        "POINT (43.252 -126.453)"}},
      5);

  validateDecodedResult(
      "___yomm|u{jT~~~r_klgxdvaA", {{"POINT (38.5 -120.2)"}}, 16);
}

TEST_F(GooglePolylineFunctionsTest, decodeErrorCases) {
  VELOX_ASSERT_THROW(
      evaluateDecode("A"),
      "Invalid polyline encoding: unexpected end of input");

  VELOX_ASSERT_THROW(
      evaluateDecode("_p~iF~ps|U", 0),
      "Polyline precision must be greater or equal to 1");

  VELOX_ASSERT_THROW(
      evaluateDecode("_p~iF~ps|U", -5),
      "Polyline precision must be greater or equal to 1");

  VELOX_ASSERT_THROW(
      evaluateDecode("_p~iF~ps|U", 17),
      "Polyline precision exponent must not exceed 16");

  VELOX_ASSERT_THROW(
      evaluateDecode("_p~iF~ps|U", 100),
      "Polyline precision exponent must not exceed 16");
}

TEST_F(GooglePolylineFunctionsTest, roundTripTests) {
  auto points1 = std::optional<std::vector<std::optional<std::string>>>{
      {"POINT (38.5 -120.2)", "POINT (40.7 -120.95)"}};
  auto encoded1 = evaluateEncode(points1);
  ASSERT_EQ(encoded1, "_p~iF~ps|U_ulLnnqC");
  ASSERT_TRUE(encoded1.has_value());
  ASSERT_TRUE(points1.has_value());

  validateDecodedResult(encoded1.value(), points1.value());
  auto roundTrip1 = evaluateRoundTrip(points1);
  ASSERT_EQ(roundTrip1, encoded1);

  auto points2 = std::optional<std::vector<std::optional<std::string>>>{
      {"POINT (38.5 -120.2)"}};
  auto encoded2 = evaluateEncode(points2, 1);
  ASSERT_EQ(encoded2, "aWbjA");
  ASSERT_TRUE(encoded2.has_value());
  ASSERT_TRUE(points2.has_value());

  validateDecodedResult(encoded2.value(), points2.value(), 1);
  auto roundTrip2 = evaluateRoundTrip(points2, 1);
  ASSERT_EQ(roundTrip2, encoded2);

  auto points3 = std::optional<std::vector<std::optional<std::string>>>{
      {"POINT (38.5 -120.2)",
       "POINT (40.7 -120.95)",
       "POINT (43.252 -126.453)"}};
  auto encoded3 = evaluateEncode(points3, 6);
  ASSERT_EQ(encoded3, "_izlhA~rlgdF_{geC~ywl@_kwzCn`{nI");
  ASSERT_TRUE(encoded3.has_value());
  ASSERT_TRUE(points3.has_value());

  validateDecodedResult(encoded3.value(), points3.value(), 6);
  auto roundTrip3 = evaluateRoundTrip(points3, 6);
  ASSERT_EQ(roundTrip3, encoded3);

  auto points4 = std::optional<std::vector<std::optional<std::string>>>{
      {"POINT (38.5 -120.2)"}};
  auto encoded4 = evaluateEncode(points4, 16);
  ASSERT_EQ(encoded4, "___yomm|u{jT~~~r_klgxdvaA");
  ASSERT_TRUE(encoded4.has_value());
  ASSERT_TRUE(points4.has_value());

  validateDecodedResult(encoded4.value(), points4.value(), 16);
  auto roundTrip4 = evaluateRoundTrip(points4, 16);
  ASSERT_EQ(roundTrip4, encoded4);
}

} // namespace
} // namespace facebook::velox::functions
