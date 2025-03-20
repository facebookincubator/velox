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
#include <folly/base64.h>
#include "velox/functions/lib/TDigest.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/TDigestRegistration.h"
#include "velox/functions/prestosql/types/TDigestType.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions::test;

class TDigestFunctionsTest : public FunctionBaseTest {
 protected:
  void SetUp() override {
    FunctionBaseTest::SetUp();
    registerTDigestType();
  }

 protected:
  std::string decodeBase64(std::string_view input) {
    std::string decoded(folly::base64DecodedSize(input), '\0');
    folly::base64Decode(input, decoded.data());
    return decoded;
  }

  double getLowerBoundQuantile(double quantile, double error) {
    return std::max(0.0, quantile - error);
  }

  double getUpperBoundQuantile(double quantile, double error) {
    return std::min(1.0, quantile + error);
  }
  double getUpperBoundValue(
      double quantile,
      double error,
      const std::vector<double>& values) {
    int index = static_cast<int>(std::min(
        NUMBER_OF_ENTRIES * (quantile + error),
        static_cast<double>(values.size() - 1)));
    return values[index];
  }
  int NUMBER_OF_ENTRIES = 1000000;
  double error = 0.01;
  double quantiles[19] = {
      0.0001,
      0.0200,
      0.0300,
      0.04000,
      0.0500,
      0.1000,
      0.2000,
      0.3000,
      0.4000,
      0.5000,
      0.6000,
      0.7000,
      0.8000,
      0.9000,
      0.9500,
      0.9600,
      0.9700,
      0.9800,
      0.9999};
};

TEST_F(TDigestFunctionsTest, valueAtQuantile) {
  const TypePtr type = TDIGEST(DOUBLE());
  const auto valueAtQuantile = [&](const std::optional<std::string>& input,
                                   const std::optional<double>& quantile) {
    return evaluateOnce<double>(
        "value_at_quantile(c0, c1)", type, input, quantile);
  };
  const std::string input = decodeBase64(
      "AQAAAAAAAADwPwAAAAAAABRAAAAAAAAALkAAAAAAAABZQAAAAAAAABRABQAAAAAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAABAAAAAAAAACEAAAAAAAAAQQAAAAAAAABRA");
  ASSERT_EQ(1.0, valueAtQuantile(input, 0.1));
  ASSERT_EQ(3.0, valueAtQuantile(input, 0.5));
  ASSERT_EQ(5.0, valueAtQuantile(input, 0.9));
  ASSERT_EQ(5.0, valueAtQuantile(input, 0.99));
};

TEST_F(TDigestFunctionsTest, valuesAtQuantiles) {
  const TypePtr type = TDIGEST(DOUBLE());
  const std::string input = decodeBase64(
      "AQAAAAAAAADwPwAAAAAAABRAAAAAAAAALkAAAAAAAABZQAAAAAAAABRABQAAAAAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAABAAAAAAAAACEAAAAAAAAAQQAAAAAAAABRA");
  VectorPtr arg0 = makeFlatVector<std::string>({input}, type);
  VectorPtr arg1 = makeNullableArrayVector<double>({{0.1, 0.5, 0.9, 0.99}});
  auto expected = makeNullableArrayVector<double>({{1.0, 3.0, 5.0, 5.0}});
  auto result =
      evaluate("values_at_quantiles(c0, c1)", makeRowVector({arg0, arg1}));
  test::assertEqualVectors(expected, result);
}

TEST_F(TDigestFunctionsTest, nullTDigestGetQuantileAtValue) {
  const TypePtr type = TDIGEST(DOUBLE());
  const auto quantileAtValue = [&](const std::optional<std::string>& input,
                                   const std::optional<double>& value) {
    return evaluateOnce<double>(
        "quantile_at_value(c0, c1)", type, input, value);
  };
  ASSERT_EQ(std::nullopt, quantileAtValue(std::nullopt, 0.3));
}

TEST_F(TDigestFunctionsTest, quantileAtValueWithinBound) {
  const TypePtr type = TDIGEST(DOUBLE());
  const auto quantileAtValue = [&](const std::optional<std::string>& input,
                                   const std::optional<double>& value) {
    return evaluateOnce<double>(
        "quantile_at_value(c0, c1)", type, input, value);
  };
  facebook::velox::functions::TDigest<> tDigest;
  std::vector<int16_t> positions;
  std::vector<double> values;
  for (int i = 0; i < NUMBER_OF_ENTRIES; ++i) {
    double value = static_cast<double>(rand()) / RAND_MAX * NUMBER_OF_ENTRIES;
    tDigest.add(positions, value);
    values.push_back(value);
  }
  tDigest.compress(positions);
  std::sort(values.begin(), values.end());
  int serializedSize = tDigest.serializedByteSize();
  std::vector<char> buffer(serializedSize);
  tDigest.serialize(buffer.data());
  std::string serializedDigest(buffer.begin(), buffer.end());
  for (auto quantile : quantiles) {
    int index = static_cast<int>(NUMBER_OF_ENTRIES * quantile);
    auto quantileValueOpt = quantileAtValue(serializedDigest, values[index]);
    ASSERT_TRUE(quantileValueOpt.has_value());
    double quantileValue = quantileValueOpt.value();
    double lowerBoundQuantile = getLowerBoundQuantile(quantile, error);
    double upperBoundQuantile = getUpperBoundQuantile(quantile, error);
    ASSERT_LE(quantileValue, upperBoundQuantile);
    ASSERT_GE(quantileValue, lowerBoundQuantile);
  }
}

TEST_F(TDigestFunctionsTest, quantileAtValueOutsideRange) {
  const TypePtr type = TDIGEST(DOUBLE());
  const auto quantileAtValue = [&](const std::optional<std::string>& input,
                                   const std::optional<double>& value) {
    return evaluateOnce<double>(
        "quantile_at_value(c0, c1)", type, input, value);
  };
  facebook::velox::functions::TDigest<> tDigest;
  std::vector<int16_t> positions;
  for (int i = 0; i < NUMBER_OF_ENTRIES; ++i) {
    double value = static_cast<double>(rand()) / RAND_MAX * NUMBER_OF_ENTRIES;
    tDigest.add(positions, value);
  }
  tDigest.compress(positions);
  int serializedSize = tDigest.serializedByteSize();
  std::vector<char> buffer(serializedSize);
  tDigest.serialize(buffer.data());
  std::string serializedDigest(buffer.begin(), buffer.end());
  ASSERT_EQ(1.0, quantileAtValue(serializedDigest, 1000000000.0));
  ASSERT_EQ(0.0, quantileAtValue(serializedDigest, -500.0));
}
