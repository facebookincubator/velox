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

#include <gmock/gmock.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double kDoubleMax = std::numeric_limits<double>::max();
// TMP: For floating types, min returns the smallest positive
// value the type can encode, not the lowest (largest neg value).
// See numeric_limits<double>::lowest();
constexpr double kDoubleMin = std::numeric_limits<double>::min();
constexpr int64_t kBigIntMax = std::numeric_limits<int64_t>::max();
constexpr int64_t kBigIntMin = std::numeric_limits<int64_t>::min();

MATCHER(IsNan, "is NaN") {
  return arg && std::isnan(*arg);
}

MATCHER(IsInf, "is Infinity") {
  return arg && std::isinf(*arg);
}

class ProbabilityTest : public functions::test::FunctionBaseTest {};

TEST_F(ProbabilityTest, betaCDF) {
  const auto betaCDF = [&](std::optional<double> a,
                           std::optional<double> b,
                           std::optional<double> value) {
    return evaluateOnce<double>("beta_cdf(c0, c1, c2)", a, b, value);
  };

  EXPECT_EQ(0.09888000000000001, betaCDF(3, 4, 0.2));
  EXPECT_EQ(0.0, betaCDF(3, 3.6, 0.0));
  EXPECT_EQ(1.0, betaCDF(3, 3.6, 1.0));
  EXPECT_EQ(0.21764809997679951, betaCDF(3, 3.6, 0.3));
  EXPECT_EQ(0.9972502881611551, betaCDF(3, 3.6, 0.9));
  EXPECT_EQ(0.0, betaCDF(kInf, 3, 0.2));
  EXPECT_EQ(0.0, betaCDF(3, kInf, 0.2));
  EXPECT_EQ(0.0, betaCDF(kInf, kInf, 0.2));
  EXPECT_EQ(0.0, betaCDF(kDoubleMax, kDoubleMax, 0.3));
  EXPECT_EQ(0.5, betaCDF(kDoubleMin, kDoubleMin, 0.3));

  VELOX_ASSERT_THROW(
      betaCDF(3, 3, kInf), "value must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(betaCDF(0, 3, 0.5), "a must be > 0");
  VELOX_ASSERT_THROW(betaCDF(3, 0, 0.5), "b must be > 0");
  VELOX_ASSERT_THROW(
      betaCDF(3, 5, -0.1), "value must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      betaCDF(3, 5, 1.1), "value must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(betaCDF(kNan, 3, 0.5), "a must be > 0");
  VELOX_ASSERT_THROW(betaCDF(3, kNan, 0.5), "b must be > 0");
  VELOX_ASSERT_THROW(
      betaCDF(3, 3, kNan), "value must be in the interval [0, 1]");
}

TEST_F(ProbabilityTest, normalCDF) {
  const auto normal_cdf = [&](std::optional<double> mean,
                              std::optional<double> sd,
                              std::optional<double> value) {
    return evaluateOnce<double>("normal_cdf(c0, c1, c2)", mean, sd, value);
  };

  EXPECT_EQ(0.97500210485177963, normal_cdf(0, 1, 1.96));
  EXPECT_EQ(0.5, normal_cdf(10, 9, 10));
  EXPECT_EQ(0.0013498980316301035, normal_cdf(-1.5, 2.1, -7.8));
  EXPECT_EQ(1.0, normal_cdf(0, 1, kInf));
  EXPECT_EQ(0.0, normal_cdf(0, 1, -kInf));
  EXPECT_EQ(0.0, normal_cdf(kInf, 1, 0));
  EXPECT_EQ(1.0, normal_cdf(-kInf, 1, 0));
  EXPECT_EQ(0.5, normal_cdf(0, kInf, 0));
  EXPECT_THAT(normal_cdf(kNan, 1, 0), IsNan());
  EXPECT_THAT(normal_cdf(0, 1, kNan), IsNan());
  EXPECT_EQ(normal_cdf(0, 1, kDoubleMax), 1);
  EXPECT_EQ(normal_cdf(0, kDoubleMax, 0), 0.5);
  EXPECT_EQ(0.0, normal_cdf(kDoubleMax, 1, 0));
  EXPECT_EQ(0.5, normal_cdf(0, 1, kDoubleMin));
  EXPECT_EQ(0.5, normal_cdf(kDoubleMin, 1, 0));
  EXPECT_EQ(0, normal_cdf(1, kDoubleMin, 0));
  EXPECT_THAT(normal_cdf(kDoubleMax, kDoubleMax, kInf), IsNan());
  EXPECT_EQ(0.5, normal_cdf(kDoubleMax, kDoubleMax, kDoubleMax));
  EXPECT_EQ(0.5, normal_cdf(kDoubleMin, kDoubleMin, kDoubleMin));
  EXPECT_EQ(0.5, normal_cdf(kDoubleMax, 1, kDoubleMax));
  EXPECT_EQ(0.5, normal_cdf(10, kDoubleMax, kDoubleMax));
  EXPECT_EQ(0.5, normal_cdf(kDoubleMax, kDoubleMax, 1.96));
  EXPECT_EQ(std::nullopt, normal_cdf(std::nullopt, 1, 1.96));
  EXPECT_EQ(std::nullopt, normal_cdf(1, 1, std::nullopt));
  EXPECT_EQ(std::nullopt, normal_cdf(std::nullopt, 1, std::nullopt));
  EXPECT_EQ(std::nullopt, normal_cdf(std::nullopt, std::nullopt, std::nullopt));

  VELOX_ASSERT_THROW(normal_cdf(0, 0, 0.1985), "standardDeviation must be > 0");
  VELOX_ASSERT_THROW(
      normal_cdf(0, kNan, 0.1985), "standardDeviation must be > 0");
}

TEST_F(ProbabilityTest, binomialCDF) {
  const auto binomialCDF = [&](std::optional<int64_t> numberOfTrials,
                               std::optional<double> successProbability,
                               std::optional<int64_t> value) {
    return evaluateOnce<double>(
        "binomial_cdf(c0, c1, c2)", numberOfTrials, successProbability, value);
  };

  EXPECT_EQ(binomialCDF(5, 0.5, 5), 1.0);
  EXPECT_EQ(binomialCDF(5, 0.5, 0), 0.03125);
  EXPECT_EQ(binomialCDF(3, 0.5, 1), 0.5);
  EXPECT_EQ(binomialCDF(20, 1.0, 0), 0.0);
  EXPECT_EQ(binomialCDF(20, 0.3, 6), 0.60800981220092398);
  EXPECT_EQ(binomialCDF(200, 0.3, 60), 0.5348091761606989);
  EXPECT_EQ(binomialCDF(kBigIntMax, 0.5, 2), 0.0);
  EXPECT_EQ(binomialCDF(kBigIntMax, 0.5, kBigIntMax), 0.0);
  EXPECT_EQ(binomialCDF(10, 0.5, kBigIntMax), 1.0);
  EXPECT_EQ(binomialCDF(10, 0.1, kBigIntMin), 0.0);
  EXPECT_EQ(binomialCDF(10, 0.1, -2), 0.0);
  EXPECT_EQ(binomialCDF(25, 0.5, -100), 0.0);

  // Invalid inputs
  VELOX_ASSERT_THROW(
      binomialCDF(5, -0.5, 3),
      "successProbability must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      binomialCDF(5, 2, 3),
      "successProbability must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      binomialCDF(5, kBigIntMax, 3),
      "successProbability must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      binomialCDF(5, kNan, 3),
      "successProbability must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      binomialCDF(-1, 0.5, 2), "numberOfTrials must be greater than 0");
  VELOX_ASSERT_THROW(
      binomialCDF(kBigIntMin, 0.5, 1), "numberOfTrials must be greater than 0");
  VELOX_ASSERT_THROW(
      binomialCDF(-2, 2, -1),
      "successProbability must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      binomialCDF(-2, 0.5, -1), "numberOfTrials must be greater than 0");
}

TEST_F(ProbabilityTest, inverseChiSquaredCDF) {
  const auto invChiSquaredCDF = [&](std::optional<double> df,
                                    std::optional<double> p) {
    return evaluateOnce<double>("inverse_chi_squared_cdf(c0, c1)", df, p);
  };

  EXPECT_EQ(0.0, invChiSquaredCDF(3, 0.0));
  EXPECT_EQ(1.4236522430352796, invChiSquaredCDF(3, 0.3));
  EXPECT_EQ(11.344866730144370, invChiSquaredCDF(3, 0.99));
  EXPECT_EQ(12.287969413016436, invChiSquaredCDF(7.2, 0.90));
  EXPECT_EQ(37.714228443623732, invChiSquaredCDF(25.05, 0.95));
  EXPECT_EQ(22.178420945304087, invChiSquaredCDF(15.057, 0.895));

  EXPECT_EQ(2.8093008564672404e-123, invChiSquaredCDF(5, kDoubleMin));
  EXPECT_EQ(1.7976931348623157e+308, invChiSquaredCDF(kDoubleMax, 0.95));
  EXPECT_EQ(4.4501477170144018e-308, invChiSquaredCDF(kDoubleMin, 0.752));

  EXPECT_EQ(std::nullopt, invChiSquaredCDF(std::nullopt, 0.94));
  EXPECT_EQ(std::nullopt, invChiSquaredCDF(142.345, std::nullopt));
  EXPECT_EQ(std::nullopt, invChiSquaredCDF(std::nullopt, std::nullopt));

  // Test invalid inputs for df.
  VELOX_ASSERT_THROW(invChiSquaredCDF(-3, 0.3), "df must be greater than 0");
  VELOX_ASSERT_THROW(invChiSquaredCDF(kNan, 0.99), "df must be greater than 0");
  VELOX_ASSERT_THROW(
      invChiSquaredCDF(kInf, 0.99),
      "Error in function boost::math::chi_squared_distribution<double>::chi_squared_distribution: Degrees of freedom argument is inf, but must be > 0 !");
  VELOX_ASSERT_THROW(
      invChiSquaredCDF(kBigIntMin, 0.15), "df must be greater than 0");

  // Test invalid inputs for p.
  VELOX_ASSERT_THROW(
      invChiSquaredCDF(3, 1.001), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invChiSquaredCDF(40, kNan), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invChiSquaredCDF(40, kInf), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invChiSquaredCDF(11, kDoubleMax), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invChiSquaredCDF(500, kBigIntMin), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invChiSquaredCDF(234, kBigIntMax), "p must be in the interval [0, 1]");

  // Test invalid inputs for both params.
  VELOX_ASSERT_THROW(
      invChiSquaredCDF(-3, -0.001), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invChiSquaredCDF(kNan, kNan), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invChiSquaredCDF(kInf, kInf), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invChiSquaredCDF(kBigIntMin, kBigIntMax),
      "p must be in the interval [0, 1]");
}

} // namespace
} // namespace facebook::velox
