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
#include <random>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::window::test {

class ApproxWinsorizeTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    window::prestosql::registerAllWindowFunctions();
  }

  RowVectorPtr runWindow(
      const RowVectorPtr& input,
      const std::string& functionCall,
      const std::string& overClause) {
    return AssertQueryBuilder(
               PlanBuilder()
                   .values({input})
                   .window({functionCall + " " + overClause + " as w"})
                   .planNode())
        .copyResults(pool());
  }
};

TEST_F(ApproxWinsorizeTest, fullRange) {
  // With bounds (0.0, 1.0), no clamping — values pass through unchanged.
  auto input = makeRowVector({
      makeFlatVector<double>({1.0, 2.0, 3.0, 4.0, 5.0}),
  });
  auto result = runWindow(input, "approx_winsorize(c0, 0.0, 1.0)", "OVER ()");
  auto originalCol = result->childAt(0)->asFlatVector<double>();
  auto winsorized = result->childAt(1)->asFlatVector<double>();
  for (int i = 0; i < 5; ++i) {
    ASSERT_NEAR(winsorized->valueAt(i), originalCol->valueAt(i), 0.1);
  }
}

TEST_F(ApproxWinsorizeTest, upperCapping) {
  // Upper winsorization should cap outliers.
  constexpr int kNumValues = 100;
  std::vector<double> values(kNumValues);
  for (int i = 0; i < kNumValues - 1; ++i) {
    values[i] = static_cast<double>(i);
  }
  values[kNumValues - 1] = 1'000.0; // outlier
  auto input = makeRowVector({makeFlatVector<double>(values)});
  auto result = runWindow(input, "approx_winsorize(c0, 0.0, 0.95)", "OVER ()");
  auto originalCol = result->childAt(0)->asFlatVector<double>();
  auto winsorized = result->childAt(1)->asFlatVector<double>();
  for (int i = 0; i < kNumValues; ++i) {
    double originalValue{originalCol->valueAt(i)};
    double winsorizedValue{winsorized->valueAt(i)};
    if (originalValue == 1'000.0) {
      // The outlier should be capped.
      ASSERT_LT(winsorizedValue, 1'000.0);
    } else if (originalValue < 50) {
      // Low values are unchanged (lower bound is 0.0).
      ASSERT_NEAR(winsorizedValue, originalValue, 1.0);
    }
  }
}

TEST_F(ApproxWinsorizeTest, twoSidedCapping) {
  // Two-sided winsorization with enough data for TDigest resolution.
  // Verify per-row invariant: each winsorized value is clamped to
  // [lowerBound, upperBound] and middle values are unchanged.
  constexpr int kNumValues = 1'000;
  std::vector<double> values(kNumValues);
  for (int i = 0; i < kNumValues; ++i) {
    values[i] = static_cast<double>(i);
  }
  auto input = makeRowVector({makeFlatVector<double>(values)});
  auto result = runWindow(input, "approx_winsorize(c0, 0.05, 0.95)", "OVER ()");
  auto originalCol = result->childAt(0)->asFlatVector<double>();
  auto winsorized = result->childAt(1)->asFlatVector<double>();
  double minWinsorized{winsorized->valueAt(0)};
  double maxWinsorized{winsorized->valueAt(0)};
  for (int i = 0; i < kNumValues; ++i) {
    double winsorizedValue{winsorized->valueAt(i)};
    double originalValue{originalCol->valueAt(i)};
    minWinsorized = std::min(minWinsorized, winsorizedValue);
    maxWinsorized = std::max(maxWinsorized, winsorizedValue);
    // If original is in the middle, winsorized should equal original.
    if (originalValue > 100 && originalValue < 900) {
      ASSERT_NEAR(winsorizedValue, originalValue, 1.0);
    }
  }
  // The range of winsorized values should be narrower than original.
  ASSERT_GE(minWinsorized, 30.0);
  ASSERT_LE(maxWinsorized, 970.0);
}

TEST_F(ApproxWinsorizeTest, partitionBy) {
  // Test with PARTITION BY — each group gets its own TDigest.
  constexpr int perGroup = 100;
  std::vector<int32_t> keys(2 * perGroup);
  std::vector<double> values(2 * perGroup);
  for (int i = 0; i < perGroup; ++i) {
    keys[i] = 1;
    values[i] = static_cast<double>(i);
  }
  values[perGroup - 1] = 1'000.0; // outlier in group 1
  for (int i = 0; i < perGroup; ++i) {
    keys[perGroup + i] = 2;
    values[perGroup + i] = static_cast<double>(i + 10);
  }
  auto input = makeRowVector({
      makeFlatVector<int32_t>(keys),
      makeFlatVector<double>(values),
  });
  auto result = runWindow(
      input, "approx_winsorize(c1, 0.0, 0.95)", "OVER (PARTITION BY c0)");
  // Row order may change. Check invariants per row.
  auto originalCol = result->childAt(1)->asFlatVector<double>();
  auto winsorized = result->childAt(2)->asFlatVector<double>();
  bool foundCappedOutlier{false};
  for (int i = 0; i < 2 * perGroup; ++i) {
    double originalValue{originalCol->valueAt(i)};
    double winsorizedValue{winsorized->valueAt(i)};
    if (originalValue == 1'000.0) {
      ASSERT_LT(winsorizedValue, 1'000.0);
      foundCappedOutlier = true;
    }
    // Winsorized value should never exceed original for upper-only capping.
    ASSERT_LE(winsorizedValue, originalValue + 1.0);
  }
  ASSERT_TRUE(foundCappedOutlier);
}

TEST_F(ApproxWinsorizeTest, nullHandling) {
  auto input = makeRowVector({
      makeNullableFlatVector<double>(
          {1.0, std::nullopt, 3.0, std::nullopt, 5.0}),
  });
  auto result = runWindow(input, "approx_winsorize(c0, 0.0, 1.0)", "OVER ()");
  auto originalCol = result->childAt(0);
  auto winsorized = result->childAt(1);
  int nullCount{0};
  int nonNullCount{0};
  for (int i = 0; i < 5; ++i) {
    if (originalCol->isNullAt(i)) {
      // Null input should produce null output.
      ASSERT_TRUE(winsorized->isNullAt(i));
      ++nullCount;
    } else {
      ASSERT_FALSE(winsorized->isNullAt(i));
      ++nonNullCount;
    }
  }
  ASSERT_EQ(nullCount, 2);
  ASSERT_EQ(nonNullCount, 3);
}

TEST_F(ApproxWinsorizeTest, allNulls) {
  auto input = makeRowVector({
      makeNullableFlatVector<double>(
          {std::nullopt, std::nullopt, std::nullopt}),
  });
  auto result = runWindow(input, "approx_winsorize(c0, 0.0, 1.0)", "OVER ()");
  auto winsorized = result->childAt(1);
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(winsorized->isNullAt(i));
  }
}

TEST_F(ApproxWinsorizeTest, withCompression) {
  constexpr int kNumValues = 1'000;
  std::vector<double> values(kNumValues);
  for (int i = 0; i < kNumValues; ++i) {
    values[i] = static_cast<double>(i);
  }
  auto input = makeRowVector({makeFlatVector<double>(values)});
  auto result =
      runWindow(input, "approx_winsorize(c0, 0.05, 0.95, 200.0)", "OVER ()");
  auto originalCol = result->childAt(0)->asFlatVector<double>();
  auto winsorized = result->childAt(1)->asFlatVector<double>();
  double minWinsorized{winsorized->valueAt(0)};
  double maxWinsorized{winsorized->valueAt(0)};
  for (int i = 0; i < kNumValues; ++i) {
    double originalValue{originalCol->valueAt(i)};
    double winsorizedValue{winsorized->valueAt(i)};
    minWinsorized = std::min(minWinsorized, winsorizedValue);
    maxWinsorized = std::max(maxWinsorized, winsorizedValue);
    // Middle values should be unchanged.
    if (originalValue > 100 && originalValue < 900) {
      ASSERT_NEAR(winsorizedValue, originalValue, 1.0);
    }
  }
  // Range should be narrowed by winsorization.
  ASSERT_GE(minWinsorized, 30.0);
  ASSERT_LE(maxWinsorized, 970.0);
}

TEST_F(ApproxWinsorizeTest, invalidBounds) {
  auto input = makeRowVector({makeFlatVector<double>({1.0, 2.0, 3.0})});
  VELOX_ASSERT_THROW(
      runWindow(input, "approx_winsorize(c0, -0.1, 0.5)", "OVER ()"), ">= 0");
}

TEST_F(ApproxWinsorizeTest, boundsReversed) {
  auto input = makeRowVector({makeFlatVector<double>({1.0, 2.0, 3.0})});
  VELOX_ASSERT_THROW(
      runWindow(input, "approx_winsorize(c0, 0.9, 0.1)", "OVER ()"), "<=");
}

TEST_F(ApproxWinsorizeTest, nanInput) {
  auto input = makeRowVector({
      makeFlatVector<double>({1.0, std::numeric_limits<double>::quiet_NaN()}),
  });
  VELOX_ASSERT_THROW(
      runWindow(input, "approx_winsorize(c0, 0.0, 1.0)", "OVER ()"), "NaN");
}

TEST_F(ApproxWinsorizeTest, accuracy) {
  // Accuracy test: compare approx_winsorize output against exact computation.
  constexpr int kNumValues = 10'000;
  std::vector<double> values(kNumValues);
  std::default_random_engine generator(42);
  std::lognormal_distribution<> distribution(0, 1.0);
  for (int i = 0; i < kNumValues; ++i) {
    values[i] = distribution(generator);
  }

  auto input = makeRowVector({makeFlatVector<double>(values)});
  auto result = runWindow(input, "approx_winsorize(c0, 0.01, 0.99)", "OVER ()");
  auto originalCol = result->childAt(0)->asFlatVector<double>();
  auto winsorized = result->childAt(1)->asFlatVector<double>();

  // Compute exact winsorized values from the original column.
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  double exactLower{sorted[static_cast<int>(kNumValues * 0.01)]};
  double exactUpper{sorted[static_cast<int>(kNumValues * 0.99)]};

  // Check that the mean of winsorized values is close to exact.
  double approxSum{0};
  double exactSum{0};
  for (int i = 0; i < kNumValues; ++i) {
    approxSum += winsorized->valueAt(i);
    double originalValue{originalCol->valueAt(i)};
    exactSum += std::max(exactLower, std::min(exactUpper, originalValue));
  }
  double approxMean{approxSum / kNumValues};
  double exactMean{exactSum / kNumValues};
  ASSERT_NEAR(approxMean, exactMean, std::abs(exactMean) * 0.05);
}

} // namespace facebook::velox::window::test
