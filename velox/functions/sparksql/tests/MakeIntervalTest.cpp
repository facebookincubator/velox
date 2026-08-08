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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/type/CalendarInterval.h"
#include "velox/type/Type.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class MakeIntervalTest : public SparkFunctionBaseTest {
 protected:
  struct IntervalResult {
    int32_t months;
    int32_t days;
    int64_t microseconds;
  };

  // Helper to evaluate and unpack CalendarInterval from int128_t.
  std::optional<IntervalResult> evaluateInterval(const std::string& expr) {
    auto result = evaluate(expr, makeRowVector(ROW({}), 1));
    SelectivityVector rows(1);
    DecodedVector decoded(*result, rows);
    if (decoded.isNullAt(0)) {
      return std::nullopt;
    }
    auto packed = decoded.valueAt<int128_t>(0);
    auto interval = CalendarInterval::unpack(packed);
    return IntervalResult{
        interval.months, interval.days, interval.microseconds};
  }

  // Helper to evaluate with input data and unpack at a specific row.
  std::optional<IntervalResult> evaluateIntervalAt(
      const std::string& expr,
      const RowVectorPtr& data,
      vector_size_t row = 0) {
    auto result = evaluate(expr, data);
    SelectivityVector rows(result->size());
    DecodedVector decoded(*result, rows);
    if (decoded.isNullAt(row)) {
      return std::nullopt;
    }
    auto packed = decoded.valueAt<int128_t>(row);
    auto interval = CalendarInterval::unpack(packed);
    return IntervalResult{
        interval.months, interval.days, interval.microseconds};
  }
};

TEST_F(MakeIntervalTest, noArgs) {
  // make_interval() → (0, 0, 0)
  auto result = evaluateInterval("make_interval()");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->months, 0);
  EXPECT_EQ(result->days, 0);
  EXPECT_EQ(result->microseconds, 0);
}

TEST_F(MakeIntervalTest, yearsOnly) {
  // make_interval(2) → (24, 0, 0)
  auto data = makeRowVector({makeFlatVector<int32_t>({2})});
  auto result = evaluateIntervalAt("make_interval(c0)", data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->months, 24); // 2*12
  EXPECT_EQ(result->days, 0);
  EXPECT_EQ(result->microseconds, 0);
}

TEST_F(MakeIntervalTest, yearsAndMonths) {
  // make_interval(1, 6) → (18, 0, 0)
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({1}), makeFlatVector<int32_t>({6})});
  auto result = evaluateIntervalAt("make_interval(c0, c1)", data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->months, 18); // 1*12 + 6
  EXPECT_EQ(result->days, 0);
  EXPECT_EQ(result->microseconds, 0);
}

TEST_F(MakeIntervalTest, weeksAndDays) {
  // make_interval(0, 0, 2, 3) → (0, 17, 0) — 2*7 + 3 = 17 days
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({2}),
       makeFlatVector<int32_t>({3})});
  auto result = evaluateIntervalAt("make_interval(c0, c1, c2, c3)", data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->months, 0);
  EXPECT_EQ(result->days, 17);
  EXPECT_EQ(result->microseconds, 0);
}

TEST_F(MakeIntervalTest, hoursMinutes) {
  // make_interval(0, 0, 0, 0, 2, 30) → (0, 0, 2*3600000000 + 30*60000000)
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({2}),
       makeFlatVector<int32_t>({30})});
  auto result =
      evaluateIntervalAt("make_interval(c0, c1, c2, c3, c4, c5)", data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->months, 0);
  EXPECT_EQ(result->days, 0);
  int64_t expectedMicros = 2LL * 3600000000LL + 30LL * 60000000LL;
  EXPECT_EQ(result->microseconds, expectedMicros);
}

TEST_F(MakeIntervalTest, allParameters) {
  // make_interval(1, 2, 3, 4, 5, 6, 7.890000)
  // months: 1*12 + 2 = 14
  // days: 3*7 + 4 = 25
  // micros: 5*3600000000 + 6*60000000 + 7890000 (7.89 secs as micros)
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int32_t>({2}),
       makeFlatVector<int32_t>({3}),
       makeFlatVector<int32_t>({4}),
       makeFlatVector<int32_t>({5}),
       makeFlatVector<int32_t>({6}),
       makeFlatVector(std::vector<int64_t>{7890000}, DECIMAL(8, 6))});
  auto result =
      evaluateIntervalAt("make_interval(c0, c1, c2, c3, c4, c5, c6)", data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->months, 14);
  EXPECT_EQ(result->days, 25);
  int64_t expectedMicros = 5LL * 3600000000LL + 6LL * 60000000LL + 7890000LL;
  EXPECT_EQ(result->microseconds, expectedMicros);
}

TEST_F(MakeIntervalTest, negativeValues) {
  // make_interval(-1, -2, -3, -4, -5, -6, -7.5)
  // months: -1*12 + (-2) = -14
  // days: -3*7 + (-4) = -25
  // micros: -5*3600000000 + (-6)*60000000 + (-7500000)
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({-1}),
       makeFlatVector<int32_t>({-2}),
       makeFlatVector<int32_t>({-3}),
       makeFlatVector<int32_t>({-4}),
       makeFlatVector<int32_t>({-5}),
       makeFlatVector<int32_t>({-6}),
       makeFlatVector(std::vector<int64_t>{-7500000}, DECIMAL(8, 6))});
  auto result =
      evaluateIntervalAt("make_interval(c0, c1, c2, c3, c4, c5, c6)", data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->months, -14);
  EXPECT_EQ(result->days, -25);
  int64_t expectedMicros =
      -5LL * 3600000000LL + (-6LL) * 60000000LL + (-7500000LL);
  EXPECT_EQ(result->microseconds, expectedMicros);
}

TEST_F(MakeIntervalTest, monthsOverflowReturnsNull) {
  // years = 178956971 → 178956971 * 12 = 2147483652 > INT32_MAX
  auto data = makeRowVector({makeFlatVector<int32_t>({178956971})});
  auto result = evaluateIntervalAt("try(make_interval(c0))", data);
  EXPECT_FALSE(result.has_value());
}

TEST_F(MakeIntervalTest, daysOverflowReturnsNull) {
  // weeks = 306783379 → 306783379 * 7 = 2147483653 > INT32_MAX
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({306783379})});
  auto result = evaluateIntervalAt("try(make_interval(c0, c1, c2))", data);
  EXPECT_FALSE(result.has_value());
}

TEST_F(MakeIntervalTest, microsOverflowReturnsNull) {
  // hours = 1, secs = INT64_MAX → secsUnscaled + hoursMicros overflows.
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({1}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector(
           std::vector<int64_t>{std::numeric_limits<int64_t>::max()},
           DECIMAL(18, 6))});
  auto result = evaluateIntervalAt(
      "try(make_interval(c0, c1, c2, c3, c4, c5, c6))", data);
  EXPECT_FALSE(result.has_value());
}

TEST_F(MakeIntervalTest, monthsAddOverflowReturnsNull) {
  // years = 1, months = INT32_MAX → 12 + INT32_MAX overflows.
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int32_t>({std::numeric_limits<int32_t>::max()})});
  auto result = evaluateIntervalAt("try(make_interval(c0, c1))", data);
  EXPECT_FALSE(result.has_value());
}

TEST_F(MakeIntervalTest, daysAddOverflowReturnsNull) {
  // weeks = 1, days = INT32_MAX → 7 + INT32_MAX overflows.
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({1}),
       makeFlatVector<int32_t>({std::numeric_limits<int32_t>::max()})});
  auto result = evaluateIntervalAt("try(make_interval(c0, c1, c2, c3))", data);
  EXPECT_FALSE(result.has_value());
}

TEST_F(MakeIntervalTest, microsMinsMicrosAddOverflowReturnsNull) {
  // secs = INT64_MAX, hours = 0, mins = 1 → INT64_MAX + 60000000 overflows.
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({1}),
       makeFlatVector(
           std::vector<int64_t>{std::numeric_limits<int64_t>::max()},
           DECIMAL(18, 6))});
  auto result = evaluateIntervalAt(
      "try(make_interval(c0, c1, c2, c3, c4, c5, c6))", data);
  EXPECT_FALSE(result.has_value());
}

TEST_F(MakeIntervalTest, microsAssociativityMatchesSpark) {
  // Verify: hours=1, mins=-60, secs=INT64_MAX
  // Spark order: INT64_MAX + (1*3600000000) → overflow → null
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({1}),
       makeFlatVector<int32_t>({-60}),
       makeFlatVector(
           std::vector<int64_t>{std::numeric_limits<int64_t>::max()},
           DECIMAL(18, 6))});
  auto result = evaluateIntervalAt(
      "try(make_interval(c0, c1, c2, c3, c4, c5, c6))", data);
  EXPECT_FALSE(result.has_value());
}

TEST_F(MakeIntervalTest, multipleRows) {
  // Verify vectorized execution across multiple rows.
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({1, 0, -1}),
       makeFlatVector<int32_t>({0, 6, -3})});
  auto r0 = evaluateIntervalAt("make_interval(c0, c1)", data, 0);
  auto r1 = evaluateIntervalAt("make_interval(c0, c1)", data, 1);
  auto r2 = evaluateIntervalAt("make_interval(c0, c1)", data, 2);
  ASSERT_TRUE(r0.has_value());
  ASSERT_TRUE(r1.has_value());
  ASSERT_TRUE(r2.has_value());
  EXPECT_EQ(r0->months, 12); // 1*12 + 0
  EXPECT_EQ(r1->months, 6); // 0*12 + 6
  EXPECT_EQ(r2->months, -15); // -1*12 + (-3)
}

TEST_F(MakeIntervalTest, overflowWithTryReturnsNull) {
  // years=178956971 → overflow
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({178956971}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector(std::vector<int64_t>{0}, DECIMAL(18, 6))});
  auto result = evaluateIntervalAt(
      "try(make_interval(c0, c1, c2, c3, c4, c5, c6))", data);
  EXPECT_FALSE(result.has_value());
}

TEST_F(MakeIntervalTest, overflowWithoutTryThrows) {
  // Overflow without try() throws (ANSI mode behavior).
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({178956971}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector(std::vector<int64_t>{0}, DECIMAL(18, 6))});
  VELOX_ASSERT_THROW(
      evaluate("make_interval(c0, c1, c2, c3, c4, c5, c6)", data),
      "Integer overflow in make_interval");
}

TEST_F(MakeIntervalTest, nullInputReturnsNull) {
  // Default null behavior: any null input produces null output.
  auto data = makeRowVector(
      {makeNullableFlatVector<int32_t>({std::nullopt}),
       makeFlatVector<int32_t>({5}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({10}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector(std::vector<int64_t>{0}, DECIMAL(18, 6))});
  auto result =
      evaluateIntervalAt("make_interval(c0, c1, c2, c3, c4, c5, c6)", data);
  EXPECT_FALSE(result.has_value());

  // Null in middle argument also produces null.
  auto data2 = makeRowVector(
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int32_t>({2}),
       makeNullableFlatVector<int32_t>({std::nullopt}),
       makeFlatVector<int32_t>({10}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector(std::vector<int64_t>{0}, DECIMAL(18, 6))});
  auto result2 =
      evaluateIntervalAt("make_interval(c0, c1, c2, c3, c4, c5, c6)", data2);
  EXPECT_FALSE(result2.has_value());

  // Null in last argument (secs DECIMAL) also produces null.
  auto data3 = makeRowVector(
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int32_t>({2}),
       makeFlatVector<int32_t>({0}),
       makeFlatVector<int32_t>({10}),
       makeFlatVector<int32_t>({3}),
       makeFlatVector<int32_t>({4}),
       makeNullableFlatVector(
           std::vector<std::optional<int64_t>>{std::nullopt}, DECIMAL(18, 6))});
  auto result3 =
      evaluateIntervalAt("make_interval(c0, c1, c2, c3, c4, c5, c6)", data3);
  EXPECT_FALSE(result3.has_value());
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
