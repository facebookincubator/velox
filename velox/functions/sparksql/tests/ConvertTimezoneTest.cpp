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

namespace facebook::velox::functions::sparksql::test {
namespace {

class ConvertTimezoneTest : public SparkFunctionBaseTest {
 protected:
  void setQueryTimeZone(const std::string& timeZone) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kSessionTimezone, timeZone},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
    });
  }

  std::optional<Timestamp> convertTimezone(
      const std::string& sourceTz,
      const std::string& targetTz,
      const std::string& tsStr) {
    auto ts = std::make_optional(parseTimestamp(tsStr));
    return evaluateOnce<Timestamp>(
        "convert_timezone(c0, c1, c2)",
        {VARCHAR(), VARCHAR(), TIMESTAMP_UTC()},
        std::make_optional(sourceTz),
        std::make_optional(targetTz),
        ts);
  }

  std::optional<Timestamp> convertTimezone(
      const std::string& targetTz,
      const std::string& tsStr) {
    auto ts = std::make_optional(parseTimestamp(tsStr));
    return evaluateOnce<Timestamp>(
        "convert_timezone(c0, c1)",
        {VARCHAR(), TIMESTAMP_UTC()},
        std::make_optional(targetTz),
        ts);
  }

  // For calls needing a nullable argument or a raw Timestamp not
  // expressible as a parseable date string.
  std::optional<Timestamp> evalConvertTimezoneNullable(
      std::optional<std::string> sourceTz,
      std::optional<std::string> targetTz,
      std::optional<Timestamp> ts) {
    return evaluateOnce<Timestamp>(
        "convert_timezone(c0, c1, c2)",
        {VARCHAR(), VARCHAR(), TIMESTAMP_UTC()},
        std::move(sourceTz),
        std::move(targetTz),
        ts);
  }
};

TEST_F(ConvertTimezoneTest, basicConversion) {
  EXPECT_EQ(
      parseTimestamp("2021-12-05 15:00:00"),
      convertTimezone(
          "Europe/Brussels", "America/Los_Angeles", "2021-12-06 00:00:00"));
}

TEST_F(ConvertTimezoneTest, sessionTimezoneAsSource) {
  setQueryTimeZone("America/Los_Angeles");
  EXPECT_EQ(
      parseTimestamp("2021-12-06 00:00:00"),
      convertTimezone("Europe/Brussels", "2021-12-05 15:00:00"));
}

TEST_F(ConvertTimezoneTest, constantTimezoneArgs) {
  // Timezone names as SQL literals are constant-folded, exercising the
  // initialize()-time caching path instead of the per-row fallback.
  const auto evalConvertTimezoneLiteral = [&](const std::string& functionCall,
                                              std::optional<Timestamp> ts) {
    return evaluateOnce<Timestamp>(functionCall, {TIMESTAMP_UTC()}, ts);
  };

  auto ts = std::make_optional(parseTimestamp("2021-12-06 00:00:00"));
  EXPECT_EQ(
      parseTimestamp("2021-12-05 15:00:00"),
      evalConvertTimezoneLiteral(
          "convert_timezone('Europe/Brussels', 'America/Los_Angeles', c0)",
          ts));

  setQueryTimeZone("America/Los_Angeles");
  auto ts2 = std::make_optional(parseTimestamp("2021-12-05 15:00:00"));
  EXPECT_EQ(
      parseTimestamp("2021-12-06 00:00:00"),
      evalConvertTimezoneLiteral(
          "convert_timezone('Europe/Brussels', c0)", ts2));
}

// 1941-12-25 in Hong Kong: clocks jumped from HKWT (UTC+8) to JST (UTC+9),
// making midnight nonexistent. Spark adjusts to 00:30:00 JST, which is
// 1941-12-24 15:30:00 UTC.
TEST_F(ConvertTimezoneTest, timezoneGapCorrection) {
  auto result = evalConvertTimezoneNullable(
      std::make_optional<std::string>("Asia/Hong_Kong"),
      std::make_optional<std::string>("UTC"),
      std::make_optional(Timestamp(-884'217'600, 0)));
  EXPECT_EQ(Timestamp(-884'248'200, 0), result);
}

TEST_F(ConvertTimezoneTest, unknownTimezone) {
  VELOX_ASSERT_THROW(
      convertTimezone("Asia/Ooty", "UTC", "2021-12-06 00:00:00"),
      "Unknown time zone: 'Asia/Ooty'");
  VELOX_ASSERT_THROW(
      convertTimezone("UTC", "Asia/Ooty", "2021-12-06 00:00:00"),
      "Unknown time zone: 'Asia/Ooty'");
  VELOX_ASSERT_THROW(
      convertTimezone("Asia/Ooty", "2021-12-06 00:00:00"),
      "Unknown time zone: 'Asia/Ooty'");
}

TEST_F(ConvertTimezoneTest, nullPropagation) {
  EXPECT_EQ(
      std::nullopt,
      evalConvertTimezoneNullable(
          std::optional<std::string>(std::nullopt),
          std::make_optional<std::string>("UTC"),
          std::make_optional(parseTimestamp("2021-12-06 00:00:00"))));
  EXPECT_EQ(
      std::nullopt,
      evalConvertTimezoneNullable(
          std::make_optional<std::string>("UTC"),
          std::optional<std::string>(std::nullopt),
          std::make_optional(parseTimestamp("2021-12-06 00:00:00"))));
  EXPECT_EQ(
      std::nullopt,
      evalConvertTimezoneNullable(
          std::make_optional<std::string>("UTC"),
          std::make_optional<std::string>("UTC"),
          std::optional<Timestamp>(std::nullopt)));
}

TEST_F(ConvertTimezoneTest, sameSourceAndTargetTimezone) {
  EXPECT_EQ(
      parseTimestamp("2015-01-24 05:30:00"),
      convertTimezone("Asia/Riyadh", "Asia/Riyadh", "2015-01-24 05:30:00"));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
