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

#include "velox/functions/prestosql/tests/CastBaseTest.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

namespace facebook::velox {
namespace {

class TimeWithTimezoneCastTest : public functions::test::CastBaseTest {
 protected:
  void setQueryTimeZone(const std::string& timeZone) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kSessionTimezone, timeZone},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
    });
  }

  void disableAdjustTimestampToTimezone() {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kAdjustTimestampToTimezone, "false"},
    });
  }

  // Helper to pack TIME WITH TIME ZONE value
  int64_t packTimeWithTZ(int64_t timeMillis, int16_t offsetMinutes) {
    auto encodedOffset = TimeWithTimezoneType::biasEncode(offsetMinutes);
    return pack(timeMillis, encodedOffset);
  }
};

TEST_F(TimeWithTimezoneCastTest, fromTime) {
  {
    // Test casting TIME to TIME WITH TIME ZONE with various times
    auto input = makeFlatVector<int64_t>(
        {
            0, // 00:00:00.000 (midnight)
            3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond +
                321, // 03:04:05.321
            12 * kMillisInHour, // 12:00:00.000 (noon)
            23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                999, // 23:59:59.999 (end of day)
        },
        TIME());

    setQueryTimeZone("America/Los_Angeles");

    // LA is UTC-8, so offset is -8 * 60 = -480 minutes
    auto expected = makeFlatVector<int64_t>(
        {
            packTimeWithTZ(0, -480),
            packTimeWithTZ(
                3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond +
                    321,
                -480),
            packTimeWithTZ(12 * kMillisInHour, -480),
            packTimeWithTZ(
                23 * kMillisInHour + 59 * kMillisInMinute +
                    59 * kMillisInSecond + 999,
                -480),
        },
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }

  {
    // Test with nulls
    auto input = makeNullableFlatVector<int64_t>(
        {0,
         std::nullopt,
         12 * kMillisInHour,
         std::nullopt,
         23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
             999},
        TIME());

    setQueryTimeZone("America/Los_Angeles");

    auto expected = makeNullableFlatVector<int64_t>(
        {packTimeWithTZ(0, -480),
         std::nullopt,
         packTimeWithTZ(12 * kMillisInHour, -480),
         std::nullopt,
         packTimeWithTZ(
             23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                 999,
             -480)},
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }

  {
    // Test with different timezone (Asia/Shanghai, UTC+8)
    auto input = makeFlatVector<int64_t>(
        {
            0,
            12 * kMillisInHour,
            23 * kMillisInHour + 59 * kMillisInMinute + 59 * kMillisInSecond +
                999,
        },
        TIME());

    setQueryTimeZone("Asia/Shanghai");

    // Shanghai is UTC+8, so offset is 8 * 60 = 480 minutes
    auto expected = makeFlatVector<int64_t>(
        {
            packTimeWithTZ(0, 480),
            packTimeWithTZ(12 * kMillisInHour, 480),
            packTimeWithTZ(
                23 * kMillisInHour + 59 * kMillisInMinute +
                    59 * kMillisInSecond + 999,
                480),
        },
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }

  {
    // Test with adjustTimestampToTimezone disabled
    // When disabled, offset should be 0 (UTC)
    auto input = makeFlatVector<int64_t>(
        {
            0,
            3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond + 321,
            12 * kMillisInHour,
        },
        TIME());

    setQueryTimeZone("Asia/Shanghai");
    disableAdjustTimestampToTimezone();

    // When adjustTimestampToTimezone is false, offset is 0 (UTC)
    auto expected = makeFlatVector<int64_t>(
        {
            packTimeWithTZ(0, 0),
            packTimeWithTZ(
                3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond +
                    321,
                0),
            packTimeWithTZ(12 * kMillisInHour, 0),
        },
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }

  {
    // Test constant TIME vector cast to TIME WITH TIME ZONE (non-null)
    auto constantTimeVector = BaseVector::wrapInConstant(
        1000, 0, makeFlatVector<int64_t>({12 * kMillisInHour}, TIME()));

    setQueryTimeZone("America/New_York");

    // NY is UTC-5, so offset is -5 * 60 = -300 minutes
    auto expected = BaseVector::wrapInConstant(
        1000,
        0,
        makeFlatVector<int64_t>(
            {packTimeWithTZ(12 * kMillisInHour, -300)}, TIME_WITH_TIME_ZONE()));

    testCast(constantTimeVector, expected);
  }

  {
    // Test constant TIME vector cast to TIME WITH TIME ZONE (null)
    auto nullTimeVector = BaseVector::createNullConstant(TIME(), 500, pool());

    setQueryTimeZone("America/Los_Angeles");

    auto expected =
        BaseVector::createNullConstant(TIME_WITH_TIME_ZONE(), 500, pool());

    testCast(nullTimeVector, expected);
  }

  {
    // Test constant TIME vector with different sizes
    setQueryTimeZone("America/Los_Angeles");

    for (auto size : {1, 10, 100, 1000}) {
      auto constantTimeVector = BaseVector::wrapInConstant(
          size,
          0,
          makeFlatVector<int64_t>(
              {3 * kMillisInHour + 4 * kMillisInMinute + 5 * kMillisInSecond +
               321},
              TIME()));

      auto expected = BaseVector::wrapInConstant(
          size,
          0,
          makeFlatVector<int64_t>(
              {packTimeWithTZ(
                  3 * kMillisInHour + 4 * kMillisInMinute +
                      5 * kMillisInSecond + 321,
                  -480)},
              TIME_WITH_TIME_ZONE()));

      testCast(constantTimeVector, expected);
    }
  }

  {
    // Test basic cast for completeness
    auto input =
        makeFlatVector<int64_t>({0, 12 * kMillisInHour, 86399999}, TIME());

    setQueryTimeZone("America/Los_Angeles");

    auto expected = makeFlatVector<int64_t>(
        {
            packTimeWithTZ(0, -480),
            packTimeWithTZ(12 * kMillisInHour, -480),
            packTimeWithTZ(86399999, -480),
        },
        TIME_WITH_TIME_ZONE());

    testCast(input, expected);
  }
}

} // namespace
} // namespace facebook::velox
