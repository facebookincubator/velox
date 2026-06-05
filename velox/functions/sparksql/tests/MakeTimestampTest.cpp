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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class MakeTimestampTest : public SparkFunctionBaseTest {
 protected:
  void setQueryTimeZone(const std::string& timeZone) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kSessionTimezone, timeZone},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
    });
  }

  void setAnsiEnabled(bool enabled) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled),
          enabled ? "true" : "false"}});
  }
};

TEST_F(MakeTimestampTest, basic) {
  const auto microsType = DECIMAL(16, 6);
  const auto testMakeTimestamp = [&](const RowVectorPtr& data,
                                     const VectorPtr& expected,
                                     bool hasTimeZone) {
    auto result = hasTimeZone
        ? evaluate("make_timestamp(c0, c1, c2, c3, c4, c5, c6)", data)
        : evaluate("make_timestamp(c0, c1, c2, c3, c4, c5)", data);
    facebook::velox::test::assertEqualVectors(expected, result);
  };
  const auto testConstantTimezone = [&](const RowVectorPtr& data,
                                        const std::string& timezone,
                                        const VectorPtr& expected) {
    auto result = evaluate(
        fmt::format("make_timestamp(c0, c1, c2, c3, c4, c5, '{}')", timezone),
        data);
    facebook::velox::test::assertEqualVectors(expected, result);
  };

  // Valid cases w/o time zone argument.
  {
    const auto year = makeFlatVector<int32_t>({2021, 2021, 2021, 2021, 2021});
    const auto month = makeFlatVector<int32_t>({7, 7, 7, 7, 7});
    const auto day = makeFlatVector<int32_t>({11, 11, 11, 11, 11});
    const auto hour = makeFlatVector<int32_t>({6, 6, 6, 6, 6});
    const auto minute = makeFlatVector<int32_t>({30, 30, 30, 30, 30});
    const auto micros = makeNullableFlatVector<int64_t>(
        {45678000, 1e6, 6e7, 59999999, std::nullopt}, microsType);
    auto data = makeRowVector({year, month, day, hour, minute, micros});

    setQueryTimeZone("GMT");
    auto expectedGMT = makeNullableFlatVector<Timestamp>(
        {parseTimestamp("2021-07-11 06:30:45.678"),
         parseTimestamp("2021-07-11 06:30:01"),
         parseTimestamp("2021-07-11 06:31:00"),
         parseTimestamp("2021-07-11 06:30:59.999999"),
         std::nullopt});
    testMakeTimestamp(data, expectedGMT, false);
    testConstantTimezone(data, "GMT", expectedGMT);

    setQueryTimeZone("Asia/Shanghai");
    auto expectedSessionTimezone = makeNullableFlatVector<Timestamp>(
        {parseTimestamp("2021-07-10 22:30:45.678"),
         parseTimestamp("2021-07-10 22:30:01"),
         parseTimestamp("2021-07-10 22:31:00"),
         parseTimestamp("2021-07-10 22:30:59.999999"),
         std::nullopt});
    testMakeTimestamp(data, expectedSessionTimezone, false);
    // Session time zone will be ignored if time zone is specified in argument.
    testConstantTimezone(data, "GMT", expectedGMT);
  }

  // Valid cases w/ time zone argument.
  {
    setQueryTimeZone("Asia/Shanghai");
    const auto year = makeFlatVector<int32_t>({2021, 2021, 1, 2024});
    const auto month = makeFlatVector<int32_t>({07, 07, 1, 3});
    const auto day = makeFlatVector<int32_t>({11, 11, 1, 10});
    const auto hour = makeFlatVector<int32_t>({6, 6, 1, 2});
    const auto minute = makeFlatVector<int32_t>({30, 30, 1, 1});
    const auto micros =
        makeFlatVector<int64_t>({45678000, 45678000, 0, 45678000}, microsType);
    const auto timeZone = makeNullableFlatVector<StringView>(
        {"GMT", "CET", std::nullopt, "America/Chicago"});
    auto data =
        makeRowVector({year, month, day, hour, minute, micros, timeZone});
    // Session time zone will be ignored if time zone is specified in argument.
    auto expected = makeNullableFlatVector<Timestamp>(
        {parseTimestamp("2021-07-11 06:30:45.678"),
         parseTimestamp("2021-07-11 04:30:45.678"),
         std::nullopt,
         parseTimestamp("2024-03-10 08:01:45.678")});
    testMakeTimestamp(data, expected, true);
  }
}

TEST_F(MakeTimestampTest, errors) {
  const auto microsType = DECIMAL(16, 6);
  const auto testInvalidInputs = [&](const RowVectorPtr& data) {
    std::vector<std::optional<Timestamp>> nullResults(
        data->size(), std::nullopt);
    auto expected = makeNullableFlatVector<Timestamp>(nullResults);
    auto result = evaluate("make_timestamp(c0, c1, c2, c3, c4, c5)", data);
    facebook::velox::test::assertEqualVectors(expected, result);
  };
  std::optional<int32_t> one = 1;
  const auto testInvalidSeconds = [&](std::optional<int64_t> microsec) {
    auto result = evaluateOnce<Timestamp>(
        "make_timestamp(c0, c1, c2, c3, c4, c5)",
        {INTEGER(), INTEGER(), INTEGER(), INTEGER(), INTEGER(), microsType},
        one,
        one,
        one,
        one,
        one,
        microsec);
    EXPECT_EQ(result, std::nullopt);
  };
  const auto testInvalidArguments = [&](std::optional<int64_t> microsec,
                                        const TypePtr& microsType) {
    return evaluateOnce<Timestamp>(
        "make_timestamp(c0, c1, c2, c3, c4, c5)",
        {INTEGER(), INTEGER(), INTEGER(), INTEGER(), INTEGER(), microsType},
        one,
        one,
        one,
        one,
        one,
        microsec);
  };

  // Throw if no session time zone.
  VELOX_ASSERT_USER_THROW(
      testInvalidArguments(60007000, DECIMAL(16, 6)),
      "make_timestamp requires session time zone to be set.");

  setQueryTimeZone("Asia/Shanghai");
  // ANSI off: invalid input returns NULL instead of throwing.
  setAnsiEnabled(false);
  // Invalid input returns null.
  const auto year = makeFlatVector<int32_t>(
      {facebook::velox::util::kMinYear - 1,
       facebook::velox::util::kMaxYear + 1,
       1,
       1,
       1,
       1,
       1,
       1,
       1,
       1});
  const auto month = makeFlatVector<int32_t>({1, 1, 0, 13, 1, 1, 1, 1, 1, 1});
  const auto day = makeFlatVector<int32_t>({1, 1, 1, 1, 0, 32, 1, 1, 1, 1});
  const auto hour = makeFlatVector<int32_t>({1, 1, 1, 1, 1, 1, 25, 1, 24, 1});
  const auto minute = makeFlatVector<int32_t>({1, 1, 1, 1, 1, 1, 1, 61, 1, 60});
  const auto micros =
      makeFlatVector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, microsType);
  auto data = makeRowVector({year, month, day, hour, minute, micros});
  testInvalidInputs(data);

  // Seconds should be either in the range of [0,59], or 60 with zero
  // microseconds.
  testInvalidSeconds(61e6);
  testInvalidSeconds(99999999);
  testInvalidSeconds(999999999);
  testInvalidSeconds(60007000);

  // Throw if data type for microseconds is invalid.
  VELOX_ASSERT_THROW(
      testInvalidArguments(1e6, DECIMAL(20, 6)),
      "Seconds must be short decimal type but got DECIMAL(20, 6)");
  VELOX_ASSERT_THROW(
      testInvalidArguments(1e6, DECIMAL(16, 8)),
      "Scalar function signature is not supported: "
      "make_timestamp(INTEGER, INTEGER, INTEGER, INTEGER, INTEGER, "
      "DECIMAL(16, 8)).");
}

TEST_F(MakeTimestampTest, ansiErrors) {
  // Under ANSI mode each invalid argument throws with a field-specific
  // message rather than returning NULL.
  const auto microsType = DECIMAL(16, 6);
  setQueryTimeZone("Asia/Shanghai");
  setAnsiEnabled(true);

  const auto eval = [&](int32_t year,
                        int32_t month,
                        int32_t day,
                        int32_t hour,
                        int32_t minute,
                        int64_t micros) {
    return evaluateOnce<Timestamp>(
        "make_timestamp(c0, c1, c2, c3, c4, c5)",
        {INTEGER(), INTEGER(), INTEGER(), INTEGER(), INTEGER(), microsType},
        std::optional<int32_t>(year),
        std::optional<int32_t>(month),
        std::optional<int32_t>(day),
        std::optional<int32_t>(hour),
        std::optional<int32_t>(minute),
        std::optional<int64_t>(micros));
  };

  // Hour out of range.
  VELOX_ASSERT_USER_THROW(
      eval(2021, 7, 11, 24, 30, 0),
      "Invalid value for hour, must be in [0, 24): 24");
  VELOX_ASSERT_USER_THROW(
      eval(2021, 7, 11, -1, 30, 0),
      "Invalid value for hour, must be in [0, 24): -1");
  VELOX_ASSERT_USER_THROW(
      eval(2021, 7, 11, 25, 30, 0),
      "Invalid value for hour, must be in [0, 24): 25");

  // Minute out of range.
  VELOX_ASSERT_USER_THROW(
      eval(2021, 7, 11, 6, 60, 0),
      "Invalid value for minute, must be in [0, 60): 60");
  VELOX_ASSERT_USER_THROW(
      eval(2021, 7, 11, 6, -1, 0),
      "Invalid value for minute, must be in [0, 60): -1");

  // Negative microseconds.
  VELOX_ASSERT_USER_THROW(
      eval(2021, 7, 11, 6, 30, -1),
      "Invalid value for second microseconds, must be non-negative: -1");

  // Seconds out of range.
  VELOX_ASSERT_USER_THROW(
      eval(2021, 7, 11, 6, 30, 61'000'000),
      "Invalid value for second, must be in [0, 60] with 0 microseconds at 60: 61.000000");
  VELOX_ASSERT_USER_THROW(
      eval(2021, 7, 11, 6, 30, 60'007'000),
      "Invalid value for second, must be in [0, 60] with 0 microseconds at 60: 60.007000");

  // Invalid date components.
  VELOX_ASSERT_USER_THROW(eval(2021, 0, 11, 6, 30, 0), "Date out of range");
  VELOX_ASSERT_USER_THROW(eval(2021, 13, 11, 6, 30, 0), "Date out of range");
  VELOX_ASSERT_USER_THROW(eval(2021, 7, 0, 6, 30, 0), "Date out of range");
  VELOX_ASSERT_USER_THROW(eval(2021, 7, 32, 6, 30, 0), "Date out of range");
  VELOX_ASSERT_USER_THROW(
      eval(facebook::velox::util::kMinYear - 1, 7, 11, 6, 30, 0),
      "Date out of range");
  VELOX_ASSERT_USER_THROW(
      eval(facebook::velox::util::kMaxYear + 1, 7, 11, 6, 30, 0),
      "Date out of range");
}

TEST_F(MakeTimestampTest, invalidTimezone) {
  const auto microsType = DECIMAL(16, 6);
  const auto year = makeFlatVector<int32_t>({2021, 2021, 2021, 2021, 2021});
  const auto month = makeFlatVector<int32_t>({7, 7, 7, 7, 7});
  const auto day = makeFlatVector<int32_t>({11, 11, 11, 11, 11});
  const auto hour = makeFlatVector<int32_t>({6, 6, 6, -6, 6});
  const auto minute = makeFlatVector<int32_t>({30, 30, 30, 30, 30});
  const auto micros = makeNullableFlatVector<int64_t>(
      {45678000, 1e6, 6e7, 59999999, std::nullopt}, microsType);
  auto data = makeRowVector({year, month, day, hour, minute, micros});

  // Time zone is not set.
  VELOX_ASSERT_USER_THROW(
      evaluate("make_timestamp(c0, c1, c2, c3, c4, c5)", data),
      "make_timestamp requires session time zone to be set.");

  // Invalid constant time zone.
  setQueryTimeZone("GMT");
  for (auto timeZone : {"Invalid", ""}) {
    SCOPED_TRACE(fmt::format("timezone: {}", timeZone));
    VELOX_ASSERT_USER_THROW(
        evaluate(
            fmt::format(
                "make_timestamp(c0, c1, c2, c3, c4, c5, '{}')", timeZone),
            data),
        fmt::format("Unknown time zone: '{}'", timeZone));
  }

  // Invalid timezone from vector.
  auto timeZones = makeFlatVector<StringView>(
      {"GMT", "CET", "Asia/Shanghai", "Invalid", "GMT"});
  data = makeRowVector({year, month, day, hour, minute, micros, timeZones});
  VELOX_ASSERT_USER_THROW(
      evaluate("make_timestamp(c0, c1, c2, c3, c4, c5, c6)", data),
      "Unknown time zone: 'Invalid'");

  timeZones =
      makeFlatVector<StringView>({"GMT", "CET", "Asia/Shanghai", "", "GMT"});
  data = makeRowVector({year, month, day, hour, minute, micros, timeZones});
  VELOX_ASSERT_USER_THROW(
      evaluate("make_timestamp(c0, c1, c2, c3, c4, c5, c6)", data),
      "Unknown time zone: ''");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
