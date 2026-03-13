/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/functions/iceberg/tests/IcebergFunctionBaseTest.h"

namespace facebook::velox::functions::iceberg {
namespace {

class DateTimeFunctionsTest
    : public functions::iceberg::test::IcebergFunctionBaseTest {};

TEST_F(DateTimeFunctionsTest, years) {
  const auto years = [&](const std::string& ts) {
    return evaluateOnce<int32_t>(
        "years(c0)", std::make_optional<Timestamp>(parseTimestamp(ts)));
  };
  EXPECT_EQ(47, years("2017-12-01 10:12:55.038194"));
  EXPECT_EQ(0, years("1970-01-01 00:00:01.000001"));
  EXPECT_EQ(-1, years("1969-12-31 23:59:58.999999"));

  const auto yearsDate = [&](const std::string& date) {
    return evaluateOnce<int32_t>(
        "years(c0)", DATE(), std::make_optional<int32_t>(parseDate(date)));
  };
  EXPECT_EQ(47, yearsDate("2017-12-01"));
  EXPECT_EQ(0, yearsDate("1970-01-01"));
  EXPECT_EQ(-1, yearsDate("1969-12-31"));
}

TEST_F(DateTimeFunctionsTest, months) {
  const auto months = [&](const std::string& ts) {
    return evaluateOnce<int32_t>(
        "months(c0)", std::make_optional<Timestamp>(parseTimestamp(ts)));
  };
  EXPECT_EQ(575, months("2017-12-01 10:12:55.038194"));
  EXPECT_EQ(0, months("1970-01-01 00:00:01.000001"));
  EXPECT_EQ(-1, months("1969-12-31 23:59:58.999999"));

  const auto monthsDate = [&](const std::string& date) {
    return evaluateOnce<int32_t>(
        "months(c0)", DATE(), std::make_optional<int32_t>(parseDate(date)));
  };
  EXPECT_EQ(575, monthsDate("2017-12-01"));
  EXPECT_EQ(0, monthsDate("1970-01-01"));
  EXPECT_EQ(-1, monthsDate("1969-12-31"));
}

TEST_F(DateTimeFunctionsTest, days) {
  const auto days = [&](const std::string& ts) {
    return evaluateOnce<int32_t>(
        "days(c0)", std::make_optional<Timestamp>(parseTimestamp(ts)));
  };

  const auto daysDate = [&](const std::string& date) {
    return evaluateOnce<int32_t>(
        "days(c0)", DATE(), std::make_optional<int32_t>(parseDate(date)));
  };
  EXPECT_EQ(days("2017-12-01 10:12:55.038194"), daysDate("2017-12-01"));
  EXPECT_EQ(days("1970-01-01 00:00:01.000001"), daysDate("1970-01-01"));
  EXPECT_EQ(days("1969-12-31 23:59:58.999999"), daysDate("1969-12-31"));
}

TEST_F(DateTimeFunctionsTest, hours) {
  const auto hours = [&](const std::string& ts) {
    return evaluateOnce<int32_t>(
        "hours(c0)", std::make_optional<Timestamp>(parseTimestamp(ts)));
  };
  EXPECT_EQ(420034, hours("2017-12-01 10:12:55.038194"));
  EXPECT_EQ(0, hours("1970-01-01 00:00:01.000001"));
  EXPECT_EQ(-1, hours("1969-12-31 23:59:58.999999"));
}

} // namespace
} // namespace facebook::velox::functions::iceberg
