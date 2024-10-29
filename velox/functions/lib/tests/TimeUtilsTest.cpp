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

#include "velox/functions/lib/TimeUtils.h"
#include <gtest/gtest.h>

namespace facebook::velox::functions {
namespace {

TEST(TimeUtilsTest, fromDateTimeUnitString) {
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("", false));

  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("microsecond", false));
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("dd", false));
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("mon", false));
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("mm", false));
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("yyyy", false));
  ASSERT_EQ(std::nullopt, fromDateTimeUnitString("yy", false));

  ASSERT_EQ(
      std::optional(DateTimeUnit::kMillisecond),
      fromDateTimeUnitString("millisecond", false));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kSecond),
      fromDateTimeUnitString("second", false));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kMinute),
      fromDateTimeUnitString("minute", false));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kHour),
      fromDateTimeUnitString("hour", false));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kDay), fromDateTimeUnitString("day", false));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kWeek),
      fromDateTimeUnitString("week", false));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kMonth),
      fromDateTimeUnitString("month", false));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kQuarter),
      fromDateTimeUnitString("quarter", false));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kYear),
      fromDateTimeUnitString("year", false));

  ASSERT_EQ(
      std::optional(DateTimeUnit::kMicrosecond),
      fromDateTimeUnitString("microsecond", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kMillisecond),
      fromDateTimeUnitString("millisecond", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kSecond),
      fromDateTimeUnitString("second", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kMinute),
      fromDateTimeUnitString("minute", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kHour),
      fromDateTimeUnitString("hour", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kDay),
      fromDateTimeUnitString("day", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kDay),
      fromDateTimeUnitString("dd", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kWeek),
      fromDateTimeUnitString("week", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kMonth),
      fromDateTimeUnitString("month", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kMonth),
      fromDateTimeUnitString("mon", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kMonth),
      fromDateTimeUnitString("mm", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kQuarter),
      fromDateTimeUnitString("quarter", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kYear),
      fromDateTimeUnitString("year", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kYear),
      fromDateTimeUnitString("yyyy", false, true, true));
  ASSERT_EQ(
      std::optional(DateTimeUnit::kYear),
      fromDateTimeUnitString("yy", false, true, true));
}

} // namespace
} // namespace facebook::velox::functions
