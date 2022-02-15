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
#include "velox/functions/lib/DateTimeFormatterBuilder.h"
#include "velox/functions/lib/DateTimeFormatter.h"
#include "velox/common/base/Exceptions.h"

#include <gtest/gtest.h>

using namespace facebook::velox;

namespace facebook::velox::functions {

class DateTimeFormatterTest : public testing::Test {};

TEST_F(DateTimeFormatterTest, fixedLengthTokenBuilder) {
  DateTimeFormatterBuilder builder(100);
  std::string expectedLiterals;
  std::vector<DateTimeToken> expectedTokens;

  // Test fixed length tokens
  builder.appendEra();
  builder.appendLiteral("-");
  auto formatter = builder.appendHalfDayOfDay().build();

  expectedLiterals = "-";
  std::string_view actualLiterals(
      formatter->literalBuf().get(), formatter->bufSize());
  EXPECT_EQ(actualLiterals, expectedLiterals);
  expectedTokens = {
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::ERA, 2}),
      DateTimeToken("-"),
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::HALFDAY_OF_DAY, 2})};
  EXPECT_EQ(formatter->tokens(), expectedTokens);
}
TEST_F(DateTimeFormatterTest, variableLengthTokenBuilder) {
  // Test variable length tokens
  DateTimeFormatterBuilder builder(100);
  std::string expectedLiterals;
  std::vector<DateTimeToken> expectedTokens;

  auto formatter = builder.appendCenturyOfEra(3)
                       .appendLiteral("-")
                       .appendYearOfEra(4)
                       .appendLiteral("/")
                       .appendWeekYear(3)
                       .appendLiteral("//")
                       .appendWeekOfWeekYear(3)
                       .appendLiteral("-00-")
                       .appendDayOfWeek0Based(3)
                       .appendDayOfWeek1Based(4)
                       .appendLiteral("--")
                       .appendDayOfWeekText(6)
                       .appendLiteral("---")
                       .appendYear(5)
                       .appendLiteral("///")
                       .appendDayOfYear(4)
                       .appendMonthOfYear(2)
                       .appendMonthOfYearText(4)
                       .appendDayOfMonth(4)
                       .appendHourOfHalfDay(2)
                       .appendClockHourOfHalfDay(3)
                       .appendClockHourOfDay(2)
                       .appendHourOfDay(2)
                       .appendMinuteOfHour(2)
                       .appendSecondOfMinute(1)
                       .appendFractionOfSecond(6)
                       .appendTimeZone(3)
                       .appendTimeZoneOffsetId(3)
                       .build();

  expectedLiterals = "-///-00------///";
  auto actualLiterals =
      std::string_view(formatter->literalBuf().get(), formatter->bufSize());
  EXPECT_EQ(actualLiterals, expectedLiterals);
  expectedTokens = {
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::CENTURY_OF_ERA, 3}),
      DateTimeToken("-"),
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::YEAR_OF_ERA, 4}),
      DateTimeToken("/"),
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::WEEK_YEAR, 3}),
      DateTimeToken("//"),
      DateTimeToken(
          FormatPattern{DateTimeFormatSpecifier::WEEK_OF_WEEK_YEAR, 3}),
      DateTimeToken("-00-"),
      DateTimeToken(
          FormatPattern{DateTimeFormatSpecifier::DAY_OF_WEEK_0_BASED, 3}),
      DateTimeToken(
          FormatPattern{DateTimeFormatSpecifier::DAY_OF_WEEK_1_BASED, 4}),
      DateTimeToken("--"),
      DateTimeToken(
          FormatPattern{DateTimeFormatSpecifier::DAY_OF_WEEK_TEXT, 6}),
      DateTimeToken("---"),
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::YEAR, 5}),
      DateTimeToken("///"),
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::DAY_OF_YEAR, 4}),
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::MONTH_OF_YEAR, 2}),
      DateTimeToken(
          FormatPattern{DateTimeFormatSpecifier::MONTH_OF_YEAR_TEXT, 4}),
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::DAY_OF_MONTH, 4}),
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::HOUR_OF_HALFDAY, 2}),
      DateTimeToken(
          FormatPattern{DateTimeFormatSpecifier::CLOCK_HOUR_OF_HALFDAY, 3}),
      DateTimeToken(
          FormatPattern{DateTimeFormatSpecifier::CLOCK_HOUR_OF_DAY, 2}),
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::HOUR_OF_DAY, 2}),
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::MINUTE_OF_HOUR, 2}),
      DateTimeToken(
          FormatPattern{DateTimeFormatSpecifier::SECOND_OF_MINUTE, 1}),
      DateTimeToken(
          FormatPattern{DateTimeFormatSpecifier::FRACTION_OF_SECOND, 6}),
      DateTimeToken(FormatPattern{DateTimeFormatSpecifier::TIMEZONE, 3}),
      DateTimeToken(
          FormatPattern{DateTimeFormatSpecifier::TIMEZONE_OFFSET_ID, 3})};
  EXPECT_EQ(formatter->tokens(), expectedTokens);
}

} // namespace facebook::velox::functions