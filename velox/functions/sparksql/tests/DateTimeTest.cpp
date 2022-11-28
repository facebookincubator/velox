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

#include <optional>
#include <string>
#include <string_view>
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Date.h"
#include "velox/type/Timestamp.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class DateTimeTest : public SparkFunctionBaseTest {
 protected:
  std::string daysShort[7] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};

  std::string daysLong[7] = {
      "Monday",
      "Tuesday",
      "Wednesday",
      "Thursday",
      "Friday",
      "Saturday",
      "Sunday"};

  std::string monthsShort[12] = {
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec"};

  std::string monthsLong[12] = {
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December"};

  std::string padNumber(int number) {
    return number < 10 ? "0" + std::to_string(number) : std::to_string(number);
  }

  void setQueryTimeZone(const std::string& timeZone) {
    queryCtx_->setConfigOverridesUnsafe({
        {core::QueryConfig::kSessionTimezone, timeZone},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
    });
  }

  void disableAdjustTimestampToTimezone() {
    queryCtx_->setConfigOverridesUnsafe({
        {core::QueryConfig::kAdjustTimestampToTimezone, "false"},
    });
  }

 public:
  struct TimestampWithTimezone {
    TimestampWithTimezone(int64_t milliSeconds, int16_t timezoneId)
        : milliSeconds_(milliSeconds), timezoneId_(timezoneId) {}

    int64_t milliSeconds_{0};
    int16_t timezoneId_{0};
  };

  std::optional<TimestampWithTimezone> parseDatetime(
      const std::optional<std::string>& input,
      const std::optional<std::string>& format) {
    auto resultVector = evaluate(
        "parse_datetime(c0, c1)",
        makeRowVector(
            {makeNullableFlatVector<std::string>({input}),
             makeNullableFlatVector<std::string>({format})}));
    EXPECT_EQ(1, resultVector->size());

    if (resultVector->isNullAt(0)) {
      return std::nullopt;
    }

    auto rowVector = resultVector->as<RowVector>();
    return TimestampWithTimezone{
        rowVector->children()[0]->as<SimpleVector<int64_t>>()->valueAt(0),
        rowVector->children()[1]->as<SimpleVector<int16_t>>()->valueAt(0)};
  }

  std::optional<Timestamp> dateParse(
      const std::optional<std::string>& input,
      const std::optional<std::string>& format) {
    auto resultVector = evaluate(
        "date_parse(c0, c1)",
        makeRowVector(
            {makeNullableFlatVector<std::string>({input}),
             makeNullableFlatVector<std::string>({format})}));
    EXPECT_EQ(1, resultVector->size());

    if (resultVector->isNullAt(0)) {
      return std::nullopt;
    }
    return resultVector->as<SimpleVector<Timestamp>>()->valueAt(0);
  }

  std::optional<std::string> dateFormat(
      std::optional<Timestamp> timestamp,
      const std::string& format) {
    auto resultVector = evaluate(
        "date_format(c0, c1)",
        makeRowVector(
            {makeNullableFlatVector<Timestamp>({timestamp}),
             makeNullableFlatVector<std::string>({format})}));
    return resultVector->as<SimpleVector<StringView>>()->valueAt(0);
  }

  std::optional<std::string> formatDatetime(
      std::optional<Timestamp> timestamp,
      const std::string& format) {
    auto resultVector = evaluate(
        "format_datetime(c0, c1)",
        makeRowVector(
            {makeNullableFlatVector<Timestamp>({timestamp}),
             makeNullableFlatVector<std::string>({format})}));
    return resultVector->as<SimpleVector<StringView>>()->valueAt(0);
  }

  template <typename T>
  std::optional<T> evaluateWithTimestampWithTimezone(
      const std::string& expression,
      std::optional<int64_t> timestamp,
      const std::optional<std::string>& timeZoneName) {
    if (!timestamp.has_value() || !timeZoneName.has_value()) {
      return evaluateOnce<T>(
          expression,
          makeRowVector({makeRowVector(
              {
                  makeNullableFlatVector<int64_t>({std::nullopt}),
                  makeNullableFlatVector<int16_t>({std::nullopt}),
              },
              [](vector_size_t /*row*/) { return true; })}));
    }

    const std::optional<int64_t> tzid =
        util::getTimeZoneID(timeZoneName.value());
    return evaluateOnce<T>(
        expression,
        makeRowVector({makeRowVector({
            makeNullableFlatVector<int64_t>({timestamp}),
            makeNullableFlatVector<int16_t>({tzid}),
        })}));
  }

  VectorPtr evaluateWithTimestampWithTimezone(
      const std::string& expression,
      std::optional<int64_t> timestamp,
      const std::optional<std::string>& timeZoneName) {
    if (!timestamp.has_value() || !timeZoneName.has_value()) {
      return evaluate(
          expression,
          makeRowVector({makeRowVector(
              {
                  makeNullableFlatVector<int64_t>({std::nullopt}),
                  makeNullableFlatVector<int16_t>({std::nullopt}),
              },
              [](vector_size_t /*row*/) { return true; })}));
    }

    const std::optional<int64_t> tzid =
        util::getTimeZoneID(timeZoneName.value());
    return evaluate(
        expression,
        makeRowVector({makeRowVector({
            makeNullableFlatVector<int64_t>({timestamp}),
            makeNullableFlatVector<int16_t>({tzid}),
        })}));
  }
};

bool operator==(
    const DateTimeTest::TimestampWithTimezone& a,
    const DateTimeTest::TimestampWithTimezone& b) {
  return a.milliSeconds_ == b.milliSeconds_ && a.timezoneId_ == b.timezoneId_;
}

TEST_F(DateTimeTest, year) {
  const auto year = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int32_t>("year(c0)", date);
  };
  EXPECT_EQ(std::nullopt, year(std::nullopt));
  EXPECT_EQ(1970, year(Timestamp(0, 0)));
  EXPECT_EQ(1969, year(Timestamp(-1, 9000)));
  EXPECT_EQ(2096, year(Timestamp(4000000000, 0)));
  EXPECT_EQ(2096, year(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(2001, year(Timestamp(998474645, 321000000)));
  EXPECT_EQ(2001, year(Timestamp(998423705, 321000000)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, year(std::nullopt));
  EXPECT_EQ(1969, year(Timestamp(0, 0)));
  EXPECT_EQ(1969, year(Timestamp(-1, 12300000000)));
  EXPECT_EQ(2096, year(Timestamp(4000000000, 0)));
  EXPECT_EQ(2096, year(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(2001, year(Timestamp(998474645, 321000000)));
  EXPECT_EQ(2001, year(Timestamp(998423705, 321000000)));
}

TEST_F(DateTimeTest, yearDate) {
  const auto year = [&](std::optional<Date> date) {
    return evaluateOnce<int32_t>("year(c0)", date);
  };
  EXPECT_EQ(std::nullopt, year(std::nullopt));
  EXPECT_EQ(1970, year(Date(0)));
  EXPECT_EQ(1969, year(Date(-1)));
  EXPECT_EQ(2020, year(Date(18262)));
  EXPECT_EQ(1920, year(Date(-18262)));
}

TEST_F(DateTimeTest, yearTimestampWithTimezone) {
  EXPECT_EQ(
      1969,
      evaluateWithTimestampWithTimezone<int32_t>("year(c0)", 0, "-01:00"));
  EXPECT_EQ(
      1970,
      evaluateWithTimestampWithTimezone<int32_t>("year(c0)", 0, "+00:00"));
  EXPECT_EQ(
      1973,
      evaluateWithTimestampWithTimezone<int32_t>(
          "year(c0)", 123456789000, "+14:00"));
  EXPECT_EQ(
      1966,
      evaluateWithTimestampWithTimezone<int32_t>(
          "year(c0)", -123456789000, "+03:00"));
  EXPECT_EQ(
      2001,
      evaluateWithTimestampWithTimezone<int32_t>(
          "year(c0)", 987654321000, "-07:00"));
  EXPECT_EQ(
      1938,
      evaluateWithTimestampWithTimezone<int32_t>(
          "year(c0)", -987654321000, "-13:00"));
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "year(c0)", std::nullopt, std::nullopt));
}

TEST_F(DateTimeTest, quarter) {
  const auto quarter = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int32_t>("quarter(c0)", date);
  };
  EXPECT_EQ(std::nullopt, quarter(std::nullopt));
  EXPECT_EQ(1, quarter(Timestamp(0, 0)));
  EXPECT_EQ(4, quarter(Timestamp(-1, 9000)));
  EXPECT_EQ(4, quarter(Timestamp(4000000000, 0)));
  EXPECT_EQ(4, quarter(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(2, quarter(Timestamp(990000000, 321000000)));
  EXPECT_EQ(3, quarter(Timestamp(998423705, 321000000)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, quarter(std::nullopt));
  EXPECT_EQ(4, quarter(Timestamp(0, 0)));
  EXPECT_EQ(4, quarter(Timestamp(-1, 12300000000)));
  EXPECT_EQ(4, quarter(Timestamp(4000000000, 0)));
  EXPECT_EQ(4, quarter(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(2, quarter(Timestamp(990000000, 321000000)));
  EXPECT_EQ(3, quarter(Timestamp(998423705, 321000000)));
}

TEST_F(DateTimeTest, quarterDate) {
  const auto quarter = [&](std::optional<Date> date) {
    return evaluateOnce<int32_t>("quarter(c0)", date);
  };
  EXPECT_EQ(std::nullopt, quarter(std::nullopt));
  EXPECT_EQ(1, quarter(Date(0)));
  EXPECT_EQ(4, quarter(Date(-1)));
  EXPECT_EQ(4, quarter(Date(-40)));
  EXPECT_EQ(2, quarter(Date(110)));
  EXPECT_EQ(3, quarter(Date(200)));
  EXPECT_EQ(1, quarter(Date(18262)));
  EXPECT_EQ(1, quarter(Date(-18262)));
}

TEST_F(DateTimeTest, quarterTimestampWithTimezone) {
  EXPECT_EQ(
      4,
      evaluateWithTimestampWithTimezone<int32_t>("quarter(c0)", 0, "-01:00"));
  EXPECT_EQ(
      1,
      evaluateWithTimestampWithTimezone<int32_t>("quarter(c0)", 0, "+00:00"));
  EXPECT_EQ(
      4,
      evaluateWithTimestampWithTimezone<int32_t>(
          "quarter(c0)", 123456789000, "+14:00"));
  EXPECT_EQ(
      1,
      evaluateWithTimestampWithTimezone<int32_t>(
          "quarter(c0)", -123456789000, "+03:00"));
  EXPECT_EQ(
      2,
      evaluateWithTimestampWithTimezone<int32_t>(
          "quarter(c0)", 987654321000, "-07:00"));
  EXPECT_EQ(
      3,
      evaluateWithTimestampWithTimezone<int32_t>(
          "quarter(c0)", -987654321000, "-13:00"));
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "quarter(c0)", std::nullopt, std::nullopt));
}

TEST_F(DateTimeTest, month) {
  const auto month = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int32_t>("month(c0)", date);
  };
  EXPECT_EQ(std::nullopt, month(std::nullopt));
  EXPECT_EQ(1, month(Timestamp(0, 0)));
  EXPECT_EQ(12, month(Timestamp(-1, 9000)));
  EXPECT_EQ(10, month(Timestamp(4000000000, 0)));
  EXPECT_EQ(10, month(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(8, month(Timestamp(998474645, 321000000)));
  EXPECT_EQ(8, month(Timestamp(998423705, 321000000)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, month(std::nullopt));
  EXPECT_EQ(12, month(Timestamp(0, 0)));
  EXPECT_EQ(12, month(Timestamp(-1, 12300000000)));
  EXPECT_EQ(10, month(Timestamp(4000000000, 0)));
  EXPECT_EQ(10, month(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(8, month(Timestamp(998474645, 321000000)));
  EXPECT_EQ(8, month(Timestamp(998423705, 321000000)));
}

TEST_F(DateTimeTest, monthDate) {
  const auto month = [&](std::optional<Date> date) {
    return evaluateOnce<int32_t>("month(c0)", date);
  };
  EXPECT_EQ(std::nullopt, month(std::nullopt));
  EXPECT_EQ(1, month(Date(0)));
  EXPECT_EQ(12, month(Date(-1)));
  EXPECT_EQ(11, month(Date(-40)));
  EXPECT_EQ(2, month(Date(40)));
  EXPECT_EQ(1, month(Date(18262)));
  EXPECT_EQ(1, month(Date(-18262)));
}

TEST_F(DateTimeTest, monthTimestampWithTimezone) {
  EXPECT_EQ(
      12, evaluateWithTimestampWithTimezone<int32_t>("month(c0)", 0, "-01:00"));
  EXPECT_EQ(
      1, evaluateWithTimestampWithTimezone<int32_t>("month(c0)", 0, "+00:00"));
  EXPECT_EQ(
      11,
      evaluateWithTimestampWithTimezone<int32_t>(
          "month(c0)", 123456789000, "+14:00"));
  EXPECT_EQ(
      2,
      evaluateWithTimestampWithTimezone<int32_t>(
          "month(c0)", -123456789000, "+03:00"));
  EXPECT_EQ(
      4,
      evaluateWithTimestampWithTimezone<int32_t>(
          "month(c0)", 987654321000, "-07:00"));
  EXPECT_EQ(
      9,
      evaluateWithTimestampWithTimezone<int32_t>(
          "month(c0)", -987654321000, "-13:00"));
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "month(c0)", std::nullopt, std::nullopt));
}

TEST_F(DateTimeTest, hour) {
  const auto hour = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int32_t>("hour(c0)", date);
  };
  EXPECT_EQ(std::nullopt, hour(std::nullopt));
  EXPECT_EQ(0, hour(Timestamp(0, 0)));
  EXPECT_EQ(23, hour(Timestamp(-1, 9000)));
  EXPECT_EQ(7, hour(Timestamp(4000000000, 0)));
  EXPECT_EQ(7, hour(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(10, hour(Timestamp(998474645, 321000000)));
  EXPECT_EQ(19, hour(Timestamp(998423705, 321000000)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, hour(std::nullopt));
  EXPECT_EQ(13, hour(Timestamp(0, 0)));
  EXPECT_EQ(12, hour(Timestamp(-1, 12300000000)));
  // Disabled for now because the TZ for Pacific/Apia in 2096 varies between
  // systems.
  // EXPECT_EQ(21, hour(Timestamp(4000000000, 0)));
  // EXPECT_EQ(21, hour(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(23, hour(Timestamp(998474645, 321000000)));
  EXPECT_EQ(8, hour(Timestamp(998423705, 321000000)));
}

TEST_F(DateTimeTest, hourTimestampWithTimezone) {
  EXPECT_EQ(
      20,
      evaluateWithTimestampWithTimezone<int32_t>(
          "hour(c0)", 998423705000, "+01:00"));
  EXPECT_EQ(
      12,
      evaluateWithTimestampWithTimezone<int32_t>(
          "hour(c0)", 41028000, "+01:00"));
  EXPECT_EQ(
      13,
      evaluateWithTimestampWithTimezone<int32_t>(
          "hour(c0)", 41028000, "+02:00"));
  EXPECT_EQ(
      14,
      evaluateWithTimestampWithTimezone<int32_t>(
          "hour(c0)", 41028000, "+03:00"));
  EXPECT_EQ(
      8,
      evaluateWithTimestampWithTimezone<int32_t>(
          "hour(c0)", 41028000, "-03:00"));
  EXPECT_EQ(
      1,
      evaluateWithTimestampWithTimezone<int32_t>(
          "hour(c0)", 41028000, "+14:00"));
  EXPECT_EQ(
      9,
      evaluateWithTimestampWithTimezone<int32_t>(
          "hour(c0)", -100000, "-14:00"));
  EXPECT_EQ(
      2,
      evaluateWithTimestampWithTimezone<int32_t>(
          "hour(c0)", -41028000, "+14:00"));
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "hour(c0)", std::nullopt, std::nullopt));
}

TEST_F(DateTimeTest, hourDate) {
  const auto hour = [&](std::optional<Date> date) {
    return evaluateOnce<int32_t>("hour(c0)", date);
  };
  EXPECT_EQ(std::nullopt, hour(std::nullopt));
  EXPECT_EQ(0, hour(Date(0)));
  EXPECT_EQ(0, hour(Date(-1)));
  EXPECT_EQ(0, hour(Date(-40)));
  EXPECT_EQ(0, hour(Date(40)));
  EXPECT_EQ(0, hour(Date(18262)));
  EXPECT_EQ(0, hour(Date(-18262)));
}

TEST_F(DateTimeTest, dayOfMonth) {
  const auto day = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int32_t>("day_of_month(c0)", date);
  };
  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(1, day(Timestamp(0, 0)));
  EXPECT_EQ(31, day(Timestamp(-1, 9000)));
  EXPECT_EQ(30, day(Timestamp(1632989700, 0)));
  EXPECT_EQ(1, day(Timestamp(1633076100, 0)));
  EXPECT_EQ(6, day(Timestamp(1633508100, 0)));
  EXPECT_EQ(31, day(Timestamp(1635668100, 0)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(31, day(Timestamp(0, 0)));
  EXPECT_EQ(31, day(Timestamp(-1, 9000)));
  EXPECT_EQ(30, day(Timestamp(1632989700, 0)));
  EXPECT_EQ(1, day(Timestamp(1633076100, 0)));
  EXPECT_EQ(6, day(Timestamp(1633508100, 0)));
  EXPECT_EQ(31, day(Timestamp(1635668100, 0)));
}

TEST_F(DateTimeTest, dayOfMonthDate) {
  const auto day = [&](std::optional<Date> date) {
    return evaluateOnce<int32_t>("day_of_month(c0)", date);
  };
  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(1, day(Date(0)));
  EXPECT_EQ(31, day(Date(-1)));
  EXPECT_EQ(22, day(Date(-40)));
  EXPECT_EQ(10, day(Date(40)));
  EXPECT_EQ(1, day(Date(18262)));
  EXPECT_EQ(2, day(Date(-18262)));
}

TEST_F(DateTimeTest, dayOfMonthTimestampWithTimezone) {
  EXPECT_EQ(
      31,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_month(c0)", 0, "-01:00"));
  EXPECT_EQ(
      1,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_month(c0)", 0, "+00:00"));
  EXPECT_EQ(
      30,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_month(c0)", 123456789000, "+14:00"));
  EXPECT_EQ(
      2,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_month(c0)", -123456789000, "+03:00"));
  EXPECT_EQ(
      18,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_month(c0)", 987654321000, "-07:00"));
  EXPECT_EQ(
      14,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_month(c0)", -987654321000, "-13:00"));
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_month(c0)", std::nullopt, std::nullopt));
}

TEST_F(DateTimeTest, dayOfWeek) {
  const auto day = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int32_t>("day_of_week(c0)", date);
  };
  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(5, day(Timestamp(0, 0)));
  EXPECT_EQ(4, day(Timestamp(-1, 9000)));
  EXPECT_EQ(2, day(Timestamp(1633940100, 0)));
  EXPECT_EQ(3, day(Timestamp(1634026500, 0)));
  EXPECT_EQ(4, day(Timestamp(1634112900, 0)));
  EXPECT_EQ(5, day(Timestamp(1634199300, 0)));
  EXPECT_EQ(6, day(Timestamp(1634285700, 0)));
  EXPECT_EQ(7, day(Timestamp(1634372100, 0)));
  EXPECT_EQ(1, day(Timestamp(1633853700, 0)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(4, day(Timestamp(0, 0)));
  EXPECT_EQ(4, day(Timestamp(-1, 9000)));
  EXPECT_EQ(2, day(Timestamp(1633940100, 0)));
  EXPECT_EQ(3, day(Timestamp(1634026500, 0)));
  EXPECT_EQ(4, day(Timestamp(1634112900, 0)));
  EXPECT_EQ(5, day(Timestamp(1634199300, 0)));
  EXPECT_EQ(6, day(Timestamp(1634285700, 0)));
  EXPECT_EQ(7, day(Timestamp(1634372100, 0)));
  EXPECT_EQ(1, day(Timestamp(1633853700, 0)));
}

TEST_F(DateTimeTest, dayOfWeekDate) {
  const auto day = [&](std::optional<Date> date) {
    return evaluateOnce<int32_t>("day_of_week(c0)", date);
  };
  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(5, day(Date(0)));
  EXPECT_EQ(4, day(Date(-1)));
  EXPECT_EQ(7, day(Date(-40)));
  EXPECT_EQ(3, day(Date(40)));
  EXPECT_EQ(4, day(Date(18262)));
  EXPECT_EQ(6, day(Date(-18262)));
}

TEST_F(DateTimeTest, dayOfWeekTimestampWithTimezone) {
  EXPECT_EQ(
      4,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_week(c0)", 0, "-01:00"));
  EXPECT_EQ(
      5,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_week(c0)", 0, "+00:00"));
  EXPECT_EQ(
      6,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_week(c0)", 123456789000, "+14:00"));
  EXPECT_EQ(
      4,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_week(c0)", -123456789000, "+03:00"));
  EXPECT_EQ(
      4,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_week(c0)", 987654321000, "-07:00"));
  EXPECT_EQ(
      4,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_week(c0)", -987654321000, "-13:00"));
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_week(c0)", std::nullopt, std::nullopt));
}

TEST_F(DateTimeTest, dayOfYear) {
  const auto day = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int32_t>("day_of_year(c0)", date);
  };
  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(1, day(Timestamp(0, 0)));
  EXPECT_EQ(365, day(Timestamp(-1, 9000)));
  EXPECT_EQ(273, day(Timestamp(1632989700, 0)));
  EXPECT_EQ(274, day(Timestamp(1633076100, 0)));
  EXPECT_EQ(279, day(Timestamp(1633508100, 0)));
  EXPECT_EQ(304, day(Timestamp(1635668100, 0)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(365, day(Timestamp(0, 0)));
  EXPECT_EQ(365, day(Timestamp(-1, 9000)));
  EXPECT_EQ(273, day(Timestamp(1632989700, 0)));
  EXPECT_EQ(274, day(Timestamp(1633076100, 0)));
  EXPECT_EQ(279, day(Timestamp(1633508100, 0)));
  EXPECT_EQ(304, day(Timestamp(1635668100, 0)));
}

TEST_F(DateTimeTest, dayOfYearDate) {
  const auto day = [&](std::optional<Date> date) {
    return evaluateOnce<int32_t>("day_of_year(c0)", date);
  };
  EXPECT_EQ(std::nullopt, day(std::nullopt));
  EXPECT_EQ(1, day(Date(0)));
  EXPECT_EQ(365, day(Date(-1)));
  EXPECT_EQ(326, day(Date(-40)));
  EXPECT_EQ(41, day(Date(40)));
  EXPECT_EQ(1, day(Date(18262)));
  EXPECT_EQ(2, day(Date(-18262)));
}

TEST_F(DateTimeTest, dayOfYearTimestampWithTimezone) {
  EXPECT_EQ(
      365,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_year(c0)", 0, "-01:00"));
  EXPECT_EQ(
      1,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_year(c0)", 0, "+00:00"));
  EXPECT_EQ(
      334,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_year(c0)", 123456789000, "+14:00"));
  EXPECT_EQ(
      33,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_year(c0)", -123456789000, "+03:00"));
  EXPECT_EQ(
      108,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_year(c0)", 987654321000, "-07:00"));
  EXPECT_EQ(
      257,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_year(c0)", -987654321000, "-13:00"));
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "day_of_year(c0)", std::nullopt, std::nullopt));
}

TEST_F(DateTimeTest, yearOfWeek) {
  const auto yow = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int32_t>("year_of_week(c0)", date);
  };
  EXPECT_EQ(std::nullopt, yow(std::nullopt));
  EXPECT_EQ(1970, yow(Timestamp(0, 0)));
  EXPECT_EQ(1970, yow(Timestamp(-1, 0)));
  EXPECT_EQ(1969, yow(Timestamp(-345600, 0)));
  EXPECT_EQ(1970, yow(Timestamp(-259200, 0)));
  EXPECT_EQ(1970, yow(Timestamp(31536000, 0)));
  EXPECT_EQ(1970, yow(Timestamp(31708800, 0)));
  EXPECT_EQ(1971, yow(Timestamp(31795200, 0)));
  EXPECT_EQ(2021, yow(Timestamp(1632989700, 0)));

  setQueryTimeZone("Pacific/Apia");

  EXPECT_EQ(std::nullopt, yow(std::nullopt));
  EXPECT_EQ(1970, yow(Timestamp(0, 0)));
  EXPECT_EQ(1970, yow(Timestamp(-1, 0)));
  EXPECT_EQ(1969, yow(Timestamp(-345600, 0)));
  EXPECT_EQ(1969, yow(Timestamp(-259200, 0)));
  EXPECT_EQ(1970, yow(Timestamp(31536000, 0)));
  EXPECT_EQ(1970, yow(Timestamp(31708800, 0)));
  EXPECT_EQ(1970, yow(Timestamp(31795200, 0)));
  EXPECT_EQ(2021, yow(Timestamp(1632989700, 0)));
}

TEST_F(DateTimeTest, yearOfWeekDate) {
  const auto yow = [&](std::optional<Date> date) {
    return evaluateOnce<int32_t>("year_of_week(c0)", date);
  };
  EXPECT_EQ(std::nullopt, yow(std::nullopt));
  EXPECT_EQ(1970, yow(Date(0)));
  EXPECT_EQ(1970, yow(Date(-1)));
  EXPECT_EQ(1969, yow(Date(-4)));
  EXPECT_EQ(1970, yow(Date(-3)));
  EXPECT_EQ(1970, yow(Date(365)));
  EXPECT_EQ(1970, yow(Date(367)));
  EXPECT_EQ(1971, yow(Date(368)));
  EXPECT_EQ(2021, yow(Date(18900)));
}

TEST_F(DateTimeTest, yearOfWeekTimestampWithTimezone) {
  EXPECT_EQ(
      1970,
      evaluateWithTimestampWithTimezone<int32_t>(
          "year_of_week(c0)", 0, "-01:00"));
  EXPECT_EQ(
      1970,
      evaluateWithTimestampWithTimezone<int32_t>(
          "year_of_week(c0)", 0, "+00:00"));
  EXPECT_EQ(
      1973,
      evaluateWithTimestampWithTimezone<int32_t>(
          "year_of_week(c0)", 123456789000, "+14:00"));
  EXPECT_EQ(
      1966,
      evaluateWithTimestampWithTimezone<int32_t>(
          "year_of_week(c0)", -123456789000, "+03:00"));
  EXPECT_EQ(
      2001,
      evaluateWithTimestampWithTimezone<int32_t>(
          "year_of_week(c0)", 987654321000, "-07:00"));
  EXPECT_EQ(
      1938,
      evaluateWithTimestampWithTimezone<int32_t>(
          "year_of_week(c0)", -987654321000, "-13:00"));
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "year_of_week(c0)", std::nullopt, std::nullopt));
}

TEST_F(DateTimeTest, minute) {
  const auto minute = [&](std::optional<Timestamp> date) {
    return evaluateOnce<int32_t>("minute(c0)", date);
  };
  EXPECT_EQ(std::nullopt, minute(std::nullopt));
  EXPECT_EQ(0, minute(Timestamp(0, 0)));
  EXPECT_EQ(59, minute(Timestamp(-1, 9000)));
  EXPECT_EQ(6, minute(Timestamp(4000000000, 0)));
  EXPECT_EQ(6, minute(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(4, minute(Timestamp(998474645, 321000000)));
  EXPECT_EQ(55, minute(Timestamp(998423705, 321000000)));

  setQueryTimeZone("Asia/Kolkata");

  EXPECT_EQ(std::nullopt, minute(std::nullopt));
  EXPECT_EQ(30, minute(Timestamp(0, 0)));
  EXPECT_EQ(29, minute(Timestamp(-1, 9000)));
  EXPECT_EQ(36, minute(Timestamp(4000000000, 0)));
  EXPECT_EQ(36, minute(Timestamp(4000000000, 123000000)));
  EXPECT_EQ(34, minute(Timestamp(998474645, 321000000)));
  EXPECT_EQ(25, minute(Timestamp(998423705, 321000000)));
}

TEST_F(DateTimeTest, minuteDate) {
  const auto minute = [&](std::optional<Date> date) {
    return evaluateOnce<int32_t>("minute(c0)", date);
  };
  EXPECT_EQ(std::nullopt, minute(std::nullopt));
  EXPECT_EQ(0, minute(Date(0)));
  EXPECT_EQ(0, minute(Date(-1)));
  EXPECT_EQ(0, minute(Date(-40)));
  EXPECT_EQ(0, minute(Date(40)));
  EXPECT_EQ(0, minute(Date(18262)));
  EXPECT_EQ(0, minute(Date(-18262)));
}

TEST_F(DateTimeTest, minuteTimestampWithTimezone) {
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "minute(c0)", std::nullopt, std::nullopt));
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "minute(c0)", std::nullopt, "Asia/Kolkata"));
  EXPECT_EQ(
      0, evaluateWithTimestampWithTimezone<int32_t>("minute(c0)", 0, "+00:00"));
  EXPECT_EQ(
      30,
      evaluateWithTimestampWithTimezone<int32_t>("minute(c0)", 0, "+05:30"));
  EXPECT_EQ(
      6,
      evaluateWithTimestampWithTimezone<int32_t>(
          "minute(c0)", 4000000000000, "+00:00"));
  EXPECT_EQ(
      36,
      evaluateWithTimestampWithTimezone<int32_t>(
          "minute(c0)", 4000000000000, "+05:30"));
  EXPECT_EQ(
      4,
      evaluateWithTimestampWithTimezone<int32_t>(
          "minute(c0)", 998474645000, "+00:00"));
  EXPECT_EQ(
      34,
      evaluateWithTimestampWithTimezone<int32_t>(
          "minute(c0)", 998474645000, "+05:30"));
  EXPECT_EQ(
      59,
      evaluateWithTimestampWithTimezone<int32_t>(
          "minute(c0)", -1000, "+00:00"));
  EXPECT_EQ(
      29,
      evaluateWithTimestampWithTimezone<int32_t>(
          "minute(c0)", -1000, "+05:30"));
}

TEST_F(DateTimeTest, second) {
  const auto second = [&](std::optional<Timestamp> timestamp) {
    return evaluateOnce<int32_t>("second(c0)", timestamp);
  };
  EXPECT_EQ(std::nullopt, second(std::nullopt));
  EXPECT_EQ(0, second(Timestamp(0, 0)));
  EXPECT_EQ(40, second(Timestamp(4000000000, 0)));
  EXPECT_EQ(59, second(Timestamp(-1, 123000000)));
  EXPECT_EQ(59, second(Timestamp(-1, 12300000000)));
}

TEST_F(DateTimeTest, secondDate) {
  const auto second = [&](std::optional<Date> date) {
    return evaluateOnce<int32_t>("second(c0)", date);
  };
  EXPECT_EQ(std::nullopt, second(std::nullopt));
  EXPECT_EQ(0, second(Date(0)));
  EXPECT_EQ(0, second(Date(-1)));
  EXPECT_EQ(0, second(Date(-40)));
  EXPECT_EQ(0, second(Date(40)));
  EXPECT_EQ(0, second(Date(18262)));
  EXPECT_EQ(0, second(Date(-18262)));
}

TEST_F(DateTimeTest, secondTimestampWithTimezone) {
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "second(c0)", std::nullopt, std::nullopt));
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "second(c0)", std::nullopt, "+05:30"));
  EXPECT_EQ(
      0, evaluateWithTimestampWithTimezone<int32_t>("second(c0)", 0, "+00:00"));
  EXPECT_EQ(
      0, evaluateWithTimestampWithTimezone<int32_t>("second(c0)", 0, "+05:30"));
  EXPECT_EQ(
      40,
      evaluateWithTimestampWithTimezone<int32_t>(
          "second(c0)", 4000000000000, "+00:00"));
  EXPECT_EQ(
      40,
      evaluateWithTimestampWithTimezone<int32_t>(
          "second(c0)", 4000000000000, "+05:30"));
  EXPECT_EQ(
      59,
      evaluateWithTimestampWithTimezone<int32_t>(
          "second(c0)", -1000, "+00:00"));
  EXPECT_EQ(
      59,
      evaluateWithTimestampWithTimezone<int32_t>(
          "second(c0)", -1000, "+05:30"));
}

TEST_F(DateTimeTest, millisecond) {
  const auto millisecond = [&](std::optional<Timestamp> timestamp) {
    return evaluateOnce<int32_t>("millisecond(c0)", timestamp);
  };
  EXPECT_EQ(std::nullopt, millisecond(std::nullopt));
  EXPECT_EQ(0, millisecond(Timestamp(0, 0)));
  EXPECT_EQ(0, millisecond(Timestamp(4000000000, 0)));
  EXPECT_EQ(123, millisecond(Timestamp(-1, 123000000)));
  EXPECT_EQ(12300, millisecond(Timestamp(-1, 12300000000)));
}

TEST_F(DateTimeTest, millisecondDate) {
  const auto millisecond = [&](std::optional<Date> date) {
    return evaluateOnce<int32_t>("millisecond(c0)", date);
  };
  EXPECT_EQ(std::nullopt, millisecond(std::nullopt));
  EXPECT_EQ(0, millisecond(Date(0)));
  EXPECT_EQ(0, millisecond(Date(-1)));
  EXPECT_EQ(0, millisecond(Date(-40)));
  EXPECT_EQ(0, millisecond(Date(40)));
  EXPECT_EQ(0, millisecond(Date(18262)));
  EXPECT_EQ(0, millisecond(Date(-18262)));
}

TEST_F(DateTimeTest, millisecondTimestampWithTimezone) {
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "millisecond(c0)", std::nullopt, std::nullopt));
  EXPECT_EQ(
      std::nullopt,
      evaluateWithTimestampWithTimezone<int32_t>(
          "millisecond(c0)", std::nullopt, "+05:30"));
  EXPECT_EQ(
      0,
      evaluateWithTimestampWithTimezone<int32_t>(
          "millisecond(c0)", 0, "+00:00"));
  EXPECT_EQ(
      0,
      evaluateWithTimestampWithTimezone<int32_t>(
          "millisecond(c0)", 0, "+05:30"));
  EXPECT_EQ(
      123,
      evaluateWithTimestampWithTimezone<int32_t>(
          "millisecond(c0)", 4000000000123, "+00:00"));
  EXPECT_EQ(
      123,
      evaluateWithTimestampWithTimezone<int32_t>(
          "millisecond(c0)", 4000000000123, "+05:30"));
  EXPECT_EQ(
      20,
      evaluateWithTimestampWithTimezone<int32_t>(
          "millisecond(c0)", -980, "+00:00"));
  EXPECT_EQ(
      20,
      evaluateWithTimestampWithTimezone<int32_t>(
          "millisecond(c0)", -980, "+05:30"));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
