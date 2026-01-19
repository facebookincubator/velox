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

#include "velox/core/QueryConfig.h"
#include "velox/functions/prestosql/tests/CastBaseTest.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
namespace facebook::velox::test {
namespace {

class SparkCastExprTest : public functions::test::CastBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::sparksql::registerFunctions("");
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void setAnsiSupport(bool value) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kSparkAnsiEnabled, std::to_string(value)}});
  }

  template <typename T>
  void testDecimalToIntegralCasts() {
    auto shortFlat = makeNullableFlatVector<int64_t>(
        {-300,
         -260,
         -230,
         -200,
         -100,
         0,
         5500,
         5749,
         5755,
         6900,
         7200,
         std::nullopt},
        DECIMAL(6, 2));
    testCast(
        shortFlat,
        makeNullableFlatVector<T>(
            {-3,
             -2 /*-2.6 truncated to -2*/,
             -2 /*-2.3 truncated to -2*/,
             -2,
             -1,
             0,
             55,
             57 /*57.49 truncated to 57*/,
             57 /*57.55 truncated to 57*/,
             69,
             72,
             std::nullopt}));
    auto longFlat = makeNullableFlatVector<int128_t>(
        {-30'000'000'000,
         -25'500'000'000,
         -24'500'000'000,
         -20'000'000'000,
         -10'000'000'000,
         0,
         550'000'000'000,
         554'900'000'000,
         559'900'000'000,
         690'000'000'000,
         720'000'000'000,
         std::nullopt},
        DECIMAL(20, 10));
    testCast(
        longFlat,
        makeNullableFlatVector<T>(
            {-3,
             -2 /*-2.55 truncated to -2*/,
             -2 /*-2.45 truncated to -2*/,
             -2,
             -1,
             0,
             55,
             55 /* 55.49 truncated to 55*/,
             55 /* 55.99 truncated to 55*/,
             69,
             72,
             std::nullopt}));
  }

  template <typename T>
  void testIntegralToTimestampCast() {
    testCast(
        makeNullableFlatVector<T>({
            0,
            1,
            std::numeric_limits<T>::max(),
            std::numeric_limits<T>::min(),
            std::nullopt,
        }),
        makeNullableFlatVector<Timestamp>(
            {Timestamp(0, 0),
             Timestamp(1, 0),
             Timestamp(std::numeric_limits<T>::max(), 0),
             Timestamp(std::numeric_limits<T>::min(), 0),
             std::nullopt}));
  }

  template <typename T>
  void testTimestampToIntegralCast() {
    testCast(
        makeFlatVector<Timestamp>({
            Timestamp(0, 0),
            Timestamp(1, 0),
            Timestamp(std::numeric_limits<T>::max(), 0),
            Timestamp(std::numeric_limits<T>::min(), 0),
        }),
        makeFlatVector<T>({
            0,
            1,
            std::numeric_limits<T>::max(),
            std::numeric_limits<T>::min(),
        }));
  }

  template <typename T>
  void testTimestampToIntegralCastOverflow(std::vector<T> expected) {
    testCast(
        makeFlatVector<Timestamp>({
            Timestamp(1740470426, 0),
            Timestamp(2147483647, 0),
            Timestamp(9223372036854, 775'807'000),
            Timestamp(-9223372036855, 224'192'000),
        }),
        makeFlatVector<T>(expected));
  }
  void testStringToDate() {
    testCast<std::string, int32_t>(
        "date",
        {"1970-01-01",
         "2020-01-01",
         "2135-11-09",
         "1969-12-27",
         "1812-04-15",
         "1920-01-02",
         "12345-12-18",
         "1970-1-2",
         "1970-01-2",
         "1970-1-02",
         "+1970-01-02",
         " 1970-01-01",
         std::nullopt},
        {0,
         18262,
         60577,
         -5,
         -57604,
         -18262,
         3789742,
         1,
         1,
         1,
         1,
         0,
         std::nullopt},
        VARCHAR(),
        DATE());
    testCast<std::string, int32_t>(
        "date",
        {"12345",
         "2015",
         "2015-03",
         "2015-03-18T",
         "2015-03-18T123123",
         "2015-03-18 123142",
         "2015-03-18 (BC)"},
        {3789391, 16436, 16495, 16512, 16512, 16512, 16512},
        VARCHAR(),
        DATE());
  }

  void testDecimalToIntegral() {
    testDecimalToIntegralCasts<int64_t>();
    testDecimalToIntegralCasts<int32_t>();
    testDecimalToIntegralCasts<int16_t>();
    testDecimalToIntegralCasts<int8_t>();
  }

  void testInvalidDate() {
    testInvalidCast<int8_t>(
        "date", {12}, "Cast from TINYINT to DATE is not supported", TINYINT());
    testInvalidCast<int16_t>(
        "date",
        {1234},
        "Cast from SMALLINT to DATE is not supported",
        SMALLINT());
    testInvalidCast<int32_t>(
        "date",
        {1234},
        "Cast from INTEGER to DATE is not supported",
        INTEGER());
    testInvalidCast<int64_t>(
        "date", {1234}, "Cast from BIGINT to DATE is not supported", BIGINT());

    testInvalidCast<float>(
        "date", {12.99}, "Cast from REAL to DATE is not supported", REAL());
    testInvalidCast<double>(
        "date", {12.99}, "Cast from DOUBLE to DATE is not supported", DOUBLE());
  }

  void testStringToTimestamp() {
    std::vector<std::optional<std::string>> input{
        "1970-01-01",
        "1970-01-01 00:00:00-02:00",
        "1970-01-01 00:00:00 +02:00",
        "2000-01-01",
        "1970-01-01 00:00:00",
        "2000-01-01 12:21:56",
        std::nullopt,
        "2015-03-18T12:03:17",
        "2015-03-18T12:03:17Z",
        "2015-03-18 12:03:17",
        "2015-03-18T12:03:17",
        "2015-03-18 12:03:17.123",
        "2015-03-18T12:03:17.123",
        "2015-03-18T12:03:17.456",
        "2015-03-18 12:03:17.456",
    };
    std::vector<std::optional<Timestamp>> expected{
        Timestamp(0, 0),
        Timestamp(7200, 0),
        Timestamp(-7200, 0),
        Timestamp(946684800, 0),
        Timestamp(0, 0),
        Timestamp(946729316, 0),
        std::nullopt,
        Timestamp(1426680197, 0),
        Timestamp(1426680197, 0),
        Timestamp(1426680197, 0),
        Timestamp(1426680197, 0),
        Timestamp(1426680197, 123000000),
        Timestamp(1426680197, 123000000),
        Timestamp(1426680197, 456000000),
        Timestamp(1426680197, 456000000),
    };
    testCast<std::string, Timestamp>("timestamp", input, expected);

    setTimezone("Asia/Shanghai");
    testCast<std::string, Timestamp>(
        "timestamp",
        {"1970-01-01 00:00:00",
         "1970-01-01 08:00:00",
         "1970-01-01 08:00:59",
         "1970"},
        {Timestamp(-8 * 3600, 0),
         Timestamp(0, 0),
         Timestamp(59, 0),
         Timestamp(-8 * 3600, 0)});
  }

  void testIntToTimestamp() {
    // Cast bigint as timestamp.
    testCast(
        makeNullableFlatVector<int64_t>({
            0,
            1727181032,
            -1727181032,
            9223372036855,
            -9223372036856,
            std::numeric_limits<int64_t>::max(),
            std::numeric_limits<int64_t>::min(),
        }),
        makeNullableFlatVector<Timestamp>({
            Timestamp(0, 0),
            Timestamp(1727181032, 0),
            Timestamp(-1727181032, 0),
            Timestamp(9223372036854, 775'807'000),
            Timestamp(-9223372036855, 224'192'000),
            Timestamp(9223372036854, 775'807'000),
            Timestamp(-9223372036855, 224'192'000),
        }));

    // Cast tinyint/smallint/integer as timestamp.
    testIntegralToTimestampCast<int8_t>();
    testIntegralToTimestampCast<int16_t>();
    testIntegralToTimestampCast<int32_t>();
  }

  void testTimestampToInt() {
    // Cast timestamp as bigint.
    testCast(
        makeFlatVector<Timestamp>({
            Timestamp(0, 0),
            Timestamp(1, 0),
            Timestamp(10, 0),
            Timestamp(-1, 0),
            Timestamp(-10, 0),
            Timestamp(-1, 500000),
            Timestamp(-2, 999999),
            Timestamp(-10, 999999),
            Timestamp(1, 999999),
            Timestamp(-1, 1),
            Timestamp(1234567, 500000),
            Timestamp(-9876543, 1234),
            Timestamp(1727181032, 0),
            Timestamp(-1727181032, 0),
            Timestamp(9223372036854, 775'807'000),
            Timestamp(-9223372036855, 224'192'000),
        }),
        makeNullableFlatVector<int64_t>({
            0,
            1,
            10,
            -1,
            -10,
            -1,
            -2,
            -10,
            1,
            -1,
            1234567,
            -9876543,
            1727181032,
            -1727181032,
            9223372036854,
            -9223372036855,
        }));

    // Cast timestamp as tinyint/smallint/integer.
    testTimestampToIntegralCast<int8_t>();
    testTimestampToIntegralCast<int16_t>();
    testTimestampToIntegralCast<int32_t>();
  }

  void testDoubleToTimestamp() {
    testCast(
        makeFlatVector<double>({
            0.0,
            1727181032.0,
            -1727181032.0,
            9223372036855.999,
            -9223372036856.999,
            1.79769e+308,
            std::numeric_limits<double>::max(),
            -std::numeric_limits<double>::max(),
            std::numeric_limits<double>::min(),
        }),
        makeNullableFlatVector<Timestamp>({
            Timestamp(0, 0),
            Timestamp(1727181032, 0),
            Timestamp(-1727181032, 0),
            Timestamp(9223372036854, 775'807'000),
            Timestamp(-9223372036855, 224'192'000),
            Timestamp(9223372036854, 775'807'000),
            Timestamp(9223372036854, 775'807'000),
            Timestamp(-9223372036855, 224'192'000),
            Timestamp(0, 0),
        }));
  }

  void testFloatToTimestamp() {
    testCast(
        makeFlatVector<float>({
            0.0,
            1727181032.0,
            -1727181032.0,
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::min(),
        }),
        makeNullableFlatVector<Timestamp>({
            Timestamp(0, 0),
            Timestamp(1727181056, 0),
            Timestamp(-1727181056, 0),
            Timestamp(9223372036854, 775'807'000),
            Timestamp(0, 0),
        }));
  }

  void testPrimitiveValidCornerCases() {
    // To integer.
    {
      // Valid strings.
      testCast<std::string, int8_t>("tinyint", {"+1"}, {1});
      testCast<std::string, int8_t>("tinyint", {"-1"}, {-1});

      testCast<double, int8_t>("tinyint", {127.1}, {127});

      testCast<double, int64_t>("bigint", {12345.12}, {12345});
      testCast<double, int64_t>("bigint", {12345.67}, {12345});
    }
    

    // To floating-point.
    {
      testCast<std::string, float>("real", {"1."}, {1.0});
      testCast<std::string, float>("real", {"1"}, {1});
      testCast<std::string, float>("real", {"infinity"}, {kInf});
      testCast<std::string, float>("real", {"-infinity"}, {-kInf});
      testCast<std::string, float>("real", {"nan"}, {kNan});
      testCast<std::string, float>("real", {"InfiNiTy"}, {kInf});
      testCast<std::string, float>("real", {"-InfiNiTy"}, {-kInf});
      testCast<std::string, float>("real", {"nAn"}, {kNan});
    }

    // To boolean.
    {
      testCast<int8_t, bool>("boolean", {1}, {true});
      testCast<int8_t, bool>("boolean", {0}, {false});
      testCast<int8_t, bool>("boolean", {12}, {true});
      testCast<int8_t, bool>("boolean", {-1}, {true});

      testCast<std::string, bool>("boolean", {"1"}, {true});
      testCast<std::string, bool>("boolean", {"t"}, {true});
      testCast<std::string, bool>("boolean", {"y"}, {true});
      testCast<std::string, bool>("boolean", {"yes"}, {true});
      testCast<std::string, bool>("boolean", {"true"}, {true});

      testCast<std::string, bool>("boolean", {"0"}, {false});
      testCast<std::string, bool>("boolean", {"f"}, {false});
      testCast<std::string, bool>("boolean", {"n"}, {false});
      testCast<std::string, bool>("boolean", {"no"}, {false});
      testCast<std::string, bool>("boolean", {"false"}, {false});

      testCast<std::string, bool>(
          "boolean",
          {"t",
           "T",
           "true",
           "TRUE",
           "TrUe",
           "y",
           "Y",
           "yes",
           "YES",
           "YeS",
           "1"},
          {true, true, true, true, true, true, true, true, true, true, true});

      testCast<std::string, bool>(
          "boolean",
          {"f",
           "F",
           "false",
           "FALSE",
           "FaLsE",
           "n",
           "N",
           "no",
           "NO",
           "nO",
           "0"},
          {false,
           false,
           false,
           false,
           false,
           false,
           false,
           false,
           false,
           false,
           false});

      testCast<std::string, bool>(
          "boolean",
          {" true", "false ", " 1 ", "  yes  ", "  no  "},
          {true, false, true, true, false});

      testCast<std::string, bool>(
          "boolean",
          {"true", std::nullopt, "false", std::nullopt},
          {true, std::nullopt, false, std::nullopt});
    }

    // To string.
    {
      testCast<float, std::string>("varchar", {kInf}, {"Infinity"});
      testCast<float, std::string>("varchar", {kNan}, {"NaN"});
    }
  }

  void testTruncate() {
    testCast<Timestamp, int32_t>("integer", {Timestamp(17, 500'000'000)}, {17});

    testCast<Timestamp, int32_t>("integer", {Timestamp(17, 1'000)}, {17});

    testCast<Timestamp, int32_t>("integer", {Timestamp(17, 999'999'000)}, {17});

    testCast<Timestamp, int32_t>("integer", {Timestamp(17, 600'000'000)}, {17});

    testCast<Timestamp, int32_t>("integer", {Timestamp(17, 400'000'000)}, {17});

    testCast<Timestamp, int32_t>("integer", {Timestamp(17, 555'000'000)}, {17});

    testCast<Timestamp, int32_t>("integer", {Timestamp(12 * 3600, 0)}, {43200});
    testCast<Timestamp, int32_t>(
        "integer", {Timestamp(12 * 3600, 120'000'000)}, {43200});

    testCast<Timestamp, int32_t>(
        "integer", {Timestamp(12 * 3600, 345'600'000)}, {43200});

    testCast<Timestamp, int32_t>(
        "integer", {Timestamp(1 * 3600 + 2 * 60 + 3, 555'550'000)}, {3723});

    testCast<Timestamp, int64_t>(
        "bigint", {Timestamp(1 * 3600 + 2 * 60 + 3, 555'550'000)}, {3723L});

    testCast<Timestamp, int32_t>(
        "integer", {Timestamp(23 * 3600 + 59 * 60 + 59, 999'900'000)}, {86399});

    testCast<Timestamp, int64_t>(
        "bigint", {Timestamp(23 * 3600 + 59 * 60 + 59, 999'900'000)}, {86399L});
  }

  void testStringToBoolean() {
    // Valid strings for true (case-insensitive): t, true, y, yes, 1.
    testCast<std::string, bool>(
        "boolean",
        {"t", "T", "true", "TRUE", "TrUe", "y", "Y", "yes", "YES", "YeS", "1"},
        {true, true, true, true, true, true, true, true, true, true, true});

    // Valid strings for false (case-insensitive): f, false, n, no, 0.
    testCast<std::string, bool>(
        "boolean",
        {"f", "F", "false", "FALSE", "FaLsE", "n", "N", "no", "NO", "nO", "0"},
        {false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false,
         false});

    // Whitespace should be trimmed.
    testCast<std::string, bool>(
        "boolean",
        {" true", "false ", " 1 ", "  yes  ", "  no  "},
        {true, false, true, true, false});

    // NULL values should remain NULL.
    testCast<std::string, bool>(
        "boolean",
        {"true", std::nullopt, "false", std::nullopt},
        {true, std::nullopt, false, std::nullopt});
  }

  void testTryCasts() {
    testTryCast<std::string, int8_t>(
        "tinyint",
        {"-",
         "-0",
         " @w 123",
         "123 ",
         "  122",
         "",
         "-12-3",
         "1234",
         "-129",
         "1.1.1",
         "1..",
         "1.abc",
         "..",
         "-..",
         "125.5",
         "127",
         "-128",
         "1.2"},
        {std::nullopt,
         0,
         std::nullopt,
         123,
         122,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         127,
         -128,
         std::nullopt});

    testTryCast<double, int32_t>(
        "integer",
        {1e12, 2.5, 3.6, 100.44, -100.101},
        {std::nullopt, 2, 3, 100, -100});
    testTryCast<int64_t, int8_t>("tinyint", {456}, {std::nullopt});
    testTryCast<int64_t, int16_t>("smallint", {1234567}, {std::nullopt});
    testTryCast<int64_t, int32_t>("integer", {2147483649}, {std::nullopt});

    testTryCast<std::string, int16_t>("smallint", {"52769"}, {std::nullopt});
    testTryCast<std::string, int32_t>(
        "integer", {"17515055537"}, {std::nullopt});
    testTryCast<std::string, int32_t>(
        "integer", {"-17515055537"}, {std::nullopt});
    testTryCast<std::string, int64_t>(
        "bigint", {"9663372036854775809"}, {std::nullopt});
    testTryCast<int64_t, int8_t>(
        "tinyint", {456}, {std::nullopt}, DECIMAL(6, 0));
  }

  void testOverflow() {
    testCast<int16_t, int8_t>("tinyint", {456}, {-56});
    testCast<int32_t, int8_t>("tinyint", {266}, {10});
    testCast<int64_t, int8_t>("tinyint", {1234}, {-46});
    testCast<int64_t, int16_t>("smallint", {1234567}, {-10617});
    testCast<double, int8_t>("tinyint", {127.8}, {127});
    testCast<double, int8_t>("tinyint", {129.9}, {-127});
    testCast<double, int16_t>("smallint", {1234567.89}, {-10617});
    testCast<double, int64_t>(
        "bigint", {std::numeric_limits<double>::max()}, {9223372036854775807});
    testCast<double, int64_t>(
        "bigint", {std::numeric_limits<double>::quiet_NaN()}, {0});
    auto shortFlat = makeNullableFlatVector<int64_t>(
        {-3000,
         -2600,
         -2300,
         -2000,
         -1000,
         0,
         55000,
         57490,
         5755,
         6900,
         7200,
         std::nullopt},
        DECIMAL(5, 1));
    testCast(
        shortFlat,
        makeNullableFlatVector<int8_t>(
            {-44, -4, 26, 56, -100, 0, 124, 117, 63, -78, -48, std::nullopt}),
        false);
    testCast(
        makeNullableFlatVector<int64_t>({214748364890}, DECIMAL(12, 2)),
        makeNullableFlatVector<int8_t>({0}),
        false);
    testCast(
        makeNullableFlatVector<int64_t>({214748364890}, DECIMAL(12, 2)),
        makeNullableFlatVector<int32_t>({-2147483648}),
        false);
    testCast(
        makeNullableFlatVector<int64_t>({214748364890}, DECIMAL(12, 2)),
        makeNullableFlatVector<int64_t>({2147483648}),
        false);
  }

  void testTimestampToString() {
    testCast<Timestamp, std::string>(
        "string",
        {
            Timestamp(-946684800, 0),
            Timestamp(-7266, 0),
            Timestamp(0, 0),
            Timestamp(946684800, 0),
            Timestamp(9466848000, 0),
            Timestamp(94668480000, 0),
            Timestamp(946729316, 0),
            Timestamp(946729316, 123),
            Timestamp(946729316, 100000000),
            Timestamp(946729316, 129900000),
            Timestamp(946729316, 123456789),
            Timestamp(7266, 0),
            Timestamp(-50049331200, 0),
            Timestamp(253405036800, 0),
            Timestamp(-62480037600, 0),
            std::nullopt,
        },
        {
            "1940-01-02 00:00:00",
            "1969-12-31 21:58:54",
            "1970-01-01 00:00:00",
            "2000-01-01 00:00:00",
            "2269-12-29 00:00:00",
            "4969-12-04 00:00:00",
            "2000-01-01 12:21:56",
            "2000-01-01 12:21:56",
            "2000-01-01 12:21:56.1",
            "2000-01-01 12:21:56.1299",
            "2000-01-01 12:21:56.123456",
            "1970-01-01 02:01:06",
            "0384-01-01 08:00:00",
            "+10000-02-01 16:00:00",
            "-0010-02-01 10:00:00",
            std::nullopt,
        });

    std::vector<std::optional<Timestamp>> input = {
        Timestamp(-946684800, 0),
        Timestamp(-7266, 0),
        Timestamp(0, 0),
        Timestamp(61, 10),
        Timestamp(3600, 0),
        Timestamp(946684800, 0),

        Timestamp(946729316, 0),
        Timestamp(946729316, 123),
        Timestamp(946729316, 100000000),
        Timestamp(946729316, 129900000),
        Timestamp(946729316, 123456789),
        Timestamp(7266, 0),
        std::nullopt,
    };

    setTimezone("America/Los_Angeles");
    testCast<Timestamp, std::string>(
        "string",
        input,
        {
            "1940-01-01 16:00:00",
            "1969-12-31 13:58:54",
            "1969-12-31 16:00:00",
            "1969-12-31 16:01:01",
            "1969-12-31 17:00:00",
            "1999-12-31 16:00:00",
            "2000-01-01 04:21:56",
            "2000-01-01 04:21:56",
            "2000-01-01 04:21:56.1",
            "2000-01-01 04:21:56.1299",
            "2000-01-01 04:21:56.123456",
            "1969-12-31 18:01:06",
            std::nullopt,
        });
    setTimezone("Asia/Shanghai");
    testCast<Timestamp, std::string>(
        "string",
        input,
        {
            "1940-01-02 08:00:00",
            "1970-01-01 05:58:54",
            "1970-01-01 08:00:00",
            "1970-01-01 08:01:01",
            "1970-01-01 09:00:00",
            "2000-01-01 08:00:00",
            "2000-01-01 20:21:56",
            "2000-01-01 20:21:56",
            "2000-01-01 20:21:56.1",
            "2000-01-01 20:21:56.1299",
            "2000-01-01 20:21:56.123456",
            "1970-01-01 10:01:06",
            std::nullopt,
        });
  }

  void testFromString() {
    // String with leading and trailing whitespaces.
    testCast<std::string, int8_t>(
        "tinyint", {"\n\f\r\t\n\u001F 123\u000B\u001C\u001D\u001E"}, {123});
    testCast<std::string, int32_t>(
        "integer", {"\n\f\r\t\n\u001F 123\u000B\u001C\u001D\u001E"}, {123});
    testCast<std::string, int64_t>(
        "bigint", {"\n\f\r\t\n\u001F 123\u000B\u001C\u001D\u001E"}, {123});
    testCast<std::string, int32_t>(
        "date",
        {"\n\f\r\t\n\u001F 2015-03-18T\u000B\u001C\u001D\u001E"},
        {16512},
        VARCHAR(),
        DATE());
    testCast<std::string, float>(
        "real", {"\n\f\r\t\n\u001F 123.0\u000B\u001C\u001D\u001E"}, {123.0});
    testCast<std::string, double>(
        "double", {"\n\f\r\t\n\u001F 123.0\u000B\u001C\u001D\u001E"}, {123.0});
    testCast<std::string, Timestamp>(
        "timestamp",
        {"\n\f\r\t\n\u001F 2000-01-01 12:21:56\u000B\u001C\u001D\u001E"},
        {Timestamp(946729316, 0)});
    testCast(
        makeFlatVector<StringView>(
            {" 9999999999.99",
             "9999999999.99 ",
             "\n\f\r\t\n\u001F 9999999999.99\u000B\u001C\u001D\u001E",
             " -3E+2",
             "-3E+2 ",
             "\u000B\u001C\u001D-3E+2\u001E\n\f\r\t\n\u001F "}),
        makeFlatVector<int64_t>(
            {999'999'999'999,
             999'999'999'999,
             999'999'999'999,
             -30000,
             -30000,
             -30000},
            DECIMAL(12, 2)));
  }

  void testTinyIntToBinary() {
    testCast<int8_t, std::string>(
        TINYINT(),
        VARBINARY(),
        {18,
         -26,
         0,
         110,
         std::numeric_limits<int8_t>::max(),
         std::numeric_limits<int8_t>::min()},
        {std::string("\x12", 1),
         std::string("\xE6", 1),
         std::string("\0", 1),
         std::string("\x6E", 1),
         std::string("\x7F", 1),
         std::string("\x80", 1)});
  }

  void testSmallintToBinary() {
    testCast<int16_t, std::string>(
        SMALLINT(),
        VARBINARY(),
        {180,
         -199,
         0,
         12300,
         std::numeric_limits<int16_t>::max(),
         std::numeric_limits<int16_t>::min()},
        {std::string("\0\xB4", 2),
         std::string("\xFF\x39", 2),
         std::string("\0\0", 2),
         std::string("\x30\x0C", 2),
         std::string("\x7F\xFF", 2),
         std::string("\x80\00", 2)});
  }

  void testIntegerToBinary() {
    testCast<int32_t, std::string>(
        INTEGER(),
        VARBINARY(),
        {18,
         -26,
         0,
         180000,
         std::numeric_limits<int32_t>::max(),
         std::numeric_limits<int32_t>::min()},
        {std::string("\0\0\0\x12", 4),
         std::string("\xFF\xFF\xFF\xE6", 4),
         std::string("\0\0\0\0", 4),
         std::string("\0\x02\xBF\x20", 4),
         std::string("\x7F\xFF\xFF\xFF", 4),
         std::string("\x80\0\0\0", 4)});
  }

  void testBigIntToBinary() {
    testCast<int64_t, std::string>(
        BIGINT(),
        VARBINARY(),
        {123456,
         -256789,
         0,
         180000,
         std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::min()},
        {std::string("\0\0\0\0\0\x01\xE2\x40", 8),
         std::string("\xFF\xFF\xFF\xFF\xFF\xFC\x14\xEB", 8),
         std::string("\0\0\0\0\0\0\0\0", 8),
         std::string("\0\0\0\0\0\x02\xBF\x20", 8),
         std::string("\x7F\xFF\xFF\xFF\xFF\xFF\xFF\xFF", 8),
         std::string("\x80\x00\x00\x00\x00\x00\x00\x00", 8)});
  }

  void testDecimalToString() {
    testCast(
        makeFlatVector<int64_t>(
            {100, 1230, 12345, 0, -100, -1230, -12345}, DECIMAL(10, 2)),
        makeFlatVector<std::string>(
            {"1.00", "12.30", "123.45", "0.00", "-1.00", "-12.30", "-123.45"}));

    testCast(
        makeFlatVector<int64_t>(
            {100000, 123000, 123450, 0, -100000, -123000, -123450},
            DECIMAL(10, 3)),
        makeFlatVector<std::string>(
            {"100.000",
             "123.000",
             "123.450",
             "0.000",
             "-100.000",
             "-123.000",
             "-123.450"}));

    testCast(
        makeFlatVector<int128_t>(
            {HugeInt::build(0, 10000000000),
             HugeInt::build(0, 12300000000),
             HugeInt::build(0, 12345000000),
             HugeInt::build(0, 0),
             -HugeInt::build(0, 10000000000),
             -HugeInt::build(0, 12300000000),
             -HugeInt::build(0, 12345000000)},
            DECIMAL(20, 10)),
        makeFlatVector<std::string>(
            {"1.0000000000",
             "1.2300000000",
             "1.2345000000",
             "0.0000000000",
             "-1.0000000000",
             "-1.2300000000",
             "-1.2345000000"}));

    testCast(
        makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(10, 0)),
        makeFlatVector<std::string>({"100", "200", "300"}));

    testCast(
        makeFlatVector<int64_t>({12, 120, 1200, -12, -120}, DECIMAL(10, 8)),
        makeFlatVector<std::string>(
            {"0.00000012",
             "0.00000120",
             "0.00001200",
             "-0.00000012",
             "-0.00000120"}));

    testCast(
        makeNullableFlatVector<int64_t>(
            {100, std::nullopt, 1230, std::nullopt}, DECIMAL(10, 2)),
        makeNullableFlatVector<std::string>(
            {"1.00", std::nullopt, "12.30", std::nullopt}));
  }

  void testBoolToTimestamp() {
    testCast(
        makeFlatVector<bool>({true, false}),
        makeFlatVector<Timestamp>({
            Timestamp(0, 1000),
            Timestamp(0, 0),
        }));
  }

  void testRecursiveTryCast() {
    // Test array elements.
    testCast(
        makeArrayVector<StringView>({
            {"1", "2", "3"},
            {"4", "a", "6"},
            {"b", "c", "d"},
        }),
        makeNullableArrayVector<int64_t>({
            {1, 2, 3},
            {4, std::nullopt, 6},
            {std::nullopt, std::nullopt, std::nullopt},
        }),
        true);

    // Test map values (Spark doesn't allow casting if the map keys can become
    // null).
    testCast(
        makeMapVectorFromJson<int64_t, std::string>({
            R"( {1:"1", 2:"2", 3:"3"} )",
            R"( {1:"4", 2:"a", 3:"6"} )",
            R"( {1:"b", 2:"c", 3:"d"} )",
        }),
        makeMapVectorFromJson<int64_t, int64_t>({
            "{1:1, 2:2, 3:3}",
            "{1:4, 2:null, 3:6}",
            "{1:null, 2:null, 3:null}",
        }),
        true);

    // Test row fields.
    testCast(
        makeRowVector(
            {makeFlatVector<StringView>({"1", "4", "b"}),
             makeFlatVector<StringView>({"2", "a", "c"}),
             makeFlatVector<StringView>({"3", "6", "d"})}),
        makeRowVector(
            {makeNullableFlatVector<int64_t>({1, 4, std::nullopt}),
             makeNullableFlatVector<int64_t>({2, std::nullopt, std::nullopt}),
             makeNullableFlatVector<int64_t>({3, 6, std::nullopt})}),
        true);

    // Test nested arrays.
    testCast(
        makeNestedArrayVectorFromJson<std::string>({
            R"( [["1", "2", "3"], ["4", "a", "6"]] )",
            R"( [["b", "c", "d"], ["x", "7", "z"]] )",
        }),
        makeNestedArrayVectorFromJson<int64_t>({
            "[[1, 2, 3], [4, null, 6]]",
            "[[null, null, null], [null, 7, null]]",
        }),
        true);
  }
};
class SparkCastExprTestAnsiOn : public SparkCastExprTest {
 protected:
  void SetUp() override {
    SparkCastExprTest::SetUp();
    setAnsiSupport(true);
  }
};

class SparkCastExprTestAnsiOff : public SparkCastExprTest {
 protected:
  void SetUp() override {
    SparkCastExprTest::SetUp();
    setAnsiSupport(false);
  }
};

TEST_F(SparkCastExprTestAnsiOn, stringToDate) {
  testStringToDate();
  auto testInvalidDate = [this](const std::string& value) {
    auto input = makeRowVector({makeFlatVector<std::string>({value})});
    VELOX_ASSERT_THROW(
        (evaluateCast(VARCHAR(), DATE(), input, false)),
        "Unable to parse date value");
  };

  testInvalidDate("2012-Oct-23");
  testInvalidDate("2015/03/18");
  testInvalidDate("2015-13-01");
  testInvalidDate("2015-02-30");
}

TEST_F(SparkCastExprTestAnsiOn, decimalToIntegral) {
  testDecimalToIntegral();
}

TEST_F(SparkCastExprTestAnsiOn, testInvalidDate) {
  auto expected = [](const std::string& v) {
    return fmt::format(
        "Cannot cast VARCHAR '{}' to DATE.  Unable to parse date value: \"{}\". "
        "Valid date string patterns include ([y]y*, [y]y*-[m]m*, "
        "[y]y*-[m]m*-[d]d*, [y]y*-[m]m*-[d]d* *, "
        "[y]y*-[m]m*-[d]d*T*), and any pattern prefixed with [+-]",
        v,
        v);
  };
  testInvalidCast<std::string>(
      "date", {"2012-Oct-23"}, expected("2012-Oct-23"), VARCHAR());
  testInvalidCast<std::string>(
      "date", {"2015-03-18X"}, expected("2015-03-18X"), VARCHAR());
  testInvalidCast<std::string>(
      "date", {"2015/03/18"}, expected("2015/03/18"), VARCHAR());
  testInvalidCast<std::string>(
      "date", {"2015.03.18"}, expected("2015.03.18"), VARCHAR());
  testInvalidCast<std::string>(
      "date", {"20150318"}, expected("20150318"), VARCHAR());
  testInvalidCast<std::string>(
      "date", {"2015-031-8"}, expected("2015-031-8"), VARCHAR());
  testInvalidCast<std::string>(
      "date", {"-1-1-1"}, expected("-1-1-1"), VARCHAR());
  testInvalidCast<std::string>(
      "date", {"-11-1-1"}, expected("-11-1-1"), VARCHAR());
  testInvalidCast<std::string>(
      "date", {"-111-1-1"}, expected("-111-1-1"), VARCHAR());
  testInvalidCast<std::string>(
      "date", {"- 1111-1-1"}, expected("- 1111-1-1"), VARCHAR());
  testInvalidDate();
}

TEST_F(SparkCastExprTestAnsiOn, stringToTimestamp) {
  testStringToTimestamp();
}

TEST_F(SparkCastExprTestAnsiOn, intToTimestamp) {
  testIntToTimestamp();
}

TEST_F(SparkCastExprTestAnsiOn, timestampToInt) {
  testTimestampToInt();
}

TEST_F(SparkCastExprTestAnsiOn, doubleToTimestamp) {
  testDoubleToTimestamp();
}

TEST_F(SparkCastExprTestAnsiOn, floatToTimestamp) {
  testFloatToTimestamp();
}

TEST_F(SparkCastExprTestAnsiOn, primitiveInvalidCornerCase) {
  // To integer - ANSI ON (throws on error).
  {
    auto testInvalidThrows =
        [this](const std::string& type, const std::string& value) {
          auto input = makeRowVector({makeFlatVector<std::string>({value})});
          VELOX_ASSERT_THROW(
              (evaluate(fmt::format("cast(c0 as {})", type), input)), "");
        };

    // Invalid strings should throw.
    testInvalidThrows("tinyint", "1234567");
    testInvalidThrows("tinyint", "1a");
    testInvalidThrows("tinyint", "");
    testInvalidThrows("integer", "1'234'567");
    testInvalidThrows("integer", "1,234,567");
    testInvalidThrows("bigint", "infinity");
    testInvalidThrows("bigint", "nan");
    testInvalidThrows("bigint", "abc");
    testInvalidThrows("bigint", "+");
    testInvalidThrows("bigint", "-");

    // Overflow should throw.
    testInvalidThrows("tinyint", "128");
    testInvalidThrows("tinyint", "-129");
    testInvalidThrows("smallint", "32768");
    testInvalidThrows("smallint", "-32769");
    testInvalidThrows("integer", "2147483648");
    testInvalidThrows("integer", "-2147483649");
  }
}

TEST_F(SparkCastExprTestAnsiOn, primitiveValidCornerCases) {
  testPrimitiveValidCornerCases();

  auto testInvalidString = [this](const std::string& value) {
    auto input = makeRowVector({makeFlatVector<std::string>({value})});
    VELOX_ASSERT_THROW(evaluate("cast(c0 as boolean)", input), "Cannot cast");
  };

  testInvalidString("invalid");
  testInvalidString("tru");
  testInvalidString("2");
  testInvalidString("-1");
  testInvalidString("on");
  testInvalidString("off");
  testInvalidString("");
  testInvalidString(" ");
  testInvalidString("nan");
  auto testInvalidThrows =
      [this](const std::string& type, const std::string& value) {
        auto input = makeRowVector({makeFlatVector<std::string>({value})});
        VELOX_ASSERT_THROW(
            (evaluate(fmt::format("cast(c0 as {})", type), input)), "");
      };

  // Invalid strings should throw.
  testInvalidThrows("tinyint", "1234567");
  testInvalidThrows("tinyint", "1a");
  testInvalidThrows("tinyint", "");
  testInvalidThrows("integer", "1'234'567");
  testInvalidThrows("integer", "1,234,567");
  testInvalidThrows("bigint", "infinity");
  testInvalidThrows("bigint", "nan");
  testInvalidThrows("bigint", "abc");
  testInvalidThrows("bigint", "+");
  testInvalidThrows("bigint", "-");

  // Overflow should throw.
  testInvalidThrows("tinyint", "128");
  testInvalidThrows("tinyint", "-129");
  testInvalidThrows("smallint", "32768");
  testInvalidThrows("smallint", "-32769");
  testInvalidThrows("integer", "2147483648");
  testInvalidThrows("integer", "-2147483649");

  testCast<std::string, int64_t>(
        "bigint", {"123", "-456", "+789", "  999  "}, {123, -456, 789, 999});
    testCast<std::string, int32_t>(
        "integer", {"123", "-456", "+789", "  999  "}, {123, -456, 789, 999});
    testCast<std::string, int16_t>(
        "smallint", {"123", "-456", "+789", "  999  "}, {123, -456, 789, 999});
    testCast<std::string, int8_t>(
        "tinyint", {"12", "-45", "+78", "  99  "}, {12, -45, 78, 99});

    auto testDecimalThrows =
        [this](const std::string& type, const std::string& value) {
          auto input = makeRowVector({makeFlatVector<std::string>({value})});
          VELOX_ASSERT_THROW(
              (evaluate(fmt::format("cast(c0 as {})", type), input)),
              "Cannot cast");
        };

    testDecimalThrows("bigint", "123.45");
    testDecimalThrows("integer", "456.78");
    testDecimalThrows("smallint", "78.9");
    testDecimalThrows("tinyint", "12.3");
    testDecimalThrows("tinyint", "1.2");
    testDecimalThrows("tinyint", "-1.8");
}

TEST_F(SparkCastExprTestAnsiOn, truncate) {
  testTruncate();
}

TEST_F(SparkCastExprTestAnsiOn, tryCasts) {
  testTryCasts();
}
TEST_F(SparkCastExprTestAnsiOn, overflow) {
}
TEST_F(SparkCastExprTestAnsiOn, timestampToString) {
  testTimestampToString();
}
TEST_F(SparkCastExprTestAnsiOn, fromString) {
  testFromString();
}
TEST_F(SparkCastExprTestAnsiOn, tinyIntToBinary) {
  testTinyIntToBinary();
}

TEST_F(SparkCastExprTestAnsiOn, smallintToBinary) {
  testSmallintToBinary();
}
TEST_F(SparkCastExprTestAnsiOn, integerToBinary) {
  testIntegerToBinary();
}
TEST_F(SparkCastExprTestAnsiOn, bigIntToBinary) {
  testBigIntToBinary();
}

TEST_F(SparkCastExprTestAnsiOn, boolToTimestamp) {
  testBoolToTimestamp();
}

TEST_F(SparkCastExprTestAnsiOn, decimalToString) {
  testDecimalToString();
}

TEST_F(SparkCastExprTestAnsiOn, recursiveTryCast) {
  testRecursiveTryCast();
}

TEST_F(SparkCastExprTestAnsiOn, stringToBoolean) {
  testStringToBoolean();
    auto testInvalidString = [this](const std::string& value) {
    auto input = makeRowVector({makeFlatVector<std::string>({value})});
    VELOX_ASSERT_THROW(evaluate("cast(c0 as boolean)", input), "Cannot cast");
  };

  testInvalidString("invalid");
  testInvalidString("tru");
  testInvalidString("2");
  testInvalidString("-1");
  testInvalidString("on");
  testInvalidString("off");
  testInvalidString("");
  testInvalidString(" ");
  testInvalidString("nan");
}

TEST_F(SparkCastExprTestAnsiOff, stringToDate) {
  testStringToDate();
  testCast<std::string, int32_t>(
      "date",
      {"2012-Oct-23", /*Invalid format*/
       "2015/03/18", /*Wrong separator*/
       "2015-13-01", /*Invalid month*/
       "2015-02-30"}, /*Invalid day*/
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt},
      VARCHAR(),
      DATE());
}

TEST_F(SparkCastExprTestAnsiOff, decimalToIntegral) {
  testDecimalToIntegral();
}

TEST_F(SparkCastExprTestAnsiOff, testInvalidDate) {
  testInvalidDate();
  // Parsing ill-formated dates.
  testCast<std::string, int32_t>(
      "date", {"2012-Oct-23"}, {std::nullopt}, VARCHAR(), DATE());
  testCast<std::string, int32_t>(
      "date", {"2015-03-18X"}, {std::nullopt}, VARCHAR(), DATE());
  testCast<std::string, int32_t>(
      "date", {"2015/03/18"}, {std::nullopt}, VARCHAR(), DATE());
  testCast<std::string, int32_t>(
      "date", {"2015.03.18"}, {std::nullopt}, VARCHAR(), DATE());
  testCast<std::string, int32_t>(
      "date", {"20150318"}, {std::nullopt}, VARCHAR(), DATE());
  testCast<std::string, int32_t>(
      "date", {"2015-031-8"}, {std::nullopt}, VARCHAR(), DATE());
  testCast<std::string, int32_t>(
      "date", {"-1-1-1"}, {std::nullopt}, VARCHAR(), DATE());
  testCast<std::string, int32_t>(
      "date", {"-11-1-1"}, {std::nullopt}, VARCHAR(), DATE());
  testCast<std::string, int32_t>(
      "date", {"-111-1-1"}, {std::nullopt}, VARCHAR(), DATE());
  testCast<std::string, int32_t>(
      "date", {"- 1111-1-1"}, {std::nullopt}, VARCHAR(), DATE());
}

TEST_F(SparkCastExprTestAnsiOff, stringToTimestamp) {
  testStringToTimestamp();
  testCast<std::string, Timestamp>("timestamp", {"INVALID"}, {std::nullopt});
}

TEST_F(SparkCastExprTestAnsiOff, intToTimestamp) {
  testIntToTimestamp();
}

TEST_F(SparkCastExprTestAnsiOff, decimalToString) {
  testDecimalToString();
}

TEST_F(SparkCastExprTestAnsiOff, timestampToInt) {
  testTimestampToInt();
  testCast<Timestamp, int64_t>(
      "bigint", {Timestamp(9223372036856, 0)}, {std::nullopt});
  testTimestampToIntegralCastOverflow<int8_t>({
      -102,
      -1,
      -10,
      9,
  });
  testTimestampToIntegralCastOverflow<int16_t>({
      30874,
      -1,
      23286,
      -23287,
  });
  testTimestampToIntegralCastOverflow<int32_t>({
      1740470426,
      2147483647,
      2077252342,
      -2077252343,
  });
}

TEST_F(SparkCastExprTestAnsiOff, doubleToTimestamp) {
  testDoubleToTimestamp();
  testCast(
      makeFlatVector<double>({
          kInf,
          kNan,
          -kInf,
      }),
      makeNullableFlatVector<Timestamp>({
          std::nullopt,
          std::nullopt,
          std::nullopt,
      }));
}

TEST_F(SparkCastExprTestAnsiOff, floatToTimestamp) {
  testFloatToTimestamp();
  testCast(
      makeFlatVector<float>({kInf, kNan, -kInf}),
      makeNullableFlatVector<Timestamp>({
          std::nullopt,
          std::nullopt,
          std::nullopt,
      }));
}

TEST_F(SparkCastExprTestAnsiOff, primitiveInvalidCornerCase) {
  // To floating-point - invalid strings return null
  testCast<std::string, float>("real", {"1.2a"}, {std::nullopt});
  testCast<std::string, float>("real", {"1.2.3"}, {std::nullopt});

  // To boolean - invalid strings return null
  testCast<std::string, bool>("boolean", {"1.7E308"}, {std::nullopt});
  testCast<std::string, bool>("boolean", {"nan"}, {std::nullopt});
  testCast<std::string, bool>("boolean", {"12"}, {std::nullopt});
  testCast<std::string, bool>("boolean", {"-1"}, {std::nullopt});
  testCast<std::string, bool>("boolean", {"tr"}, {std::nullopt});
  testCast<std::string, bool>("boolean", {"tru"}, {std::nullopt});
  testCast<std::string, bool>("boolean", {"on"}, {std::nullopt});
  testCast<std::string, bool>("boolean", {"off"}, {std::nullopt});

  // To integer
  {
    // Invalid strings.
    testCast<std::string, int8_t>("tinyint", {"1234567"}, {std::nullopt});
    testCast<std::string, int8_t>("tinyint", {"1a"}, {std::nullopt});
    testCast<std::string, int8_t>("tinyint", {""}, {std::nullopt});
    testCast<std::string, int32_t>("integer", {"1'234'567"}, {std::nullopt});
    testCast<std::string, int32_t>("integer", {"1,234,567"}, {std::nullopt});
    testCast<std::string, int64_t>("bigint", {"infinity"}, {std::nullopt});
    testCast<std::string, int64_t>("bigint", {"nan"}, {std::nullopt});
    testCast<std::string, int64_t>(
        "bigint",
        {"abc", "+", "-", "  "},
        {std::nullopt, std::nullopt, std::nullopt, std::nullopt});

    // Overflow cases.
    testCast<std::string, int8_t>(
        "tinyint",
        {"128", "-129", "1000"},
        {std::nullopt, std::nullopt, std::nullopt});
    testCast<std::string, int16_t>(
        "smallint", {"32768", "-32769"}, {std::nullopt, std::nullopt});
  }

}

TEST_F(SparkCastExprTestAnsiOff, primitiveValidCornerCases) {
  testPrimitiveValidCornerCases();

  testCast<std::string, int64_t>(
        "bigint", {"123", "-456", "+789", "  999  "}, {123, -456, 789, 999});
    testCast<std::string, int32_t>(
        "integer", {"123", "-456", "+789", "  999  "}, {123, -456, 789, 999});
    testCast<std::string, int16_t>(
        "smallint", {"123", "-456", "+789", "  999  "}, {123, -456, 789, 999});
    testCast<std::string, int8_t>(
        "tinyint", {"12", "-45", "+78", "  99  "}, {12, -45, 78, 99});

  testCast<float, int64_t>("bigint", {kNan}, {0});
  testCast<float, int32_t>("integer", {kNan}, {0});
  testCast<float, int16_t>("smallint", {kNan}, {0});
  testCast<float, int8_t>("tinyint", {kNan}, {0});

  testCast<std::string, int8_t>("tinyint", {"1.2"}, {1});
  testCast<std::string, int8_t>("tinyint", {"1.23444"}, {1});
  testCast<std::string, int8_t>("tinyint", {".2355"}, {0});
  testCast<std::string, int8_t>("tinyint", {"-1.8"}, {-1});
  testCast<std::string, int8_t>("tinyint", {"1."}, {1});
  testCast<std::string, int8_t>("tinyint", {"-1."}, {-1});
  testCast<std::string, int8_t>("tinyint", {"0."}, {0});
  testCast<std::string, int8_t>("tinyint", {"."}, {0});
  testCast<std::string, int8_t>("tinyint", {"-."}, {0});

  testCast<int32_t, int8_t>("tinyint", {1234567}, {-121});
  testCast<int32_t, int8_t>("tinyint", {-1234567}, {121});

  testCast<double, int8_t>("tinyint", {12345.67}, {57});
  testCast<double, int8_t>("tinyint", {-12345.67}, {-57});
  testCast<float, int64_t>("bigint", {kInf}, {9223372036854775807});

  testCast<double, float>("real", {1.7E308}, {kInf});

  testCast<double, bool>("boolean", {1.0}, {true});
  testCast<double, bool>("boolean", {1.1}, {true});
  testCast<double, bool>("boolean", {0.1}, {true});
  testCast<double, bool>("boolean", {-0.1}, {true});
  testCast<double, bool>("boolean", {-1.0}, {true});
  testCast<float, bool>("boolean", {kNan}, {false});
  testCast<float, bool>("boolean", {kInf}, {true});
  testCast<double, bool>("boolean", {0.0000000000001}, {true});
  

  testCast<std::string, bool>(
      "boolean",
      {"invalid", "tru", "2", "-1", "on", "off", "nan", "", " "},
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});

  testCast<std::string, bool>(
      "boolean",
      {"true", "invalid", "false", std::nullopt, "1", "2", "0"},
      {true, std::nullopt, false, std::nullopt, true, std::nullopt, false});

    {
      // Invalid strings.
      testCast<std::string, int8_t>("tinyint", {"1234567"}, {std::nullopt});
      testCast<std::string, int8_t>("tinyint", {"1a"}, {std::nullopt});
      testCast<std::string, int8_t>("tinyint", {""}, {std::nullopt});
      testCast<std::string, int32_t>("integer", {"1'234'567"}, {std::nullopt});
      testCast<std::string, int32_t>("integer", {"1,234,567"}, {std::nullopt});
      testCast<std::string, int64_t>("bigint", {"infinity"}, {std::nullopt});
      testCast<std::string, int64_t>("bigint", {"nan"}, {std::nullopt});
      testCast<std::string, int64_t>(
          "bigint",
          {"abc", "+", "-", "  "},
          {std::nullopt, std::nullopt, std::nullopt, std::nullopt});

      // Overflow cases.
      testCast<std::string, int8_t>(
          "tinyint",
          {"128", "-129", "1000"},
          {std::nullopt, std::nullopt, std::nullopt});
      testCast<std::string, int16_t>(
          "smallint", {"32768", "-32769"}, {std::nullopt, std::nullopt});
    }
}

TEST_F(SparkCastExprTestAnsiOff, truncate) {
  testCast<int32_t, int8_t>(
      "tinyint", {1111111, 2, 3, 1000, -100101}, {71, 2, 3, -24, -5});
  testTruncate();
}

TEST_F(SparkCastExprTestAnsiOff, tryCasts) {
  testTryCasts();
}
TEST_F(SparkCastExprTestAnsiOff, overflow) {
  testOverflow();

  // String overflow cases return null in non-ANSI mode
  testCast<std::string, int8_t>("tinyint", {"166"}, {std::nullopt});
  testCast<std::string, int16_t>("smallint", {"52769"}, {std::nullopt});
  testCast<std::string, int32_t>("integer", {"17515055537"}, {std::nullopt});
  testCast<std::string, int32_t>("integer", {"-17515055537"}, {std::nullopt});
  testCast<std::string, int64_t>(
      "bigint", {"9663372036854775809"}, {std::nullopt});
}
TEST_F(SparkCastExprTestAnsiOff, timestampToString) {
  testTimestampToString();
}
TEST_F(SparkCastExprTestAnsiOff, fromString) {
  testFromString();
}
TEST_F(SparkCastExprTestAnsiOff, tinyIntToBinary) {
  testTinyIntToBinary();
}

TEST_F(SparkCastExprTestAnsiOff, smallintToBinary) {
  testSmallintToBinary();
}
TEST_F(SparkCastExprTestAnsiOff, integerToBinary) {
  testIntegerToBinary();
}
TEST_F(SparkCastExprTestAnsiOff, bigIntToBinary) {
  testBigIntToBinary();
}

TEST_F(SparkCastExprTestAnsiOff, stringToBoolean) {
  testStringToBoolean();

  testCast<std::string, bool>(
      "boolean",
      {"invalid", "tru", "2", "-1", "on", "off", "nan", "", " "},
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});

  testCast<std::string, bool>(
      "boolean",
      {"true", "invalid", "false", std::nullopt, "1", "2", "0"},
      {true, std::nullopt, false, std::nullopt, true, std::nullopt, false});
}

TEST_F(SparkCastExprTestAnsiOff, boolToTimestamp) {
  testBoolToTimestamp();
}

TEST_F(SparkCastExprTestAnsiOff, recursiveTryCast) {
  testRecursiveTryCast();
}
} // namespace
} // namespace facebook::velox::test
