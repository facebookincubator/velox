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

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/types/BingTileType.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"
#include "velox/functions/prestosql/types/IPAddressType.h"
#include "velox/functions/prestosql/types/IPPrefixType.h"
#include "velox/functions/prestosql/types/JsonType.h"
#include "velox/functions/prestosql/types/KHyperLogLogType.h"
#include "velox/functions/prestosql/types/P4HyperLogLogType.h"
#include "velox/functions/prestosql/types/PrestoTypes.h"
#include "velox/functions/prestosql/types/QDigestType.h"
#include "velox/functions/prestosql/types/SetDigestType.h"
#include "velox/functions/prestosql/types/SfmSketchType.h"
#include "velox/functions/prestosql/types/TDigestType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/functions/prestosql/types/UuidType.h"
#include "velox/type/Time.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox {
namespace {

TEST(PrestoTypeSqlTest, primitive) {
  EXPECT_EQ(PrestoTypes::toSql(BOOLEAN()), "BOOLEAN");
  EXPECT_EQ(PrestoTypes::toSql(TINYINT()), "TINYINT");
  EXPECT_EQ(PrestoTypes::toSql(SMALLINT()), "SMALLINT");
  EXPECT_EQ(PrestoTypes::toSql(INTEGER()), "INTEGER");
  EXPECT_EQ(PrestoTypes::toSql(BIGINT()), "BIGINT");
  EXPECT_EQ(PrestoTypes::toSql(REAL()), "REAL");
  EXPECT_EQ(PrestoTypes::toSql(DOUBLE()), "DOUBLE");
  EXPECT_EQ(PrestoTypes::toSql(VARCHAR()), "VARCHAR");
  EXPECT_EQ(PrestoTypes::toSql(VARBINARY()), "VARBINARY");
  EXPECT_EQ(PrestoTypes::toSql(TIMESTAMP()), "TIMESTAMP");
  EXPECT_EQ(PrestoTypes::toSql(DATE()), "DATE");
  EXPECT_EQ(PrestoTypes::toSql(UNKNOWN()), "UNKNOWN");
}

TEST(PrestoTypeSqlTest, custom) {
  EXPECT_EQ(
      PrestoTypes::toSql(TIMESTAMP_WITH_TIME_ZONE()),
      "TIMESTAMP WITH TIME ZONE");
  EXPECT_EQ(PrestoTypes::toSql(JSON()), "JSON");
  EXPECT_EQ(PrestoTypes::toSql(HYPERLOGLOG()), "HYPERLOGLOG");
  EXPECT_EQ(PrestoTypes::toSql(KHYPERLOGLOG()), "KHYPERLOGLOG");
  EXPECT_EQ(PrestoTypes::toSql(P4HYPERLOGLOG()), "P4HYPERLOGLOG");
  EXPECT_EQ(PrestoTypes::toSql(TDIGEST(DOUBLE())), "TDIGEST(DOUBLE)");
  EXPECT_EQ(PrestoTypes::toSql(QDIGEST(DOUBLE())), "QDIGEST(DOUBLE)");
  EXPECT_EQ(PrestoTypes::toSql(QDIGEST(BIGINT())), "QDIGEST(BIGINT)");
  EXPECT_EQ(PrestoTypes::toSql(SETDIGEST()), "SETDIGEST");
  EXPECT_EQ(PrestoTypes::toSql(SFMSKETCH()), "SFMSKETCH");
  EXPECT_EQ(PrestoTypes::toSql(IPADDRESS()), "IPADDRESS");
  EXPECT_EQ(PrestoTypes::toSql(IPPREFIX()), "IPPREFIX");
  EXPECT_EQ(PrestoTypes::toSql(UUID()), "UUID");
  EXPECT_EQ(PrestoTypes::toSql(BINGTILE()), "BINGTILE");
}

TEST(PrestoTypeSqlTest, complex) {
  EXPECT_EQ(PrestoTypes::toSql(ARRAY(BOOLEAN())), "ARRAY(BOOLEAN)");
  EXPECT_EQ(
      PrestoTypes::toSql(ARRAY(ARRAY(INTEGER()))), "ARRAY(ARRAY(INTEGER))");
  EXPECT_EQ(
      PrestoTypes::toSql(MAP(BOOLEAN(), INTEGER())), "MAP(BOOLEAN, INTEGER)");
  EXPECT_EQ(
      PrestoTypes::toSql(MAP(VARCHAR(), ARRAY(BIGINT()))),
      "MAP(VARCHAR, ARRAY(BIGINT))");
}

TEST(PrestoTypeSqlTest, row) {
  EXPECT_EQ(
      PrestoTypes::toSql(ROW({{"a", BOOLEAN()}, {"b", INTEGER()}})),
      "ROW(a BOOLEAN, b INTEGER)");
  EXPECT_EQ(
      PrestoTypes::toSql(
          ROW({{"a_", BOOLEAN()}, {"b$", INTEGER()}, {"c d", INTEGER()}})),
      "ROW(a_ BOOLEAN, \"b$\" INTEGER, \"c d\" INTEGER)");
  // Unnamed fields.
  EXPECT_EQ(
      PrestoTypes::toSql(ROW({INTEGER(), BOOLEAN()})), "ROW(INTEGER, BOOLEAN)");
  // Field names that need quoting.
  EXPECT_EQ(
      PrestoTypes::toSql(ROW(
          {{"a with spaces", INTEGER()}, {"b with \"quotes\"", VARCHAR()}})),
      "ROW(\"a with spaces\" INTEGER, \"b with \"\"quotes\"\"\" VARCHAR)");
  EXPECT_EQ(
      PrestoTypes::toSql(ROW({{"123", INTEGER()}})), "ROW(\"123\" INTEGER)");
}

TEST(PrestoTypeSqlTest, decimal) {
  EXPECT_EQ(PrestoTypes::toSql(DECIMAL(10, 2)), "DECIMAL(10, 2)");
  EXPECT_EQ(PrestoTypes::toSql(DECIMAL(38, 0)), "DECIMAL(38, 0)");
  EXPECT_EQ(PrestoTypes::toSql(DECIMAL(3, 3)), "DECIMAL(3, 3)");
}

// Tests PrestoTypes::valueToString which extends Type::valueToString with
// Presto custom type formatting. Tests are grouped by physical type and use
// the same raw values to illustrate the difference in output.
TEST(PrestoTypeSqlTest, valueToString) {
  // int64_t types: BIGINT, DECIMAL, TIME, BINGTILE, TIMESTAMP WITH TIME ZONE,
  // TIME WITH TIME ZONE. Same raw value 45'123 is formatted differently
  // depending on the logical type.
  {
    const int64_t value = 45'123;

    {
      TypePtr type = BIGINT();
      EXPECT_EQ(PrestoTypes::valueToString(value, type), "45123");
    }
    {
      TypePtr type = DECIMAL(10, 2);
      EXPECT_EQ(PrestoTypes::valueToString(value, type), "451.23");
    }
    {
      // 45'123 ms since midnight = 00:00:45.123.
      TypePtr type = TIME();
      EXPECT_EQ(PrestoTypes::valueToString(value, type), "00:00:45.123");
    }
    {
      TypePtr type = TIME_WITH_TIME_ZONE();
      EXPECT_EQ(PrestoTypes::valueToString(value, type), "11:06:00.011-12:54");
    }
    {
      TypePtr type = TIMESTAMP_WITH_TIME_ZONE();
      EXPECT_EQ(
          PrestoTypes::valueToString(value, type),
          "1969-12-31 11:06:00.011 -12:54");
    }
    {
      TypePtr type = BINGTILE();
      EXPECT_EQ(
          PrestoTypes::valueToString(value, type), "{x=0, y=45123, zoom=0}");
    }
  }

  // int128_t types: HUGEINT, DECIMAL, IPADDRESS, UUID.
  // Same raw value 45'123 is formatted differently depending on the logical
  // type.
  {
    const int128_t value = 45'123;

    {
      TypePtr type = DECIMAL(38, 2);
      EXPECT_EQ(PrestoTypes::valueToString(value, type), "451.23");
    }
    {
      TypePtr type = IPADDRESS();
      EXPECT_EQ(PrestoTypes::valueToString(value, type), "::b043");
    }
    {
      TypePtr type = UUID();
      EXPECT_EQ(
          PrestoTypes::valueToString(value, type),
          "00000000-0000-0000-0000-00000000b043");
    }
  }

  // int32_t types: INTEGER, DATE.
  {
    const int32_t value = 45'123;

    {
      TypePtr type = INTEGER();
      EXPECT_EQ(PrestoTypes::valueToString(value, type), "45123");
    }
    {
      TypePtr type = DATE();
      EXPECT_EQ(PrestoTypes::valueToString(value, type), "2093-07-17");
    }
  }

  // StringView types: VARCHAR and JSON render the string content as-is.
  // JSON values are already normalized by json_parse at execution time.
  {
    StringView value(R"({"a":1,"b":2})");

    EXPECT_EQ(PrestoTypes::valueToString(value, VARCHAR()), value.str());
    EXPECT_EQ(PrestoTypes::valueToString(value, JSON()), value.str());
  }

  // VARBINARY-based Presto types render as space-separated hex bytes.
  {
    StringView value("hello");
    const std::string expected = "68 65 6c 6c 6f";

    EXPECT_EQ(PrestoTypes::valueToString(value, VARBINARY()), expected);
    EXPECT_EQ(PrestoTypes::valueToString(value, HYPERLOGLOG()), expected);
    EXPECT_EQ(PrestoTypes::valueToString(value, KHYPERLOGLOG()), expected);
    EXPECT_EQ(PrestoTypes::valueToString(value, P4HYPERLOGLOG()), expected);
    EXPECT_EQ(PrestoTypes::valueToString(value, TDIGEST(DOUBLE())), expected);
    EXPECT_EQ(PrestoTypes::valueToString(value, QDIGEST(DOUBLE())), expected);
    EXPECT_EQ(PrestoTypes::valueToString(value, SETDIGEST()), expected);
    EXPECT_EQ(PrestoTypes::valueToString(value, SFMSKETCH()), expected);
  }

  // Remaining scalar types.
  EXPECT_EQ(PrestoTypes::valueToString(true, BOOLEAN()), "true");
  EXPECT_EQ(
      PrestoTypes::valueToString(static_cast<int8_t>(42), TINYINT()), "42");
  EXPECT_EQ(
      PrestoTypes::valueToString(static_cast<int16_t>(42), SMALLINT()), "42");
  EXPECT_EQ(PrestoTypes::valueToString(1.5f, REAL()), "1.5");
  EXPECT_EQ(PrestoTypes::valueToString(1.5, DOUBLE()), "1.5");
  EXPECT_EQ(
      PrestoTypes::valueToString(Timestamp(0, 0), TIMESTAMP()),
      "1970-01-01 00:00:00.000");
  // Non-zero nanos: verify millisecond truncation.
  EXPECT_EQ(
      PrestoTypes::valueToString(
          Timestamp(1'705'320'000, 123'456'789), TIMESTAMP()),
      "2024-01-15 12:00:00.123");
  // Pre-1000 year: verify zero-padded year.
  EXPECT_EQ(
      PrestoTypes::valueToString(Timestamp(-62'135'596'800, 0), TIMESTAMP()),
      "0001-01-01 00:00:00.000");
}

// Tests PrestoTypes::valueToString(DecodedVector) which handles nulls,
// Presto custom types, and complex types. Uses dictionary-encoded vectors
// with reversed indices to verify proper index handling.
class PrestoTypesDecodedTest : public testing::Test,
                               public test::VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance({});
  }
};

TEST_F(PrestoTypesDecodedTest, nulls) {
  auto dictionary = wrapInDictionary(
      makeIndices({1, 0}),
      makeNullableFlatVector<int64_t>({std::nullopt, int64_t{45'123}}));
  DecodedVector decoded(*dictionary);
  EXPECT_EQ(PrestoTypes::valueToString(decoded, 0, BIGINT()), "45123");
  EXPECT_EQ(PrestoTypes::valueToString(decoded, 1, BIGINT()), "null");
}

// int64_t: BIGINT (built-in) vs TIMESTAMP WITH TIME ZONE (custom).
// Same underlying vector, different type interpretation.
TEST_F(PrestoTypesDecodedTest, bigintAndTimestampWithTimeZone) {
  auto dictionary = wrapInDictionary(
      makeIndices({1, 0}),
      makeFlatVector<int64_t>({int64_t{45'123}, int64_t{45'124}}));
  DecodedVector decoded(*dictionary);

  EXPECT_EQ(PrestoTypes::valueToString(decoded, 0, BIGINT()), "45124");
  EXPECT_EQ(PrestoTypes::valueToString(decoded, 1, BIGINT()), "45123");

  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 0, TIMESTAMP_WITH_TIME_ZONE()),
      "1969-12-31 11:07:00.011 -12:53");
  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 1, TIMESTAMP_WITH_TIME_ZONE()),
      "1969-12-31 11:06:00.011 -12:54");
}

// Array: same underlying vector interpreted as ARRAY(INTEGER) vs
// ARRAY(DATE). Uses > 5 elements to verify no truncation.
TEST_F(PrestoTypesDecodedTest, array) {
  auto dictionary = wrapInDictionary(
      makeIndices({1, 0}),
      makeArrayVector<int32_t>({
          {0, 1, 2, 3, 4, 5, 6},
          {18'262, 18'263, 18'264, 18'265, 18'266, 18'267},
      }));

  DecodedVector decoded(*dictionary);

  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 0, ARRAY(INTEGER())),
      "[18262, 18263, 18264, 18265, 18266, 18267]");
  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 1, ARRAY(INTEGER())),
      "[0, 1, 2, 3, 4, 5, 6]");

  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 0, ARRAY(DATE())),
      "[2020-01-01, 2020-01-02, 2020-01-03, 2020-01-04, 2020-01-05, 2020-01-06]");
  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 1, ARRAY(DATE())),
      "[1970-01-01, 1970-01-02, 1970-01-03, 1970-01-04, 1970-01-05, 1970-01-06, 1970-01-07]");

  // Empty array — same vector, different element types.
  {
    auto emptyDictionary =
        wrapInDictionary(makeIndices({0}), makeArrayVector<int32_t>({{}}));
    DecodedVector emptyDecoded(*emptyDictionary);
    EXPECT_EQ(
        PrestoTypes::valueToString(emptyDecoded, 0, ARRAY(INTEGER())), "[]");
    EXPECT_EQ(
        PrestoTypes::valueToString(emptyDecoded, 0, ARRAY(UNKNOWN())), "[]");
    EXPECT_EQ(PrestoTypes::valueToString(emptyDecoded, 0, ARRAY(DATE())), "[]");
  }
}

// Map: same underlying vector interpreted as MAP(INTEGER, INTEGER) vs
// MAP(INTEGER, DATE). Uses > 5 entries to verify no truncation.
TEST_F(PrestoTypesDecodedTest, map) {
  auto dictionary = wrapInDictionary(
      makeIndices({1, 0}),
      makeMapVector<int32_t, int32_t>(
          {{{1, 0}, {2, 1}, {3, 2}, {4, 3}, {5, 4}, {6, 5}},
           {
               {7, 18'262},
               {8, 18'263},
               {9, 18'264},
               {10, 18'265},
               {11, 18'266},
               {12, 18'267},
           }}));

  DecodedVector decoded(*dictionary);

  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 0, MAP(INTEGER(), INTEGER())),
      "{7=18262, 8=18263, 9=18264, 10=18265, 11=18266, 12=18267}");
  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 1, MAP(INTEGER(), INTEGER())),
      "{1=0, 2=1, 3=2, 4=3, 5=4, 6=5}");

  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 0, MAP(INTEGER(), DATE())),
      "{7=2020-01-01, 8=2020-01-02, 9=2020-01-03, 10=2020-01-04, 11=2020-01-05, 12=2020-01-06}");
  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 1, MAP(INTEGER(), DATE())),
      "{1=1970-01-01, 2=1970-01-02, 3=1970-01-03, 4=1970-01-04, 5=1970-01-05, 6=1970-01-06}");

  // Empty map — same vector, different value types.
  {
    auto emptyDictionary = wrapInDictionary(
        makeIndices({0}), makeMapVector<int32_t, int32_t>({{}}));
    DecodedVector emptyDecoded(*emptyDictionary);
    EXPECT_EQ(
        PrestoTypes::valueToString(emptyDecoded, 0, MAP(INTEGER(), INTEGER())),
        "{}");
    EXPECT_EQ(
        PrestoTypes::valueToString(emptyDecoded, 0, MAP(INTEGER(), DATE())),
        "{}");
  }
}

// ROW (built-in) vs IPPREFIX (custom ROW type). Same underlying RowVector,
// different type produces different output.
TEST_F(PrestoTypesDecodedTest, rowAndIpPrefix) {
  auto dictionary = wrapInDictionary(
      makeIndices({1, 0}),
      makeRowVector({
          makeFlatVector<int128_t>({int128_t{45'123}, int128_t{45'124}}),
          makeFlatVector<int8_t>({24, 8}),
      }));

  DecodedVector decoded(*dictionary);

  EXPECT_EQ(PrestoTypes::valueToString(decoded, 0, IPPREFIX()), "::b044/8");
  EXPECT_EQ(PrestoTypes::valueToString(decoded, 1, IPPREFIX()), "::b043/24");

  // Same data interpreted as a plain ROW renders raw values.
  {
    auto rowType = ROW({{"ip", HUGEINT()}, {"prefix", TINYINT()}});
    EXPECT_EQ(
        PrestoTypes::valueToString(decoded, 0, rowType),
        "{ip=45124, prefix=8}");
    EXPECT_EQ(
        PrestoTypes::valueToString(decoded, 1, rowType),
        "{ip=45123, prefix=24}");
  }

  // Same data interpreted as ROW(UUID(), TINYINT()).
  {
    auto rowType = ROW({{"id", UUID()}, {"prefix", TINYINT()}});
    EXPECT_EQ(
        PrestoTypes::valueToString(decoded, 0, rowType),
        "{id=00000000-0000-0000-0000-00000000b044, prefix=8}");
    EXPECT_EQ(
        PrestoTypes::valueToString(decoded, 1, rowType),
        "{id=00000000-0000-0000-0000-00000000b043, prefix=24}");
  }
}

// ROW with > 5 fields to verify no truncation.
TEST_F(PrestoTypesDecodedTest, rowManyFields) {
  auto dictionary = wrapInDictionary(
      makeIndices({1, 0}),
      makeRowVector({
          makeFlatVector<int32_t>({1, 10}),
          makeFlatVector<int32_t>({2, 20}),
          makeFlatVector<int32_t>({3, 30}),
          makeFlatVector<int32_t>({4, 40}),
          makeFlatVector<int32_t>({5, 50}),
          makeFlatVector<int32_t>({6, 60}),
      }));

  DecodedVector decoded(*dictionary);

  // Same underlying data, but each field uses a different INTEGER-based type.
  auto rowType = ROW(
      {{"a", INTEGER()},
       {"b", DATE()},
       {"c", INTERVAL_YEAR_MONTH()},
       {"d", INTEGER()},
       {"e", DATE()},
       {"f", INTERVAL_YEAR_MONTH()}});
  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 0, rowType),
      "{a=10, b=1970-01-21, c=2-6, d=40, e=1970-02-20, f=5-0}");
  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 1, rowType),
      "{a=1, b=1970-01-03, c=0-3, d=4, e=1970-01-06, f=0-6}");

  // Unnamed fields use field0, field1, etc. like Presto.
  auto unnamedRowType = ROW(folly::copy(rowType->children()));
  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 0, unnamedRowType),
      "{field0=10, field1=1970-01-21, field2=2-6, field3=40, field4=1970-02-20, field5=5-0}");
  EXPECT_EQ(
      PrestoTypes::valueToString(decoded, 1, unnamedRowType),
      "{field0=1, field1=1970-01-03, field2=0-3, field3=4, field4=1970-01-06, field5=0-6}");
}

} // namespace
} // namespace facebook::velox
