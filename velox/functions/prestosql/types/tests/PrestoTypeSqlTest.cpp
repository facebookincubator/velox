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
#include "velox/functions/prestosql/types/PrestoTypeSql.h"
#include "velox/functions/prestosql/types/QDigestType.h"
#include "velox/functions/prestosql/types/SetDigestType.h"
#include "velox/functions/prestosql/types/SfmSketchType.h"
#include "velox/functions/prestosql/types/TDigestType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/functions/prestosql/types/UuidType.h"

namespace facebook::velox {
namespace {

TEST(PrestoTypeSqlTest, primitive) {
  EXPECT_EQ(toPrestoTypeSql(BOOLEAN()), "BOOLEAN");
  EXPECT_EQ(toPrestoTypeSql(TINYINT()), "TINYINT");
  EXPECT_EQ(toPrestoTypeSql(SMALLINT()), "SMALLINT");
  EXPECT_EQ(toPrestoTypeSql(INTEGER()), "INTEGER");
  EXPECT_EQ(toPrestoTypeSql(BIGINT()), "BIGINT");
  EXPECT_EQ(toPrestoTypeSql(REAL()), "REAL");
  EXPECT_EQ(toPrestoTypeSql(DOUBLE()), "DOUBLE");
  EXPECT_EQ(toPrestoTypeSql(VARCHAR()), "VARCHAR");
  EXPECT_EQ(toPrestoTypeSql(VARBINARY()), "VARBINARY");
  EXPECT_EQ(toPrestoTypeSql(TIMESTAMP()), "TIMESTAMP");
  EXPECT_EQ(toPrestoTypeSql(DATE()), "DATE");
  EXPECT_EQ(toPrestoTypeSql(UNKNOWN()), "UNKNOWN");
}

TEST(PrestoTypeSqlTest, custom) {
  EXPECT_EQ(
      toPrestoTypeSql(TIMESTAMP_WITH_TIME_ZONE()), "TIMESTAMP WITH TIME ZONE");
  EXPECT_EQ(toPrestoTypeSql(JSON()), "JSON");
  EXPECT_EQ(toPrestoTypeSql(HYPERLOGLOG()), "HYPERLOGLOG");
  EXPECT_EQ(toPrestoTypeSql(KHYPERLOGLOG()), "KHYPERLOGLOG");
  EXPECT_EQ(toPrestoTypeSql(P4HYPERLOGLOG()), "P4HYPERLOGLOG");
  EXPECT_EQ(toPrestoTypeSql(TDIGEST(DOUBLE())), "TDIGEST(DOUBLE)");
  EXPECT_EQ(toPrestoTypeSql(QDIGEST(DOUBLE())), "QDIGEST(DOUBLE)");
  EXPECT_EQ(toPrestoTypeSql(QDIGEST(BIGINT())), "QDIGEST(BIGINT)");
  EXPECT_EQ(toPrestoTypeSql(SETDIGEST()), "SETDIGEST");
  EXPECT_EQ(toPrestoTypeSql(SFMSKETCH()), "SFMSKETCH");
  EXPECT_EQ(toPrestoTypeSql(IPADDRESS()), "IPADDRESS");
  EXPECT_EQ(toPrestoTypeSql(IPPREFIX()), "IPPREFIX");
  EXPECT_EQ(toPrestoTypeSql(UUID()), "UUID");
  EXPECT_EQ(toPrestoTypeSql(BINGTILE()), "BINGTILE");
}

TEST(PrestoTypeSqlTest, complex) {
  EXPECT_EQ(toPrestoTypeSql(ARRAY(BOOLEAN())), "ARRAY(BOOLEAN)");
  EXPECT_EQ(toPrestoTypeSql(ARRAY(ARRAY(INTEGER()))), "ARRAY(ARRAY(INTEGER))");
  EXPECT_EQ(
      toPrestoTypeSql(MAP(BOOLEAN(), INTEGER())), "MAP(BOOLEAN, INTEGER)");
  EXPECT_EQ(
      toPrestoTypeSql(MAP(VARCHAR(), ARRAY(BIGINT()))),
      "MAP(VARCHAR, ARRAY(BIGINT))");
}

TEST(PrestoTypeSqlTest, row) {
  EXPECT_EQ(
      toPrestoTypeSql(ROW({{"a", BOOLEAN()}, {"b", INTEGER()}})),
      "ROW(a BOOLEAN, b INTEGER)");
  EXPECT_EQ(
      toPrestoTypeSql(
          ROW({{"a_", BOOLEAN()}, {"b$", INTEGER()}, {"c d", INTEGER()}})),
      "ROW(a_ BOOLEAN, \"b$\" INTEGER, \"c d\" INTEGER)");
  // Unnamed fields.
  EXPECT_EQ(
      toPrestoTypeSql(ROW({INTEGER(), BOOLEAN()})), "ROW(INTEGER, BOOLEAN)");
  // Field names that need quoting.
  EXPECT_EQ(
      toPrestoTypeSql(ROW(
          {{"a with spaces", INTEGER()}, {"b with \"quotes\"", VARCHAR()}})),
      "ROW(\"a with spaces\" INTEGER, \"b with \"\"quotes\"\"\" VARCHAR)");
  EXPECT_EQ(toPrestoTypeSql(ROW({{"123", INTEGER()}})), "ROW(\"123\" INTEGER)");
}

TEST(PrestoTypeSqlTest, decimal) {
  EXPECT_EQ(toPrestoTypeSql(DECIMAL(10, 2)), "DECIMAL(10, 2)");
  EXPECT_EQ(toPrestoTypeSql(DECIMAL(38, 0)), "DECIMAL(38, 0)");
  EXPECT_EQ(toPrestoTypeSql(DECIMAL(3, 3)), "DECIMAL(3, 3)");
}

} // namespace
} // namespace facebook::velox
