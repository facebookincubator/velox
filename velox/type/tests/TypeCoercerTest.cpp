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

#include "velox/type/TypeCoercer.h"
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/type/CastRegistry.h"

namespace facebook::velox {
namespace {

void testCoercion(const TypePtr& fromType, const TypePtr& toType) {
  auto coercion =
      TypeCoercer::defaults().coerceTypeBase(fromType, toType->name());
  ASSERT_TRUE(coercion.has_value());
  VELOX_EXPECT_EQ_TYPES(coercion->type, toType);
}

void testBaseCoercion(const TypePtr& fromType) {
  auto coercion =
      TypeCoercer::defaults().coerceTypeBase(fromType, fromType->kindName());
  ASSERT_TRUE(coercion.has_value());
  VELOX_EXPECT_EQ_TYPES(coercion->type, fromType);
}

void testNoCoercion(const TypePtr& fromType, const TypePtr& toType) {
  auto coercion =
      TypeCoercer::defaults().coerceTypeBase(fromType, toType->name());
  ASSERT_FALSE(coercion.has_value());
}

TEST(TypeCoercerTest, basic) {
  testCoercion(TINYINT(), TINYINT());
  testCoercion(TINYINT(), BIGINT());
  testCoercion(TINYINT(), REAL());

  testNoCoercion(TINYINT(), VARCHAR());
  testNoCoercion(TINYINT(), DATE());

  testBaseCoercion(ARRAY(TINYINT()));
  testNoCoercion(ARRAY(TINYINT()), MAP(INTEGER(), REAL()));
}

TEST(TypeCoercerTest, decimal) {
  testCoercion(DECIMAL(10, 2), REAL());
  testCoercion(DECIMAL(10, 2), DOUBLE());
  testCoercion(DECIMAL(38, 6), REAL());
  testCoercion(DECIMAL(38, 6), DOUBLE());

  testNoCoercion(DECIMAL(10, 2), VARCHAR());
  testNoCoercion(DECIMAL(10, 2), BIGINT());

  ASSERT_TRUE(TypeCoercer::defaults().coercible(DECIMAL(10, 2), DOUBLE()));
  ASSERT_TRUE(TypeCoercer::defaults().coercible(DECIMAL(10, 2), REAL()));
  ASSERT_FALSE(TypeCoercer::defaults().coercible(DOUBLE(), DECIMAL(10, 2)));
}

TEST(TypeCoercerTest, integerToDecimal) {
  testCoercion(TINYINT(), DECIMAL(3, 0));
  testCoercion(SMALLINT(), DECIMAL(5, 0));
  testCoercion(INTEGER(), DECIMAL(10, 0));
  testCoercion(BIGINT(), DECIMAL(19, 0));

  ASSERT_TRUE(TypeCoercer::defaults().coercible(TINYINT(), DECIMAL(10, 2)));
  ASSERT_TRUE(TypeCoercer::defaults().coercible(SMALLINT(), DECIMAL(10, 2)));
  ASSERT_TRUE(TypeCoercer::defaults().coercible(INTEGER(), DECIMAL(38, 4)));
  ASSERT_TRUE(TypeCoercer::defaults().coercible(BIGINT(), DECIMAL(38, 4)));

  ASSERT_TRUE(TypeCoercer::defaults().coercible(TINYINT(), DECIMAL(3, 0)));
  ASSERT_TRUE(TypeCoercer::defaults().coercible(SMALLINT(), DECIMAL(5, 0)));
  ASSERT_TRUE(TypeCoercer::defaults().coercible(INTEGER(), DECIMAL(10, 0)));
  ASSERT_TRUE(TypeCoercer::defaults().coercible(BIGINT(), DECIMAL(19, 0)));

  ASSERT_FALSE(TypeCoercer::defaults().coercible(TINYINT(), DECIMAL(2, 0)));
  ASSERT_FALSE(TypeCoercer::defaults().coercible(SMALLINT(), DECIMAL(4, 0)));
  ASSERT_FALSE(TypeCoercer::defaults().coercible(INTEGER(), DECIMAL(9, 0)));
  ASSERT_FALSE(TypeCoercer::defaults().coercible(BIGINT(), DECIMAL(18, 0)));
  ASSERT_FALSE(TypeCoercer::defaults().coercible(INTEGER(), DECIMAL(4, 2)));
  ASSERT_FALSE(TypeCoercer::defaults().coercible(BIGINT(), DECIMAL(10, 2)));

  ASSERT_FALSE(TypeCoercer::defaults().coercible(DECIMAL(10, 2), INTEGER()));
  ASSERT_FALSE(TypeCoercer::defaults().coercible(DECIMAL(10, 2), BIGINT()));
}

TEST(TypeCoercerTest, integerToDecimalLeastCommonSuperType) {
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(INTEGER(), DECIMAL(38, 4)),
      DECIMAL(38, 4));
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(DECIMAL(38, 4), INTEGER()),
      DECIMAL(38, 4));
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(BIGINT(), DECIMAL(10, 2)),
      DECIMAL(21, 2));
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(DECIMAL(10, 2), BIGINT()),
      DECIMAL(21, 2));
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(INTEGER(), DECIMAL(4, 2)),
      DECIMAL(12, 2));
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(SMALLINT(), DECIMAL(4, 2)),
      DECIMAL(7, 2));
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(TINYINT(), DECIMAL(10, 4)),
      DECIMAL(10, 4));
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(BIGINT(), DECIMAL(19, 0)),
      DECIMAL(19, 0));

  ASSERT_EQ(
      TypeCoercer::defaults().leastCommonSuperType(BIGINT(), DECIMAL(38, 20)),
      nullptr);
}

TEST(TypeCoercerTest, decimalDecimalLeastCommonSuperType) {
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(
          DECIMAL(10, 2), DECIMAL(20, 4)),
      DECIMAL(20, 4));
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(
          DECIMAL(20, 2), DECIMAL(10, 4)),
      DECIMAL(22, 4));
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(
          DECIMAL(38, 4), DECIMAL(38, 4)),
      DECIMAL(38, 4));
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(
          DECIMAL(10, 2), DECIMAL(10, 2)),
      DECIMAL(10, 2));
}

TEST(TypeCoercerTest, decimalLeastCommonSuperType) {
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(DECIMAL(10, 2), DOUBLE()),
      DOUBLE());
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(DOUBLE(), DECIMAL(10, 2)),
      DOUBLE());
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(DECIMAL(10, 2), REAL()),
      REAL());
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(REAL(), DECIMAL(10, 2)),
      REAL());
}

TEST(TypeCoercerTest, date) {
  testCoercion(DATE(), DATE());
  testCoercion(DATE(), TIMESTAMP());

  testNoCoercion(DATE(), BIGINT());
}

TEST(TypeCoercerTest, unknown) {
  ASSERT_TRUE(TypeCoercer::defaults().coercible(UNKNOWN(), BOOLEAN()));
  ASSERT_TRUE(TypeCoercer::defaults().coercible(UNKNOWN(), BIGINT()));
  ASSERT_TRUE(TypeCoercer::defaults().coercible(UNKNOWN(), VARCHAR()));
  ASSERT_TRUE(TypeCoercer::defaults().coercible(UNKNOWN(), ARRAY(INTEGER())));
}

TEST(TypeCoercerTest, coerceTypeBaseFromUnknown) {
  // Test coercion from UNKNOWN to various types.
  testCoercion(UNKNOWN(), TINYINT());
  testCoercion(UNKNOWN(), BOOLEAN());
  testCoercion(UNKNOWN(), SMALLINT());
  testCoercion(UNKNOWN(), INTEGER());
  testCoercion(UNKNOWN(), BIGINT());
  testCoercion(UNKNOWN(), REAL());
  testCoercion(UNKNOWN(), DOUBLE());
  testCoercion(UNKNOWN(), VARCHAR());
  testCoercion(UNKNOWN(), VARBINARY());
}

TEST(TypeCoercerTest, noCost) {
  auto assertNoCost = [](const TypePtr& type) {
    SCOPED_TRACE(type->toString());
    auto cost = TypeCoercer::defaults().coercible(type, type);
    ASSERT_TRUE(cost.has_value());
    EXPECT_EQ(cost.value(), 0);
  };

  assertNoCost(UNKNOWN());
  assertNoCost(BOOLEAN());
  assertNoCost(TINYINT());
  assertNoCost(SMALLINT());
  assertNoCost(INTEGER());
  assertNoCost(BIGINT());
  assertNoCost(REAL());
  assertNoCost(DOUBLE());
  assertNoCost(VARCHAR());
  assertNoCost(VARBINARY());
  assertNoCost(TIMESTAMP());
  assertNoCost(DATE());

  assertNoCost(ARRAY(INTEGER()));
  assertNoCost(ARRAY(UNKNOWN()));
  assertNoCost(MAP(INTEGER(), REAL()));
  assertNoCost(MAP(UNKNOWN(), UNKNOWN()));
  assertNoCost(ROW({INTEGER(), REAL()}));
  assertNoCost(ROW({UNKNOWN(), UNKNOWN()}));
}

TEST(TypeCoercerTest, array) {
  ASSERT_TRUE(
      TypeCoercer::defaults().coercible(ARRAY(UNKNOWN()), ARRAY(INTEGER())));
  ASSERT_TRUE(
      TypeCoercer::defaults().coercible(
          ARRAY(UNKNOWN()), ARRAY(ARRAY(VARCHAR()))));

  ASSERT_FALSE(
      TypeCoercer::defaults().coercible(ARRAY(BIGINT()), ARRAY(REAL())));
  ASSERT_FALSE(
      TypeCoercer::defaults().coercible(
          ARRAY(UNKNOWN()), MAP(INTEGER(), REAL())));
  ASSERT_FALSE(
      TypeCoercer::defaults().coercible(ARRAY(UNKNOWN()), ROW({UNKNOWN()})));
  ASSERT_FALSE(TypeCoercer::defaults().coercible(ARRAY(VARCHAR()), VARCHAR()));
}

TEST(TypeCoercerTest, map) {
  ASSERT_TRUE(
      TypeCoercer::defaults().coercible(
          MAP(UNKNOWN(), UNKNOWN()), MAP(INTEGER(), REAL())));
  ASSERT_TRUE(
      TypeCoercer::defaults().coercible(
          MAP(VARCHAR(), REAL()), MAP(VARCHAR(), DOUBLE())));
  ASSERT_TRUE(
      TypeCoercer::defaults().coercible(
          MAP(INTEGER(), REAL()), MAP(BIGINT(), DOUBLE())));

  ASSERT_FALSE(
      TypeCoercer::defaults().coercible(
          MAP(INTEGER(), REAL()), MAP(BIGINT(), INTEGER())));
  ASSERT_FALSE(
      TypeCoercer::defaults().coercible(
          MAP(UNKNOWN(), UNKNOWN()), ARRAY(BIGINT())));
  ASSERT_FALSE(
      TypeCoercer::defaults().coercible(
          MAP(UNKNOWN(), UNKNOWN()), ROW({INTEGER(), BIGINT()})));
}

TEST(TypeCoercerTest, row) {
  ASSERT_TRUE(
      TypeCoercer::defaults().coercible(
          ROW({UNKNOWN(), INTEGER(), REAL()}),
          ROW({SMALLINT(), BIGINT(), DOUBLE()})));

  ASSERT_FALSE(
      TypeCoercer::defaults().coercible(
          ROW({UNKNOWN(), INTEGER(), REAL()}),
          ROW({SMALLINT(), VARCHAR(), DOUBLE()})));

  ASSERT_FALSE(
      TypeCoercer::defaults().coercible(
          ROW({UNKNOWN(), INTEGER(), REAL()}), ARRAY(INTEGER())));
  ASSERT_FALSE(
      TypeCoercer::defaults().coercible(
          ROW({UNKNOWN(), INTEGER(), REAL()}), MAP(INTEGER(), REAL())));
  ASSERT_FALSE(
      TypeCoercer::defaults().coercible(
          ROW({UNKNOWN(), INTEGER(), REAL()}), ROW({UNKNOWN(), INTEGER()})));
  ASSERT_FALSE(
      TypeCoercer::defaults().coercible(
          ROW({UNKNOWN(), INTEGER(), REAL()}), BIGINT()));
}

TEST(TypeCoercerTest, leastCommonSuperType) {
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(INTEGER(), BIGINT()),
      BIGINT());

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(
          ARRAY(INTEGER()), ARRAY(TINYINT())),
      ARRAY(INTEGER()));

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(
          MAP(TINYINT(), DOUBLE()), MAP(INTEGER(), REAL())),
      MAP(INTEGER(), DOUBLE()));

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(
          ROW({TINYINT(), DOUBLE()}), ROW({INTEGER(), REAL()})),
      ROW({INTEGER(), DOUBLE()}));

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(
          ROW({"", "", ""}, INTEGER()), ROW({"", "", ""}, SMALLINT())),
      ROW({"", "", ""}, INTEGER()));

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(
          ROW({"a", "b", "c"}, INTEGER()), ROW({"a", "b", "c"}, SMALLINT())),
      ROW({"a", "b", "c"}, INTEGER()));

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(
          ROW({"", "", ""}, INTEGER()), ROW({"a", "b", "c"}, SMALLINT())),
      ROW({"", "", ""}, INTEGER()));

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::defaults().leastCommonSuperType(
          ROW({"a", "bb", ""}, INTEGER()), ROW({"a", "b", "c"}, SMALLINT())),
      ROW({"a", "", ""}, INTEGER()));

  ASSERT_TRUE(
      TypeCoercer::defaults().leastCommonSuperType(VARCHAR(), TINYINT()) ==
      nullptr);

  ASSERT_TRUE(
      TypeCoercer::defaults().leastCommonSuperType(
          ARRAY(TINYINT()), TINYINT()) == nullptr);

  ASSERT_TRUE(
      TypeCoercer::defaults().leastCommonSuperType(
          ARRAY(TINYINT()), ARRAY(VARCHAR())) == nullptr);

  ASSERT_TRUE(
      TypeCoercer::defaults().leastCommonSuperType(
          ROW({""}, TINYINT()), ROW({"", ""}, TINYINT())) == nullptr);

  ASSERT_TRUE(
      TypeCoercer::defaults().leastCommonSuperType(
          MAP(INTEGER(), REAL()), ROW({INTEGER(), REAL()})) == nullptr);
}

TEST(TypeCoercerTest, parametricBuiltinTargetDoesNotThrow) {
  // Parametric built-in factories throw on empty params. Verify graceful
  // handling.
  EXPECT_EQ(
      TypeCoercer::defaults().coerceTypeBase(BIGINT(), "ARRAY"), std::nullopt);
}

TEST(TypeCoercerTest, ctorRejectsDuplicateCostForSameSource) {
  VELOX_ASSERT_THROW(
      TypeCoercer({{INTEGER(), BIGINT(), 1}, {INTEGER(), DOUBLE(), 1}}),
      "Duplicate cost 1 for source type INTEGER");
}

TEST(TypeCoercerTest, ctorRejectsNullEntries) {
  VELOX_ASSERT_THROW(
      TypeCoercer({{nullptr, BIGINT(), 1}}),
      "CoercionEntry.from must not be null");
  VELOX_ASSERT_THROW(
      TypeCoercer({{INTEGER(), nullptr, 1}}),
      "CoercionEntry.to must not be null");
}

TEST(TypeCoercerTest, ctorRejectsDecimalToDecimal) {
  // DECIMAL -> DECIMAL is hardcoded in the type system; rule entries
  // wouldn't be honored, so reject them at construction.
  VELOX_ASSERT_THROW(
      TypeCoercer({{DECIMAL(1, 0), DECIMAL(20, 4), 1}}),
      "DECIMAL -> DECIMAL coercion is not customizable");
}

TEST(TypeCoercerTest, ctorRejectsNonCanonicalSourceDecimal) {
  // Source DECIMAL must be the canonical placeholder DECIMAL(1, 0).
  VELOX_ASSERT_THROW(
      TypeCoercer({{DECIMAL(10, 2), DOUBLE(), 1}}),
      "Source DECIMAL in CoercionEntry must be DECIMAL(1, 0)");

  // The canonical placeholder is accepted.
  EXPECT_NO_THROW(TypeCoercer({{DECIMAL(1, 0), DOUBLE(), 1}}));
}

TEST(TypeCoercerTest, ctorRejectsDuplicateRule) {
  // Plain duplicate (same source -> same target name).
  VELOX_ASSERT_THROW(
      TypeCoercer({{INTEGER(), BIGINT(), 1}, {INTEGER(), BIGINT(), 2}}),
      "Duplicate coercion rule INTEGER -> BIGINT");

  // DECIMAL footgun: both rules collide on map key (TINYINT, DECIMAL).
  VELOX_ASSERT_THROW(
      TypeCoercer(
          {{TINYINT(), DECIMAL(3, 0), 1}, {TINYINT(), DECIMAL(10, 2), 2}}),
      "Duplicate coercion rule TINYINT -> DECIMAL");
}

} // namespace
} // namespace facebook::velox
