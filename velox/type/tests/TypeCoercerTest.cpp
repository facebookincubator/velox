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

namespace facebook::velox {
namespace {

void testCoercion(const TypePtr& fromType, const TypePtr& toType) {
  auto coercion = TypeCoercer::coerceTypeBase(fromType, toType->name());
  ASSERT_TRUE(coercion.has_value());
  VELOX_EXPECT_EQ_TYPES(coercion->type, toType);
}

void testBaseCoercion(const TypePtr& fromType) {
  auto coercion = TypeCoercer::coerceTypeBase(fromType, fromType->kindName());
  ASSERT_TRUE(coercion.has_value());
  VELOX_EXPECT_EQ_TYPES(coercion->type, fromType);
}

void testNoCoercion(const TypePtr& fromType, const TypePtr& toType) {
  auto coercion = TypeCoercer::coerceTypeBase(fromType, toType->name());
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

  // UNKNOWN coerces to any primitive type.
  testCoercion(UNKNOWN(), VARCHAR());
  testCoercion(UNKNOWN(), BIGINT());
  testCoercion(UNKNOWN(), BOOLEAN());
}

TEST(TypeCoercerTest, date) {
  testCoercion(DATE(), DATE());
  testCoercion(DATE(), TIMESTAMP());

  testNoCoercion(DATE(), BIGINT());
}

TEST(TypeCoercerTest, unknown) {
  ASSERT_TRUE(TypeCoercer::coercible(UNKNOWN(), BOOLEAN()));
  ASSERT_TRUE(TypeCoercer::coercible(UNKNOWN(), BIGINT()));
  ASSERT_TRUE(TypeCoercer::coercible(UNKNOWN(), VARCHAR()));
  ASSERT_TRUE(TypeCoercer::coercible(UNKNOWN(), ARRAY(INTEGER())));
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
    auto cost = TypeCoercer::coercible(type, type);
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
  ASSERT_TRUE(TypeCoercer::coercible(ARRAY(UNKNOWN()), ARRAY(INTEGER())));
  ASSERT_TRUE(
      TypeCoercer::coercible(ARRAY(UNKNOWN()), ARRAY(ARRAY(VARCHAR()))));

  ASSERT_FALSE(TypeCoercer::coercible(ARRAY(BIGINT()), ARRAY(REAL())));
  ASSERT_FALSE(
      TypeCoercer::coercible(ARRAY(UNKNOWN()), MAP(INTEGER(), REAL())));
  ASSERT_FALSE(TypeCoercer::coercible(ARRAY(UNKNOWN()), ROW({UNKNOWN()})));
  ASSERT_FALSE(TypeCoercer::coercible(ARRAY(VARCHAR()), VARCHAR()));
}

TEST(TypeCoercerTest, map) {
  ASSERT_TRUE(
      TypeCoercer::coercible(
          MAP(UNKNOWN(), UNKNOWN()), MAP(INTEGER(), REAL())));
  ASSERT_TRUE(
      TypeCoercer::coercible(MAP(VARCHAR(), REAL()), MAP(VARCHAR(), DOUBLE())));
  ASSERT_TRUE(
      TypeCoercer::coercible(MAP(INTEGER(), REAL()), MAP(BIGINT(), DOUBLE())));

  ASSERT_FALSE(
      TypeCoercer::coercible(MAP(INTEGER(), REAL()), MAP(BIGINT(), INTEGER())));
  ASSERT_FALSE(
      TypeCoercer::coercible(MAP(UNKNOWN(), UNKNOWN()), ARRAY(BIGINT())));
  ASSERT_FALSE(
      TypeCoercer::coercible(
          MAP(UNKNOWN(), UNKNOWN()), ROW({INTEGER(), BIGINT()})));
}

TEST(TypeCoercerTest, row) {
  ASSERT_TRUE(
      TypeCoercer::coercible(
          ROW({UNKNOWN(), INTEGER(), REAL()}),
          ROW({SMALLINT(), BIGINT(), DOUBLE()})));

  ASSERT_FALSE(
      TypeCoercer::coercible(
          ROW({UNKNOWN(), INTEGER(), REAL()}),
          ROW({SMALLINT(), VARCHAR(), DOUBLE()})));

  ASSERT_FALSE(
      TypeCoercer::coercible(
          ROW({UNKNOWN(), INTEGER(), REAL()}), ARRAY(INTEGER())));
  ASSERT_FALSE(
      TypeCoercer::coercible(
          ROW({UNKNOWN(), INTEGER(), REAL()}), MAP(INTEGER(), REAL())));
  ASSERT_FALSE(
      TypeCoercer::coercible(
          ROW({UNKNOWN(), INTEGER(), REAL()}), ROW({UNKNOWN(), INTEGER()})));
  ASSERT_FALSE(
      TypeCoercer::coercible(ROW({UNKNOWN(), INTEGER(), REAL()}), BIGINT()));
}

TEST(TypeCoercerTest, leastCommonSuperType) {
  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::leastCommonSuperType(INTEGER(), BIGINT()), BIGINT());

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::leastCommonSuperType(ARRAY(INTEGER()), ARRAY(TINYINT())),
      ARRAY(INTEGER()));

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::leastCommonSuperType(
          MAP(TINYINT(), DOUBLE()), MAP(INTEGER(), REAL())),
      MAP(INTEGER(), DOUBLE()));

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::leastCommonSuperType(
          ROW({TINYINT(), DOUBLE()}), ROW({INTEGER(), REAL()})),
      ROW({INTEGER(), DOUBLE()}));

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::leastCommonSuperType(
          ROW({"", "", ""}, INTEGER()), ROW({"", "", ""}, SMALLINT())),
      ROW({"", "", ""}, INTEGER()));

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::leastCommonSuperType(
          ROW({"a", "b", "c"}, INTEGER()), ROW({"a", "b", "c"}, SMALLINT())),
      ROW({"a", "b", "c"}, INTEGER()));

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::leastCommonSuperType(
          ROW({"", "", ""}, INTEGER()), ROW({"a", "b", "c"}, SMALLINT())),
      ROW({"", "", ""}, INTEGER()));

  VELOX_ASSERT_EQ_TYPES(
      TypeCoercer::leastCommonSuperType(
          ROW({"a", "bb", ""}, INTEGER()), ROW({"a", "b", "c"}, SMALLINT())),
      ROW({"a", "", ""}, INTEGER()));

  ASSERT_TRUE(
      TypeCoercer::leastCommonSuperType(VARCHAR(), TINYINT()) == nullptr);

  ASSERT_TRUE(
      TypeCoercer::leastCommonSuperType(ARRAY(TINYINT()), TINYINT()) ==
      nullptr);

  ASSERT_TRUE(
      TypeCoercer::leastCommonSuperType(ARRAY(TINYINT()), ARRAY(VARCHAR())) ==
      nullptr);

  ASSERT_TRUE(
      TypeCoercer::leastCommonSuperType(
          ROW({""}, TINYINT()), ROW({"", ""}, TINYINT())) == nullptr);

  ASSERT_TRUE(
      TypeCoercer::leastCommonSuperType(
          MAP(INTEGER(), REAL()), ROW({INTEGER(), REAL()})) == nullptr);
}

} // namespace
} // namespace facebook::velox
