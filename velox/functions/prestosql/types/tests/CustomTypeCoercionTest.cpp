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
#include "velox/functions/prestosql/types/QDigestRegistration.h"
#include "velox/functions/prestosql/types/QDigestType.h"
#include "velox/functions/prestosql/types/TDigestRegistration.h"
#include "velox/functions/prestosql/types/TDigestType.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneRegistration.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneRegistration.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/TypeCoercer.h"

namespace facebook::velox {
namespace {

// Verifies implicit coercion rules registered by Presto custom types.
class CustomTypeCoercionTest : public testing::Test {
 protected:
  void SetUp() override {
    registerTimestampWithTimeZoneType();
    registerTimeWithTimezoneType();
    registerQDigestType();
    registerTDigestType();
  }
};

TEST_F(CustomTypeCoercionTest, timestampWithTimeZone) {
  auto coercion = TypeCoercer::coerceTypeBase(
      TIMESTAMP(), TIMESTAMP_WITH_TIME_ZONE()->name());
  ASSERT_TRUE(coercion.has_value());
  VELOX_EXPECT_EQ_TYPES(coercion->type, TIMESTAMP_WITH_TIME_ZONE());
  EXPECT_EQ(coercion->cost, 1);

  coercion =
      TypeCoercer::coerceTypeBase(DATE(), TIMESTAMP_WITH_TIME_ZONE()->name());
  ASSERT_TRUE(coercion.has_value());
  VELOX_EXPECT_EQ_TYPES(coercion->type, TIMESTAMP_WITH_TIME_ZONE());
  EXPECT_EQ(coercion->cost, 2);

  // Reverse directions are explicit-only, not coercible.
  ASSERT_FALSE(
      TypeCoercer::coerceTypeBase(
          TIMESTAMP_WITH_TIME_ZONE(), TIMESTAMP()->name()));
  ASSERT_FALSE(
      TypeCoercer::coerceTypeBase(TIMESTAMP_WITH_TIME_ZONE(), DATE()->name()));
}

TEST_F(CustomTypeCoercionTest, timeWithTimeZone) {
  auto coercion =
      TypeCoercer::coerceTypeBase(TIME(), TIME_WITH_TIME_ZONE()->name());
  ASSERT_TRUE(coercion.has_value());
  VELOX_EXPECT_EQ_TYPES(coercion->type, TIME_WITH_TIME_ZONE());

  // Reverse directions are explicit-only, not coercible.
  ASSERT_FALSE(
      TypeCoercer::coerceTypeBase(TIME_WITH_TIME_ZONE(), TIME()->name()));
  ASSERT_FALSE(
      TypeCoercer::coerceTypeBase(TIME_WITH_TIME_ZONE(), DATE()->name()));
}

TEST_F(CustomTypeCoercionTest, digests) {
  EXPECT_FALSE(
      TypeCoercer::coerceTypeBase(TDIGEST(DOUBLE()), "QDIGEST").has_value());
  EXPECT_FALSE(
      TypeCoercer::coerceTypeBase(QDIGEST(DOUBLE()), "TDIGEST").has_value());
}

} // namespace
} // namespace facebook::velox
