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
#include "velox/functions/prestosql/coercion/PrestoCoercions.h"

#include <gtest/gtest.h>

namespace facebook::velox::functions::prestosql {
namespace {

// Verifies the BIGINT row of the Presto coercion table. The relative
// ordering DECIMAL < REAL < DOUBLE is load-bearing for overload resolution:
// for `divide(real, bigint)`, BIGINT -> REAL (cost 2) must beat
// REAL -> DOUBLE + BIGINT -> DOUBLE (cost 1 + 3 = 4) so the resolver picks
// divide(real, real) and the result type is REAL -- matching Presto's
// behavior.
TEST(PrestoCoercionsTest, bigintRow) {
  const auto& tc = typeCoercer();

  auto toDecimal = tc.coerceTypeBase(BIGINT(), DECIMAL(19, 0));
  auto toReal = tc.coerceTypeBase(BIGINT(), REAL());
  auto toDouble = tc.coerceTypeBase(BIGINT(), DOUBLE());

  ASSERT_TRUE(toDecimal.has_value());
  ASSERT_TRUE(toReal.has_value());
  ASSERT_TRUE(toDouble.has_value());

  EXPECT_LT(toDecimal->cost, toReal->cost);
  EXPECT_LT(toReal->cost, toDouble->cost);
}

// Sanity check on a representative subset of the Presto rule set.
TEST(PrestoCoercionsTest, sanity) {
  const auto& tc = typeCoercer();

  EXPECT_TRUE(tc.coercible(INTEGER(), BIGINT()).has_value());
  EXPECT_TRUE(tc.coercible(BIGINT(), DOUBLE()).has_value());
  EXPECT_TRUE(tc.coercible(DATE(), TIMESTAMP()).has_value());
  EXPECT_TRUE(tc.coercible(REAL(), DOUBLE()).has_value());
  EXPECT_TRUE(tc.coercible(UNKNOWN(), VARCHAR()).has_value());
}

} // namespace
} // namespace facebook::velox::functions::prestosql
