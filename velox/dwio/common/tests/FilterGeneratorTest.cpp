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

#include "velox/dwio/common/tests/utils/FilterGenerator.h"

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;

namespace {

// Enough draws that a 1-in-10 branch is overwhelmingly likely to be taken at
// least once, without making the test slow.
constexpr int32_t kDraws = 200;

int32_t countRowGroupSkipSpecs(const RowTypePtr& rowType) {
  auto type = rowType;
  FilterGenerator generator(type, /*seed=*/12345);
  int32_t count = 0;
  for (int32_t i = 0; i < kDraws; ++i) {
    auto filterable = generator.makeFilterables(rowType->size(), 100);
    for (const auto& spec : generator.makeRandomSpecs(filterable, 0)) {
      count += spec.isForRowGroupSkip ? 1 : 0;
    }
  }
  return count;
}

} // namespace

// The two FilterSpec constructors used to disagree on isForRowGroupSkip -- the
// explicit one defaulted it true, the in-class initializer false -- and
// makeRandomSpecs builds specs with the default constructor. Pins them
// together, since a divergence silently disables or enables the whole
// row-group-skip path.
TEST(FilterGeneratorTest, filterSpecDefaultsAgree) {
  FilterSpec defaultConstructed;
  FilterSpec explicitlyConstructed("field");
  EXPECT_EQ(
      defaultConstructed.isForRowGroupSkip,
      explicitlyConstructed.isForRowGroupSkip);
  EXPECT_FALSE(defaultConstructed.isForRowGroupSkip);
}

TEST(FilterGeneratorTest, rowGroupSkipSpecsForSupportedTypes) {
  auto rowType = ROW(
      {{"tiny", TINYINT()},
       {"small", SMALLINT()},
       {"int", INTEGER()},
       {"big", BIGINT()}});
  EXPECT_GT(countRowGroupSkipSpecs(rowType), 0);
}

// makeRowGroupSkipRangeFilter has no specialization for these kinds: the
// generic template funnels the max through getIntegerValue(), which would
// narrow a double or an int128 into a BigintRange of the wrong type, and
// ComplexColumnStats fails outright. VARCHAR is excluded for a different
// reason -- its specialization keys off kMaxString, a sentinel chosen to
// exceed test data, which selects nothing against real data.
TEST(FilterGeneratorTest, noRowGroupSkipSpecsForUnsupportedTypes) {
  auto rowType = ROW(
      {{"real", REAL()},
       {"double", DOUBLE()},
       {"huge", HUGEINT()},
       {"bool", BOOLEAN()},
       {"string", VARCHAR()},
       {"row", ROW({{"nested", BIGINT()}})},
       {"array", ARRAY(BIGINT())},
       {"map", MAP(INTEGER(), BIGINT())}});
  EXPECT_EQ(countRowGroupSkipSpecs(rowType), 0);
}

// A row-group-skip spec replaces the value filter entirely, so pairing it with
// a null kind would silently drop the null semantics the category asked for.
TEST(FilterGeneratorTest, rowGroupSkipNeverPairedWithNullKinds) {
  auto rowType = ROW({{"int", INTEGER()}, {"big", BIGINT()}});
  auto type = rowType;
  FilterGenerator generator(type, /*seed=*/999);
  int32_t observed = 0;
  for (int32_t i = 0; i < kDraws; ++i) {
    auto filterable = generator.makeFilterables(rowType->size(), 100);
    for (const auto& spec : generator.makeRandomSpecs(filterable, 0)) {
      if (spec.isForRowGroupSkip) {
        ++observed;
        EXPECT_NE(spec.filterKind, common::FilterKind::kIsNull);
        EXPECT_NE(spec.filterKind, common::FilterKind::kIsNotNull);
      }
    }
  }
  // Without this the test would pass with zero assertions run if the draw were
  // ever disabled again -- which is the bug this whole change fixes.
  EXPECT_GT(observed, 0);
}
