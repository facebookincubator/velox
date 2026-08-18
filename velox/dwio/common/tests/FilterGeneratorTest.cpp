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

#include "velox/vector/tests/utils/VectorTestBase.h"

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

namespace {

class FilterGeneratorSeedTest : public testing::Test,
                                public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::vector<RowVectorPtr> makeBatches(const RowTypePtr& rowType) {
    std::vector<RowVectorPtr> batches;
    for (int32_t batch = 0; batch < 2; ++batch) {
      batches.push_back(makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>(
               kBatchRows, [&](auto row) { return row + batch * kBatchRows; }),
           makeFlatVector<int32_t>(
               kBatchRows, [&](auto row) { return (row * 7) % 101; })}));
    }
    return batches;
  }

  // Renders the generated filters into a stable string so two runs can be
  // compared without depending on filter identity.
  std::string generateFilterDescription(
      const RowTypePtr& rowType,
      uint32_t seed) {
    auto type = rowType;
    FilterGenerator generator(type, seed);
    auto filterable = generator.makeFilterables(rowType->size(), 100);
    auto specs = generator.makeRandomSpecs(filterable, 0);
    auto batches = makeBatches(rowType);
    std::vector<uint64_t> hitRows;
    auto filters =
        generator.makeSubfieldFilters(specs, batches, nullptr, hitRows);

    std::vector<std::string> rendered;
    for (const auto& [subfield, filter] : filters) {
      rendered.push_back(
          fmt::format("{}:{}", subfield.toString(), filter->toString()));
    }
    std::sort(rendered.begin(), rendered.end());
    return folly::join(",", rendered);
  }

  static constexpr vector_size_t kBatchRows = 500;
};

} // namespace

// Filter kind selection used to be driven by a process-global counter on
// AbstractColumnStats rather than by the seed, so an unrelated generator run
// earlier in the same process shifted the choice. A validation service whose
// whole premise is "replay this failing seed" cannot afford that.
TEST_F(FilterGeneratorSeedTest, filterKindIsIndependentOfEarlierGenerators) {
  auto rowType = ROW({{"big", BIGINT()}, {"int", INTEGER()}});
  const auto before = generateFilterDescription(rowType, /*seed=*/7);

  // Run unrelated generators in between; with a shared global counter these
  // perturb the next run's filter kinds.
  for (uint32_t otherSeed = 100; otherSeed < 105; ++otherSeed) {
    generateFilterDescription(rowType, otherSeed);
  }

  EXPECT_EQ(before, generateFilterDescription(rowType, /*seed=*/7));
}

TEST_F(FilterGeneratorSeedTest, differentSeedsProduceDifferentFilters) {
  auto rowType = ROW({{"big", BIGINT()}, {"int", INTEGER()}});
  std::set<std::string> descriptions;
  for (uint32_t seed = 1; seed <= 20; ++seed) {
    descriptions.insert(generateFilterDescription(rowType, seed));
  }
  // Not asserting all 20 differ -- collisions are legitimate -- only that the
  // seed actually drives the outcome.
  EXPECT_GT(descriptions.size(), 1);
}

// The old selectivity scheme drew from {1, 10, 20, 30} and {76, 80, ... 100}.
// Nothing landed between 31% and 75%, and 100% -- a filter that excludes
// nothing -- came up roughly one draw in thirteen.
TEST(FilterGeneratorTest, selectivityCoversTheMidBandAndNeverReachesAll) {
  auto rowType = ROW({{"big", BIGINT()}, {"int", INTEGER()}});
  auto type = rowType;
  FilterGenerator generator(type, /*seed=*/4242);

  int32_t low = 0;
  int32_t mid = 0;
  int32_t high = 0;
  for (int32_t i = 0; i < kDraws; ++i) {
    auto filterable = generator.makeFilterables(rowType->size(), 100);
    for (const auto& spec : generator.makeRandomSpecs(filterable, 0)) {
      if (spec.filterKind == common::FilterKind::kIsNull ||
          spec.filterKind == common::FilterKind::kIsNotNull) {
        continue;
      }
      EXPECT_GT(spec.selectPct, 0);
      EXPECT_LT(spec.selectPct, 100);
      // startPct must leave room for the range it introduces.
      EXPECT_GE(spec.startPct, 0);
      EXPECT_LT(spec.startPct, 100 - spec.selectPct);

      if (spec.selectPct < 10) {
        ++low;
      } else if (spec.selectPct < 75) {
        ++mid;
      } else {
        ++high;
      }
    }
  }

  EXPECT_GT(low, 0);
  EXPECT_GT(mid, 0);
  EXPECT_GT(high, 0);
}
