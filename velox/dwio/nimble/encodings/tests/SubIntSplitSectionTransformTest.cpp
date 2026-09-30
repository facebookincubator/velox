/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <span>
#include <vector>

#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"

using namespace facebook::nimble;
using namespace facebook::nimble::subintsplit;

namespace {

// The shapes a bit-range section actually takes: near-random low bits,
// low-cardinality high bits, a monotone counter, and a constant.
enum class Shape { Uniform, LowCardinality, Monotone, Constant };

std::vector<uint64_t>
makeSection(size_t count, int width, Shape shape, uint64_t seed) {
  std::mt19937_64 rng(seed);
  const uint64_t mask =
      width >= 64 ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
  std::vector<uint64_t> values(count);
  for (size_t i = 0; i < count; ++i) {
    switch (shape) {
      case Shape::Uniform:
        values[i] = rng() & mask;
        break;
      case Shape::LowCardinality:
        values[i] = ((i / 37) % 5) & mask;
        break;
      case Shape::Monotone:
        values[i] = (i * 3) & mask;
        break;
      case Shape::Constant:
        values[i] = 7 & mask;
        break;
    }
  }
  return values;
}

std::vector<TransformId> allTransforms() {
  return {TransformId::KeyDerived};
}

} // namespace

// A transform that scores well is only useful if it is exactly invertible, so
// this is the gate on every other claim about them.
TEST(SectionTransformTest, roundTripsEveryShapeAndWidth) {
  for (auto id : allTransforms()) {
    const auto* transform = transformFor(id);
    ASSERT_NE(transform, nullptr) << toString(id);
    // Widths include several that do not divide 64 evenly and the two
    // degenerate ends (1 and 64). Counts include ones that are not a multiple
    // of any word or chunk size.
    for (size_t count :
         {size_t{1},
          size_t{2},
          size_t{7},
          size_t{256},
          size_t{1024},
          size_t{4'133}}) {
      for (int width : {1, 3, 5, 7, 8, 13, 16, 32, 40, 63, 64}) {
        for (auto shape :
             {Shape::Uniform,
              Shape::LowCardinality,
              Shape::Monotone,
              Shape::Constant}) {
          const auto original =
              makeSection(count, width, shape, count * 31 + width);
          const auto key =
              makeSection(count, 6, Shape::LowCardinality, count + 11);

          std::vector<uint64_t> values = original;
          TransformContext context{.keySection = key, .width = width};
          TransformState state;
          transform->apply(values, context, state);
          transform->invert(values, context, state);

          EXPECT_EQ(values, original)
              << toString(id) << " count=" << count << " width=" << width
              << " shape=" << static_cast<int>(shape);
        }
      }
    }
  }
}

// The permutation is recovered by re-sorting the key, so a section sorted by
// itself must come back in the same order it went in.
TEST(SectionTransformTest, keyDerivedOnItsOwnKeyIsIdentityOrder) {
  const auto key = makeSection(512, 6, Shape::LowCardinality, 5);
  std::vector<uint64_t> values = key;
  const auto* transform = transformFor(TransformId::KeyDerived);
  TransformContext context{.keySection = key, .width = 6};
  TransformState state;
  transform->apply(values, context, state);
  EXPECT_TRUE(std::is_sorted(values.begin(), values.end()));
  transform->invert(values, context, state);
  EXPECT_EQ(values, key);
}

// A caller holding one key across several sections builds the permutation once
// and hands it back, which is only sound if what it hands back is exactly the
// permutation the transform would have built for itself. Pinned rather than
// trusted: an order belonging to some other key would reorder rows by a key the
// stream does not name, and the result would look like ordinary data.
TEST(SectionTransformTest, suppliedKeyOrderMatchesTheOneItWouldBuild) {
  constexpr size_t kCount = 4'000;
  constexpr int kWidth = 18;
  std::mt19937_64 rng(11);
  std::vector<uint64_t> key(kCount);
  for (auto& value : key) {
    value = rng() % 37;
  }
  const auto original = makeSection(kCount, kWidth, Shape::Uniform, 5);
  const auto* transform = transformFor(TransformId::KeyDerived);

  TransformContext derivedContext{.keySection = key, .width = kWidth};
  TransformState state;
  std::vector<uint64_t> derived = original;
  transform->apply(derived, derivedContext, state);

  const auto order = buildKeyOrder(key);
  TransformContext suppliedContext{
      .keySection = key, .width = kWidth, .keyOrder = order};
  std::vector<uint64_t> supplied = original;
  transform->apply(supplied, suppliedContext, state);

  EXPECT_EQ(supplied, derived);

  // The decoder is never handed an order, so a supplied one still has to leave
  // a stream the derived path can undo.
  transform->invert(supplied, derivedContext, state);
  EXPECT_EQ(supplied, original);
}

// Undoing the permutation walks one cursor per distinct key, so how many keys
// there are decides both what it costs and how much bookkeeping there is to get
// wrong. Every other round-trip test here uses low-cardinality keys, so this
// pushes the run count into the thousands, through the path that derives the
// run ids itself rather than being handed them.
TEST(SectionTransformTest, keyDerivedRoundTripsWithManyDistinctKeys) {
  constexpr size_t kCount = 20'000;
  constexpr size_t kDistinctKeys = 5'000;
  std::mt19937_64 rng(42);
  std::vector<uint64_t> key(kCount);
  for (auto& value : key) {
    value = rng() % kDistinctKeys;
  }
  const auto original = makeSection(kCount, 20, Shape::Uniform, 99);
  std::vector<uint64_t> values = original;

  const auto* transform = transformFor(TransformId::KeyDerived);
  TransformContext context{.keySection = key, .width = 20};
  TransformState state;
  transform->apply(values, context, state);
  transform->invert(values, context, state);
  EXPECT_EQ(values, original);
}

// Same large-k round trip, but through the path where the key's own encoding
// already hands back dense run ids, as a dictionary-backed key section would.
// Ids follow the dictionary's sorted numbering rather than first-appearance
// order, so this also checks that the merge does not assume the two coincide.
TEST(
    SectionTransformTest,
    keyDerivedRoundTripsWithManyDistinctKeysGivenRunIds) {
  constexpr size_t kCount = 20'000;
  constexpr size_t kDistinctKeys = 5'000;
  std::mt19937_64 rng(7);
  std::vector<uint64_t> key(kCount);
  for (auto& value : key) {
    value = rng() % kDistinctKeys;
  }

  std::vector<uint64_t> distinctKeys(key.begin(), key.end());
  std::sort(distinctKeys.begin(), distinctKeys.end());
  distinctKeys.erase(
      std::unique(distinctKeys.begin(), distinctKeys.end()),
      distinctKeys.end());
  std::vector<uint32_t> runIds(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    runIds[i] = static_cast<uint32_t>(
        std::lower_bound(distinctKeys.begin(), distinctKeys.end(), key[i]) -
        distinctKeys.begin());
  }

  const auto original = makeSection(kCount, 24, Shape::Uniform, 123);
  std::vector<uint64_t> values = original;
  const auto* transform = transformFor(TransformId::KeyDerived);
  TransformContext context{
      .keySection = key,
      .width = 24,
      .keyRunIds = runIds,
      .keyRunValues = distinctKeys,
  };
  TransformState state;
  transform->apply(values, context, state);
  transform->invert(values, context, state);
  EXPECT_EQ(values, original);
}

// Point access is what decides whether a transform may serve a point-lookup
// read, so it is asserted rather than left to a comment.
TEST(SectionTransformTest, reportsHowItMapsPositions) {
  // Rows move, but the key section says where to, and it is stored in original
  // order, so a probe follows the map rather than rebuilding anything.
  EXPECT_EQ(
      transformFor(TransformId::KeyDerived)->positionMapping(),
      PositionMapping::Permuted);
  EXPECT_TRUE(transformFor(TransformId::KeyDerived)->supportsPointAccess());

  // Asserted rather than left implicit, so that adding a transform that
  // cannot answer a point read on its own has to change this test and account
  // for the cost.
  for (auto id : allTransforms()) {
    EXPECT_TRUE(transformFor(id)->supportsPointAccess()) << toString(id);
  }
}

// A permuted mapping is only meaningful if it agrees with the transform it
// describes: position i must be where apply() actually put row i.
TEST(SectionTransformTest, positionMapMatchesWhereTheTransformPutEachRow) {
  std::mt19937_64 rng(99);
  std::vector<uint64_t> keys(5000);
  std::vector<uint64_t> values(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    keys[i] = rng() % 17;
    values[i] = i;
  }

  const auto* transform = transformFor(TransformId::KeyDerived);
  auto transformed = values;
  TransformState state;
  const TransformContext context{.keySection = keys, .width = 16};
  transform->apply(transformed, context, state);

  std::vector<uint32_t> positions(values.size());
  transform->positionMap(context, state, positions);
  for (size_t i = 0; i < values.size(); ++i) {
    ASSERT_EQ(transformed[positions[i]], values[i]) << "row " << i;
  }
}

TEST(SectionTransformTest, onlyKeyDerivedNeedsAKeySection) {
  EXPECT_TRUE(transformFor(TransformId::KeyDerived)->needsKeySection());
}

TEST(SectionTransformTest, noneHasNoTransform) {
  EXPECT_EQ(transformFor(TransformId::None), nullptr);
}

// An unrecognised transform yields wrong values rather than obviously broken
// ones, so a reader must fail instead of decoding.
TEST(SectionTransformTest, unknownTransformIdThrows) {
  EXPECT_THROW(transformForRaw(2), NimbleUserError);
  EXPECT_THROW(transformForRaw(9), NimbleUserError);
  EXPECT_THROW(transformForRaw(200), NimbleUserError);
  EXPECT_NO_THROW(transformForRaw(0));
  EXPECT_NO_THROW(
      transformForRaw(static_cast<uint8_t>(TransformId::KeyDerived)));
}

// A key-derived permutation is restored from the key section alone, so it
// must not silently cost anything to store.
TEST(SectionTransformTest, keyDerivedStoresNothing) {
  auto values = makeSection(256, 8, Shape::LowCardinality, 3);
  const auto key = makeSection(256, 6, Shape::LowCardinality, 4);
  TransformContext context{.keySection = key, .width = 8};
  TransformState state;
  transformFor(TransformId::KeyDerived)->apply(values, context, state);
  EXPECT_TRUE(state.codebook.empty());
}

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
