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

#include "velox/dwio/common/Reader.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

namespace facebook::velox::dwio::common {
namespace {

using namespace facebook::velox::common;

class ReaderTest : public testing::Test, public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }
};

TEST_F(ReaderTest, getOrCreateChild) {
  auto input = makeRowVector(
      {"c.0", "c.1"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>({2, 4, 6, 7, 8}),
      });

  common::ScanSpec spec("<root>");
  spec.addField("c.0", 0);
  // Create child from name.
  spec.getOrCreateChild("c.1")->setFilter(
      common::createBigintValues({2, 4, 6}, false));

  auto actual = RowReader::projectColumns(input, spec, nullptr);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  test::assertEqualVectors(expected, actual);

  // Create child from subfield.
  spec.getOrCreateChild(common::Subfield("c.1"))
      ->setFilter(common::createBigintValues({2, 4, 6}, false));
  VELOX_ASSERT_USER_THROW(
      RowReader::projectColumns(input, spec, nullptr),
      "Field not found: c. Available fields are: c.0, c.1.");
}

TEST_F(ReaderTest, projectColumnsFilterStruct) {
  constexpr int kSize = 10;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeRowVector({
          makeFlatVector<int64_t>(kSize, folly::identity),
      }),
  });
  common::ScanSpec spec("<root>");
  spec.addField("c0", 0);
  spec.getOrCreateChild(common::Subfield("c1.c0"))
      ->setFilter(common::createBigintValues({2, 4, 6}, false));
  auto actual = RowReader::projectColumns(input, spec, nullptr);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({2, 4, 6}),
  });
  test::assertEqualVectors(expected, actual);
}

TEST_F(ReaderTest, projectColumnsNullField) {
  constexpr int kSize = 10;
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kSize),
       makeRowVector({makeFlatVector<int64_t>(kSize, folly::identity)}),
       makeFlatVector<int64_t>(kSize, folly::identity)});
  input->childAt(0) = nullptr;

  common::ScanSpec spec("<root>");
  spec.addField("c0", 0);
  spec.addField("c1", 1);
  spec.getOrCreateChild(common::Subfield("c1.c0"))
      ->setFilter(common::createBigintValues({2, 4, 6}, false));
  auto actual = RowReader::projectColumns(input, spec, nullptr);

  auto expected = makeRowVector(
      {makeFlatVector<int64_t>(3),
       makeRowVector({makeFlatVector<int64_t>({2, 4, 6})})});
  expected->childAt(0) = nullptr;
  test::assertEqualVectors(expected, actual);
}

TEST_F(ReaderTest, projectColumnsFilterArray) {
  constexpr int kSize = 10;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeArrayVector<int64_t>(
          kSize,
          [](auto) { return 1; },
          [](auto i) { return i; },
          [](auto i) { return i % 2 != 0; }),
  });
  common::ScanSpec spec("<root>");
  spec.addField("c0", 0);
  auto* c1 = spec.getOrCreateChild(common::Subfield("c1"));
  {
    SCOPED_TRACE("IS NULL");
    c1->setFilter(std::make_unique<common::IsNull>());
    auto actual = RowReader::projectColumns(input, spec, nullptr);
    auto expected = makeRowVector({
        makeFlatVector<int64_t>({1, 3, 5, 7, 9}),
    });
    test::assertEqualVectors(expected, actual);
  }
  {
    SCOPED_TRACE("IS NOT NULL");
    c1->setFilter(std::make_unique<common::IsNotNull>());
    auto actual = RowReader::projectColumns(input, spec, nullptr);
    auto expected = makeRowVector({
        makeFlatVector<int64_t>({0, 2, 4, 6, 8}),
    });
    test::assertEqualVectors(expected, actual);
  }
}

TEST_F(ReaderTest, projectColumnsMutation) {
  constexpr int kSize = 10;
  auto input = makeRowVector({makeFlatVector<int64_t>(kSize, folly::identity)});
  common::ScanSpec spec("<root>");
  spec.addAllChildFields(*input->type());
  std::vector<uint64_t> deleted(bits::nwords(kSize));
  bits::setBit(deleted.data(), 2);
  Mutation mutation;
  mutation.deletedRows = deleted.data();
  auto actual = RowReader::projectColumns(input, spec, &mutation);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 3, 4, 5, 6, 7, 8, 9}),
  });
  test::assertEqualVectors(expected, actual);

  constexpr auto kNumRounds = 1U << 6;

  size_t numNonZero = 0;
  size_t numNonMax = 0;

  // Test with random skip - use property-based testing instead of hardcoded
  // outputs to avoid brittleness when folly::Random implementation changes.
  std::mt19937 seeds;
  for (size_t round = 0; round < kNumRounds; ++round) {
    const auto seed = seeds();

    random::setSeed(folly::to_narrow(seed));
    random::RandomSkipTracker randomSkip(0.5);
    mutation.randomSkip = &randomSkip;
    actual = RowReader::projectColumns(input, spec, &mutation);

    // Property 1: Result size should be less than input size (some rows
    // skipped). With 0.5 sample rate and 9 eligible rows (excluding deleted row
    // 2), we expect roughly 4-5 rows, but allow wider range for RNG variance.
    EXPECT_GE(actual->size(), 0);
    EXPECT_LE(actual->size(), kSize - 1);

    numNonZero += actual->size() > 0;
    numNonMax += actual->size() < kSize - 1;

    // The result is a RowVector with one child column. Assume it.
    auto res = actual->as<RowVector>()->childAt(0)->as<SimpleVector<int64_t>>();
    std::vector<int64_t> vec;
    vec.reserve(actual->size());
    for (vector_size_t i = 0; i < actual->size(); ++i) {
      vec.push_back(res->valueAt(i));
    }

    // Property 2: All values in result must be from original input.
    for (auto val : vec) {
      // Each value must be in valid range
      EXPECT_GE(val, 0);
      EXPECT_LT(val, kSize);
      // Deleted row should never appear
      EXPECT_NE(val, 2);
    }

    // Property 3: Values should be in ascending order (projectColumns preserves
    // order).
    EXPECT_TRUE(std::is_sorted(vec.begin(), vec.end()));

    // Property 4: No duplicate values (each input row appears at most once).
    EXPECT_TRUE(std::adjacent_find(vec.begin(), vec.end()) == vec.end());

    // Property 5: With a fixed seed, the result should be deterministic
    // (same seed = same output, even if we don't know what that output is)
    random::setSeed(folly::to_narrow(seed));
    random::RandomSkipTracker randomSkip2(0.5);
    mutation.randomSkip = &randomSkip2;
    auto actual2 = RowReader::projectColumns(input, spec, &mutation);
    test::assertEqualVectors(actual, actual2);
  }

  EXPECT_NE(0, numNonZero);
  EXPECT_NE(0, numNonMax);
}

TEST_F(ReaderTest, rowRangeEmpty) {
  // Empty when startRow >= endRow
  EXPECT_TRUE((RowRange{0, 0}.empty()));
  EXPECT_TRUE((RowRange{5, 5}.empty()));
  EXPECT_TRUE((RowRange{10, 5}.empty()));

  // Not empty when startRow < endRow
  EXPECT_FALSE((RowRange{0, 1}.empty()));
  EXPECT_FALSE((RowRange{0, 10}.empty()));
  EXPECT_FALSE((RowRange{5, 10}.empty()));
}

} // namespace
} // namespace facebook::velox::dwio::common
