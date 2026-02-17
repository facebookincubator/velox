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

// Test that projectColumns preserves top level nulls when the input RowVector
// has null rows.
TEST_F(ReaderTest, projectColumnsTopLevelNulls) {
  constexpr int kSize = 10;
  // All nulls
  {
    SCOPED_TRACE("All nulls");
    auto child = makeFlatVector<int64_t>(kSize, folly::identity);
    auto input = makeRowVector({child});

    auto nulls = AlignedBuffer::allocate<bool>(kSize, pool());
    auto* rawNulls = nulls->asMutable<uint64_t>();
    // Set all bits to 0 (all null)
    memset(rawNulls, 0, bits::nbytes(kSize));
    input->setNulls(nulls);

    ASSERT_EQ(BaseVector::countNulls(input->nulls(), kSize), kSize);

    common::ScanSpec spec("<root>");
    spec.addAllChildFields(*input->type());

    auto actual = RowReader::projectColumns(input, spec, nullptr);

    ASSERT_NE(actual->nulls(), nullptr);
    EXPECT_EQ(BaseVector::countNulls(actual->nulls(), actual->size()), kSize);

    // Verify the output has the same size
    EXPECT_EQ(actual->size(), kSize);
  }

  // Partial nulls.
  {
    SCOPED_TRACE("Partial nulls");
    auto child = makeFlatVector<int64_t>(kSize, folly::identity);
    auto input = makeRowVector({child});

    auto nulls = AlignedBuffer::allocate<bool>(kSize, pool());
    auto* rawNulls = nulls->asMutable<uint64_t>();

    // Set rows 0, 2, 4, 6, 8 as null (even indices)
    memset(rawNulls, 0xFF, bits::nbytes(kSize));
    for (int i = 0; i < kSize; i += 2) {
      bits::setNull(rawNulls, i);
    }
    input->setNulls(nulls);

    ASSERT_EQ(BaseVector::countNulls(input->nulls(), kSize), 5); // 5 null rows

    common::ScanSpec spec("<root>");
    spec.addAllChildFields(*input->type());

    // Test without mutation (no filtering)
    auto actual = RowReader::projectColumns(input, spec, nullptr);

    // Verify nulls are preserved
    ASSERT_NE(actual->nulls(), nullptr);
    EXPECT_EQ(BaseVector::countNulls(actual->nulls(), actual->size()), 5);
    EXPECT_EQ(actual->size(), kSize);

    // Verify specific null positions
    for (int i = 0; i < kSize; ++i) {
      EXPECT_EQ(actual->isNullAt(i), (i % 2 == 0))
          << "Row " << i << " null status mismatch";
    }
  }

  // Partial nulls with constant encoding child
  {
    SCOPED_TRACE("Constant encoding child");
    // Create a constant vector (all values are the same)
    auto constantChild =
        BaseVector::createConstant(INTEGER(), 42, kSize, pool());
    auto input = makeRowVector({constantChild});

    auto nulls = AlignedBuffer::allocate<bool>(kSize, pool());
    auto* rawNulls = nulls->asMutable<uint64_t>();

    // Set rows 0, 2, 4, 6, 8 as null (even indices)
    memset(rawNulls, 0xFF, bits::nbytes(kSize));
    for (int i = 0; i < kSize; i += 2) {
      bits::setNull(rawNulls, i);
    }
    input->setNulls(nulls);

    ASSERT_EQ(BaseVector::countNulls(input->nulls(), kSize), 5); // 5 null rows
    ASSERT_TRUE(constantChild->isConstantEncoding());

    common::ScanSpec spec("<root>");
    spec.addAllChildFields(*input->type());

    auto actual = RowReader::projectColumns(input, spec, nullptr);

    // Verify nulls are preserved
    ASSERT_NE(actual->nulls(), nullptr);
    EXPECT_EQ(BaseVector::countNulls(actual->nulls(), actual->size()), 5);
    EXPECT_EQ(actual->size(), kSize);

    // Verify specific null positions
    for (int i = 0; i < kSize; ++i) {
      EXPECT_EQ(actual->isNullAt(i), (i % 2 == 0))
          << "Row " << i << " null status mismatch";
    }
  }

  // Partial nulls with dictionary encoding child
  {
    SCOPED_TRACE("Dictionary encoding child");
    // Create a dictionary vector wrapping a flat vector
    auto baseValues = makeFlatVector<int64_t>({10, 20, 30, 40, 50});
    // Create indices that map to the base values
    auto indices = makeIndices(kSize, [](auto i) { return i % 5; });
    auto dictionaryChild =
        BaseVector::wrapInDictionary(nullptr, indices, kSize, baseValues);
    auto input = makeRowVector({dictionaryChild});

    auto nulls = AlignedBuffer::allocate<bool>(kSize, pool());
    auto* rawNulls = nulls->asMutable<uint64_t>();

    // Set rows 0, 2, 4, 6, 8 as null (even indices)
    memset(rawNulls, 0xFF, bits::nbytes(kSize));
    for (int i = 0; i < kSize; i += 2) {
      bits::setNull(rawNulls, i);
    }
    input->setNulls(nulls);

    ASSERT_EQ(BaseVector::countNulls(input->nulls(), kSize), 5); // 5 null rows
    ASSERT_EQ(dictionaryChild->encoding(), VectorEncoding::Simple::DICTIONARY);

    common::ScanSpec spec("<root>");
    spec.addAllChildFields(*input->type());

    auto actual = RowReader::projectColumns(input, spec, nullptr);

    // Verify nulls are preserved
    ASSERT_NE(actual->nulls(), nullptr);
    EXPECT_EQ(BaseVector::countNulls(actual->nulls(), actual->size()), 5);
    EXPECT_EQ(actual->size(), kSize);

    // Verify specific null positions
    for (int i = 0; i < kSize; ++i) {
      EXPECT_EQ(actual->isNullAt(i), (i % 2 == 0))
          << "Row " << i << " null status mismatch";
    }
  }

  // Two columns: constant encoding and dictionary encoding
  {
    SCOPED_TRACE("Two columns: constant and dictionary encoding");
    // Create a constant vector
    auto constantChild =
        BaseVector::createConstant(INTEGER(), 42, kSize, pool());

    // Create a dictionary vector
    auto baseValues = makeFlatVector<int64_t>({100, 200, 300, 400, 500});
    auto indices = makeIndices(kSize, [](auto i) { return i % 5; });
    auto dictionaryChild =
        BaseVector::wrapInDictionary(nullptr, indices, kSize, baseValues);

    auto input = makeRowVector({constantChild, dictionaryChild});

    auto nulls = AlignedBuffer::allocate<bool>(kSize, pool());
    auto* rawNulls = nulls->asMutable<uint64_t>();

    // Set rows 0, 2, 4, 6, 8 as null (even indices)
    memset(rawNulls, 0xFF, bits::nbytes(kSize));
    for (int i = 0; i < kSize; i += 2) {
      bits::setNull(rawNulls, i);
    }
    input->setNulls(nulls);

    ASSERT_EQ(BaseVector::countNulls(input->nulls(), kSize), 5);
    ASSERT_TRUE(constantChild->isConstantEncoding());
    ASSERT_EQ(dictionaryChild->encoding(), VectorEncoding::Simple::DICTIONARY);

    common::ScanSpec spec("<root>");
    spec.addAllChildFields(*input->type());

    auto actual = RowReader::projectColumns(input, spec, nullptr);

    // Verify nulls are preserved
    ASSERT_NE(actual->nulls(), nullptr);
    EXPECT_EQ(BaseVector::countNulls(actual->nulls(), actual->size()), 5);
    EXPECT_EQ(actual->size(), kSize);

    // Verify both children exist
    auto rowResult = actual->as<RowVector>();
    ASSERT_EQ(rowResult->childrenSize(), 2);

    // Verify specific null positions
    for (int i = 0; i < kSize; ++i) {
      EXPECT_EQ(actual->isNullAt(i), (i % 2 == 0))
          << "Row " << i << " null status mismatch";
    }
  }
}

// Test that projectColumns correctly filters row-level nulls with mutation
TEST_F(ReaderTest, projectColumnsFiltersRowNullsWithMutation) {
  constexpr int kSize = 10;

  // Create a RowVector with some null rows
  auto child = makeFlatVector<int64_t>(kSize, folly::identity);
  auto input = makeRowVector({child});

  // Set rows 0, 2, 4, 6, 8 as null (even indices)
  auto nulls = AlignedBuffer::allocate<bool>(kSize, pool());
  auto* rawNulls = nulls->asMutable<uint64_t>();
  memset(rawNulls, 0xFF, bits::nbytes(kSize));
  for (int i = 0; i < kSize; i += 2) {
    bits::setNull(rawNulls, i);
  }
  input->setNulls(nulls);

  common::ScanSpec spec("<root>");
  spec.addAllChildFields(*input->type());

  // Delete rows 1 and 3 (odd indices, which are not null)
  std::vector<uint64_t> deleted(bits::nwords(kSize));
  bits::setBit(deleted.data(), 1);
  bits::setBit(deleted.data(), 3);
  Mutation mutation;
  mutation.deletedRows = deleted.data();

  auto actual = RowReader::projectColumns(input, spec, &mutation);

  // 8 rows should remain (10 - 2 deleted)
  EXPECT_EQ(actual->size(), 8);

  // Verify nulls buffer exists
  ASSERT_NE(actual->nulls(), nullptr);

  // After filtering, the remaining rows are: 0, 2, 4, 5, 6, 7, 8, 9
  // Of these, 0, 2, 4, 6, 8 were null in the original (indices 0, 1, 2, 4, 6)
  // So in the output: positions 0, 1, 2, 4, 6 should be null
  EXPECT_TRUE(actual->isNullAt(0)); // was row 0
  EXPECT_TRUE(actual->isNullAt(1)); // was row 2
  EXPECT_TRUE(actual->isNullAt(2)); // was row 4
  EXPECT_FALSE(actual->isNullAt(3)); // was row 5
  EXPECT_TRUE(actual->isNullAt(4)); // was row 6
  EXPECT_FALSE(actual->isNullAt(5)); // was row 7
  EXPECT_TRUE(actual->isNullAt(6)); // was row 8
  EXPECT_FALSE(actual->isNullAt(7)); // was row 9
}

} // namespace
} // namespace facebook::velox::dwio::common
