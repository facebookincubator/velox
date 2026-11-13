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
#include "velox/connectors/hive/HiveBucketPruning.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::connector::hive {
namespace {

// The expected buckets follow Hive's hash. TINYINT, SMALLINT and INTEGER
// values hash to themselves, a BIGINT to its high and low 32 bits XORed, a
// BOOLEAN to 1 or 0, and a string to h = 31 * h + byte. A row of bucket
// columns hashes to h = 31 * h + column hash. The bucket is
// (hash & INT32_MAX) % numBuckets.
class HiveBucketPruningTest : public ::testing::Test,
                              public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::optional<std::vector<int32_t>> buckets(
      const common::Filter& filter,
      const TypePtr& type,
      int32_t numBuckets) {
    return HiveBucketPruning::buckets({&filter}, {type}, numBuckets);
  }
};

TEST_F(HiveBucketPruningTest, integers) {
  EXPECT_THAT(
      buckets(*exec::equal(42), INTEGER(), 8),
      testing::Optional(testing::UnorderedElementsAre(2)));
  EXPECT_THAT(
      buckets(*exec::in({1, 6, 9}), BIGINT(), 4),
      testing::Optional(testing::UnorderedElementsAre(1, 2)));
  EXPECT_THAT(
      buckets(*exec::between(5, 7), SMALLINT(), 4),
      testing::Optional(testing::UnorderedElementsAre(1, 2, 3)));

  // A value list stored in a hash table: 1'000'000 is in bucket 0.
  EXPECT_THAT(
      buckets(*exec::in({1, 1'000'000}), BIGINT(), 8),
      testing::Optional(testing::UnorderedElementsAre(0, 1)));

  // -1 hashes to 0xFFFFFFFF as an INTEGER, so its bucket is
  // 0x7FFFFFFF % 8 = 7. As a BIGINT, its high and low 32 bits cancel out.
  EXPECT_THAT(
      buckets(*exec::equal(-1), INTEGER(), 8),
      testing::Optional(testing::UnorderedElementsAre(7)));
  EXPECT_THAT(
      buckets(*exec::equal(-1), BIGINT(), 8),
      testing::Optional(testing::UnorderedElementsAre(0)));

  // A TINYINT column holds only 120 to 127 of the range.
  EXPECT_THAT(
      buckets(*exec::greaterThanOrEqual(120), TINYINT(), 16),
      testing::Optional(
          testing::UnorderedElementsAre(8, 9, 10, 11, 12, 13, 14, 15)));

  // A TINYINT column can't hold 1'000.
  EXPECT_THAT(
      buckets(*exec::in({3, 1'000}), TINYINT(), 8),
      testing::Optional(testing::UnorderedElementsAre(3)));
}

TEST_F(HiveBucketPruningTest, strings) {
  EXPECT_THAT(
      buckets(*exec::in({"a", "b"}), VARCHAR(), 4),
      testing::Optional(testing::UnorderedElementsAre(1, 2)));
  EXPECT_THAT(
      buckets(*exec::between("a", "a"), VARCHAR(), 4),
      testing::Optional(testing::UnorderedElementsAre(1)));
}

TEST_F(HiveBucketPruningTest, booleans) {
  EXPECT_THAT(
      buckets(*exec::boolEqual(true), BOOLEAN(), 4),
      testing::Optional(testing::UnorderedElementsAre(1)));
  EXPECT_THAT(
      buckets(*exec::boolEqual(false), BOOLEAN(), 4),
      testing::Optional(testing::UnorderedElementsAre(0)));
}

// The buckets are the ones HivePartitionFunction, which the writer uses,
// gives rows with the filter's values.
TEST_F(HiveBucketPruningTest, matchesWriter) {
  constexpr int32_t kNumBuckets{7};
  auto check = [&](const common::Filter& filter, const VectorPtr& values) {
    SCOPED_TRACE(values->toString(0, values->size()));
    HivePartitionFunction partitionFunction(kNumBuckets, {0});
    std::vector<uint32_t> partitions;
    partitionFunction.partition(*makeRowVector({values}), partitions);
    std::vector<int32_t> expected(partitions.begin(), partitions.end());
    std::sort(expected.begin(), expected.end());
    expected.erase(
        std::unique(expected.begin(), expected.end()), expected.end());

    EXPECT_THAT(
        buckets(filter, values->type(), kNumBuckets),
        testing::Optional(testing::UnorderedElementsAreArray(expected)));
  };

  const std::vector<int64_t> small{-100, -1, 0, 1, 7, 100};
  check(*exec::in(small), makeFlatVector<int8_t>({-100, -1, 0, 1, 7, 100}));
  check(*exec::in(small), makeFlatVector<int16_t>({-100, -1, 0, 1, 7, 100}));
  check(*exec::in(small), makeFlatVector<int32_t>({-100, -1, 0, 1, 7, 100}));

  const std::vector<int64_t> large{
      -5'000'000'000, -1, 0, 1, 5'000'000'000, 123'456'789'012};
  check(*exec::in(large), makeFlatVector<int64_t>(large));

  const std::vector<std::string> strings{"", "a", "abc", "x y"};
  check(*exec::in(strings), makeFlatVector<std::string>(strings));

  check(*exec::boolEqual(true), makeFlatVector<bool>({true}));
  check(*exec::boolEqual(false), makeFlatVector<bool>({false}));
}

TEST_F(HiveBucketPruningTest, multipleColumns) {
  // Every combination of the columns' values: (1, 'a') hashes to
  // 31 * 1 + 97 = 128, (1, 'b') to 129, (2, 'a') to 159 and (2, 'b') to 160.
  // 128 and 160 share bucket 0.
  {
    auto first = exec::in({1, 2});
    auto second = exec::in({"a", "b"});
    EXPECT_THAT(
        HiveBucketPruning::buckets(
            {first.get(), second.get()}, {BIGINT(), VARCHAR()}, 16),
        testing::Optional(testing::UnorderedElementsAre(0, 1, 15)));
  }

  // Three columns: (1, 'a', c) hashes to 31 * 128 + c = 3968 + c, and
  // (2, 'a', c) to 31 * 159 + c = 4929 + c.
  {
    auto first = exec::in({1, 2});
    auto second = exec::equal("a");
    auto third = exec::in({0, 5});
    EXPECT_THAT(
        HiveBucketPruning::buckets(
            {first.get(), second.get(), third.get()},
            {BIGINT(), VARCHAR(), BIGINT()},
            64),
        testing::Optional(testing::UnorderedElementsAre(0, 1, 5, 6)));
  }

  // The bucket depends on the order of the columns: (2, 'a') hashes to
  // 31 * 2 + 97 = 159, and ('a', 2) to 31 * 97 + 2 = 3009.
  {
    auto number = exec::equal(2);
    auto string = exec::equal("a");
    EXPECT_THAT(
        HiveBucketPruning::buckets(
            {number.get(), string.get()}, {BIGINT(), VARCHAR()}, 16),
        testing::Optional(testing::UnorderedElementsAre(15)));
    EXPECT_THAT(
        HiveBucketPruning::buckets(
            {string.get(), number.get()}, {VARCHAR(), BIGINT()}, 16),
        testing::Optional(testing::UnorderedElementsAre(1)));
  }
}

TEST_F(HiveBucketPruningTest, noRows) {
  common::AlwaysFalse alwaysFalse;
  EXPECT_THAT(
      buckets(alwaysFalse, BIGINT(), 4), testing::Optional(testing::IsEmpty()));

  // Whatever the other bucket columns' filters are.
  EXPECT_THAT(
      HiveBucketPruning::buckets(
          {nullptr, &alwaysFalse}, {BIGINT(), BIGINT()}, 4),
      testing::Optional(testing::IsEmpty()));

  // A filter that allows no value of the column's type.
  EXPECT_THAT(
      buckets(*exec::greaterThanOrEqual(200), TINYINT(), 4),
      testing::Optional(testing::IsEmpty()));
}

TEST_F(HiveBucketPruningTest, notNarrowed) {
  // A bucket column without a filter.
  {
    auto filter = exec::equal(1);
    EXPECT_EQ(
        HiveBucketPruning::buckets(
            {filter.get(), nullptr}, {BIGINT(), BIGINT()}, 4),
        std::nullopt);
  }

  // A filter that passes NULL: the bucket of NULL may not be among the buckets
  // of the filter's values.
  EXPECT_EQ(
      buckets(*exec::equal(1, /*nullAllowed=*/true), BIGINT(), 4),
      std::nullopt);

  // A range and a value list over the limit.
  EXPECT_EQ(
      buckets(
          *exec::between(0, HiveBucketPruning::kMaxCombinations), BIGINT(), 4),
      std::nullopt);
  {
    std::vector<int64_t> values;
    for (auto i = 0; i <= HiveBucketPruning::kMaxCombinations; ++i) {
      values.push_back(i * 2);
    }
    EXPECT_EQ(buckets(*exec::in(values), BIGINT(), 4), std::nullopt);
  }

  // Combinations over the limit, though each column's values are within it.
  {
    auto filter = exec::between(1, 200);
    EXPECT_EQ(
        HiveBucketPruning::buckets(
            {filter.get(), filter.get()}, {BIGINT(), BIGINT()}, 4),
        std::nullopt);
  }

  // Filters whose values can't be listed.
  EXPECT_EQ(buckets(*exec::greaterThanOrEqual(0), BIGINT(), 4), std::nullopt);
  EXPECT_EQ(buckets(*exec::between("a", "b"), VARCHAR(), 4), std::nullopt);
  EXPECT_EQ(buckets(*exec::betweenDouble(1.0, 1.0), DOUBLE(), 4), std::nullopt);
}

} // namespace
} // namespace facebook::velox::connector::hive
