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

#include <map>

#include "velox/functions/lib/KHyperLogLog.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/prestosql/types/KHyperLogLogRegistration.h"
#include "velox/functions/prestosql/types/KHyperLogLogType.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;
using namespace facebook::velox::memory;

namespace facebook::velox::aggregate::test {
namespace {
class KHyperLogLogAggregateTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerKHyperLogLogType();
    // Disable tests which do raw binary comparison.
    disableTestStreaming();
    disableTestIncremental();
    pool_ = facebook::velox::memory::memoryManager()->addLeafPool();
    hsa_ = std::make_unique<HashStringAllocator>(pool_.get());
    allocator_ = hsa_.get();
  }
  void TearDown() override {
    // Clean up allocator before pool.
    hsa_.reset();
    pool_.reset();
    AggregationTestBase::TearDown();
  }

  // Helper to create KHLL with test data
  std::string createKHLL(
      const std::vector<int64_t>& values,
      const std::vector<int64_t>& uiis) {
    EXPECT_EQ(values.size(), uiis.size());
    auto khll = std::make_unique<
        common::hll::KHyperLogLog<int64_t, HashStringAllocator>>(allocator_);

    for (size_t i = 0; i < values.size(); ++i) {
      khll->add(values[i], uiis[i]);
    }

    std::string serialized(khll->estimatedSerializedSize(), '\0');
    khll->serialize(serialized.data());
    return serialized;
  }

  // Extract cardinality from KHLL varbinary.
  int64_t getCardinality(const StringView& khllData) {
    auto khll = common::hll::KHyperLogLog<int32_t, MemoryPool>::deserialize(
        khllData.data(), khllData.size(), pool_.get());
    VELOX_CHECK(!khll.hasError());
    return khll.value()->cardinality();
  }

  // Helper to verify aggregation results have expected cardinalities.
  // Uses testAggregations with a custom assert function to test all
  // aggregation plan paths.
  void testAggregationWithCardinality(
      const std::vector<RowVectorPtr>& data,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::map<int32_t, int64_t>& keysToExpectedCardinalities) {
    testAggregations(
        [&](PlanBuilder& builder) { builder.values(data); },
        groupingKeys,
        aggregates,
        {},
        [&](AssertQueryBuilder& builder) {
          std::shared_ptr<exec::Task> task;
          auto result = builder.copyResults(pool_.get(), task);
          EXPECT_EQ(result->size(), keysToExpectedCardinalities.size());

          if (groupingKeys.empty()) {
            // Global aggregation - single row result
            EXPECT_EQ(keysToExpectedCardinalities.size(), 1);
            auto expectedCardinality =
                keysToExpectedCardinalities.begin()->second;

            auto khllVector = result->childAt(0)->as<FlatVector<StringView>>();
            EXPECT_EQ(
                getCardinality(khllVector->valueAt(0)), expectedCardinality);
          } else {
            // Group-by aggregation - map keys to cardinalities
            auto khllVector = result->childAt(groupingKeys.size())
                                  ->as<FlatVector<StringView>>();
            auto keyVector = result->childAt(0)->as<FlatVector<int32_t>>();

            // Build map of key -> actual cardinality
            std::map<int32_t, int64_t> actualCardinalities;
            for (size_t i = 0; i < result->size(); ++i) {
              int32_t key = keyVector->valueAt(i);
              if (!khllVector->isNullAt(i)) {
                actualCardinalities[key] =
                    getCardinality(khllVector->valueAt(i));
              }
            }

            // Verify each key has expected cardinality
            for (const auto& [key, expectedCardinality] :
                 keysToExpectedCardinalities) {
              auto it = actualCardinalities.find(key);
              EXPECT_TRUE(it != actualCardinalities.end())
                  << "Missing result for key " << key;
              EXPECT_EQ(it->second, expectedCardinality)
                  << "Cardinality mismatch for key " << key;
            }
          }
          return task;
        });
  }

 protected:
  std::shared_ptr<MemoryPool> pool_;
  std::unique_ptr<HashStringAllocator> hsa_;
  HashStringAllocator* allocator_{};
};

TEST_F(KHyperLogLogAggregateTest, globalAggIntegers) {
  vector_size_t size = 1'000;
  auto values =
      makeFlatVector<int64_t>(size, [](auto row) { return row % 17; });
  auto uii = makeFlatVector<int64_t>(size, [](auto /*row*/) { return 1; });

  testAggregationWithCardinality(
      {makeRowVector({values, uii})},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {{0, 17}});

  // Test high cardinality
  values = makeFlatVector<int64_t>(size, [](auto row) { return row; });
  uii = makeFlatVector<int64_t>(size, [](auto /*row*/) { return 1; });

  testAggregationWithCardinality(
      {makeRowVector({values, uii})},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {{0, 1000}});
}

TEST_F(KHyperLogLogAggregateTest, globalAggIntegersWithNulls) {
  testAggregationWithCardinality(
      {makeRowVector({
          makeNullableFlatVector<int64_t>({1, 2, std::nullopt, 4, 5}),
          makeNullableFlatVector<int64_t>({10, 10, 20, std::nullopt, 30}),
      })},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {{0, 3}});

  // All nulls
  auto expected = makeRowVector({makeNullConstant(TypeKind::VARBINARY, 1)});
  testAggregations(
      {makeRowVector({
          makeNullableFlatVector<int64_t>(
              {std::nullopt, std::nullopt, std::nullopt}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt, std::nullopt, std::nullopt}),
      })},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, globalAggStrings) {
  vector_size_t size = 1'000;
  static const std::vector<std::string> kFruits = {
      "apple",
      "banana",
      "cherry",
      "unknown fruit with a long name",
      "watermelon"};

  auto values = makeFlatVector<StringView>(size, [&](auto row) {
    return StringView(kFruits[row % kFruits.size()]);
  });
  auto uii = makeFlatVector<int64_t>(size, [](auto /*row*/) { return 1; });

  testAggregationWithCardinality(
      {makeRowVector({values, uii})},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {{0, static_cast<int64_t>(kFruits.size())}});
}

TEST_F(KHyperLogLogAggregateTest, globalAggVarbinary) {
  vector_size_t size = 1'000;
  static const std::vector<std::string> kFruits = {
      "apple", "banana", "unknown fruit with a long name", "watermelon"};

  auto values = makeFlatVector<std::string>(
      size,
      [&](auto row) { return kFruits[row % kFruits.size()]; },
      nullptr,
      VARBINARY());
  auto uii = makeFlatVector<int64_t>(size, [](auto /*row*/) { return 1; });

  testAggregationWithCardinality(
      {makeRowVector({values, uii})},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {{0, static_cast<int64_t>(kFruits.size())}});
}

TEST_F(KHyperLogLogAggregateTest, globalAggTimeStamp) {
  auto data = makeFlatVector<Timestamp>(
      1'000, [](auto row) { return Timestamp::fromMillis(row); });
  auto uii = makeFlatVector<int64_t>(1'000, [](auto /*row*/) { return 1; });

  testAggregationWithCardinality(
      {makeRowVector({data, uii})},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {{0, 1000}});
}

TEST_F(KHyperLogLogAggregateTest, globalAggBoolean) {
  vector_size_t size = 2'000;
  auto uii = makeFlatVector<int64_t>(size, [](auto /*row*/) { return 1; });

  auto values =
      makeFlatVector<bool>(size, [](auto row) { return row % 2 == 0; });
  testAggregationWithCardinality(
      {makeRowVector({values, uii})},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {{0, 2}});
}

TEST_F(KHyperLogLogAggregateTest, groupByIntegers) {
  vector_size_t size = 1'000;
  auto keys = makeFlatVector<int32_t>(size, [](auto row) { return row % 2; });
  auto values = makeFlatVector<int64_t>(
      size, [](auto row) { return row % 2 == 0 ? row % 17 : row % 23; });
  auto uii = makeFlatVector<int64_t>(size, [](auto /*row*/) { return 1; });

  testAggregationWithCardinality(
      {makeRowVector({keys, values, uii})},
      {"c0"},
      {"khyperloglog_agg(c1, c2)"},
      {{0, 17}, {1, 23}});
}

TEST_F(KHyperLogLogAggregateTest, groupByHighCardinalityIntegers) {
  vector_size_t size = 1'000;
  auto keys = makeFlatVector<int32_t>(size, [](auto row) { return row % 2; });
  auto values = makeFlatVector<int64_t>(size, [](auto row) { return row; });
  auto uii = makeFlatVector<int64_t>(size, [](auto /*row*/) { return 1; });

  testAggregationWithCardinality(
      {makeRowVector({keys, values, uii})},
      {"c0"},
      {"khyperloglog_agg(c1, c2)"},
      {{0, 500}, {1, 500}});
}

TEST_F(KHyperLogLogAggregateTest, groupByAllNulls) {
  vector_size_t size = 1'000;
  auto keys = makeFlatVector<int32_t>(size, [](auto row) { return row % 2; });
  auto values = makeFlatVector<int32_t>(
      size, [](auto row) { return row % 2 == 0 ? 27 : row % 3; }, nullEvery(2));
  auto uii = makeFlatVector<int64_t>(size, [](auto /*row*/) { return 1; });

  auto vectors = makeRowVector({keys, values, uii});

  // When a group has all nulls, khyperloglog_agg returns NULL, not KHLL with
  // cardinality 0
  auto result = AssertQueryBuilder(
                    PlanBuilder()
                        .values({vectors})
                        .singleAggregation({"c0"}, {"khyperloglog_agg(c1, c2)"})
                        .planNode())
                    .copyResults(pool_.get());

  ASSERT_EQ(result->size(), 2);
  auto khllVector = result->childAt(1)->as<FlatVector<StringView>>();

  // At group 0, all values are null, so result is NULL.
  ASSERT_TRUE(khllVector->isNullAt(0));

  // Group 1 should have (0, 1, 2)
  ASSERT_FALSE(khllVector->isNullAt(1));
  EXPECT_EQ(getCardinality(khllVector->valueAt(1)), 3);
}

TEST_F(KHyperLogLogAggregateTest, mergeKHyperLogLog) {
  // Create two separate KHLL sketches
  auto khll1 = createKHLL({1, 2, 3, 4, 5}, {1, 1, 1, 1, 1});
  auto khll2 = createKHLL({6, 7, 8, 9, 10}, {1, 1, 1, 1, 1});

  auto khllData = makeFlatVector<StringView>(
      {StringView(khll1), StringView(khll2)}, KHYPERLOGLOG());

  // Merge the two sketches - should have cardinality 10
  testAggregationWithCardinality(
      {makeRowVector({khllData})}, {}, {"merge(c0)"}, {{0, 10}});
}

TEST_F(KHyperLogLogAggregateTest, mergeToIntermediate) {
  constexpr int kSize = 1000;
  auto input = makeRowVector({
      makeFlatVector<int32_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, [](auto /*row*/) { return 1; }),
      makeFlatVector<int64_t>(kSize, [](auto /*row*/) { return 1; }),
  });

  // Create KHLL sketches per group
  auto plan = PlanBuilder()
                  .values({input})
                  .singleAggregation({"c0"}, {"khyperloglog_agg(c1, c2)"})
                  .planNode();

  auto digests = split(AssertQueryBuilder(plan).copyResults(pool()), 2);

  // Merge the split digests - each group should still have cardinality 1
  std::map<int32_t, int64_t> expectedCardinalities;
  for (int32_t i = 0; i < kSize; ++i) {
    expectedCardinalities[i] = 1;
  }
  testAggregationWithCardinality(
      digests, {"c0"}, {"merge(a0)"}, expectedCardinalities);
}

} // namespace
} // namespace facebook::velox::aggregate::test
