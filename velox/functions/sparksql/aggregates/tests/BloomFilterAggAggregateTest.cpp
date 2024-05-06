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

#include "velox/common/base/BloomFilter.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

namespace facebook::velox::functions::aggregate::sparksql::test {
namespace {
class BloomFilterAggAggregateTest
    : public aggregate::test::AggregationTestBase {
 public:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("");
    allowInputShuffle();
  }

  VectorPtr getSerializedBloomFilter(
      int32_t capacity,
      bool shouldInsertValue = true) {
    BloomFilter bloomFilter;
    bloomFilter.reset(capacity);
    if (shouldInsertValue) {
      for (auto i = 0; i < 9; ++i) {
        bloomFilter.insert(folly::hasher<int64_t>()(i));
      }
    }
    std::string data;
    data.resize(bloomFilter.serializedSize());
    bloomFilter.serialize(data.data());
    return makeConstant(StringView(data), 1, VARBINARY());
  }
};
} // namespace

TEST_F(BloomFilterAggAggregateTest, basic) {
  auto vectors = {makeRowVector({makeFlatVector<int64_t>(
      100, [](vector_size_t row) { return row % 9; })})};
  auto expected = {makeRowVector({getSerializedBloomFilter(4)})};
  testAggregations(vectors, {}, {"bloom_filter_agg(c0, 5, 64)"}, expected);
}

TEST_F(BloomFilterAggAggregateTest, bloomFilterAggArgument) {
  auto vectors = {makeRowVector({makeFlatVector<int64_t>(
      100, [](vector_size_t row) { return row % 9; })})};

  auto expected1 = {makeRowVector({getSerializedBloomFilter(3)})};
  testAggregations(vectors, {}, {"bloom_filter_agg(c0, 6)"}, expected1);

  // This capacity is kMaxNumBits / 16.
  auto expected2 = {makeRowVector({getSerializedBloomFilter(262144)})};
  testAggregations(vectors, {}, {"bloom_filter_agg(c0)"}, expected2);
}

TEST_F(BloomFilterAggAggregateTest, emptyInput) {
  auto vectors = {makeRowVector({makeFlatVector<int64_t>({})})};
  auto expected = {makeRowVector({makeNullConstant(TypeKind::VARBINARY, 1)})};
  testAggregations(vectors, {}, {"bloom_filter_agg(c0, 5, 64)"}, expected);
}

TEST_F(BloomFilterAggAggregateTest, withNullInput) {
  auto vectors = {makeRowVector({makeNullableFlatVector<int64_t>(
      {0,
       1,
       2,
       std::nullopt,
       3,
       4,
       5,
       std::nullopt,
       std::nullopt,
       6,
       7,
       8,
       std::nullopt,
       std::nullopt})})};
  // The expected value is the serialized bloom filter
  auto expected = {makeRowVector({getSerializedBloomFilter(4)})};
  testAggregations(vectors, {}, {"bloom_filter_agg(c0, 5, 64)"}, expected);
}

TEST_F(BloomFilterAggAggregateTest, allNullInput) {
  // The bloom_filter_agg do not support test streaming when it's inputs are all
  // NULL, because the result of streaming will be null
  disableTestStreaming();
  // The expected value is the serialized bloom filter
  auto expected = {makeRowVector({getSerializedBloomFilter(4, false)})};
  auto vectors1 = {makeRowVector(
      {makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt})})};
  testAggregations(vectors1, {}, {"bloom_filter_agg(c0, 5, 64)"}, expected);
  auto vectors2 = {makeRowVector({makeNullableFlatVector<int64_t>(
      {std::nullopt, std::nullopt, std::nullopt})})};
  testAggregations(vectors2, {}, {"bloom_filter_agg(c0, 5, 64)"}, expected);
  auto vectors3 = {makeRowVector({makeNullableFlatVector<int64_t>(
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt})})};
  testAggregations(vectors3, {}, {"bloom_filter_agg(c0, 5, 64)"}, expected);
}

TEST_F(BloomFilterAggAggregateTest, config) {
  auto vector = {makeRowVector({makeFlatVector<int64_t>(
      100, [](vector_size_t row) { return row % 9; })})};
  std::vector<RowVectorPtr> expected = {
      makeRowVector({getSerializedBloomFilter(100)})};

  // This config will decide the bloom filter capacity, the expected value is
  // the serialized bloom filter, it should be consistent.
  testAggregations(
      vector,
      {},
      {"bloom_filter_agg(c0)"},
      expected,
      {{core::QueryConfig::kSparkBloomFilterMaxNumBits, "1600"}});

  // Test fails without setting the config.
  auto planNode = exec::test::PlanBuilder(pool())
                      .values(vector)
                      .partialAggregation({}, {"bloom_filter_agg(c0)"})
                      .finalAggregation()
                      .planNode();
  auto actual = exec::test::AssertQueryBuilder(planNode).copyResults(pool());
  EXPECT_FALSE(
      expected[0]->childAt(0)->equalValueAt(actual->childAt(0).get(), 0, 0));
}
} // namespace facebook::velox::functions::aggregate::sparksql::test
