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

#include "velox/common/base/SplitBlockBloomFilter.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/aggregates/Register.h"

namespace facebook::velox::functions::aggregate::sparksql::test {

using functions::sparksql::SparkQueryConfig;

namespace {
class BloomFilterAggAggregateTest
    : public aggregate::test::AggregationTestBase {
 public:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("");
  }

  static int32_t numBlocksForBits(int64_t numBits) {
    return static_cast<int32_t>(std::max<int64_t>(
        1, numBits / (8 * sizeof(SplitBlockBloomFilter::Block))));
  }

  VectorPtr getSerializedBloomFilter(int32_t numBlocks) {
    std::vector<SplitBlockBloomFilter::Block> blocks(numBlocks);
    SplitBlockBloomFilter bloomFilter(blocks);
    for (auto i = 0; i < 9; ++i) {
      bloomFilter.insert(folly::hasher<int64_t>()(i));
    }
    std::string data(
        reinterpret_cast<const char*>(blocks.data()),
        blocks.size() * sizeof(SplitBlockBloomFilter::Block));
    return makeConstant(StringView(data), 1, VARBINARY());
  }
};
} // namespace

TEST_F(BloomFilterAggAggregateTest, basic) {
  auto vectors = {makeRowVector({makeFlatVector<int64_t>(
      100, [](vector_size_t row) { return row % 9; })})};
  auto expected = {makeRowVector({getSerializedBloomFilter(1)})};
  testAggregations(vectors, {}, {"bloom_filter_agg(c0, 5, 64)"}, expected);
}

TEST_F(BloomFilterAggAggregateTest, bloomFilterAggArgument) {
  auto vectors = {makeRowVector({makeFlatVector<int64_t>(
      100, [](vector_size_t row) { return row % 9; })})};

  auto expected1 = {makeRowVector({getSerializedBloomFilter(1)})};
  testAggregations(vectors, {}, {"bloom_filter_agg(c0, 6)"}, expected1);

  auto expected2 = {
      makeRowVector({getSerializedBloomFilter(numBlocksForBits(8'388'608))})};
  testAggregations(vectors, {}, {"bloom_filter_agg(c0)"}, expected2);

  // Max bits case: bloom filter is at its largest possible size.
  auto expected3 = {
      makeRowVector({getSerializedBloomFilter(numBlocksForBits(67'108'864))})};
  testAggregations(vectors, {}, {"bloom_filter_agg(c0, 10000000)"}, expected3);
}

TEST_F(BloomFilterAggAggregateTest, emptyInput) {
  auto vectors = {makeRowVector({makeFlatVector<int64_t>({})})};
  auto expected = {makeRowVector({makeNullConstant(TypeKind::VARBINARY, 1)})};
  testAggregations(vectors, {}, {"bloom_filter_agg(c0, 5, 64)"}, expected);
}

TEST_F(BloomFilterAggAggregateTest, nullBloomFilter) {
  auto vectors = {makeRowVector({makeAllNullFlatVector<int64_t>(2)})};
  auto expectedFake = {makeRowVector(
      {makeNullableFlatVector<StringView>({std::nullopt}, VARBINARY())})};
  VELOX_ASSERT_THROW(
      testAggregations(
          vectors, {}, {"bloom_filter_agg(c0, 5, 64)"}, expectedFake),
      "First argument of bloom_filter_agg cannot be null");
}

TEST_F(BloomFilterAggAggregateTest, config) {
  auto vector = {makeRowVector({makeFlatVector<int64_t>(
      100, [](vector_size_t row) { return row % 9; })})};
  std::vector<RowVectorPtr> expected = {
      makeRowVector({getSerializedBloomFilter(numBlocksForBits(1'600))})};

  // This config will decide the bloom filter capacity, the expected value is
  // the serialized bloom filter, it should be consistent.
  testAggregations(
      vector,
      {},
      {"bloom_filter_agg(c0)"},
      expected,
      {{SparkQueryConfig::qualify(SparkQueryConfig::kBloomFilterMaxNumBits),
        "1600"}});

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
