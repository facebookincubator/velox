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

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/SetDigest.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;
using namespace facebook::velox::functions;

namespace facebook::velox::aggregate::test {

namespace {

class SetDigestAggregateTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
  }

  void testAggregateWithData(
      const std::vector<RowVectorPtr>& vectors,
      const std::string& aggregate,
      const std::string& /*groupByKeys*/,
      const std::string& /*resultColumns*/) {
    auto op = PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {aggregate})
                  .planNode();

    auto result = AssertQueryBuilder(op).copyResults(pool());

    ASSERT_EQ(result->size(), 1);
    auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
    ASSERT_FALSE(resultVector->isNullAt(0));

    auto serialized = resultVector->valueAt(0);

    auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
    SetDigest digest(allocator.get());
    digest.deserialize(serialized.data(), serialized.size());

    ASSERT_GT(digest.cardinality(), 0);
  }
};

TEST_F(SetDigestAggregateTest, basicIntegerAggregation) {
  auto vectors = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 1, 2, 2, 3, 3, 3, 3}),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"make_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigest digest(allocator.get());
  digest.deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(digest.cardinality(), 3);
  EXPECT_TRUE(digest.isExact());
}

TEST_F(SetDigestAggregateTest, withNullValues) {
  auto vectors = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {1, std::nullopt, 2, std::nullopt, 3, std::nullopt, 1}),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"make_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigest digest(allocator.get());
  digest.deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(digest.cardinality(), 3);
  EXPECT_TRUE(digest.isExact());
}

TEST_F(SetDigestAggregateTest, allNullValues) {
  auto vectors = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"make_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();

  EXPECT_TRUE(resultVector->isNullAt(0));
}

TEST_F(SetDigestAggregateTest, emptyInput) {
  auto vectors = makeRowVector({
      makeFlatVector<int64_t>({}),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"make_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();

  EXPECT_TRUE(resultVector->isNullAt(0));
}

TEST_F(SetDigestAggregateTest, stringValues) {
  auto vectors = makeRowVector({
      makeFlatVector<std::string>(
          {"apple", "banana", "apple", "cherry", "banana"}),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"make_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigest digest(allocator.get());
  digest.deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(digest.cardinality(), 3);
  EXPECT_TRUE(digest.isExact());
}

TEST_F(SetDigestAggregateTest, groupBy) {
  auto vectors = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2, 1, 2}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 10, 30}),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({"c0"}, {"make_set_digest(c1)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 2);

  auto groupVector = result->childAt(0)->as<FlatVector<int32_t>>();
  auto digestVector = result->childAt(1)->as<FlatVector<StringView>>();

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());

  for (auto i = 0; i < result->size(); ++i) {
    auto group = groupVector->valueAt(i);
    auto serialized = digestVector->valueAt(i);

    SetDigest digest(allocator.get());
    digest.deserialize(serialized.data(), serialized.size());

    if (group == 1) {
      EXPECT_EQ(digest.cardinality(), 2);
      EXPECT_TRUE(digest.isExact());
    } else if (group == 2) {
      EXPECT_EQ(digest.cardinality(), 2);
      EXPECT_TRUE(digest.isExact());
    }
  }
}

TEST_F(SetDigestAggregateTest, differentNumericTypes) {
  auto testType = [&](const std::string& typeStr, auto makeVector) {
    auto vectors = makeRowVector({makeVector});

    auto op = PlanBuilder()
                  .values({vectors})
                  .singleAggregation({}, {"make_set_digest(c0)"})
                  .planNode();

    auto result = AssertQueryBuilder(op).copyResults(pool());

    ASSERT_EQ(result->size(), 1);
    auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
    ASSERT_FALSE(resultVector->isNullAt(0));

    auto serialized = resultVector->valueAt(0);

    auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
    SetDigest digest(allocator.get());
    digest.deserialize(serialized.data(), serialized.size());

    EXPECT_EQ(digest.cardinality(), 3) << "Failed for type: " << typeStr;
  };

  testType("TINYINT", makeFlatVector<int8_t>({1, 2, 3, 1, 2}));
  testType("SMALLINT", makeFlatVector<int16_t>({1, 2, 3, 1, 2}));
  testType("INTEGER", makeFlatVector<int32_t>({1, 2, 3, 1, 2}));
  testType("BIGINT", makeFlatVector<int64_t>({1, 2, 3, 1, 2}));
}

TEST_F(SetDigestAggregateTest, largeExactSet) {
  std::vector<int64_t> values;
  values.reserve(1000);
  for (int64_t i = 0; i < 1000; ++i) {
    values.push_back(i);
  }

  auto vectors = makeRowVector({makeFlatVector<int64_t>(values)});

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"make_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigest digest(allocator.get());
  digest.deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(digest.cardinality(), 1000);
  EXPECT_TRUE(digest.isExact());
}

TEST_F(SetDigestAggregateTest, largeApproximateSet) {
  std::vector<int64_t> values;
  values.reserve(50000);
  for (int64_t i = 0; i < 50000; ++i) {
    values.push_back(i);
  }

  auto vectors = makeRowVector({makeFlatVector<int64_t>(values)});

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"make_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigest digest(allocator.get());
  digest.deserialize(serialized.data(), serialized.size());

  EXPECT_FALSE(digest.isExact());

  auto estimatedCardinality = digest.cardinality();
  double errorRate = std::abs(estimatedCardinality - 50000) / 50000.0;

  EXPECT_LT(errorRate, 0.05)
      << "HLL estimate should be within 5% of actual. "
      << "Actual: 50000, Estimated: " << estimatedCardinality;
}

TEST_F(SetDigestAggregateTest, partialAggregation) {
  auto vectors = {
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})}),
      makeRowVector({makeFlatVector<int64_t>({3, 4, 5})}),
      makeRowVector({makeFlatVector<int64_t>({5, 6, 7})}),
  };

  auto op = PlanBuilder()
                .values(vectors)
                .partialAggregation({}, {"make_set_digest(c0)"})
                .intermediateAggregation()
                .finalAggregation()
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigest digest(allocator.get());
  digest.deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(digest.cardinality(), 7);
  EXPECT_TRUE(digest.isExact());
}

TEST_F(SetDigestAggregateTest, roundTripSerialization) {
  auto vectors = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"make_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  auto serialized =
      result->childAt(0)->as<FlatVector<StringView>>()->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigest digest1(allocator.get());
  digest1.deserialize(serialized.data(), serialized.size());

  auto size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  SetDigest digest2(allocator.get());
  digest2.deserialize(buffer1.data(), size1);

  auto size2 = digest2.estimatedSerializedSize();
  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());

  EXPECT_EQ(size1, size2);
  EXPECT_EQ(
      std::string(buffer1.begin(), buffer1.end()),
      std::string(buffer2.begin(), buffer2.end()));
}

TEST_F(SetDigestAggregateTest, booleanValues) {
  auto vectors = makeRowVector({
      makeFlatVector<bool>({true, false, true, false, true}),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"make_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigest digest(allocator.get());
  digest.deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(digest.cardinality(), 2);
  EXPECT_TRUE(digest.isExact());
}

TEST_F(SetDigestAggregateTest, javaCompatibilityInteger) {
  auto vectors = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 1, 2, 2}),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"make_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigest digest(allocator.get());
  digest.deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(digest.cardinality(), 2);
  EXPECT_TRUE(digest.isExact());

  auto size = digest.estimatedSerializedSize();
  std::vector<char> buffer(size);
  digest.serialize(buffer.data());

  SetDigest digest2(allocator.get());
  digest2.deserialize(buffer.data(), size);
  EXPECT_EQ(digest2.cardinality(), 2);
}

TEST_F(SetDigestAggregateTest, javaCompatibilityString) {
  auto vectors = makeRowVector({
      makeFlatVector<std::string>({"abc", "def"}),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"make_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigest digest(allocator.get());
  digest.deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(digest.cardinality(), 2);
  EXPECT_TRUE(digest.isExact());

  auto size = digest.estimatedSerializedSize();
  std::vector<char> buffer(size);
  digest.serialize(buffer.data());

  SetDigest digest2(allocator.get());
  digest2.deserialize(buffer.data(), size);
  EXPECT_EQ(digest2.cardinality(), 2);
}

TEST_F(SetDigestAggregateTest, mergeBasic) {
  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());

  SetDigest digest1(allocator.get());
  digest1.add(1);
  digest1.add(2);
  digest1.add(3);

  SetDigest digest2(allocator.get());
  digest2.add(3);
  digest2.add(4);
  digest2.add(5);

  auto size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  auto size2 = digest2.estimatedSerializedSize();
  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());

  auto vectors = makeRowVector({
      makeFlatVector<std::string>(
          {std::string(buffer1.data(), size1),
           std::string(buffer2.data(), size2)},
          VARBINARY()),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"merge_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  SetDigest mergedDigest(allocator.get());
  mergedDigest.deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(mergedDigest.cardinality(), 5);
  EXPECT_TRUE(mergedDigest.isExact());
}

TEST_F(SetDigestAggregateTest, mergeWithNullValues) {
  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());

  SetDigest digest1(allocator.get());
  digest1.add(1);
  digest1.add(2);

  auto size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  auto vectors = makeRowVector({
      makeNullableFlatVector<std::string>(
          {std::string(buffer1.data(), size1),
           std::nullopt,
           std::string(buffer1.data(), size1)},
          VARBINARY()),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"merge_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  SetDigest mergedDigest(allocator.get());
  mergedDigest.deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(mergedDigest.cardinality(), 2);
}

TEST_F(SetDigestAggregateTest, mergeAllNulls) {
  auto vectors = makeRowVector({
      makeNullableFlatVector<std::string>(
          {std::nullopt, std::nullopt, std::nullopt}, VARBINARY()),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"merge_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();

  EXPECT_TRUE(resultVector->isNullAt(0));
}

TEST_F(SetDigestAggregateTest, mergeEmptyInput) {
  auto vectors = makeRowVector({
      makeFlatVector<std::string>({}, VARBINARY()),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"merge_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();

  EXPECT_TRUE(resultVector->isNullAt(0));
}

TEST_F(SetDigestAggregateTest, mergeWithGroupBy) {
  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());

  SetDigest digest1(allocator.get());
  digest1.add(1);
  digest1.add(2);

  SetDigest digest2(allocator.get());
  digest2.add(3);
  digest2.add(4);

  auto size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  auto size2 = digest2.estimatedSerializedSize();
  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2}),
      makeFlatVector<std::string>(
          {std::string(buffer1.data(), size1),
           std::string(buffer1.data(), size1),
           std::string(buffer2.data(), size2),
           std::string(buffer2.data(), size2)},
          VARBINARY()),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({"c0"}, {"merge_set_digest(c1)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 2);

  auto digestVector = result->childAt(1)->as<FlatVector<StringView>>();

  for (auto i = 0; i < result->size(); ++i) {
    auto serialized = digestVector->valueAt(i);

    SetDigest mergedDigest(allocator.get());
    mergedDigest.deserialize(serialized.data(), serialized.size());

    EXPECT_EQ(mergedDigest.cardinality(), 2);
    EXPECT_TRUE(mergedDigest.isExact());
  }
}

TEST_F(SetDigestAggregateTest, mergeDistributedAggregation) {
  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());

  std::vector<std::string> serializedDigests;
  serializedDigests.reserve(3);

  for (int64_t i = 0; i < 3; ++i) {
    SetDigest digest(allocator.get());
    for (int64_t j = i * 10; j < (i + 1) * 10; ++j) {
      digest.add(j);
    }
    auto size = digest.estimatedSerializedSize();
    std::vector<char> buffer(size);
    digest.serialize(buffer.data());
    serializedDigests.push_back(std::string(buffer.data(), size));
  }

  auto vectors = {
      makeRowVector({makeFlatVector<std::string>(
          {serializedDigests[0], serializedDigests[1]}, VARBINARY())}),
      makeRowVector(
          {makeFlatVector<std::string>({serializedDigests[2]}, VARBINARY())}),
  };

  auto op = PlanBuilder()
                .values(vectors)
                .partialAggregation({}, {"merge_set_digest(c0)"})
                .intermediateAggregation()
                .finalAggregation()
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  SetDigest mergedDigest(allocator.get());
  mergedDigest.deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(mergedDigest.cardinality(), 30);
  EXPECT_TRUE(mergedDigest.isExact());
}

TEST_F(SetDigestAggregateTest, mergeLargeApproximateDigests) {
  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());

  std::vector<std::string> serializedDigests;
  serializedDigests.reserve(5);

  for (int64_t i = 0; i < 5; ++i) {
    SetDigest digest(allocator.get());
    for (int64_t j = i * 20000; j < (i + 1) * 20000; ++j) {
      digest.add(j);
    }
    auto size = digest.estimatedSerializedSize();
    std::vector<char> buffer(size);
    digest.serialize(buffer.data());
    serializedDigests.push_back(std::string(buffer.data(), size));
  }

  auto vectors = makeRowVector({
      makeFlatVector<std::string>(serializedDigests, VARBINARY()),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"merge_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  SetDigest mergedDigest(allocator.get());
  mergedDigest.deserialize(serialized.data(), serialized.size());

  EXPECT_FALSE(mergedDigest.isExact());

  auto estimatedCardinality = mergedDigest.cardinality();
  double errorRate = std::abs(estimatedCardinality - 100000) / 100000.0;

  EXPECT_LT(errorRate, 0.05)
      << "HLL estimate should be within 5% of actual. "
      << "Actual: 100000, Estimated: " << estimatedCardinality;
}

TEST_F(SetDigestAggregateTest, mergeMakeSetDigestOutput) {
  auto vectors = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
  });

  auto partialOp =
      PlanBuilder()
          .values({vectors})
          .singleAggregation({"c0"}, {"make_set_digest(c1) as digest"})
          .planNode();

  auto partialResult = AssertQueryBuilder(partialOp).copyResults(pool());

  auto mergeOp = PlanBuilder()
                     .values({partialResult})
                     .singleAggregation({}, {"merge_set_digest(digest)"})
                     .planNode();

  auto result = AssertQueryBuilder(mergeOp).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigest mergedDigest(allocator.get());
  mergedDigest.deserialize(serialized.data(), serialized.size());

  EXPECT_EQ(mergedDigest.cardinality(), 4);
  EXPECT_TRUE(mergedDigest.isExact());
}

TEST_F(SetDigestAggregateTest, mergeRoundTripSerialization) {
  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());

  SetDigest digest1(allocator.get());
  digest1.add(1);
  digest1.add(2);
  digest1.add(3);

  auto size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  auto vectors = makeRowVector({
      makeFlatVector<std::string>(
          {std::string(buffer1.data(), size1)}, VARBINARY()),
  });

  auto op = PlanBuilder()
                .values({vectors})
                .singleAggregation({}, {"merge_set_digest(c0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  auto serialized =
      result->childAt(0)->as<FlatVector<StringView>>()->valueAt(0);

  SetDigest digest2(allocator.get());
  digest2.deserialize(serialized.data(), serialized.size());

  auto size2 = digest2.estimatedSerializedSize();
  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());

  SetDigest digest3(allocator.get());
  digest3.deserialize(buffer2.data(), size2);

  auto size3 = digest3.estimatedSerializedSize();
  std::vector<char> buffer3(size3);
  digest3.serialize(buffer3.data());

  EXPECT_EQ(size2, size3);
  EXPECT_EQ(
      std::string(buffer2.begin(), buffer2.end()),
      std::string(buffer3.begin(), buffer3.end()));
}

} // namespace

} // namespace facebook::velox::aggregate::test
