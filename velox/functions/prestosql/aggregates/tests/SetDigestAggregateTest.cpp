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
#include "velox/functions/prestosql/types/SetDigestRegistration.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;
using namespace facebook::velox::functions;

namespace facebook::velox::aggregate::test {

namespace {

using SetDigestType = functions::SetDigest<int64_t>;

class SetDigestAggregateTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerSetDigestType();
  }

  // Helper to deserialize and verify a SetDigest result.
  void verifySetDigest(
      const RowVectorPtr& result,
      int64_t expectedCardinality,
      bool expectExact = true,
      int32_t rowIndex = 0) {
    ASSERT_LT(rowIndex, result->size());
    auto resultVector = result->childAt(result->type()->size() - 1)
                            ->as<FlatVector<StringView>>();
    ASSERT_FALSE(resultVector->isNullAt(rowIndex));

    auto serialized = resultVector->valueAt(rowIndex);
    auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
    SetDigestType digest(allocator.get());
    auto status = digest.deserialize(serialized.data(), serialized.size());
    ASSERT_TRUE(status.ok()) << status.message();

    EXPECT_EQ(digest.cardinality(), expectedCardinality);
    if (expectExact) {
      EXPECT_TRUE(digest.isExact());
    }
  }

  // Helper to verify result is null.
  void verifySetDigestNull(const RowVectorPtr& result, int32_t rowIndex = 0) {
    ASSERT_LT(rowIndex, result->size());
    auto resultVector = result->childAt(result->type()->size() - 1)
                            ->as<FlatVector<StringView>>();
    EXPECT_TRUE(resultVector->isNullAt(rowIndex));
  }

  // Helper to create a serialized SetDigest from int64_t values.
  std::string serializeSetDigest(const std::vector<int64_t>& values) {
    auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
    SetDigestType digest(allocator.get());
    for (auto value : values) {
      digest.add(value);
    }
    auto size = digest.estimatedSerializedSize();
    std::string buffer(size, '\0');
    digest.serialize(buffer.data());
    return buffer;
  }

  // Helper to run make_set_digest and get result.
  RowVectorPtr runMakeSetDigest(
      const std::vector<RowVectorPtr>& input,
      const std::vector<std::string>& groupingKeys = {}) {
    auto op = PlanBuilder()
                  .values(input)
                  .singleAggregation(groupingKeys, {"make_set_digest(c0)"})
                  .planNode();
    return AssertQueryBuilder(op).copyResults(pool());
  }

  // Helper to run merge_set_digest and get result.
  RowVectorPtr runMergeSetDigest(
      const std::vector<RowVectorPtr>& input,
      const std::vector<std::string>& groupingKeys = {}) {
    auto op = PlanBuilder()
                  .values(input)
                  .project({"cast(a0 as setdigest)"})
                  .singleAggregation(groupingKeys, {"merge_set_digest(p0)"})
                  .planNode();
    return AssertQueryBuilder(op).copyResults(pool());
  }
};

TEST_F(SetDigestAggregateTest, intValues) {
  auto vectors = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 1, 2, 2, 3, 3, 3, 3}),
  });

  auto result = runMakeSetDigest({vectors});
  verifySetDigest(result, 3);
}

TEST_F(SetDigestAggregateTest, withNullValues) {
  auto vectors = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {1, std::nullopt, 2, std::nullopt, 3, std::nullopt, 1}),
  });

  auto result = runMakeSetDigest({vectors});
  verifySetDigest(result, 3);
}

TEST_F(SetDigestAggregateTest, allNullValues) {
  auto vectors = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}),
  });

  auto result = runMakeSetDigest({vectors});
  verifySetDigestNull(result);
}

TEST_F(SetDigestAggregateTest, emptyInput) {
  auto vectors = makeRowVector({
      makeFlatVector<int64_t>({}),
  });

  auto result = runMakeSetDigest({vectors});
  verifySetDigestNull(result);
}

TEST_F(SetDigestAggregateTest, stringValues) {
  auto vectors = makeRowVector({
      makeFlatVector<std::string>(
          {"apple",
           "banana",
           "apple",
           "cherry",
           "banana",
           "a longer string with more than 12 characters",
           "another extended string for testing",
           "a longer string with more than 12 characters"}),
  });

  auto result = runMakeSetDigest({vectors});
  verifySetDigest(result, 5);
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

    SetDigestType digest(allocator.get());
    auto status = digest.deserialize(serialized.data(), serialized.size());
    ASSERT_TRUE(status.ok()) << status.message();

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
    auto result = runMakeSetDigest({makeRowVector({makeVector})});
    verifySetDigest(result, 3);
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

  auto result =
      runMakeSetDigest({makeRowVector({makeFlatVector<int64_t>(values)})});
  verifySetDigest(result, 1000);
}

TEST_F(SetDigestAggregateTest, largeApproximateSet) {
  std::vector<int64_t> values;
  values.reserve(50000);
  for (int64_t i = 0; i < 50000; ++i) {
    values.push_back(i);
  }

  auto result =
      runMakeSetDigest({makeRowVector({makeFlatVector<int64_t>(values)})});

  auto serialized =
      result->childAt(0)->as<FlatVector<StringView>>()->valueAt(0);
  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigestType digest(allocator.get());
  ASSERT_TRUE(digest.deserialize(serialized.data(), serialized.size()).ok());

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

  auto result = AssertQueryBuilder(
                    PlanBuilder()
                        .values(vectors)
                        .partialAggregation({}, {"make_set_digest(c0)"})
                        .intermediateAggregation()
                        .finalAggregation()
                        .planNode())
                    .copyResults(pool());

  verifySetDigest(result, 7);
}

TEST_F(SetDigestAggregateTest, roundTripSerialization) {
  auto result = runMakeSetDigest(
      {makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4, 5})})});

  auto serialized =
      result->childAt(0)->as<FlatVector<StringView>>()->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigestType digest1(allocator.get());
  ASSERT_TRUE(digest1.deserialize(serialized.data(), serialized.size()).ok());

  auto size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  SetDigestType digest2(allocator.get());
  ASSERT_TRUE(digest2.deserialize(buffer1.data(), size1).ok());

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

  auto result = runMakeSetDigest({vectors});
  verifySetDigest(result, 2);
}

TEST_F(SetDigestAggregateTest, javaCompatibilityInteger) {
  // Tests serialization/deserialization round-trip to ensure compatibility
  // with Java implementation. Unlike other tests which only verify
  // cardinality, this explicitly tests that the serialized format can be
  // deserialized correctly (serialize → deserialize → verify).
  auto result = runMakeSetDigest(
      {makeRowVector({makeFlatVector<int64_t>({1, 1, 1, 2, 2})})});

  verifySetDigest(result, 2);

  auto serialized =
      result->childAt(0)->as<FlatVector<StringView>>()->valueAt(0);
  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());

  SetDigestType digest(allocator.get());
  ASSERT_TRUE(digest.deserialize(serialized.data(), serialized.size()).ok());

  auto size = digest.estimatedSerializedSize();
  std::vector<char> buffer(size);
  digest.serialize(buffer.data());

  SetDigestType digest2(allocator.get());
  ASSERT_TRUE(digest2.deserialize(buffer.data(), size).ok());
  EXPECT_EQ(digest2.cardinality(), 2);
}

TEST_F(SetDigestAggregateTest, javaCompatibilityString) {
  auto result = runMakeSetDigest(
      {makeRowVector({makeFlatVector<std::string>({"abc", "def"})})});

  verifySetDigest(result, 2);

  auto serialized =
      result->childAt(0)->as<FlatVector<StringView>>()->valueAt(0);
  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());

  SetDigestType digest(allocator.get());
  ASSERT_TRUE(digest.deserialize(serialized.data(), serialized.size()).ok());

  auto size = digest.estimatedSerializedSize();
  std::vector<char> buffer(size);
  digest.serialize(buffer.data());

  SetDigestType digest2(allocator.get());
  ASSERT_TRUE(digest2.deserialize(buffer.data(), size).ok());
  EXPECT_EQ(digest2.cardinality(), 2);
}

TEST_F(SetDigestAggregateTest, floatValues) {
  auto vectors = makeRowVector({
      makeFlatVector<float>({1.2f, 2.3f, 1.2f, 3.4f, 2.3f}),
  });

  auto result = runMakeSetDigest({vectors});
  verifySetDigest(result, 3);
}

TEST_F(SetDigestAggregateTest, doubleValues) {
  auto vectors = makeRowVector({
      makeFlatVector<double>({1.5, 2.7, 1.5, 3.9, 2.7}),
  });

  auto result = runMakeSetDigest({vectors});
  verifySetDigest(result, 3);
}

TEST_F(SetDigestAggregateTest, dateValues) {
  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(
          {0, 18262, 0, -5, 18262}, // Various dates including epoch
          DATE()),
  });

  auto result = runMakeSetDigest({vectors});
  verifySetDigest(result, 3);
}

TEST_F(SetDigestAggregateTest, mergeBasic) {
  auto data1 = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto data2 = makeRowVector({makeFlatVector<int64_t>({3, 4, 5})});

  std::vector<RowVectorPtr> digestVectors;
  for (auto& data : {data1, data2}) {
    digestVectors.push_back(runMakeSetDigest({data}));
  }

  auto result = runMergeSetDigest(digestVectors);
  verifySetDigest(result, 5);
}

TEST_F(SetDigestAggregateTest, mergeWithNullValues) {
  auto data1 = makeRowVector({makeFlatVector<int64_t>({1, 2})});

  auto op1 = PlanBuilder()
                 .values({data1})
                 .singleAggregation({}, {"make_set_digest(c0)"})
                 .planNode();
  auto result1 = AssertQueryBuilder(op1).copyResults(pool());

  auto combined = makeRowVector({
      makeNullableFlatVector<StringView>(
          {result1->childAt(0)->as<FlatVector<StringView>>()->valueAt(0),
           std::nullopt,
           result1->childAt(0)->as<FlatVector<StringView>>()->valueAt(0)}),
  });

  auto mergeOp = PlanBuilder()
                     .values({combined})
                     .project({"cast(c0 as setdigest)"})
                     .singleAggregation({}, {"merge_set_digest(p0)"})
                     .planNode();

  auto result = AssertQueryBuilder(mergeOp).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);
  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(digest.cardinality(), 2);
}

TEST_F(SetDigestAggregateTest, mergeWithGroupBy) {
  auto data1 = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
  });

  auto partialOp =
      PlanBuilder()
          .values({data1})
          .singleAggregation({"c0"}, {"make_set_digest(c1) as digest"})
          .planNode();
  auto partialResult = AssertQueryBuilder(partialOp).copyResults(pool());

  auto mergeOp = PlanBuilder()
                     .values({partialResult})
                     .project({"c0", "cast(digest as setdigest) as digest"})
                     .singleAggregation({"c0"}, {"merge_set_digest(digest)"})
                     .planNode();

  auto result = AssertQueryBuilder(mergeOp).copyResults(pool());

  ASSERT_EQ(result->size(), 2);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  auto digestVector = result->childAt(1)->as<FlatVector<StringView>>();

  for (auto i = 0; i < result->size(); ++i) {
    auto serialized = digestVector->valueAt(i);

    SetDigestType mergedDigest(allocator.get());
    auto status =
        mergedDigest.deserialize(serialized.data(), serialized.size());
    ASSERT_TRUE(status.ok()) << status.message();

    EXPECT_EQ(mergedDigest.cardinality(), 2);
    EXPECT_TRUE(mergedDigest.isExact());
  }
}

TEST_F(SetDigestAggregateTest, mergeDistributedAggregation) {
  std::vector<RowVectorPtr> partialResults;

  for (int64_t i = 0; i < 3; ++i) {
    std::vector<int64_t> values;
    for (int64_t j = i * 10; j < (i + 1) * 10; ++j) {
      values.push_back(j);
    }
    partialResults.push_back(
        runMakeSetDigest({makeRowVector({makeFlatVector<int64_t>(values)})}));
  }

  auto result = runMergeSetDigest(partialResults);
  verifySetDigest(result, 30);
}

TEST_F(SetDigestAggregateTest, mergeLargeApproximateDigests) {
  std::vector<RowVectorPtr> partialResults;

  for (int64_t i = 0; i < 5; ++i) {
    std::vector<int64_t> values;
    values.reserve(20000);
    for (int64_t j = i * 20000; j < (i + 1) * 20000; ++j) {
      values.push_back(j);
    }
    partialResults.push_back(
        runMakeSetDigest({makeRowVector({makeFlatVector<int64_t>(values)})}));
  }

  auto result = runMergeSetDigest(partialResults);

  auto serialized =
      result->childAt(0)->as<FlatVector<StringView>>()->valueAt(0);
  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigestType mergedDigest(allocator.get());
  ASSERT_TRUE(
      mergedDigest.deserialize(serialized.data(), serialized.size()).ok());

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
                     .project({"cast(digest as setdigest) as digest"})
                     .singleAggregation({}, {"merge_set_digest(digest)"})
                     .planNode();

  auto result = AssertQueryBuilder(mergeOp).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultVector = result->childAt(0)->as<FlatVector<StringView>>();
  ASSERT_FALSE(resultVector->isNullAt(0));

  auto serialized = resultVector->valueAt(0);

  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
  SetDigestType mergedDigest(allocator.get());
  auto status = mergedDigest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(mergedDigest.cardinality(), 4);
  EXPECT_TRUE(mergedDigest.isExact());
}

TEST_F(SetDigestAggregateTest, mergeRoundTripSerialization) {
  auto allocator = std::make_unique<HashStringAllocator>(pool_.get());

  SetDigestType digest1(allocator.get());
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
                .project({"cast(c0 as setdigest)"})
                .singleAggregation({}, {"merge_set_digest(p0)"})
                .planNode();

  auto result = AssertQueryBuilder(op).copyResults(pool());

  auto serialized =
      result->childAt(0)->as<FlatVector<StringView>>()->valueAt(0);

  SetDigestType digest2(allocator.get());
  auto status = digest2.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

  auto size2 = digest2.estimatedSerializedSize();
  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());

  SetDigestType digest3(allocator.get());
  status = digest3.deserialize(buffer2.data(), size2);
  ASSERT_TRUE(status.ok()) << status.message();

  auto size3 = digest3.estimatedSerializedSize();
  std::vector<char> buffer3(size3);
  digest3.serialize(buffer3.data());

  EXPECT_EQ(size2, size3);
  EXPECT_EQ(digest2.cardinality(), 3);
  EXPECT_EQ(digest3.cardinality(), 3);
  EXPECT_EQ(
      std::string(buffer2.begin(), buffer2.end()),
      std::string(buffer3.begin(), buffer3.end()));
}

TEST_F(SetDigestAggregateTest, mergeSetDigestGlobalIntermediate) {
  auto batch1 = makeRowVector({makeFlatVector<std::string>(
      {serializeSetDigest({1, 2, 3})}, VARBINARY())});
  auto batch2 = makeRowVector({makeFlatVector<std::string>(
      {serializeSetDigest({4, 5, 6})}, VARBINARY())});
  auto batch3 = makeRowVector({makeFlatVector<std::string>(
      {serializeSetDigest({7, 8, 9})}, VARBINARY())});

  auto result = AssertQueryBuilder(
                    PlanBuilder()
                        .values({batch1, batch2, batch3})
                        .project({"cast(c0 as setdigest)"})
                        .partialAggregation({}, {"merge_set_digest(p0)"})
                        .finalAggregation()
                        .planNode())
                    .copyResults(pool());

  verifySetDigest(result, 9);
}

TEST_F(SetDigestAggregateTest, mergeSetDigestGroupedIntermediate) {
  auto batchWithKey1 = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<std::string>(
          {serializeSetDigest({1, 2, 3}), serializeSetDigest({10, 20})},
          VARBINARY()),
  });
  auto batchWithKey2 = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<std::string>(
          {serializeSetDigest({4, 5, 6}), serializeSetDigest({30, 40})},
          VARBINARY()),
  });
  auto batchWithKey3 = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<std::string>(
          {serializeSetDigest({7, 8, 9}), serializeSetDigest({50, 60})},
          VARBINARY()),
  });

  auto result =
      AssertQueryBuilder(
          PlanBuilder()
              .values({batchWithKey1, batchWithKey2, batchWithKey3})
              .project({"c0", "cast(c1 as setdigest) as digest"})
              .partialAggregation({"c0"}, {"merge_set_digest(digest)"})
              .finalAggregation()
              .planNode())
          .copyResults(pool());

  ASSERT_EQ(result->size(), 2);
  verifySetDigest(result, 9, true, 0);
  verifySetDigest(result, 6, true, 1);
}

} // namespace

} // namespace facebook::velox::aggregate::test
