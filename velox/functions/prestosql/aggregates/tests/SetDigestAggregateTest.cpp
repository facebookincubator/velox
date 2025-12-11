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
};

TEST_F(SetDigestAggregateTest, intValues) {
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
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

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
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

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
          {"apple",
           "banana",
           "apple",
           "cherry",
           "banana",
           "a longer string with more than 12 characters",
           "another extended string for testing",
           "a longer string with more than 12 characters"}),
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
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(digest.cardinality(), 5);
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
    SetDigestType digest(allocator.get());
    auto status = digest.deserialize(serialized.data(), serialized.size());
    ASSERT_TRUE(status.ok()) << status.message();

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
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

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
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

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
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

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
  SetDigestType digest1(allocator.get());
  auto status = digest1.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

  auto size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  SetDigestType digest2(allocator.get());
  status = digest2.deserialize(buffer1.data(), size1);
  ASSERT_TRUE(status.ok()) << status.message();

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
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(digest.cardinality(), 2);
  EXPECT_TRUE(digest.isExact());
}

TEST_F(SetDigestAggregateTest, javaCompatibilityInteger) {
  // Tests serialization/deserialization round-trip to ensure compatibility
  // with Java implementation. Unlike other tests which only verify
  // cardinality, this explicitly tests that the serialized format can be
  // deserialized correctly (serialize → deserialize → verify).
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
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(digest.cardinality(), 2);
  EXPECT_TRUE(digest.isExact());

  auto size = digest.estimatedSerializedSize();
  std::vector<char> buffer(size);
  digest.serialize(buffer.data());

  SetDigestType digest2(allocator.get());
  status = digest2.deserialize(buffer.data(), size);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_TRUE(status.ok()) << status.message();
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
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(digest.cardinality(), 2);
  EXPECT_TRUE(digest.isExact());

  auto size = digest.estimatedSerializedSize();
  std::vector<char> buffer(size);
  digest.serialize(buffer.data());

  SetDigestType digest2(allocator.get());
  status = digest2.deserialize(buffer.data(), size);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(digest2.cardinality(), 2);
}

TEST_F(SetDigestAggregateTest, floatValues) {
  auto vectors = makeRowVector({
      makeFlatVector<float>({1.2f, 2.3f, 1.2f, 3.4f, 2.3f}),
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
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(digest.cardinality(), 3);
  EXPECT_TRUE(digest.isExact());
}

TEST_F(SetDigestAggregateTest, doubleValues) {
  auto vectors = makeRowVector({
      makeFlatVector<double>({1.5, 2.7, 1.5, 3.9, 2.7}),
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
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(digest.cardinality(), 3);
  EXPECT_TRUE(digest.isExact());
}

TEST_F(SetDigestAggregateTest, dateValues) {
  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(
          {0, 18262, 0, -5, 18262}, // Various dates including epoch
          DATE()),
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
  SetDigestType digest(allocator.get());
  auto status = digest.deserialize(serialized.data(), serialized.size());
  ASSERT_TRUE(status.ok()) << status.message();

  // Should have 3 distinct dates
  EXPECT_EQ(digest.cardinality(), 3);
  EXPECT_TRUE(digest.isExact());
}

} // namespace

} // namespace facebook::velox::aggregate::test
