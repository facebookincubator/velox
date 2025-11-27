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

#include "velox/common/hyperloglog/KHyperLogLog.h"
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
    pool_ = facebook::velox::memory::memoryManager()->addLeafPool();
    hsa_ = std::make_unique<HashStringAllocator>(pool_.get());
    allocator_ = hsa_.get();
  }
  void TearDown() override {
    // Clean up allocator before pool.
    hsa_.reset();
    pool_.reset();
  }

  // Helper to create KHLL with test data
  std::string createKHLL(
      const std::vector<int64_t>& values,
      const std::vector<int64_t>& uiis) {
    EXPECT_EQ(values.size(), uiis.size());
    auto khll =
        std::make_unique<common::hll::KHyperLogLog<HashStringAllocator>>(
            allocator_);

    for (size_t i = 0; i < values.size(); ++i) {
      khll->add(values[i], uiis[i]);
    }

    std::string serialized(khll->estimatedSerializedSize(), '\0');
    khll->serialize(serialized.data());
    return serialized;
  }

 protected:
  std::shared_ptr<MemoryPool> pool_;
  std::unique_ptr<HashStringAllocator> hsa_;
  HashStringAllocator* allocator_;
};

TEST_F(KHyperLogLogAggregateTest, basicBigintBigint) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 10, 20, 20, 30}),
  });

  // Expected: 5 distinct values
  auto expected = makeRowVector({makeFlatVector<int64_t>({5})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, varcharBigint) {
  auto data = makeRowVector({
      makeFlatVector<StringView>(
          {"apple", "banana", "apple", "cherry", "banana"}),
      makeFlatVector<int64_t>({1, 1, 2, 1, 2}),
  });

  // Expected: 3 distinct values (apple, banana, cherry)
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, bigintVarchar) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 1, 2}),
      makeFlatVector<StringView>({"user1", "user1", "user2", "user2", "user1"}),
  });

  // Expected: 3 distinct values
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, varcharVarchar) {
  auto data = makeRowVector({
      makeFlatVector<StringView>(
          {"apple", "banana", "apple", "cherry", "banana"}),
      makeFlatVector<StringView>({"user1", "user1", "user2", "user1", "user2"}),
  });

  // Expected: 3 distinct values (apple, banana, cherry)
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, doubleBigint) {
  auto data = makeRowVector({
      makeFlatVector<double>({1.5, 2.5, 3.5, 1.5, 2.5}),
      makeFlatVector<int64_t>({1, 1, 2, 2, 1}),
  });

  // Expected: 3 distinct values
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, doubleVarchar) {
  auto data = makeRowVector({
      makeFlatVector<double>({1.5, 2.5, 3.5, 1.5, 2.5}),
      makeFlatVector<StringView>({"user1", "user1", "user2", "user2", "user1"}),
  });

  // Expected: 3 distinct values
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, varbinaryBigint) {
  std::vector<std::string> fruits = {
      "apple", "banana", "apple", "cherry", "banana"};
  auto data = makeRowVector({
      makeFlatVector<StringView>(
          fruits.size(),
          [&](auto row) { return StringView(fruits[row]); },
          nullptr,
          VARBINARY()),
      makeFlatVector<int64_t>({1, 1, 2, 1, 2}),
  });

  // Expected: 3 distinct values (apple, banana, cherry)
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, timestampBigint) {
  auto data = makeRowVector({
      makeFlatVector<Timestamp>({
          Timestamp::fromMillis(1000),
          Timestamp::fromMillis(2000),
          Timestamp::fromMillis(3000),
          Timestamp::fromMillis(1000),
          Timestamp::fromMillis(2000),
      }),
      makeFlatVector<int64_t>({1, 1, 2, 2, 1}),
  });

  // Expected: 3 distinct timestamps
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, smallintBigint) {
  auto data = makeRowVector({
      makeFlatVector<int16_t>({1, 2, 3, 1, 2}),
      makeFlatVector<int64_t>({10, 10, 20, 20, 30}),
  });

  // Expected: 3 distinct values
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, realBigint) {
  auto data = makeRowVector({
      makeFlatVector<float>({1.5f, 2.5f, 3.5f, 1.5f, 2.5f}),
      makeFlatVector<int64_t>({1, 1, 2, 2, 1}),
  });

  // Expected: 3 distinct values
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, bigintVarbinary) {
  std::vector<std::string> users = {
      "user1", "user1", "user2", "user2", "user1"};
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 1, 2}),
      makeFlatVector<StringView>(
          users.size(),
          [&](auto row) { return StringView(users[row]); },
          nullptr,
          VARBINARY()),
  });

  // Expected: 3 distinct values
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, varcharVarbinary) {
  std::vector<std::string> users = {
      "user1", "user1", "user2", "user1", "user2"};
  auto data = makeRowVector({
      makeFlatVector<StringView>(
          {"apple", "banana", "apple", "cherry", "banana"}),
      makeFlatVector<StringView>(
          users.size(),
          [&](auto row) { return StringView(users[row]); },
          nullptr,
          VARBINARY()),
  });

  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, bigintTimestamp) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 1, 2}),
      makeFlatVector<Timestamp>({
          Timestamp::fromMillis(1000),
          Timestamp::fromMillis(1000),
          Timestamp::fromMillis(2000),
          Timestamp::fromMillis(2000),
          Timestamp::fromMillis(1000),
      }),
  });

  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, decimalBigint) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({100, 200, 300, 100, 200}, DECIMAL(10, 2)),
      makeFlatVector<int64_t>({1, 1, 2, 2, 1}),
  });

  // Expected: 3 distinct values
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, bigintDecimal) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 1, 2}),
      makeFlatVector<int64_t>({100, 100, 200, 200, 100}, DECIMAL(10, 2)),
  });

  // Expected: 3 distinct values
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, withNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, 2, std::nullopt, 4, 5}),
      makeNullableFlatVector<int64_t>({10, 10, 20, std::nullopt, 30}),
  });

  // Expected: 3 distinct values (nulls are ignored)
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});

  testAggregations(
      {data},
      {},
      {"khyperloglog_agg(c0, c1)"},
      {"cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, allNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}),
      makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}),
  });

  // Expected: null result
  auto expected = makeRowVector({makeNullConstant(TypeKind::VARBINARY, 1)});

  testAggregations({data}, {}, {"khyperloglog_agg(c0, c1)"}, {expected});
}

TEST_F(KHyperLogLogAggregateTest, groupBy) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2, 1, 2}), // groups
      makeFlatVector<int64_t>({10, 20, 30, 40, 30, 50}), // values
      makeFlatVector<int64_t>({1, 1, 1, 1, 2, 2}), // uii
  });

  // Group 1: values {10, 20, 30} = 3 distinct
  // Group 2: values {30, 40, 50} = 3 distinct
  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int64_t>({3, 3}),
  });

  testAggregations(
      {data},
      {"c0"},
      {"khyperloglog_agg(c1, c2)"},
      {"c0", "cardinality(a0)"},
      {expected});
}

TEST_F(KHyperLogLogAggregateTest, mergeKhll) {
  // Create two separate KHLLs using serialized form
  auto khll1 = createKHLL(
      std::vector<int64_t>{1, 2, 3}, std::vector<int64_t>{10, 10, 20});

  auto khll2 = createKHLL(
      std::vector<int64_t>{3, 4, 5}, std::vector<int64_t>{20, 20, 30});
  auto data = makeRowVector({
      makeFlatVector<StringView>(
          {StringView(khll1), StringView(khll2)}, KHYPERLOGLOG()),
  });

  // After merge, should have 5 distinct values: {1, 2, 3, 4, 5}
  auto expected = makeRowVector({makeFlatVector<int64_t>({5})});

  testAggregations({data}, {}, {"merge(c0)"}, {"cardinality(a0)"}, {expected});
}

TEST_F(KHyperLogLogAggregateTest, mergeKhllGroupBy) {
  auto khll1 =
      createKHLL(std::vector<int64_t>{1, 2}, std::vector<int64_t>{10, 10});

  auto khll2 =
      createKHLL(std::vector<int64_t>{3, 4}, std::vector<int64_t>{20, 20});

  auto khll3 =
      createKHLL(std::vector<int64_t>{5, 6}, std::vector<int64_t>{30, 30});

  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2}), // groups
      makeFlatVector<StringView>(
          {StringView(khll1), StringView(khll2), StringView(khll3)},
          KHYPERLOGLOG()),
  });

  // Group 1: merge of khll1 and khll2 = 4 values
  // Group 2: khll3 = 2 values
  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int64_t>({4, 2}),
  });

  testAggregations(
      {data}, {"c0"}, {"merge(c1)"}, {"c0", "cardinality(a0)"}, {expected});
}

} // namespace
} // namespace facebook::velox::aggregate::test
