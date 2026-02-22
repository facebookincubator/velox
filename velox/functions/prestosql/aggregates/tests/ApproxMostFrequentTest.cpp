/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {
namespace {

template <typename T>
struct ApproxMostFrequentTest : AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
  }

  std::shared_ptr<FlatVector<int>> makeGroupKeys() {
    return makeFlatVector<int>(3, [](auto row) { return row; });
  }

  std::shared_ptr<FlatVector<int>> makeKeys() {
    return makeFlatVector<int>(
        1000, [](auto row) { return static_cast<int>(std::sqrt(row)) % 3; });
  }

  std::shared_ptr<FlatVector<T>> makeValues() {
    return makeFlatVector<T>(1000, [](auto row) { return std::sqrt(row); });
  }

  std::shared_ptr<FlatVector<T>> makeValuesWithNulls() {
    auto values = makeValues();
    for (int i = 0; i < values->size(); ++i) {
      if (static_cast<int>(std::sqrt(i)) % 3 == 0) {
        values->setNull(i, true);
      }
    }
    return values;
  }

  MapVectorPtr makeGlobalExpected() {
    return makeMapVector<T, int64_t>({{{30, 61}, {29, 59}, {28, 57}}});
  }

  MapVectorPtr makeGroupedExpected() {
    return makeMapVector<T, int64_t>(
        {{{24, 49}, {27, 55}, {30, 61}},
         {{22, 45}, {25, 51}, {28, 57}},
         {{23, 47}, {26, 53}, {29, 59}}});
  }

  MapVectorPtr makeEmptyGroupExpected() {
    auto expected = makeGroupedExpected();
    expected->setNull(0, true);
    return expected;
  }
};

template <>
std::shared_ptr<FlatVector<StringView>>
ApproxMostFrequentTest<StringView>::makeValues() {
  std::string s[32];
  for (int i = 0; i < 32; ++i) {
    s[i] = std::to_string(i);
  }
  return makeFlatVector<StringView>(1000, [&](auto row) {
    return StringView(s[static_cast<int>(std::sqrt(row))]);
  });
}

template <>
MapVectorPtr ApproxMostFrequentTest<StringView>::makeGlobalExpected() {
  return makeMapVector<StringView, int64_t>(
      {{{"30", 61}, {"29", 59}, {"28", 57}}});
}

template <>
MapVectorPtr ApproxMostFrequentTest<StringView>::makeGroupedExpected() {
  return makeMapVector<StringView, int64_t>(
      {{{"24", 49}, {"27", 55}, {"30", 61}},
       {{"22", 45}, {"25", 51}, {"28", 57}},
       {{"23", 47}, {"26", 53}, {"29", 59}}});
}

using ValueTypes = ::testing::Types<int, StringView>;
TYPED_TEST_SUITE(ApproxMostFrequentTest, ValueTypes);

TYPED_TEST(ApproxMostFrequentTest, global) {
  auto values = this->makeValues();
  auto expected = this->makeGlobalExpected();
  this->testAggregations(
      {this->makeRowVector({values})},
      {},
      {"approx_most_frequent(3, c0, 31)"},
      {this->makeRowVector({expected})});
}

TYPED_TEST(ApproxMostFrequentTest, grouped) {
  auto values = this->makeValues();
  auto keys = this->makeKeys();
  auto groupKeys = this->makeGroupKeys();
  auto expected = this->makeGroupedExpected();
  this->testAggregations(
      {this->makeRowVector({keys, values})},
      {"c0"},
      {"approx_most_frequent(3, c1, 11)"},
      {this->makeRowVector({groupKeys, expected})});
}

TYPED_TEST(ApproxMostFrequentTest, emptyGroup) {
  auto values = this->makeValuesWithNulls();
  auto keys = this->makeKeys();
  auto groupKeys = this->makeGroupKeys();
  auto expected = this->makeEmptyGroupExpected();
  this->testAggregations(
      {this->makeRowVector({keys, values})},
      {"c0"},
      {"approx_most_frequent(3, c1, 11)"},
      {this->makeRowVector({groupKeys, expected})});
}

using ApproxMostFrequentTestInt = ApproxMostFrequentTest<int>;

TEST_F(ApproxMostFrequentTestInt, invalidBuckets) {
  auto rootPool = memory::memoryManager()->addRootPool(
      "test-root", 1 << 21, exec::MemoryReclaimer::create());
  auto leafPool = rootPool->addLeafChild("test-leaf");
  auto run = [&](int64_t buckets) {
    auto rows = makeRowVector({
        makeConstant<int64_t>(buckets, buckets),
        makeFlatVector<int>(buckets, folly::identity),
        makeConstant<int64_t>(buckets, buckets),
    });
    auto plan = exec::test::PlanBuilder()
                    .values({rows})
                    .singleAggregation({}, {"approx_most_frequent(c0, c1, c2)"})
                    .planNode();
    return exec::test::AssertQueryBuilder(plan).copyResults(leafPool.get());
  };
  ASSERT_EQ(run(10)->size(), 1);
  try {
    run(1 << 19);
    FAIL() << "Expected an exception";
  } catch (const VeloxException& e) {
    EXPECT_EQ(e.errorCode(), error_code::kMemCapExceeded);
  }
}

using ApproxMostFrequentTestStringView = ApproxMostFrequentTest<StringView>;

TEST_F(ApproxMostFrequentTestStringView, stringLifeCycle) {
  std::string s[32];
  for (int i = 0; i < 32; ++i) {
    s[i] = std::string(StringView::kInlineSize, 'x') + std::to_string(i);
  }
  auto values = makeFlatVector<StringView>(1000, [&](auto row) {
    return StringView(s[static_cast<int>(std::sqrt(row))]);
  });
  auto rows = makeRowVector({values});
  auto expected = makeRowVector({
      makeMapVector<StringView, int64_t>(
          {{{StringView(s[30]), 122},
            {StringView(s[29]), 118},
            {StringView(s[28]), 114}}}),
  });
  testReadFromFiles(
      {rows, rows}, {}, {"approx_most_frequent(3, c0, 31)"}, {expected});
}

TEST_F(ApproxMostFrequentTestStringView, globalAggregationWithCompaction) {
  // Test setup: 1024 unique strings of 1MB each = 1GB total input.
  // With capacity=100, only 100 strings are kept at any time, so without
  // compaction, ~924 dead strings accumulate in memory.
  constexpr int kNumStrings = 1024;
  constexpr int kCapacity = 100;
  constexpr size_t kStringLen = 1 << 20; // 1MB
  constexpr size_t kTotalInputBytes = kNumStrings * kStringLen; // 1GB
  constexpr size_t kActiveBytes = kCapacity * kStringLen; // 100MB

  // Use a smaller compaction threshold for testing.
  constexpr uint64_t kCompactionBytes = 256UL << 20; // 256MB
  constexpr uint64_t kMemoryLimit = 512UL << 20; // 512MB

  std::vector<std::string> strings;
  strings.reserve(kNumStrings);
  for (int i = 0; i < kNumStrings; ++i) {
    auto& s = strings.emplace_back(kStringLen, 'a' + (i % 26));
    auto indexStr = folly::to<std::string>(i);
    s.replace(kStringLen - indexStr.size(), indexStr.size(), indexStr);
  }

  auto values = makeFlatVector<StringView>(
      kNumStrings, [&](auto row) { return StringView(strings[row]); });
  auto rows = makeRowVector({values});

  auto plan =
      exec::test::PlanBuilder()
          .values({rows})
          .partialAggregation(
              {}, {fmt::format("approx_most_frequent(10, c0, {})", kCapacity)})
          .planNode();

  struct TestSettings {
    std::string name;
    std::string compactionBytesConfig;
    uint64_t memoryLimit;
    bool expectOOM;
    std::function<void(uint64_t)> validatePeakMemory;

    std::string debugString() const {
      return fmt::format(
          "name: {}, compactionBytesConfig: {}, memoryLimit: {}, expectOOM: {}",
          name,
          compactionBytesConfig,
          memoryLimit,
          expectOOM);
    }
  };

  std::vector<TestSettings> testSettings = {
      {
          "compaction-enabled",
          folly::to<std::string>(kCompactionBytes),
          kMemoryLimit,
          false,
          [=](uint64_t peakMemory) {
            ASSERT_LT(peakMemory, kCompactionBytes + 2 * kActiveBytes)
                << "With compaction enabled, peak memory should be bounded near "
                   "compaction threshold plus active bytes";
          },
      },
      {
          "oom-without-compaction",
          "0",
          kMemoryLimit,
          true,
          nullptr,
      },
      {
          "compaction-disabled-large-memory",
          "0",
          2UL << 30,
          false,
          [=](uint64_t peakMemory) {
            // Without compaction, peak memory should exceed total input size.
            ASSERT_GT(peakMemory, kTotalInputBytes)
                << "Without compaction, peak memory should exceed total input "
                   "size due to dead string accumulation";
          },
      },
  };

  for (const auto& testSetting : testSettings) {
    SCOPED_TRACE(testSetting.debugString());

    auto queryPool = memory::memoryManager()->addRootPool(
        testSetting.name,
        testSetting.memoryLimit,
        exec::MemoryReclaimer::create());
    auto queryCtx = core::QueryCtx::create(
        executor_.get(), core::QueryConfig{{}}, {}, {}, std::move(queryPool));

    exec::test::AssertQueryBuilder queryBuilder(plan);
    queryBuilder.queryCtx(queryCtx);
    if (!testSetting.compactionBytesConfig.empty()) {
      queryBuilder.config(
          core::QueryConfig::kAggregationCompactionBytesThreshold,
          testSetting.compactionBytesConfig);
    }

    if (testSetting.expectOOM) {
      VELOX_ASSERT_THROW(
          queryBuilder.copyResults(pool()), "Exceeded memory pool capacity");
    } else {
      auto result = queryBuilder.copyResults(pool());
      ASSERT_EQ(result->size(), 1);
      ASSERT_FALSE(result->isNullAt(0));

      auto peakMemory = queryCtx->pool()->peakBytes();
      testSetting.validatePeakMemory(peakMemory);
    }
  }
}

class ApproxMostFrequentTestBoolean : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
  }
};

TEST_F(ApproxMostFrequentTestBoolean, basic) {
  auto input = makeRowVector({
      makeFlatVector<int32_t>({0, 1, 0, 1, 0, 1, 0, 1}),
      makeFlatVector<bool>(
          {true, false, true, true, false, false, false, false}),
      makeConstant(true, 8),
      makeConstant(false, 8),
      makeAllNullFlatVector<bool>(8),
      makeNullableFlatVector<bool>(
          {true, false, std::nullopt, true, false, std::nullopt, false, false}),
  });

  auto expected = makeRowVector({
      makeMapVector<bool, int64_t>({
          {{true, 3}, {false, 5}},
      }),
  });

  testAggregations(
      {input}, {}, {"approx_most_frequent(3, c1, 31)"}, {expected});

  expected = makeRowVector({
      makeFlatVector<int32_t>({0, 1}),
      makeMapVector<bool, int64_t>({
          {{true, 2}, {false, 2}},
          {{true, 1}, {false, 3}},
      }),
  });

  testAggregations(
      {input}, {"c0"}, {"approx_most_frequent(3, c1, 31)"}, {expected});

  // All 'true'.
  expected = makeRowVector({makeMapVector<bool, int64_t>({{{true, 8}}})});
  testAggregations(
      {input}, {}, {"approx_most_frequent(3, c2, 31)"}, {expected});

  expected = makeRowVector({
      makeFlatVector<int32_t>({0, 1}),
      makeMapVector<bool, int64_t>({
          {{true, 4}},
          {{true, 4}},
      }),
  });

  testAggregations(
      {input}, {"c0"}, {"approx_most_frequent(3, c2, 31)"}, {expected});

  // All 'false'.
  expected = makeRowVector({makeMapVector<bool, int64_t>({{{false, 8}}})});
  testAggregations(
      {input}, {}, {"approx_most_frequent(3, c3, 31)"}, {expected});

  expected = makeRowVector({
      makeFlatVector<int32_t>({0, 1}),
      makeMapVector<bool, int64_t>({
          {{false, 4}},
          {{false, 4}},
      }),
  });

  testAggregations(
      {input}, {"c0"}, {"approx_most_frequent(3, c3, 31)"}, {expected});

  // All nulls.
  expected = makeRowVector({
      BaseVector::createNullConstant(MAP(BOOLEAN(), BIGINT()), 1, pool()),
  });
  testAggregations(
      {input}, {}, {"approx_most_frequent(3, c4, 31)"}, {expected});

  // Some nulls.
  expected = makeRowVector({
      makeMapVector<bool, int64_t>({
          {{true, 2}, {false, 4}},
      }),
  });

  testAggregations(
      {input}, {}, {"approx_most_frequent(3, c5, 31)"}, {expected});

  expected = makeRowVector({
      makeFlatVector<int32_t>({0, 1}),
      makeMapVector<bool, int64_t>({
          {{true, 1}, {false, 2}},
          {{true, 1}, {false, 2}},
      }),
  });

  testAggregations(
      {input}, {"c0"}, {"approx_most_frequent(3, c5, 31)"}, {expected});
}

class ApproxMostFrequentTestJson : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
  }
};

TEST_F(ApproxMostFrequentTestJson, basic) {
  // JSON strings as input
  std::vector<std::string> jsonStrings = {
      "{\"type\": \"store\"}",
      "{\"type\": \"fruit\"}",
      "{\"type\": \"fruit\"}",
      "{\"type\": \"book\"}",
      "{\"type\": \"store\"}",
      "{\"type\": \"fruit\"}"};

  auto inputVector = makeFlatVector<StringView>(
      static_cast<vector_size_t>(jsonStrings.size()),
      [&](auto row) { return StringView(jsonStrings[row]); });

  MapVectorPtr expectedMap = makeMapVector<StringView, int64_t>(
      {{{StringView("{\"type\": \"fruit\"}"), 3},
        {StringView("{\"type\": \"store\"}"), 2}}});
  auto expected = makeRowVector({{expectedMap}});

  testAggregations(
      {makeRowVector({inputVector})},
      {},
      {"approx_most_frequent(2, c0, 31)"},
      {expected});
}

} // namespace
} // namespace facebook::velox::aggregate::test
