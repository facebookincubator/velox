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

#include <fmt/format.h>
#include <folly/Math.h>
#include <re2/re2.h>

#include "folly/experimental/EventCount.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/common/memory/tests/SharedArbitratorTestUtil.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/GroupingSet.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/PrefixSort.h"
#include "velox/exec/Values.h"
#include "velox/exec/prefixsort/PrefixSortEncoder.h"
#include "velox/exec/tests/utils/ArbitratorTestUtil.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/SumNonPODAggregate.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

namespace facebook::velox::exec::test {

using core::QueryConfig;
using facebook::velox::test::BatchMaker;
using namespace common::testutil;

class AggregationTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    TestValue::enable();
  }

  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  std::vector<RowVectorPtr>
  makeVectors(const RowTypePtr& rowType, size_t size, int numVectors) {
    std::vector<RowVectorPtr> vectors;
    VectorFuzzer fuzzer({.vectorSize = size}, pool());
    for (int32_t i = 0; i < numVectors; ++i) {
      vectors.push_back(fuzzer.fuzzInputRow(rowType));
    }
    return vectors;
  }

  template <typename T>
  void testSingleKey(
      const std::vector<RowVectorPtr>& vectors,
      const std::string& keyName,
      bool ignoreNullKeys,
      bool distinct) {
    std::vector<std::string> aggregates;
    if (!distinct) {
      // TODO (dm): "sum(15)", "sum(0.1)",  "min(15)",  "min(0.1)", "max(15)",
      // "max(0.1)",
      aggregates = {
          "sum(c1)",
          "sum(c2)",
          "sum(c4)",
          "sum(c5)",
          "min(c1)",
          "min(c2)",
          "min(c3)",
          "min(c4)",
          "min(c5)",
          "max(c1)",
          "max(c2)",
          "max(c3)",
          "max(c4)",
          "max(c5)"};
    }

    auto op = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {keyName},
                      aggregates,
                      {},
                      core::AggregationNode::Step::kPartial,
                      ignoreNullKeys)
                  .planNode();

    std::string fromClause = "FROM tmp";
    if (ignoreNullKeys) {
      fromClause += " WHERE " + keyName + " IS NOT NULL";
    }
    if (distinct) {
      assertQuery(op, "SELECT distinct " + keyName + " " + fromClause);
    } else {
      // TODO (dm): sum(15), sum(cast(0.1 as double)), min(15), min(0.1),
      // max(15), max(0.1),
      assertQuery(
          op,
          "SELECT " + keyName +
              ", sum(c1), sum(c2), sum(c4), sum(c5) , min(c1), min(c2), min(c3), min(c4), min(c5), max(c1), max(c2), max(c3), max(c4), max(c5) " +
              fromClause + " GROUP BY " + keyName);
    }
  }

  void testMultiKey(
      const std::vector<RowVectorPtr>& vectors,
      bool ignoreNullKeys,
      bool distinct) {
    std::vector<std::string> aggregates;
    // TODO (dm): "sum(15)", "sum(0.1)",  "min(15)",  "min(0.1)", "max(15)",
    // "max(0.1)"
    if (!distinct) {
      aggregates = {
          "sum(c4)",
          "sum(c5)",
          "min(c3)",
          "min(c4)",
          "min(c5)",
          "max(c3)",
          "max(c4)",
          "max(c5)"};
    }
    auto op = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0", "c1", "c6"},
                      aggregates,
                      {},
                      core::AggregationNode::Step::kPartial,
                      ignoreNullKeys)
                  .planNode();

    std::string fromClause = "FROM tmp";
    if (ignoreNullKeys) {
      fromClause +=
          " WHERE c0 IS NOT NULL AND c1 IS NOT NULL AND c6 IS NOT NULL";
    }
    if (distinct) {
      assertQuery(op, "SELECT distinct c0, c1, c6 " + fromClause);
    } else {
      // TODO (dm): sum(15), sum(cast(0.1 as double)), min(15), min(0.1),
      // max(15), max(0.1),, sum(1)
      assertQuery(
          op,
          "SELECT c0, c1, c6, sum(c4), sum(c5), min(c3), min(c4), min(c5),  max(c3), max(c4), max(c5) " +
              fromClause + " GROUP BY c0, c1, c6");
    }
  }

  template <typename T>
  void setTestKey(
      int64_t value,
      int32_t multiplier,
      vector_size_t row,
      FlatVector<T>* vector) {
    vector->set(row, value * multiplier);
  }

  template <typename T>
  void setKey(
      int32_t column,
      int32_t cardinality,
      int32_t multiplier,
      int32_t row,
      RowVector* batch) {
    auto vector = batch->childAt(column)->asUnchecked<FlatVector<T>>();
    auto value = folly::Random::rand32(rng_) % cardinality;
    setTestKey(value, multiplier, row, vector);
  }

  void makeModeTestKeys(
      TypePtr rowType,
      int32_t numRows,
      int32_t c0,
      int32_t c1,
      int32_t c2,
      int32_t c3,
      int32_t c4,
      int32_t c5,
      std::vector<RowVectorPtr>& batches) {
    RowVectorPtr rowVector;
    for (auto count = 0; count < numRows; ++count) {
      if (count % 1000 == 0) {
        rowVector = BaseVector::create<RowVector>(
            rowType, std::min(1000, numRows - count), pool_.get());
        batches.push_back(rowVector);
        for (auto& child : rowVector->children()) {
          child->resize(1000);
        }
      }
      setKey<int64_t>(0, c0, 6, count % 1000, rowVector.get());
      setKey<int16_t>(1, c1, 1, count % 1000, rowVector.get());
      setKey<int8_t>(2, c2, 1, count % 1000, rowVector.get());
      setKey<StringView>(3, c3, 2, count % 1000, rowVector.get());
      setKey<StringView>(4, c4, 5, count % 1000, rowVector.get());
      setKey<StringView>(5, c5, 8, count % 1000, rowVector.get());
    }
  }

  // Inserts 'key' into 'order' with random bits and a serial
  // number. The serial number makes repeats of 'key' unique and the
  // random bits randomize the order in the set.
  void insertRandomOrder(
      int64_t key,
      int64_t serial,
      folly::F14FastSet<uint64_t>& order) {
    // The word has 24 bits of grouping key, 8 random bits and 32 bits of serial
    // number.
    order.insert(
        ((folly::Random::rand32(rng_) & 0xff) << 24) | key | (serial << 32));
  }

  // Returns the key from a value inserted with insertRandomOrder().
  int32_t randomOrderKey(uint64_t key) {
    return key & ((1 << 24) - 1);
  }

  void addBatch(
      int32_t count,
      RowVectorPtr rows,
      BufferPtr& dictionary,
      std::vector<RowVectorPtr>& batches) {
    std::vector<VectorPtr> children;
    dictionary->setSize(count * sizeof(vector_size_t));
    children.push_back(BaseVector::wrapInDictionary(
        BufferPtr(nullptr), dictionary, count, rows->childAt(0)));
    children.push_back(BaseVector::wrapInDictionary(
        BufferPtr(nullptr), dictionary, count, rows->childAt(1)));
    children.push_back(children[1]);
    batches.push_back(vectorMaker_.rowVector(children));
    dictionary = AlignedBuffer::allocate<vector_size_t>(
        dictionary->capacity() / sizeof(vector_size_t), rows->pool());
  }

  // Makes batches which reference rows in 'rows' via dictionary. The
  // dictionary indices are given by 'order', wich has values with
  // indices plus random bits so as to create randomly scattered,
  // sometimes repeated values.
  void makeBatches(
      RowVectorPtr rows,
      folly::F14FastSet<uint64_t>& order,
      std::vector<RowVectorPtr>& batches) {
    constexpr int32_t kBatch = 1000;
    BufferPtr dictionary =
        AlignedBuffer::allocate<vector_size_t>(kBatch, rows->pool());
    auto rawIndices = dictionary->asMutable<vector_size_t>();
    int32_t counter = 0;
    for (auto& n : order) {
      rawIndices[counter++] = randomOrderKey(n);
      if (counter == kBatch) {
        addBatch(counter, rows, dictionary, batches);
        rawIndices = dictionary->asMutable<vector_size_t>();
        counter = 0;
      }
    }
    if (counter > 0) {
      addBatch(counter, rows, dictionary, batches);
    }
  }

  std::unique_ptr<RowContainer> makeRowContainer(
      const std::vector<TypePtr>& keyTypes,
      const std::vector<TypePtr>& dependentTypes) {
    return std::make_unique<RowContainer>(
        keyTypes,
        false,
        std::vector<Accumulator>{},
        dependentTypes,
        false,
        false,
        true,
        true,
        pool_.get());
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {BIGINT(),
           SMALLINT(),
           INTEGER(),
           BIGINT(),
           DOUBLE(), // DM: This used to be REAL() but we don't support that
           DOUBLE(),
           VARCHAR()})};
  folly::Random::DefaultGenerator rng_;
  memory::MemoryReclaimer::Stats reclaimerStats_;
  VectorFuzzer::Options fuzzerOpts_{
      .vectorSize = 1024,
      .nullRatio = 0,
      .stringLength = 1024,
      .stringVariableLength = false,
      .allowLazyVector = false};
};

template <>
void AggregationTest::setTestKey(
    int64_t value,
    int32_t multiplier,
    vector_size_t row,
    FlatVector<StringView>* vector) {
  std::string chars;
  if (multiplier == 2) {
    chars.resize(2);
    chars[0] = (value % 64) + 32;
    chars[1] = ((value / 64) % 64) + 32;
  } else {
    chars = fmt::format("{}", value);
    for (int i = 2; i < multiplier; ++i) {
      chars = chars + fmt::format("{}", i * value);
    }
  }
  vector->set(row, StringView(chars));
}

// DM: Works
TEST_F(AggregationTest, global) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // DM: removed "sum(15)","min(15)","max(15)",
  auto op = PlanBuilder()
                .values(vectors)
                .aggregation(
                    {},
                    {"sum(c1)",
                     "sum(c2)",
                     "sum(c4)",
                     "sum(c5)",

                     "min(c1)",
                     "min(c2)",
                     "min(c3)",
                     "min(c4)",
                     "min(c5)",

                     "max(c1)",
                     "max(c2)",
                     "max(c3)",
                     "max(c4)",
                     "max(c5)"},
                    {},
                    core::AggregationNode::Step::kPartial,
                    false)
                .planNode();

  // DM: removed sum(15), min(15), max(15),
  assertQuery(
      op,
      "SELECT sum(c1), sum(c2), sum(c4), sum(c5), "
      "min(c1), min(c2), min(c3), min(c4), min(c5), "
      "max(c1), max(c2), max(c3), max(c4), max(c5) FROM tmp");
}

// DM: Works
TEST_F(AggregationTest, singleBigintKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testSingleKey<int64_t>(vectors, "c0", false, false);
  testSingleKey<int64_t>(vectors, "c0", true, false);
}

// DM: Works
TEST_F(AggregationTest, singleBigintKeyDistinct) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testSingleKey<int64_t>(vectors, "c0", false, true);
  testSingleKey<int64_t>(vectors, "c0", true, true);
}

// DM: Works
TEST_F(AggregationTest, singleStringKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testSingleKey<StringView>(vectors, "c6", false, false);
  testSingleKey<StringView>(vectors, "c6", true, false);
}

// DM: Works
TEST_F(AggregationTest, singleStringKeyDistinct) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testSingleKey<StringView>(vectors, "c6", false, true);
  testSingleKey<StringView>(vectors, "c6", true, true);
}

// DM: Works
TEST_F(AggregationTest, multiKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testMultiKey(vectors, false, false);
  testMultiKey(vectors, true, false);
}

// DM: Works
TEST_F(AggregationTest, multiKeyDistinct) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testMultiKey(vectors, false, true);
  testMultiKey(vectors, true, true);
}

// DM: Works
TEST_F(AggregationTest, aggregateOfNulls) {
  auto rowVector = makeRowVector({
      BatchMaker::createVector<TypeKind::BIGINT>(
          rowType_->childAt(0), 100, *pool_),
      makeNullConstant(TypeKind::SMALLINT, 100),
  });

  auto vectors = {rowVector};
  createDuckDbTable(vectors);

  auto op = PlanBuilder()
                .values(vectors)
                .aggregation(
                    {"c0"},
                    {"sum(c1)", "min(c1)", "max(c1)"},
                    {},
                    core::AggregationNode::Step::kPartial,
                    false)
                .planNode();

  assertQuery(op, "SELECT c0, sum(c1), min(c1), max(c1) FROM tmp GROUP BY c0");

  // global aggregation
  op = PlanBuilder()
           .values(vectors)
           .aggregation(
               {},
               {"sum(c1)", "min(c1)", "max(c1)"},
               {},
               core::AggregationNode::Step::kPartial,
               false)
           .planNode();

  assertQuery(op, "SELECT sum(c1), min(c1), max(c1) FROM tmp");
}

// DM: Works
TEST_F(AggregationTest, allKeyTypes) {
  // Covers different key types. Unlike the integer/string tests, the
  // hash table begins life in the generic mode, not array or
  // normalized key. Add types here as they become supported.
  auto rowType = ROW(
      {"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
      {DOUBLE(), REAL(), BIGINT(), INTEGER(), BOOLEAN(), VARCHAR(), DOUBLE()});

  std::vector<RowVectorPtr> batches;
  for (auto i = 0; i < 10; ++i) {
    batches.push_back(std::static_pointer_cast<RowVector>(
        BatchMaker::createBatch(rowType, 100, *pool_)));
  }
  createDuckDbTable(batches);
  auto op =
      PlanBuilder()
          .values(batches)
          .singleAggregation({"c0", "c1", "c2", "c3", "c4", "c5"}, {"sum(c6)"})
          .planNode();

  // DM: Instead of sum(c6, this was sum(1) but we don't yet support constants
  assertQuery(
      op,
      "SELECT c0, c1, c2, c3, c4, c5, sum(c6) FROM tmp "
      " GROUP BY c0, c1, c2, c3, c4, c5");
}

// DM: Works
TEST_F(AggregationTest, ignoreNullKeys) {
  // Some keys are null.
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>(
          {std::nullopt, 1, std::nullopt, 2, std::nullopt, 1, 2}),
      makeFlatVector<int32_t>({-1, 1, -2, 2, -3, 3, 4}),
  });

  auto makePlan = [&](bool ignoreNullKeys) {
    return PlanBuilder()
        .values({data})
        .aggregation(
            {"c0"},
            {"sum(c1)"},
            {},
            core::AggregationNode::Step::kPartial,
            ignoreNullKeys)
        .planNode();
  };

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int64_t>({4, 6}),
  });
  AssertQueryBuilder(makePlan(true)).assertResults(expected);

  expected = makeRowVector({
      makeNullableFlatVector<int32_t>({std::nullopt, 1, 2}),
      makeFlatVector<int64_t>({-6, 4, 6}),
  });
  AssertQueryBuilder(makePlan(false)).assertResults(expected);

  // All keys are null.
  data = makeRowVector({
      makeAllNullFlatVector<int32_t>(3),
      makeFlatVector<int32_t>({1, 2, 3}),
  });

  AssertQueryBuilder(makePlan(true)).assertEmptyResults();
}

#if 0
TEST_F(AggregationTest, largeValueRangeArray) {
  // We have keys that map to integer range. The keys are
  // a little under max array hash table size apart. This wastes 16MB of
  // memory for the array hash table. Every batch will overflow the
  // max partial memory. We check that when detecting the first
  // overflow, the partial agg rehashes itself not to use a value
  // range array hash mode and will accept more batches without
  // flushing.
  std::string string1k;
  string1k.resize(1000);
  std::vector<RowVectorPtr> vectors;
  // Make two identical ectors. The first one overflows the max size
  // but gets rehashed to smaller by using value ids instead of
  // ranges. The next vector fits in the space made freed.
  for (auto i = 0; i < 2; ++i) {
    vectors.push_back(makeRowVector(
        {makeFlatVector<int64_t>(
             1000, [](auto row) { return row % 2 == 0 ? 100 : 1000000; }),
         makeFlatVector<StringView>(
             1000, [&](auto /*row*/) { return StringView(string1k); })}));
  }
  std::vector<RowVectorPtr> expected = {makeRowVector(
      {makeFlatVector<int64_t>({100, 1000000}),
       makeFlatVector<int64_t>({1000, 1000})})};

  core::PlanNodeId partialAggId;
  core::PlanNodeId finalAggId;
  auto op = PlanBuilder()
                .values({vectors})
                .partialAggregation({"c0"}, {"array_agg(c1)"})
                .capturePlanNodeId(partialAggId)
                .finalAggregation()
                .capturePlanNodeId(finalAggId)
                .project({"c0", "cardinality(a0) as l"})
                .planNode();
  auto task = test::assertQuery(op, expected);
  auto stats = toPlanStats(task->taskStats());
  auto runtimeStats = stats.at(partialAggId).customStats;

  // The partial agg is expected to exceed max size after the first batch and
  // see that it has an oversize range based array with just 2 entries. It is
  // then expected to change hash mode and rehash.
  EXPECT_EQ(1, runtimeStats.at("hashtable.numRehashes").count);

  // The partial agg is expected to flush just once. The final agg gets one
  // batch.
  EXPECT_EQ(1, stats.at(finalAggId).inputVectors);
}

TEST_F(AggregationTest, partialAggregationMemoryLimitIncrease) {
  constexpr int64_t kGB = 1 << 30;
  auto vectors = {
      makeRowVector({makeFlatVector<int32_t>(
          100, [](auto row) { return row; }, nullEvery(5))}),
      makeRowVector({makeFlatVector<int32_t>(
          110, [](auto row) { return row + 29; }, nullEvery(7))}),
      makeRowVector({makeFlatVector<int32_t>(
          90, [](auto row) { return row - 71; }, nullEvery(7))}),
  };

  createDuckDbTable(vectors);

  struct {
    int64_t initialPartialMemoryLimit;
    int64_t extendedPartialMemoryLimit;
    bool expectedPartialOutputFlush;
    bool expectedPartialAggregationMemoryLimitIncrease;

    std::string debugString() const {
      return fmt::format(
          "initialPartialMemoryLimit: {}, extendedPartialMemoryLimit: {}, expectedPartialOutputFlush: {}, expectedPartialAggregationMemoryLimitIncrease: {}",
          initialPartialMemoryLimit,
          extendedPartialMemoryLimit,
          expectedPartialOutputFlush,
          expectedPartialAggregationMemoryLimitIncrease);
    }
  } testSettings[] = {// Set with a large initial partial aggregation memory
                      // limit and expect no flush and memory limit bump.
                      {kGB, 2 * kGB, false, false},
                      // Set with a very small initial and extended partial
                      // aggregation memory limit.
                      {100, 100, true, false},
                      // Set with a very small initial partial aggregation
                      // memory limit but large extended memory limit.
                      {100, kGB, true, true}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    // Distinct aggregation.
    core::PlanNodeId aggNodeId;
    auto task = AssertQueryBuilder(duckDbQueryRunner_)
                    .config(
                        QueryConfig::kMaxPartialAggregationMemory,
                        std::to_string(testData.initialPartialMemoryLimit))
                    .config(
                        QueryConfig::kMaxExtendedPartialAggregationMemory,
                        std::to_string(testData.extendedPartialMemoryLimit))
                    .plan(PlanBuilder()
                              .values(vectors)
                              .partialAggregation({"c0"}, {})
                              .capturePlanNodeId(aggNodeId)
                              .finalAggregation()
                              .planNode())
                    .assertResults("SELECT distinct c0 FROM tmp");
    const auto runtimeStats =
        toPlanStats(task->taskStats()).at(aggNodeId).customStats;
    if (testData.expectedPartialOutputFlush > 0) {
      EXPECT_LT(0, runtimeStats.at("flushRowCount").count);
      EXPECT_LT(0, runtimeStats.at("flushRowCount").max);
      EXPECT_LT(0, runtimeStats.at("partialAggregationPct").max);
    } else {
      EXPECT_EQ(0, runtimeStats.count("flushRowCount"));
      EXPECT_EQ(0, runtimeStats.count("partialAggregationPct"));
    }
    if (testData.expectedPartialAggregationMemoryLimitIncrease) {
      EXPECT_LT(
          testData.initialPartialMemoryLimit,
          runtimeStats.at("maxExtendedPartialAggregationMemoryUsage").max);
      EXPECT_GE(
          testData.extendedPartialMemoryLimit,
          runtimeStats.at("maxExtendedPartialAggregationMemoryUsage").max);
    } else {
      EXPECT_EQ(
          0, runtimeStats.count("maxExtendedPartialAggregationMemoryUsage"));
    }
  }
}

TEST_F(AggregationTest, partialAggregationMaybeReservationReleaseCheck) {
  auto vectors = {
      makeRowVector({makeFlatVector<int32_t>(
          100, [](auto row) { return row; }, nullEvery(5))}),
      makeRowVector({makeFlatVector<int32_t>(
          110, [](auto row) { return row + 29; }, nullEvery(7))}),
      makeRowVector({makeFlatVector<int32_t>(
          90, [](auto row) { return row - 71; }, nullEvery(7))}),
  };

  createDuckDbTable(vectors);

  constexpr int64_t kGB = 1 << 30;
  const int64_t kMaxPartialMemoryUsage = 1 * kGB;
  // Make sure partial aggregation runs out of memory after first batch.
  CursorParameters params;
  params.queryCtx = core::QueryCtx::create(executor_.get());
  params.queryCtx->testingOverrideConfigUnsafe({
      {QueryConfig::kMaxPartialAggregationMemory,
       std::to_string(kMaxPartialMemoryUsage)},
      {QueryConfig::kMaxExtendedPartialAggregationMemory,
       std::to_string(kMaxPartialMemoryUsage)},
  });

  core::PlanNodeId aggNodeId;
  params.planNode = PlanBuilder()
                        .values(vectors)
                        .partialAggregation({"c0"}, {})
                        .capturePlanNodeId(aggNodeId)
                        .finalAggregation()
                        .planNode();
  auto task = assertQuery(params, "SELECT distinct c0 FROM tmp");
  const auto runtimeStats =
      toPlanStats(task->taskStats()).at(aggNodeId).customStats;
  EXPECT_EQ(0, runtimeStats.count("flushRowCount"));
  EXPECT_EQ(0, runtimeStats.count("maxExtendedPartialAggregationMemoryUsage"));
  EXPECT_EQ(0, runtimeStats.count("partialAggregationPct"));
  // Check all the reserved memory have been released.
  EXPECT_EQ(0, task->pool()->availableReservation());
  EXPECT_GT(kMaxPartialMemoryUsage, task->pool()->reservedBytes());
}

TEST_F(AggregationTest, spillAll) {
  auto inputs = makeVectors(rowType_, 100, 10);

  const auto numDistincts =
      AssertQueryBuilder(PlanBuilder()
                             .values(inputs)
                             .singleAggregation({"c0"}, {}, {})
                             .planNode())
          .copyResults(pool_.get())
          ->size();

  auto plan = PlanBuilder()
                  .values(inputs)
                  .singleAggregation({"c0"}, {"array_agg(c1)"})
                  .planNode();

  auto results = AssertQueryBuilder(plan).copyResults(pool_.get());

  for (int numPartitionBits : {1, 2, 3}) {
    auto tempDirectory = exec::test::TempDirectoryPath::create();
    auto queryCtx = core::QueryCtx::create(executor_.get());
    TestScopedSpillInjection scopedSpillInjection(100);
    auto task = AssertQueryBuilder(plan)
                    .spillDirectory(tempDirectory->getPath())
                    .config(QueryConfig::kSpillEnabled, true)
                    .config(QueryConfig::kAggregationSpillEnabled, true)
                    .config(
                        QueryConfig::kSpillNumPartitionBits,
                        std::to_string(numPartitionBits))
                    .assertResults(results);

    auto stats = task->taskStats().pipelineStats;
    ASSERT_LT(
        0, stats[0].operatorStats[1].runtimeStats[Operator::kSpillRuns].count);
    // Check spilled bytes.
    ASSERT_LT(0, stats[0].operatorStats[1].spilledInputBytes);
    ASSERT_LT(0, stats[0].operatorStats[1].spilledBytes);
    ASSERT_EQ(
        stats[0].operatorStats[1].spilledPartitions, 1 << numPartitionBits);
    // Verifies all the rows have been spilled.
    ASSERT_EQ(stats[0].operatorStats[1].spilledRows, numDistincts);
    OperatorTestBase::deleteTaskAndCheckSpillDirectory(task);
  }
}

TEST_F(AggregationTest, disableNonBooleanMasks) {
  auto data = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({1, -1, 0, -2, 10}),
       makeFlatVector<std::string>({"a", "a", "b", "c", "a"})});

  auto plan = PlanBuilder()
                  .values({data})
                  .aggregation(
                      {"c1"},
                      {"count(c0) FILTER(WHERE c0)"},
                      {},
                      core::AggregationNode::Step::kPartial,
                      false)
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "FILTER(WHERE..) clause must use masks that are BOOLEAN");

  // Planbuilder doesnt allow expressions in FILTER clauses
  plan = PlanBuilder()
             .values({data})
             .project({"c0", "c1", "c0 > 0 as mask"})
             .aggregation(
                 {"c1"},
                 {"count(c0) FILTER(WHERE mask)"},
                 {},
                 core::AggregationNode::Step::kPartial,
                 true)
             .planNode();

  AssertQueryBuilder(plan).copyResults(pool());
}

TEST_F(AggregationTest, outputBatchSizeCheckWithoutSpill) {
  const int vectorSize = 100;
  const std::string strValue(1L << 20, 'a');

  RowVectorPtr largeVector = makeRowVector(
      {makeFlatVector<int32_t>(vectorSize, [&](auto row) { return row; }),
       makeFlatVector<StringView>(
           vectorSize, [&](auto /*unused*/) { return StringView(strValue); })});
  auto largeRowType = asRowType(largeVector->type());

  RowVectorPtr smallVector = makeRowVector(
      {makeFlatVector<int32_t>(vectorSize, [&](auto row) { return row; }),
       makeFlatVector<int32_t>(vectorSize, [&](auto row) { return row; })});
  auto smallRowType = asRowType(smallVector->type());

  struct {
    bool smallInput;
    uint32_t maxOutputRows;
    uint32_t maxOutputBytes;
    uint32_t expectedNumOutputVectors;

    std::string debugString() const {
      return fmt::format(
          "smallInput: {} maxOutputRows: {}, maxOutputBytes: {}, expectedNumOutputVectors: {}",
          smallInput,
          maxOutputRows,
          succinctBytes(maxOutputBytes),
          expectedNumOutputVectors);
    }
  } testSettings[] = {
      {true, 1000, 1000'000, 1},
      {true, 10, 1000'000, 10},
      {true, 1, 1000'000, 100},
      {true, 1, 1, 100},
      {true, 10, 1, 100},
      {true, 100, 1, 100},
      {true, 1000, 1, 100},
      {false, 1000, 1, 100},
      {false, 1000, 1000'000'000, 1},
      {false, 100, 1000'000'000, 1},
      {false, 10, 1000'000'000, 10},
      {false, 1, 1000'000'000, 100},
      {false, 1, 1, 100}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    std::vector<RowVectorPtr> inputs;
    if (testData.smallInput) {
      inputs.push_back(smallVector);
    } else {
      inputs.push_back(largeVector);
    }
    createDuckDbTable(inputs);
    core::PlanNodeId aggrNodeId;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .config(
                QueryConfig::kPreferredOutputBatchBytes,
                std::to_string(testData.maxOutputBytes))
            .config(
                QueryConfig::kMaxOutputBatchRows,
                std::to_string(testData.maxOutputRows))
            .plan(PlanBuilder()
                      .values(inputs)
                      .singleAggregation({"c0"}, {"array_agg(c1)"})
                      .capturePlanNodeId(aggrNodeId)
                      .planNode())
            .assertResults("SELECT c0, array_agg(c1) FROM tmp GROUP BY 1");

    ASSERT_EQ(
        toPlanStats(task->taskStats()).at(aggrNodeId).outputVectors,
        testData.expectedNumOutputVectors);
  }
}

TEST_F(AggregationTest, distinctWithSpilling) {
  struct TestParam {
    std::vector<RowVectorPtr> inputs;
    std::function<void(uint32_t)> expectedSpillFilesCheck{nullptr};
  };

  std::vector<TestParam> testParams{
      {makeVectors(rowType_, 10, 100),
       [](uint32_t spilledFiles) { ASSERT_GE(spilledFiles, 100); }},
      {{makeRowVector(
           {"c0"},
           {makeFlatVector<int64_t>(
               2'000, [](vector_size_t /* unused */) { return 100; })})},
       [](uint32_t spilledFiles) { ASSERT_EQ(spilledFiles, 1); }}};

  for (const auto& testParam : testParams) {
    createDuckDbTable(testParam.inputs);
    auto spillDirectory = exec::test::TempDirectoryPath::create();
    core::PlanNodeId aggrNodeId;
    TestScopedSpillInjection scopedSpillInjection(100);
    auto task = AssertQueryBuilder(duckDbQueryRunner_)
                    .spillDirectory(spillDirectory->getPath())
                    .config(QueryConfig::kSpillEnabled, true)
                    .config(QueryConfig::kAggregationSpillEnabled, true)
                    .plan(PlanBuilder()
                              .values(testParam.inputs)
                              .singleAggregation({"c0"}, {}, {})
                              .capturePlanNodeId(aggrNodeId)
                              .planNode())
                    .assertResults("SELECT distinct c0 FROM tmp");

    // Verify that spilling is not triggered.
    const auto planNodeStatsMap = toPlanStats(task->taskStats());
    const auto& aggrNodeStats = planNodeStatsMap.at(aggrNodeId);
    ASSERT_GT(aggrNodeStats.spilledInputBytes, 0);
    ASSERT_EQ(aggrNodeStats.spilledPartitions, 8);
    ASSERT_GT(aggrNodeStats.spilledBytes, 0);
    testParam.expectedSpillFilesCheck(aggrNodeStats.spilledFiles);
    OperatorTestBase::deleteTaskAndCheckSpillDirectory(task);
  }
}

TEST_F(AggregationTest, preGroupedAggregationWithSpilling) {
  std::vector<RowVectorPtr> vectors;
  int64_t val = 0;
  for (int32_t i = 0; i < 4; ++i) {
    vectors.push_back(makeRowVector(
        {// Pre-grouped key.
         makeFlatVector<int64_t>(10, [&](auto /*row*/) { return val++ / 5; }),
         // Payload.
         makeFlatVector<int64_t>(10, [](auto row) { return row; }),
         makeFlatVector<int64_t>(10, [](auto row) { return row; })}));
  }
  createDuckDbTable(vectors);
  auto spillDirectory = exec::test::TempDirectoryPath::create();
  core::PlanNodeId aggrNodeId;
  TestScopedSpillInjection scopedSpillInjection(100);
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .spillDirectory(spillDirectory->getPath())
          .config(QueryConfig::kSpillEnabled, true)
          .config(QueryConfig::kAggregationSpillEnabled, true)
          .plan(PlanBuilder()
                    .values(vectors)
                    .aggregation(
                        {"c0", "c1"},
                        {"c0"},
                        {"sum(c2)"},
                        {},
                        core::AggregationNode::Step::kSingle,
                        false)
                    .capturePlanNodeId(aggrNodeId)
                    .planNode())
          .assertResults("SELECT c0, c1, sum(c2) FROM tmp GROUP BY c0, c1");
  auto stats = task->taskStats().pipelineStats;
  // Verify that spilling is not triggered.
  ASSERT_EQ(toPlanStats(task->taskStats()).at(aggrNodeId).spilledInputBytes, 0);
  ASSERT_EQ(toPlanStats(task->taskStats()).at(aggrNodeId).spilledBytes, 0);
  OperatorTestBase::deleteTaskAndCheckSpillDirectory(task);
}

TEST_F(AggregationTest, adaptiveOutputBatchRows) {
  int32_t defaultOutputBatchRows = 10;
  vector_size_t size = defaultOutputBatchRows * 5;
  auto vectors = std::vector<RowVectorPtr>(
      8,
      makeRowVector(
          {"k0", "c0"},
          {makeFlatVector<int32_t>(size, [&](auto row) { return row; }),
           makeFlatVector<int8_t>(size, [&](auto row) { return row % 2; })}));

  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .singleAggregation({"k0"}, {"sum(c0)"})
                  .planNode();

  // Test setting larger output batch bytes will create batches of greater
  // number of rows.
  {
    auto outputBatchBytes = "1000";
    auto task =
        AssertQueryBuilder(plan, duckDbQueryRunner_)
            .config(QueryConfig::kPreferredOutputBatchBytes, outputBatchBytes)
            .assertResults("SELECT k0, SUM(c0) FROM tmp GROUP BY k0");

    auto aggOpStats = task->taskStats().pipelineStats[0].operatorStats[1];
    ASSERT_GT(
        aggOpStats.outputPositions / aggOpStats.outputVectors,
        defaultOutputBatchRows);
  }

  // Test setting smaller output batch bytes will create batches of fewer
  // number of rows.
  {
    auto outputBatchBytes = "1";
    auto task =
        AssertQueryBuilder(plan, duckDbQueryRunner_)
            .config(QueryConfig::kPreferredOutputBatchBytes, outputBatchBytes)
            .assertResults("SELECT k0, SUM(c0) FROM tmp GROUP BY k0");

    auto aggOpStats = task->taskStats().pipelineStats[0].operatorStats[1];
    ASSERT_LT(
        aggOpStats.outputPositions / aggOpStats.outputVectors,
        defaultOutputBatchRows);
  }
}

TEST_F(AggregationTest, noAggregationsNoGroupingKeys) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation({}, {})
                  .finalAggregation()
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  // 1 row.
  ASSERT_EQ(result->size(), 1);
  // Zero columns.
  ASSERT_EQ(result->type()->size(), 0);
}

// Reproduces hang in partial distinct aggregation described in
// https://github.com/facebookincubator/velox/issues/7967 .
TEST_F(AggregationTest, distinctHang) {
  static const int64_t kMin = std::numeric_limits<int32_t>::min();
  static const int64_t kMax = std::numeric_limits<int32_t>::max();
  auto data = makeRowVector({
      makeFlatVector<int64_t>(
          5'000,
          [](auto row) {
            if (row % 2 == 0) {
              return kMin + row;
            } else {
              return kMax - row;
            }
          }),
      makeFlatVector<int64_t>(
          5'000,
          [](auto row) {
            if (row % 2 == 0) {
              return kMin - row;
            } else {
              return kMax + row;
            }
          }),
  });

  auto newData = makeRowVector({
      makeFlatVector<int64_t>(
          5'000, [](auto row) { return kMin + row + 5'000; }),
      makeFlatVector<int64_t>(5'000, [](auto row) { return kMin - row; }),
  });

  createDuckDbTable({data, newData});

  core::PlanNodeId aggNodeId;
  auto plan = PlanBuilder()
                  .values({data, newData, data})
                  .partialAggregation({"c0", "c1"}, {})
                  .capturePlanNodeId(aggNodeId)
                  .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .config(QueryConfig::kMaxPartialAggregationMemory, 400000)
      .assertResults("SELECT distinct c0, c1 FROM tmp");
}

// Verify that ORDER BY clause is ignored for aggregates that are not order
// sensitive.
TEST_F(AggregationTest, ignoreOrderBy) {
  auto data = makeRowVector({
      makeFlatVector<int16_t>({1, 1, 2, 2, 1, 2, 1}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60, 70}),
      makeFlatVector<int32_t>({11, 44, 22, 55, 33, 66, 77}),
  });

  createDuckDbTable({data});

  // Sorted aggregations over same inputs.
  auto plan =
      PlanBuilder()
          .values({data})
          .partialAggregation(
              {"c0"}, {"sum(c1 ORDER BY c2 DESC)", "avg(c1 ORDER BY c3)"})
          .finalAggregation()
          .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT c0, sum(c1), avg(c1) FROM tmp GROUP BY 1");
}

class TestAccumulator {
 public:
  ~TestAccumulator() {
    VELOX_FAIL("Destructor should not be called.");
  }
};

class TestAggregate : public Aggregate {
 public:
  explicit TestAggregate(TypePtr resultType) : Aggregate(resultType) {}

  void addRawInput(
      char** /*groups*/,
      const SelectivityVector& /*rows*/,
      const std::vector<VectorPtr>& /*args*/,
      bool /*mayPushdown*/) override {
    VELOX_UNSUPPORTED("This shouldn't get called.");
  }

  void extractValues(
      char** /*groups*/,
      int32_t /*numGroups*/,
      VectorPtr* /*result*/) override {
    VELOX_UNSUPPORTED("This shouldn't get called.");
  }

  void addIntermediateResults(
      char** /*groups*/,
      const SelectivityVector& /*rows*/,
      const std::vector<VectorPtr>& /*args*/,
      bool /*mayPushdown*/) override {
    VELOX_UNSUPPORTED("This shouldn't get called.");
  }

  void addSingleGroupRawInput(
      char* /*group*/,
      const SelectivityVector& /*rows*/,
      const std::vector<VectorPtr>& /*args*/,
      bool /*mayPushdown*/) override {
    VELOX_UNSUPPORTED("This shouldn't get called.");
  }

  void addSingleGroupIntermediateResults(
      char* /*group*/,
      const SelectivityVector& /*rows*/,
      const std::vector<VectorPtr>& /*args*/,
      bool /*mayPushdown*/) override {
    VELOX_UNSUPPORTED("This shouldn't get called.");
  }

  void extractAccumulators(
      char** /*groups*/,
      int32_t /*numGroups*/,
      VectorPtr* /*result*/) override {
    VELOX_UNSUPPORTED("This shouldn't get called.");
  }

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(TestAccumulator);
  }

  bool destroyCalled = false;

 protected:
  void initializeNewGroupsInternal(
      char** /*groups*/,
      folly::Range<const vector_size_t*> /*indices*/) override {
    VELOX_UNSUPPORTED("This shouldn't get called.");
  }

  void destroyInternal(folly::Range<char**> groups) override {
    destroyCalled = true;
    destroyAccumulators<TestAccumulator>(groups);
  }
};

TEST_F(AggregationTest, destroyAfterPartialInitialization) {
  TestAggregate agg(INTEGER());

  Accumulator accumulator(
      true, // isFixedSize
      sizeof(TestAccumulator), // fixedSize
      true, // usesExternalMemory, this is set to force RowContainer.clear() to
            // call eraseRows.
      1, // alignment
      INTEGER(), // spillType,
      [](folly::Range<char**>, VectorPtr&) {
        VELOX_UNSUPPORTED("This shouldn't get called.");
      },
      [&agg](folly::Range<char**> groups) { agg.destroy(groups); });

  RowContainer rows(
      {}, // keyTypes
      false, // nullableKeys
      {accumulator},
      {}, // dependentTypes
      false, // hasNext
      false, // isJoinBuild
      false, // hasProbedFlag
      false, // hasNormalizedKeys
      pool());
  const auto rowColumn = rows.columnAt(0);
  agg.setOffsets(
      rowColumn.offset(),
      rowColumn.nullByte(),
      rowColumn.nullMask(),
      rowColumn.initializedByte(),
      rowColumn.initializedMask(),
      rows.rowSizeOffset());
  rows.newRow();
  rows.clear();

  ASSERT_TRUE(agg.destroyCalled);
}

TEST_F(AggregationTest, nanKeys) {
  // Some keys are NaNs.
  auto kNaN = std::numeric_limits<double>::quiet_NaN();
  auto kSNaN = std::numeric_limits<double>::signaling_NaN();
  // Columns reused across test cases.
  auto c0 = makeFlatVector<double>({kNaN, 1, kNaN, 2, kSNaN, 1, 2});
  auto c1 = makeFlatVector<int32_t>({1, 1, 1, 1, 1, 1, 1});
  // Expected result columns reused across test cases. A deduplicated version of
  // c0 and c1.
  auto e0 = makeFlatVector<double>({1, 2, kNaN});
  auto e1 = makeFlatVector<int32_t>({1, 1, 1});

  auto testDistinctAgg = [&](std::vector<std::string> aggKeys,
                             std::vector<VectorPtr> inputCols,
                             std::vector<VectorPtr> expectedCols) {
    auto plan = PlanBuilder()
                    .values({makeRowVector(inputCols)})
                    .singleAggregation(aggKeys, {}, {})
                    .planNode();
    AssertQueryBuilder(plan).assertResults(makeRowVector(expectedCols));
  };

  // Test with a primitive type key.
  testDistinctAgg({"c0"}, {c0}, {e0});
  // Multiple key columns.
  testDistinctAgg({"c0", "c1"}, {c0, c1}, {e0, e1});

  // Test with a complex type key.
  testDistinctAgg({"c0"}, {makeRowVector({c0, c1})}, {makeRowVector({e0, e1})});
  // Multiple key columns.
  testDistinctAgg(
      {"c0", "c1"},
      {makeRowVector({c0, c1}), c1},
      {makeRowVector({e0, e1}), e1});
}
#endif
} // namespace facebook::velox::exec::test
