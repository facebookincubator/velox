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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/memory/SharedArbitrator.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using facebook::velox::test::VectorTestBase;

namespace {

class CudfNestedLoopJoinTest : public testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    FLAGS_velox_enable_memory_usage_track_in_default_memory_pool = true;
    memory::SharedArbitrator::registerFactory();

    memory::MemoryManager::Options options;
    options.allocatorCapacity = 8L << 30;
    options.arbitratorCapacity = 6L << 30;
    options.arbitratorKind = "SHARED";
    options.checkUsageLeak = true;

    using ExtraConfig = memory::SharedArbitrator::ExtraConfig;
    options.extraArbitratorConfigs = {
        {std::string(ExtraConfig::kMemoryPoolInitialCapacity),
         std::to_string(512 << 20) + "B"},
        {std::string(ExtraConfig::kGlobalArbitrationEnabled), "true"},
    };

    memory::MemoryManager::testingSetInstance(options);
    functions::prestosql::registerAllScalarFunctions();
    if (!isRegisteredVectorSerde()) {
      serializer::presto::PrestoVectorSerde::registerVectorSerde();
    }
  }

  static void TearDownTestSuite() {
    memory::SharedArbitrator::unregisterFactory();
  }

  void SetUp() override {
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    waitForAllTasksToBeDeleted();
  }

  // Each PlanBuilder must use a unique planNodeIdGenerator to avoid
  // duplicate plan node IDs.
  core::PlanNodePtr makeCrossJoinPlan(
      const std::vector<RowVectorPtr>& probeData,
      const std::vector<RowVectorPtr>& buildData,
      const std::vector<std::string>& outputLayout) {
    auto planNodeIdGenerator =
        std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values(probeData)
        .nestedLoopJoin(
            PlanBuilder(planNodeIdGenerator).values(buildData).planNode(),
            outputLayout)
        .planNode();
  }
};

TEST_F(CudfNestedLoopJoinTest, basicCrossJoin) {
  auto probeData = makeRowVector(
      {"p0", "p1"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<StringView>({"a", "b", "c"}),
      });
  auto buildData = makeRowVector(
      {"b0"},
      {
          makeFlatVector<int64_t>({10, 20}),
      });

  auto plan = makeCrossJoinPlan({probeData}, {buildData}, {"p0", "p1", "b0"});

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
      makeFlatVector<StringView>({"a", "a", "b", "b", "c", "c"}),
      makeFlatVector<int64_t>({10, 20, 10, 20, 10, 20}),
  });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfNestedLoopJoinTest, emptyBuild) {
  auto probeData = makeRowVector(
      {"p0"}, {makeFlatVector<int32_t>({1, 2, 3})});
  auto buildData = makeRowVector(
      {"b0"}, {makeFlatVector<int64_t>({})});

  auto plan = makeCrossJoinPlan({probeData}, {buildData}, {"p0", "b0"});
  auto result = AssertQueryBuilder(plan).copyResults(pool());
  ASSERT_EQ(result->size(), 0);
}

TEST_F(CudfNestedLoopJoinTest, emptyProbe) {
  auto probeData = makeRowVector(
      {"p0"}, {makeFlatVector<int32_t>({})});
  auto buildData = makeRowVector(
      {"b0"}, {makeFlatVector<int64_t>({10, 20})});

  auto plan = makeCrossJoinPlan({probeData}, {buildData}, {"p0", "b0"});
  auto result = AssertQueryBuilder(plan).copyResults(pool());
  ASSERT_EQ(result->size(), 0);
}

TEST_F(CudfNestedLoopJoinTest, bothEmpty) {
  auto probeData = makeRowVector(
      {"p0"}, {makeFlatVector<int32_t>({})});
  auto buildData = makeRowVector(
      {"b0"}, {makeFlatVector<int64_t>({})});

  auto plan = makeCrossJoinPlan({probeData}, {buildData}, {"p0", "b0"});
  auto result = AssertQueryBuilder(plan).copyResults(pool());
  ASSERT_EQ(result->size(), 0);
}

TEST_F(CudfNestedLoopJoinTest, singleRowBuild) {
  auto probeData = makeRowVector(
      {"p0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto buildData = makeRowVector(
      {"b0"}, {makeFlatVector<int64_t>({42})});

  auto plan = makeCrossJoinPlan({probeData}, {buildData}, {"p0", "b0"});

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({42, 42, 42, 42, 42}),
  });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfNestedLoopJoinTest, multipleProbeAndBuildBatches) {
  auto probeBatch1 = makeRowVector(
      {"p0"}, {makeFlatVector<int32_t>({1, 2})});
  auto probeBatch2 = makeRowVector(
      {"p0"}, {makeFlatVector<int32_t>({3})});
  auto buildBatch1 = makeRowVector(
      {"b0"}, {makeFlatVector<int64_t>({10})});
  auto buildBatch2 = makeRowVector(
      {"b0"}, {makeFlatVector<int64_t>({20})});

  auto plan = makeCrossJoinPlan(
      {probeBatch1, probeBatch2},
      {buildBatch1, buildBatch2},
      {"p0", "b0"});

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
      makeFlatVector<int64_t>({10, 20, 10, 20, 10, 20}),
  });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfNestedLoopJoinTest, outputColumnOrder) {
  auto probeData = makeRowVector(
      {"p0", "p1"},
      {
          makeFlatVector<int32_t>({1}),
          makeFlatVector<double>({1.5}),
      });
  auto buildData = makeRowVector(
      {"b0"},
      {
          makeFlatVector<int64_t>({99}),
      });

  auto planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeData})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildData})
                          .planNode(),
                      {"b0", "p0", "p1"})
                  .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({99}),
      makeFlatVector<int32_t>({1}),
      makeFlatVector<double>({1.5}),
  });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfNestedLoopJoinTest, largerCrossJoin) {
  constexpr int32_t kProbeRows = 100;
  constexpr int32_t kBuildRows = 50;

  std::vector<int32_t> probeValues(kProbeRows);
  std::iota(probeValues.begin(), probeValues.end(), 0);

  std::vector<int64_t> buildValues(kBuildRows);
  std::iota(buildValues.begin(), buildValues.end(), 1000);

  auto probeData = makeRowVector(
      {"p0"}, {makeFlatVector<int32_t>(probeValues)});
  auto buildData = makeRowVector(
      {"b0"}, {makeFlatVector<int64_t>(buildValues)});

  auto plan = makeCrossJoinPlan({probeData}, {buildData}, {"p0", "b0"});
  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), kProbeRows * kBuildRows);

  auto* probeCol = result->childAt(0)->as<FlatVector<int32_t>>();
  auto* buildCol = result->childAt(1)->as<FlatVector<int64_t>>();

  for (int32_t i = 0; i < kProbeRows * kBuildRows; ++i) {
    ASSERT_EQ(probeCol->valueAt(i), i / kBuildRows) << "at row " << i;
    ASSERT_EQ(buildCol->valueAt(i), 1000 + (i % kBuildRows)) << "at row " << i;
  }
}

// Ported from CPU NestedLoopJoinTest.emptyBuildOrProbeWithoutFilter
// (cross join subset only -- GPU only supports inner cross join).
TEST_F(CudfNestedLoopJoinTest, emptyBuildOrProbeWithoutFilter) {
  auto probeData = makeRowVector(
      {"p0"}, {makeFlatVector<int32_t>({1, 2, 3})});
  auto emptyProbe = makeRowVector(
      {"p0"}, {makeFlatVector<int32_t>({})});
  auto buildData = makeRowVector(
      {"b0"}, {makeFlatVector<int64_t>({10, 20})});
  auto emptyBuild = makeRowVector(
      {"b0"}, {makeFlatVector<int64_t>({})});

  // probe x emptyBuild = 0 rows
  auto plan1 = makeCrossJoinPlan({probeData}, {emptyBuild}, {"p0", "b0"});
  ASSERT_EQ(AssertQueryBuilder(plan1).copyResults(pool())->size(), 0);

  // emptyProbe x build = 0 rows
  auto plan2 = makeCrossJoinPlan({emptyProbe}, {buildData}, {"p0", "b0"});
  ASSERT_EQ(AssertQueryBuilder(plan2).copyResults(pool())->size(), 0);

  // emptyProbe x emptyBuild = 0 rows
  auto plan3 = makeCrossJoinPlan({emptyProbe}, {emptyBuild}, {"p0", "b0"});
  ASSERT_EQ(AssertQueryBuilder(plan3).copyResults(pool())->size(), 0);

  // probe x build = 6 rows
  auto plan4 = makeCrossJoinPlan({probeData}, {buildData}, {"p0", "b0"});
  ASSERT_EQ(AssertQueryBuilder(plan4).copyResults(pool())->size(), 6);
}

// Ported from CPU NestedLoopJoinTest.allTypes (cross join only).
TEST_F(CudfNestedLoopJoinTest, allTypesCrossJoin) {
  auto probeData = makeRowVector(
      {"p0", "p1", "p2", "p3", "p4", "p5"},
      {
          makeFlatVector<int64_t>({1, 2}),
          makeFlatVector<StringView>({"hello", "world"}),
          makeFlatVector<float>({1.1f, 2.2f}),
          makeFlatVector<double>({10.1, 20.2}),
          makeFlatVector<int32_t>({100, 200}),
          makeFlatVector<int16_t>({10, 20}),
      });
  auto buildData = makeRowVector(
      {"b0", "b1"},
      {
          makeFlatVector<int64_t>({99}),
          makeFlatVector<StringView>({"gpu"}),
      });

  auto plan = makeCrossJoinPlan(
      {probeData}, {buildData},
      {"p0", "p1", "p2", "p3", "p4", "p5", "b0", "b1"});

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeFlatVector<StringView>({"hello", "world"}),
      makeFlatVector<float>({1.1f, 2.2f}),
      makeFlatVector<double>({10.1, 20.2}),
      makeFlatVector<int32_t>({100, 200}),
      makeFlatVector<int16_t>({10, 20}),
      makeFlatVector<int64_t>({99, 99}),
      makeFlatVector<StringView>({"gpu", "gpu"}),
  });

  AssertQueryBuilder(plan).assertResults(expected);
}

// Ported from CPU NestedLoopJoinTest.mergeBuildVectors (cross join only).
TEST_F(CudfNestedLoopJoinTest, mergeBuildVectors) {
  auto buildBatch1 = makeRowVector(
      {"b0"}, {makeFlatVector<int64_t>({1, 2})});
  auto buildBatch2 = makeRowVector(
      {"b0"}, {makeFlatVector<int64_t>({3, 4})});
  auto buildBatch3 = makeRowVector(
      {"b0"}, {makeFlatVector<int64_t>({5, 6, 7})});
  auto probeData = makeRowVector(
      {"p0"}, {makeFlatVector<int32_t>({10, 20})});

  auto plan = makeCrossJoinPlan(
      {probeData},
      {buildBatch1, buildBatch2, buildBatch3},
      {"p0", "b0"});
  auto result = AssertQueryBuilder(plan).copyResults(pool());

  // 2 probe rows x 7 build rows = 14 output rows
  ASSERT_EQ(result->size(), 14);
}

// Test cross join with nullable columns.
TEST_F(CudfNestedLoopJoinTest, withNulls) {
  auto probeData = makeRowVector(
      {"p0"},
      {
          makeNullableFlatVector<int64_t>({1, std::nullopt, 3}),
      });
  auto buildData = makeRowVector(
      {"b0"},
      {
          makeNullableFlatVector<int64_t>({std::nullopt, 20}),
      });

  auto plan = makeCrossJoinPlan({probeData}, {buildData}, {"p0", "b0"});

  auto expected = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {1, 1, std::nullopt, std::nullopt, 3, 3}),
      makeNullableFlatVector<int64_t>(
          {std::nullopt, 20, std::nullopt, 20, std::nullopt, 20}),
  });

  AssertQueryBuilder(plan).assertResults(expected);
}

// Ported from CPU NestedLoopJoinTest.zeroColumnBuild.
TEST_F(CudfNestedLoopJoinTest, zeroColumnBuild) {
  auto probeData = makeRowVector(
      {"p0"}, {makeFlatVector<int32_t>({1, 2, 3})});

  // Build side with no columns but 2 rows.
  auto buildData = makeRowVector({}, 2);

  auto planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeData})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildData})
                          .planNode(),
                      {"p0"})
                  .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
  });

  AssertQueryBuilder(plan).assertResults(expected);
}

} // namespace
