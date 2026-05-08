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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <fmt/format.h>
#include <folly/String.h>
#include <gtest/gtest.h>

#include <limits>
#include <sstream>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

class CudfWindowTest : public testing::Test,
                       public facebook::velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    if (!isRegisteredVectorSerde()) {
      serializer::presto::PrestoVectorSerde::registerVectorSerde();
    }
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    window::prestosql::registerAllWindowFunctions();
    parse::registerTypeResolver();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
  }
};

TEST_F(CudfWindowTest, rowNumberPartitionOrder) {
  auto data = makeRowVector(
      {"id", "val"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 15, 25, 35}),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({"row_number() over (partition by id order by val)"})
                  .orderBy({"id ASC NULLS LAST", "val ASC NULLS LAST"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"id", "val", "w0"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 15, 25, 35}),
          makeFlatVector<int64_t>({1, 2, 3, 1, 2, 3}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfWindowTest, lagLead) {
  auto data = makeRowVector(
      {"id", "val"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({100, 200, 300, 10, 20}),
      });

  auto plan =
      PlanBuilder()
          .values({data})
          .window({
              "lag(val, 1) over (partition by id order by val) as lag_val",
              "lead(val, 1) over (partition by id order by val) as lead_val",
          })
          .orderBy({"id ASC NULLS LAST", "val ASC NULLS LAST"}, false)
          .planNode();

  auto lagValues = makeNullableFlatVector<int64_t>(
      {std::nullopt, 100, 200, std::nullopt, 10});
  auto leadValues = makeNullableFlatVector<int64_t>(
      {200, 300, std::nullopt, 20, std::nullopt});
  auto expected = makeRowVector(
      {"id", "val", "lag_val", "lead_val"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({100, 200, 300, 10, 20}),
          lagValues,
          leadValues,
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfWindowTest, rankPartitionOrder) {
  // Ties: partition 1 has two rows with val=20.
  auto data = makeRowVector(
      {"id", "val"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 2, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 20, 30, 5, 5, 15}),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({"rank() over (partition by id order by val)"})
                  .orderBy({"id ASC NULLS LAST", "val ASC NULLS LAST"}, false)
                  .planNode();

  // rank(): tied rows get the same rank, next rank skips.
  // Partition 1 (10,20,20,30): 1,2,2,4
  // Partition 2 (5,5,15): 1,1,3
  auto expected = makeRowVector(
      {"id", "val", "w0"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 2, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 20, 30, 5, 5, 15}),
          makeFlatVector<int64_t>({1, 2, 2, 4, 1, 1, 3}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfWindowTest, denseRankPartitionOrder) {
  auto data = makeRowVector(
      {"id", "val"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 2, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 20, 30, 5, 5, 15}),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({"dense_rank() over (partition by id order by val)"})
                  .orderBy({"id ASC NULLS LAST", "val ASC NULLS LAST"}, false)
                  .planNode();

  // dense_rank(): tied rows get the same rank, no gaps.
  // Partition 1 (10,20,20,30): 1,2,2,3
  // Partition 2 (5,5,15): 1,1,2
  auto expected = makeRowVector(
      {"id", "val", "w0"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 2, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 20, 30, 5, 5, 15}),
          makeFlatVector<int64_t>({1, 2, 2, 3, 1, 1, 2}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfWindowTest, firstValueLastValue) {
  auto data = makeRowVector(
      {"id", "val"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 100, 200}),
      });

  auto plan =
      PlanBuilder()
          .values({data})
          .window({
              "first_value(val) over (partition by id order by val) as fv",
              "last_value(val) over (partition by id order by val "
              "rows between unbounded preceding and unbounded following) as lv",
          })
          .orderBy({"id ASC NULLS LAST", "val ASC NULLS LAST"}, false)
          .planNode();

  auto expected = makeRowVector(
      {"id", "val", "fv", "lv"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 100, 200}),
          makeFlatVector<int64_t>({10, 10, 10, 100, 100}),
          makeFlatVector<int64_t>({30, 30, 30, 200, 200}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfWindowTest, sumWindow) {
  auto data = makeRowVector(
      {"id", "val"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 100, 200}),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({
                      "sum(val) over (partition by id "
                      "rows between unbounded preceding and "
                      "unbounded following) as total",
                  })
                  .orderBy({"id ASC NULLS LAST", "val ASC NULLS LAST"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"id", "val", "total"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 100, 200}),
          makeFlatVector<int64_t>({60, 60, 60, 300, 300}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfWindowTest, minMaxWindow) {
  auto data = makeRowVector(
      {"id", "val"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 100, 200}),
      });

  auto plan =
      PlanBuilder()
          .values({data})
          .window({
              "min(val) over (partition by id "
              "rows between unbounded preceding and unbounded following) as mn",
              "max(val) over (partition by id "
              "rows between unbounded preceding and unbounded following) as mx",
          })
          .orderBy({"id ASC NULLS LAST", "val ASC NULLS LAST"}, false)
          .planNode();

  auto expected = makeRowVector(
      {"id", "val", "mn", "mx"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 100, 200}),
          makeFlatVector<int64_t>({10, 10, 10, 100, 100}),
          makeFlatVector<int64_t>({30, 30, 30, 200, 200}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfWindowTest, countWindow) {
  auto data = makeRowVector(
      {"id", "val"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 100, 200}),
      });

  auto plan =
      PlanBuilder()
          .values({data})
          .window({
              "count(val) over (partition by id "
              "rows between unbounded preceding and unbounded following) as cnt",
          })
          .orderBy({"id ASC NULLS LAST", "val ASC NULLS LAST"}, false)
          .planNode();

  auto expected = makeRowVector(
      {"id", "val", "cnt"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 100, 200}),
          makeFlatVector<int64_t>({3, 3, 3, 2, 2}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfWindowTest, avgWindow) {
  auto data = makeRowVector(
      {"id", "val"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<double>({10.0, 20.0, 30.0, 100.0, 200.0}),
      });

  auto plan =
      PlanBuilder()
          .values({data})
          .window({
              "avg(val) over (partition by id "
              "rows between unbounded preceding and unbounded following) as average",
          })
          .orderBy({"id ASC NULLS LAST", "val ASC NULLS LAST"}, false)
          .planNode();

  auto expected = makeRowVector(
      {"id", "val", "average"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<double>({10.0, 20.0, 30.0, 100.0, 200.0}),
          makeFlatVector<double>({20.0, 20.0, 20.0, 150.0, 150.0}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

// Ports of checks from velox/exec/tests/WindowTest.cpp (CPU window tests).

TEST_F(CudfWindowTest, duplicateOrOverlappingKeys) {
  auto data = makeRowVector(
      ROW({"a", "b", "c", "d", "e"},
          {
              BIGINT(),
              BIGINT(),
              BIGINT(),
              BIGINT(),
              BIGINT(),
          }),
      10);

  auto buildPlan = [&](const std::vector<std::string>& partitionKeys,
                       const std::vector<std::string>& sortingKeys) {
    std::ostringstream sql;
    sql << "row_number() over (";
    if (!partitionKeys.empty()) {
      sql << " partition by ";
      sql << folly::join(", ", partitionKeys);
    }
    if (!sortingKeys.empty()) {
      sql << " order by ";
      sql << folly::join(", ", sortingKeys);
    }
    sql << ")";

    PlanBuilder().values({data}).window({sql.str()}).planNode();
  };

  VELOX_ASSERT_THROW(
      buildPlan({"a", "a"}, {"b"}),
      "Partitioning keys must be unique. Found duplicate key: a");

  VELOX_ASSERT_THROW(
      buildPlan({"a", "b"}, {"c", "d", "c"}),
      "Sorting keys must be unique and not overlap with partitioning keys. Found duplicate key: c");

  VELOX_ASSERT_THROW(
      buildPlan({"a", "b"}, {"c", "b"}),
      "Sorting keys must be unique and not overlap with partitioning keys. Found duplicate key: b");
}

TEST_F(CudfWindowTest, rowNumberGlobalOrderBy) {
  auto data = makeRowVector(
      {"d", "p", "s"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4}),
          makeFlatVector<int16_t>({1, 1, 2, 2, 2}),
          makeFlatVector<int32_t>({30, 10, 20, 5, 15}),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({"row_number() over (order by s)"})
                  .orderBy({"s ASC NULLS LAST"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"d", "p", "s", "w0"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4}),
          makeFlatVector<int16_t>({1, 1, 2, 2, 2}),
          makeFlatVector<int32_t>({30, 10, 20, 5, 15}),
          makeFlatVector<int64_t>({5, 2, 4, 1, 3}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfWindowTest, rankGlobalOrderBy) {
  auto data = makeRowVector({"c1"}, {makeFlatVector<int64_t>({1, 1, 1, 2, 2})});

  auto plan =
      PlanBuilder()
          .values({data})
          .window({"rank() over (order by c1 rows unbounded preceding)"})
          .orderBy({"c1 ASC NULLS LAST"}, false)
          .planNode();

  auto expected = makeRowVector(
      {"c1", "w0"},
      {
          makeFlatVector<int64_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({1, 1, 1, 4, 4}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfWindowTest, rowNumberMultiBatch) {
  auto data = makeRowVector(
      {"id", "val"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 15, 25, 35}),
      });

  auto plan = PlanBuilder()
                  .values(split(data, 3))
                  .window({"row_number() over (partition by id order by val)"})
                  .orderBy({"id ASC NULLS LAST", "val ASC NULLS LAST"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"id", "val", "w0"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 15, 25, 35}),
          makeFlatVector<int64_t>({1, 2, 3, 1, 2, 3}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfWindowTest, multiFunctionPartitionOrder) {
  // Same shape as valuesRowsStreamingWindowBuild (CPU) but non-streaming window
  // and explicit expected vectors (no DuckDB runner).
  auto data = makeRowVector(
      {"c0", "c1", "c2", "c3", "c4"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int32_t>({1, 2, 3, 1, 2}),
          makeFlatVector<int64_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int32_t>({0, 1, 2, 0, 1}),
          makeFlatVector<int32_t>({10, 20, 30, 100, 200}),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({
                      "rank() over (partition by c0, c2 order by c1, c3)",
                      "dense_rank() over (partition by c0, c2 order by c1, c3)",
                      "row_number() over (partition by c0, c2 order by c1, c3)",
                      "sum(c4) over (partition by c0, c2 order by c1, c3)",
                  })
                  .orderBy(
                      {"c0 ASC NULLS LAST",
                       "c2 ASC NULLS LAST",
                       "c1 ASC NULLS LAST",
                       "c3 ASC NULLS LAST"},
                      false)
                  .planNode();

  auto expected = makeRowVector(
      {"c0", "c1", "c2", "c3", "c4", "w0", "w1", "w2", "w3"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int32_t>({1, 2, 3, 1, 2}),
          makeFlatVector<int64_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int32_t>({0, 1, 2, 0, 1}),
          makeFlatVector<int32_t>({10, 20, 30, 100, 200}),
          makeFlatVector<int64_t>({1, 2, 3, 1, 2}),
          makeFlatVector<int64_t>({1, 2, 3, 1, 2}),
          makeFlatVector<int64_t>({1, 2, 3, 1, 2}),
          makeFlatVector<int64_t>({10, 30, 60, 100, 300}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfWindowTest, rankNaNRangeFrameBounds) {
  // rank() ignores RANGE frame bounds; port of the rank() loop from
  // WindowTest.NaNFrameBound (sum+RANGE is not mirrored here).
  const auto kNan = std::numeric_limits<double>::quiet_NaN();
  auto data = makeRowVector(
      {"c0", "s0", "off0", "off1"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4}),
          makeFlatVector<double>({1.0, 2.0, 3.0, kNan}),
          makeFlatVector<double>({0.1, 2.0, 1.9, kNan}),
          makeFlatVector<double>({kNan, 2.0, kNan, kNan}),
      });

  const auto makeFrames = [](const std::string& call) {
    std::vector<std::string> frames;

    std::vector<std::string> orders{"asc", "desc"};
    std::vector<std::string> bounds{"preceding", "following"};
    for (const std::string& order : orders) {
      for (const std::string& startBound : bounds) {
        for (const std::string& endBound : bounds) {
          if (startBound == "following" && endBound == "preceding") {
            continue;
          }
          frames.push_back(
              fmt::format(
                  "{} over (order by s0 {} range between off0 {} and off1 {})",
                  call,
                  order,
                  startBound,
                  endBound));
          frames.push_back(
              fmt::format(
                  "{} over (order by s0 {} range between off1 {} and off0 {})",
                  call,
                  order,
                  startBound,
                  endBound));
        }
      }
    }
    return frames;
  };

  auto expected =
      makeRowVector({"w0"}, {makeFlatVector<int64_t>({1, 2, 3, 4})});
  for (const auto& frame : makeFrames("rank()")) {
    auto plan =
        PlanBuilder().values({data}).window({frame}).project({"w0"}).planNode();
    AssertQueryBuilder(plan).assertResults(expected);
  }
}

// =============================================================================
// Tests ported from velox/exec/tests/WindowTest.cpp
// =============================================================================

// Disabled: Spilling is not yet implemented in cudf
TEST_F(CudfWindowTest, DISABLED_spill) {
  const vector_size_t size = 1'000;
  auto data = makeRowVector(
      {"d", "p", "s"},
      {
          // Payload.
          makeFlatVector<int64_t>(size, [](auto row) { return row; }),
          // Partition key.
          makeFlatVector<int16_t>(size, [](auto row) { return row % 11; }),
          // Sorting key.
          makeFlatVector<int32_t>(size, [](auto row) { return row; }),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({"row_number() over (partition by p order by s)"})
                  .planNode();

  auto expected = makeRowVector(
      {"d", "p", "s", "w0"},
      {
          makeFlatVector<int64_t>(size, [](auto row) { return row; }),
          makeFlatVector<int16_t>(size, [](auto row) { return row % 11; }),
          makeFlatVector<int32_t>(size, [](auto row) { return row; }),
          // row_number results would need to be computed
          makeFlatVector<int64_t>(size, [](auto) { return 0; }),
      });

  // Note: Full spill testing requires spill infrastructure not available in
  // cudf
  GTEST_SKIP() << "Spilling not implemented in cudf";
}

// Disabled: Spilling is not yet implemented in cudf
TEST_F(CudfWindowTest, DISABLED_spillBatchReadTinyPartitions) {
  GTEST_SKIP() << "Spilling not implemented in cudf";
}

// Disabled: Spilling is not yet implemented in cudf
TEST_F(CudfWindowTest, DISABLED_spillBatchReadHugePartitions) {
  GTEST_SKIP() << "Spilling not implemented in cudf";
}

// Disabled: Spilling is not yet implemented in cudf
TEST_F(CudfWindowTest, DISABLED_spillUnsupported) {
  GTEST_SKIP() << "Spilling not implemented in cudf";
}

// Disabled: Streaming window is not yet implemented in cudf
TEST_F(CudfWindowTest, DISABLED_rowBasedStreamingWindowOOM) {
  GTEST_SKIP() << "Streaming window not implemented in cudf";
}

// Disabled: Pre-partitioned sort build is not yet implemented in cudf
TEST_F(CudfWindowTest, DISABLED_prePartitionedSortBuild) {
  const vector_size_t size = 1'000;
  const int numPartitions = 37;
  auto data = makeRowVector(
      {"p", "s"},
      {
          // Partition key.
          makeFlatVector<int16_t>(
              size, [](auto row) { return row % numPartitions; }),
          // Sorting key.
          makeFlatVector<int32_t>(size, [](auto row) { return row; }),
      });

  auto plan =
      PlanBuilder()
          .values({data})
          .window({"row_number() over (partition by p order by s desc)"})
          .planNode();

  GTEST_SKIP() << "Pre-partitioned sort build not implemented in cudf";
}

// Disabled: Pre-partitioned sort build is not yet implemented in cudf
TEST_F(CudfWindowTest, DISABLED_prePartitionedSortBuildSkewed) {
  GTEST_SKIP() << "Pre-partitioned sort build not implemented in cudf";
}

// Disabled: Spilling is not yet implemented in cudf
TEST_F(CudfWindowTest, DISABLED_prePartitionedBuildWithSpill) {
  GTEST_SKIP() << "Spilling not implemented in cudf";
}

// Disabled: Negative frame arguments use regr_count which is not supported in
// cudf
TEST_F(CudfWindowTest, DISABLED_negativeFrameArg) {
  const vector_size_t size = 1'000;

  auto sizeAt = [](vector_size_t row) { return row % 5; };
  auto keyAt = [](vector_size_t row) { return row % 11; };
  auto keys = makeArrayVector<float>(size, sizeAt, keyAt);
  auto data = makeRowVector(
      {"c0", "c1", "p0", "p1", "k0", "row_number"},
      {
          // Payload.
          makeFlatVector<float>(size, [](auto row) { return row; }),
          makeFlatVector<float>(size, [](auto row) { return row; }),
          // Partition key.
          keys,
          makeFlatVector<std::string>(
              size, [](auto row) { return fmt::format("{}", row + 20); }),
          makeFlatVector<int32_t>(size, [](auto row) { return row; }),
          // Sorting key.
          makeFlatVector<int64_t>(size, [](auto row) { return row; }),
      });

  struct {
    std::string fragmentStart;
    std::string fragmentEnd;

    std::string debugString() const {
      if (fragmentStart[0] == '-') {
        return fmt::format(
            "Window frame {} offset must not be negative", fragmentStart);
      } else {
        return fmt::format(
            "Window frame {} offset must not be negative", fragmentEnd);
      }
    }
  } testSettings[] = {
      {"k0", "-1"}, // Negative end
      {"-1", "k0"}, // Negative start
      {"-1", "-3"} // Negative start, negative end
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const auto& startOffset = testData.fragmentStart;
    const auto& endOffset = testData.fragmentEnd;
    auto plan =
        PlanBuilder()
            .values({data})
            .window({fmt::format(
                "regr_count(c0, c1) over (partition by p0, p1 order by row_number ROWS between {} PRECEDING and {} FOLLOWING)",
                startOffset,
                endOffset)})
            .planNode();
    VELOX_ASSERT_USER_THROW(
        AssertQueryBuilder(plan).copyResults(pool()), testData.debugString());
  }
}

// Disabled: RANGE frames with column offsets are not yet supported in cudf
TEST_F(CudfWindowTest, DISABLED_NaNFrameBound) {
  const auto kNan = std::numeric_limits<double>::quiet_NaN();
  auto data = makeRowVector(
      {"c0", "s0", "off0", "off1"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4}),
          makeFlatVector<double>({1.0, 2.0, 3.0, kNan}),
          makeFlatVector<double>({0.1, 2.0, 1.9, kNan}),
          makeFlatVector<double>({kNan, 2.0, kNan, kNan}),
      });

  const auto makeFrames = [](const std::string& call) {
    std::vector<std::string> frames;

    std::vector<std::string> orders{"asc", "desc"};
    std::vector<std::string> bounds{"preceding", "following"};
    for (const std::string& order : orders) {
      for (const std::string& startBound : bounds) {
        for (const std::string& endBound : bounds) {
          // Frames starting from following and ending at preceding are not
          // allowed.
          if (startBound == "following" && endBound == "preceding") {
            continue;
          }
          frames.push_back(
              fmt::format(
                  "{} over (order by s0 {} range between off0 {} and off1 {})",
                  call,
                  order,
                  startBound,
                  endBound));
          frames.push_back(
              fmt::format(
                  "{} over (order by s0 {} range between off1 {} and off0 {})",
                  call,
                  order,
                  startBound,
                  endBound));
        }
      }
    }
    return frames;
  };

  auto expected = makeRowVector(
      {makeNullableFlatVector<int64_t>({std::nullopt, 2, std::nullopt, 4})});
  for (const auto& frame : makeFrames("sum(c0)")) {
    auto plan =
        PlanBuilder().values({data}).window({frame}).project({"w0"}).planNode();
    AssertQueryBuilder(plan).assertResults(expected);
  }

  // rank() should not be affected by the frames, so added this test to ensure
  // rank() produces correct results even if the frame bounds contain NaN.
  expected = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  for (const auto& frame : makeFrames("rank()")) {
    auto plan =
        PlanBuilder().values({data}).window({frame}).project({"w0"}).planNode();
    AssertQueryBuilder(plan).assertResults(expected);
  }
}

// =============================================================================
// Tests ported from velox/functions/prestosql/window/tests/RankTest.cpp
// =============================================================================

// Tests rank functions with all rows in a single partition.
TEST_F(CudfWindowTest, rankSinglePartition) {
  // All rows have the same partition key.
  auto data = makeRowVector(
      {"p", "s"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 1}),
          makeFlatVector<int64_t>({10, 20, 20, 30, 40}),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({
                      "row_number() over (partition by p order by s) as rn",
                      "rank() over (partition by p order by s) as r",
                      "dense_rank() over (partition by p order by s) as dr",
                  })
                  .orderBy({"s ASC NULLS LAST"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"p", "s", "rn", "r", "dr"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 1}),
          makeFlatVector<int64_t>({10, 20, 20, 30, 40}),
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>({1, 2, 2, 4, 5}),
          makeFlatVector<int64_t>({1, 2, 2, 3, 4}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

// Tests rank functions with single-row partitions.
TEST_F(CudfWindowTest, rankSingleRowPartitions) {
  // Each row is its own partition.
  auto data = makeRowVector(
      {"p", "s"},
      {
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>({100, 200, 300, 400, 500}),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({
                      "row_number() over (partition by p order by s) as rn",
                      "rank() over (partition by p order by s) as r",
                      "dense_rank() over (partition by p order by s) as dr",
                  })
                  .orderBy({"p ASC NULLS LAST"}, false)
                  .planNode();

  // Each partition has only one row, so all ranks are 1.
  auto expected = makeRowVector(
      {"p", "s", "rn", "r", "dr"},
      {
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>({100, 200, 300, 400, 500}),
          makeFlatVector<int64_t>({1, 1, 1, 1, 1}),
          makeFlatVector<int64_t>({1, 1, 1, 1, 1}),
          makeFlatVector<int64_t>({1, 1, 1, 1, 1}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

// Tests rank functions with nulls in sort key.
TEST_F(CudfWindowTest, rankWithNulls) {
  auto data = makeRowVector(
      {"p", "s"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 2, 2, 2}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt, 10, 20, 20, std::nullopt, std::nullopt, 30}),
      });

  auto plan =
      PlanBuilder()
          .values({data})
          .window({
              "row_number() over (partition by p order by s NULLS FIRST) as rn",
              "rank() over (partition by p order by s NULLS FIRST) as r",
              "dense_rank() over (partition by p order by s NULLS FIRST) as dr",
          })
          .orderBy({"p ASC NULLS LAST", "s ASC NULLS FIRST"}, false)
          .planNode();

  // With NULLS FIRST, null values come before non-null values.
  // Partition 1: null, 10, 20, 20 -> row_number: 1,2,3,4; rank: 1,2,3,3;
  //   dense_rank: 1,2,3,3
  // Partition 2: null, null, 30 -> row_number: 1,2,3; rank: 1,1,3;
  //   dense_rank: 1,1,2
  auto expected = makeRowVector(
      {"p", "s", "rn", "r", "dr"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 2, 2, 2}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt, 10, 20, 20, std::nullopt, std::nullopt, 30}),
          makeFlatVector<int64_t>({1, 2, 3, 4, 1, 2, 3}),
          makeFlatVector<int64_t>({1, 2, 3, 3, 1, 1, 3}),
          makeFlatVector<int64_t>({1, 2, 3, 3, 1, 1, 2}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

// =============================================================================
// Tests ported from velox/functions/prestosql/window/tests/LeadLagTest.cpp
// =============================================================================

// Tests lag/lead with zero offset.
TEST_F(CudfWindowTest, lagLeadZeroOffset) {
  auto data = makeRowVector(
      {"p", "v"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 100, 200}),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({
                      "lag(v, 0) over (partition by p order by v) as lag0",
                      "lead(v, 0) over (partition by p order by v) as lead0",
                  })
                  .orderBy({"p ASC NULLS LAST", "v ASC NULLS LAST"}, false)
                  .planNode();

  // Zero offset means the current row's value.
  auto expected = makeRowVector(
      {"p", "v", "lag0", "lead0"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 100, 200}),
          makeFlatVector<int64_t>({10, 20, 30, 100, 200}),
          makeFlatVector<int64_t>({10, 20, 30, 100, 200}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

// Tests lag/lead with larger offset.
TEST_F(CudfWindowTest, lagLeadLargeOffset) {
  auto data = makeRowVector(
      {"p", "v"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 1}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({
                      "lag(v, 3) over (partition by p order by v) as lag3",
                      "lead(v, 3) over (partition by p order by v) as lead3",
                  })
                  .orderBy({"v ASC NULLS LAST"}, false)
                  .planNode();

  // lag(3): first 3 rows are null, then 10, 20
  // lead(3): last 3 rows are null, first 2 are 40, 50
  auto expected = makeRowVector(
      {"p", "v", "lag3", "lead3"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 1}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt, std::nullopt, std::nullopt, 10, 20}),
          makeNullableFlatVector<int64_t>(
              {40, 50, std::nullopt, std::nullopt, std::nullopt}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

// Tests lag/lead with nulls in the value column.
TEST_F(CudfWindowTest, lagLeadWithNullValues) {
  auto data = makeRowVector(
      {"p", "v"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 1}),
          makeNullableFlatVector<int64_t>(
              {10, std::nullopt, 30, std::nullopt, 50}),
      });

  auto plan =
      PlanBuilder()
          .values({data})
          .window({
              "lag(v, 1) over (partition by p order by v NULLS FIRST) as lag1",
              "lead(v, 1) over (partition by p order by v NULLS FIRST) as lead1",
          })
          .orderBy({"v ASC NULLS FIRST"}, false)
          .planNode();

  // With NULLS FIRST, order is: null, null, 10, 30, 50
  // lag(1): null, null, null, 10, 30
  // lead(1): null, 10, 30, 50, null
  auto expected = makeRowVector(
      {"p", "v", "lag1", "lead1"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 1}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt, std::nullopt, 10, 30, 50}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt, std::nullopt, std::nullopt, 10, 30}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt, 10, 30, 50, std::nullopt}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

// Tests lag/lead with single-row partitions.
TEST_F(CudfWindowTest, lagLeadSingleRowPartitions) {
  auto data = makeRowVector(
      {"p", "v"},
      {
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({
                      "lag(v, 1) over (partition by p order by v) as lag1",
                      "lead(v, 1) over (partition by p order by v) as lead1",
                  })
                  .orderBy({"p ASC NULLS LAST"}, false)
                  .planNode();

  // Each partition has only one row, so lag and lead are always null.
  auto expected = makeRowVector(
      {"p", "v", "lag1", "lead1"},
      {
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt}),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

// Tests lag/lead with small partitions (5 rows each).
TEST_F(CudfWindowTest, lagLeadSmallPartitions) {
  // Create data with small partitions.
  std::vector<int32_t> partitions;
  std::vector<int64_t> values;
  for (int i = 0; i < 100; ++i) {
    partitions.push_back(i / 5); // 5 rows per partition
    values.push_back(i);
  }

  auto data = makeRowVector(
      {"p", "v"},
      {
          makeFlatVector<int32_t>(partitions),
          makeFlatVector<int64_t>(values),
      });

  auto plan = PlanBuilder()
                  .values({data})
                  .window({
                      "lag(v, 2) over (partition by p order by v) as lag2",
                      "lead(v, 2) over (partition by p order by v) as lead2",
                  })
                  .orderBy({"p ASC NULLS LAST", "v ASC NULLS LAST"}, false)
                  .planNode();

  // Compute expected results.
  std::vector<std::optional<int64_t>> expectedLag;
  std::vector<std::optional<int64_t>> expectedLead;
  for (int i = 0; i < 100; ++i) {
    int posInPartition = i % 5;
    int partitionStart = (i / 5) * 5;
    if (posInPartition < 2) {
      expectedLag.push_back(std::nullopt);
    } else {
      expectedLag.push_back(values[i - 2]);
    }
    if (posInPartition >= 3) {
      expectedLead.push_back(std::nullopt);
    } else {
      expectedLead.push_back(values[i + 2]);
    }
  }

  auto expected = makeRowVector(
      {"p", "v", "lag2", "lead2"},
      {
          makeFlatVector<int32_t>(partitions),
          makeFlatVector<int64_t>(values),
          makeNullableFlatVector<int64_t>(expectedLag),
          makeNullableFlatVector<int64_t>(expectedLead),
      });

  AssertQueryBuilder(plan).assertResults(expected);
}

} // namespace
