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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/exec/Operator.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {
using namespace facebook::velox::common::testutil;

class RowNumberTest : public OperatorTestBase {
 protected:
  RowNumberTest() {
    filesystems::registerLocalFileSystem();
    rowType_ = ROW(
        {{"c0", BIGINT()},
         {"c1", BIGINT()},
         {"c2", VARCHAR()},
         {"c3", VARCHAR()}});
    fuzzerOpts_.vectorSize = 1024;
    fuzzerOpts_.nullRatio = 0;
    fuzzerOpts_.stringVariableLength = false;
    fuzzerOpts_.stringLength = 1024;
    fuzzerOpts_.allowLazyVector = false;
  }

  RowTypePtr rowType_;
  VectorFuzzer::Options fuzzerOpts_;
};

TEST_F(RowNumberTest, basic) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 1, 2, 1, 2, 1}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7}),
  });

  createDuckDbTable({data});

  // No limit, emit row numbers.
  auto plan = PlanBuilder().values({data}).rowNumber({"c0"}).planNode();
  assertQuery(plan, "SELECT *, row_number() over (partition by c0) FROM tmp");

  // No limit, don't emit row numbers.
  plan = PlanBuilder()
             .values({data})
             .rowNumber({"c0"}, std::nullopt, false)
             .planNode();
  assertQuery(
      plan,
      "SELECT c0, c1 FROM (SELECT *, row_number() over (partition by c0) as rn FROM tmp)");

  auto testLimit = [&](int32_t limit) {
    // Limit, emit row numbers.
    auto plan =
        PlanBuilder().values({data}).rowNumber({"c0"}, limit).planNode();
    assertQuery(
        plan,
        fmt::format(
            "SELECT * FROM (SELECT *, row_number() over (partition by c0) as rn FROM tmp) "
            "WHERE rn <= {}",
            limit));

    // Limit, don't emit row numbers.
    plan =
        PlanBuilder().values({data}).rowNumber({"c0"}, limit, false).planNode();
    assertQuery(
        plan,
        fmt::format(
            "SELECT c0, c1 FROM (SELECT *, row_number() over (partition by c0) as rn FROM tmp) "
            "WHERE rn <= {}",
            limit));
  };

  testLimit(1);
  testLimit(2);
  testLimit(5);
}

TEST_F(RowNumberTest, noPartitionKeys) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>(1'000, [](auto row) { return row; }),
  });

  createDuckDbTable({data, data});

  // No limit, emit row numbers.
  auto plan = PlanBuilder().values({data, data}).rowNumber({}).planNode();
  assertQuery(plan, "SELECT *, row_number() over () FROM tmp");

  // No limit, don't emit row numbers.
  plan = PlanBuilder()
             .values({data, data})
             .rowNumber({}, std::nullopt, false)
             .planNode();
  assertQuery(
      plan, "SELECT c0 FROM (SELECT *, row_number() over () as rn FROM tmp)");

  auto testLimit = [&](int32_t limit) {
    // Emit row numbers.
    auto plan =
        PlanBuilder().values({data, data}).rowNumber({}, limit).planNode();
    assertQuery(
        plan,
        fmt::format(
            "SELECT * FROM (SELECT *, row_number() over () as rn FROM tmp) "
            "WHERE rn <= {}",
            limit));

    // Don't emit row numbers.
    plan = PlanBuilder()
               .values({data, data})
               .rowNumber({}, limit, false)
               .planNode();
    assertQuery(
        plan,
        fmt::format(
            "SELECT c0 FROM (SELECT *, row_number() over () as rn FROM tmp) "
            "WHERE rn <= {}",
            limit));
  };

  testLimit(1);
  testLimit(50);
}

TEST_F(RowNumberTest, largeInput) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>(10'000, [](auto row) { return row % 7; }),
      makeFlatVector<int64_t>(10'000, [](auto row) { return row; }),
  });

  createDuckDbTable({data, data});

  // No limit, emit row numbers.
  auto plan = PlanBuilder().values({data, data}).rowNumber({"c0"}).planNode();
  assertQuery(plan, "SELECT *, row_number() over (partition by c0) FROM tmp");

  // No limit, don't emit row numbers.
  plan = PlanBuilder()
             .values({data, data})
             .rowNumber({"c0"}, std::nullopt, false)
             .planNode();
  assertQuery(
      plan,
      "SELECT c0, c1 FROM (SELECT *, row_number() over (partition by c0) as rn FROM tmp)");

  auto testLimit = [&](int32_t limit) {
    // Emit row numbers.
    auto plan =
        PlanBuilder().values({data, data}).rowNumber({"c0"}, limit).planNode();
    assertQuery(
        plan,
        fmt::format(
            "SELECT * FROM (SELECT *, row_number() over (partition by c0) as rn FROM tmp) "
            "WHERE rn <= {}",
            limit));

    // Don't emit row numbers.
    plan = PlanBuilder()
               .values({data, data})
               .rowNumber({"c0"}, limit, false)
               .planNode();
    assertQuery(
        plan,
        fmt::format(
            "SELECT c0, c1 FROM (SELECT *, row_number() over (partition by c0) as rn FROM tmp) "
            "WHERE rn <= {}",
            limit));
  };

  testLimit(1);
  testLimit(100);
  testLimit(2'000);
  testLimit(5'000);
}

TEST_F(RowNumberTest, spill) {
  std::vector<RowVectorPtr> vectors = createVectors(8, rowType_, fuzzerOpts_);
  createDuckDbTable(vectors);

  struct {
    uint32_t spillPartitionBits;

    std::string debugString() const {
      return fmt::format("SpillPartitionBits {}", spillPartitionBits);
    }
  } testSettings[] = {{2}, {3}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const auto spillDirectory = TempDirectoryPath::create();
    auto queryCtx = core::QueryCtx::create(executor_.get());
    TestScopedSpillInjection scopedSpillInjection(100, ".*", 1);

    core::PlanNodeId rowNumberPlanNodeId;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->getPath())
            .config(core::QueryConfig::kSpillEnabled, true)
            .config(core::QueryConfig::kRowNumberSpillEnabled, true)
            .config(
                core::QueryConfig::kSpillNumPartitionBits,
                testData.spillPartitionBits)
            .queryCtx(queryCtx)
            .plan(
                PlanBuilder()
                    .values(vectors)
                    .rowNumber({"c0"})
                    .capturePlanNodeId(rowNumberPlanNodeId)
                    .planNode())
            .assertResults(
                "SELECT *, row_number() over (partition by c0) FROM tmp");
    auto taskStats = toPlanStats(task->taskStats());
    auto& planStats = taskStats.at(rowNumberPlanNodeId);
    ASSERT_GT(planStats.spilledBytes, 0);
    ASSERT_EQ(
        planStats.spilledPartitions,
        (static_cast<uint32_t>(1) << testData.spillPartitionBits) * 2);
    ASSERT_GT(planStats.spilledFiles, 0);
    ASSERT_GT(planStats.spilledRows, 0);
    auto operatorStats =
        task->taskStats().pipelineStats.back().operatorStats.at(1);
    auto runtimeStats = operatorStats.runtimeStats;
    ASSERT_EQ(
        runtimeStats.at(std::string(Operator::kSpillReadBytes)).sum,
        operatorStats.spilledBytes);
    ASSERT_GT(runtimeStats.at(std::string(Operator::kSpillReads)).sum, 0);
    ASSERT_GT(runtimeStats.at(std::string(Operator::kSpillReadTime)).sum, 0);
    ASSERT_GT(
        runtimeStats.at(std::string(Operator::kSpillDeserializationTime)).sum,
        0);

    task.reset();
    waitForAllTasksToBeDeleted();
  }
}

TEST_F(RowNumberTest, maxSpillBytes) {
  const auto rowType =
      ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), VARCHAR()});
  const auto vectors = createVectors(rowType, 1024, 15 << 20);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values(vectors)
                  .rowNumber({"c0"}, 2, false)
                  .project({"c0", "c1"})
                  .planNode();
  struct {
    int32_t maxSpilledBytes;
    bool expectedExceedLimit;
    std::string debugString() const {
      return fmt::format("maxSpilledBytes {}", maxSpilledBytes);
    }
  } testSettings[] = {{1 << 30, false}, {1 << 20, true}, {0, false}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto spillDirectory = TempDirectoryPath::create();
    auto queryCtx = core::QueryCtx::create(executor_.get());
    try {
      TestScopedSpillInjection scopedSpillInjection(100, ".*", 1);
      AssertQueryBuilder(plan)
          .spillDirectory(spillDirectory->getPath())
          .queryCtx(queryCtx)
          .config(core::QueryConfig::kSpillEnabled, true)
          .config(core::QueryConfig::kRowNumberSpillEnabled, true)
          .config(core::QueryConfig::kMaxSpillBytes, testData.maxSpilledBytes)
          .copyResults(pool_.get());
      ASSERT_FALSE(testData.expectedExceedLimit);
    } catch (const VeloxRuntimeError& e) {
      ASSERT_TRUE(testData.expectedExceedLimit);
      ASSERT_NE(
          e.message().find(
              "Query exceeded per-query local spill limit of 1.00MB"),
          std::string::npos);
      ASSERT_EQ(
          e.errorCode(), facebook::velox::error_code::kSpillLimitExceeded);
    }
  }
}

TEST_F(RowNumberTest, memoryUsage) {
  const auto rowType =
      ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), VARCHAR()});
  const auto vectors = createVectors(rowType, 1024, 100 << 20);
  core::PlanNodeId rowNumberId;
  auto plan = PlanBuilder()
                  .values(vectors)
                  .rowNumber({"c0", "c2"})
                  .capturePlanNodeId(rowNumberId)
                  .project({"c0", "c1"})
                  .planNode();

  struct {
    uint8_t numSpills;

    std::string debugString() const {
      return fmt::format("numSpills {}", numSpills);
    }
  } testSettings[] = {{1}, {3}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    int64_t peakBytesWithSpilling = 0;
    int64_t peakBytesWithOutSpilling = 0;

    for (const auto& spillEnable : {false, true}) {
      auto queryCtx = core::QueryCtx::create(executor_.get());
      auto spillDirectory = TempDirectoryPath::create();
      const std::string spillEnableConfig = std::to_string(spillEnable);

      std::shared_ptr<Task> task;
      TestScopedSpillInjection scopedSpillInjection(
          100, ".*", testData.numSpills);
      AssertQueryBuilder(plan)
          .spillDirectory(spillDirectory->getPath())
          .queryCtx(queryCtx)
          .config(core::QueryConfig::kSpillEnabled, spillEnableConfig)
          .config(core::QueryConfig::kRowNumberSpillEnabled, spillEnableConfig)
          .spillDirectory(spillDirectory->getPath())
          .copyResults(pool_.get(), task);

      if (spillEnable) {
        peakBytesWithSpilling = queryCtx->pool()->peakBytes();
        auto taskStats = exec::toPlanStats(task->taskStats());
        const auto& stats = taskStats.at(rowNumberId);

        ASSERT_GT(stats.spilledBytes, 0);
        ASSERT_GT(stats.spilledRows, 0);
        ASSERT_GT(stats.spilledFiles, 0);
        ASSERT_GT(stats.spilledPartitions, 0);
      } else {
        peakBytesWithOutSpilling = queryCtx->pool()->peakBytes();
      }
    }

    ASSERT_GE(peakBytesWithOutSpilling / peakBytesWithSpilling, 2);
  }
}

DEBUG_ONLY_TEST_F(RowNumberTest, spillOnlyDuringInputOrOutput) {
  std::vector<RowVectorPtr> vectors = createVectors(8, rowType_, fuzzerOpts_);
  createDuckDbTable(vectors);

  struct {
    std::string spillInjectionPoint;
    uint32_t spillPartitionBits;

    std::string debugString() const {
      return fmt::format(
          "Spill during {}, spillPartitionBits {}",
          spillInjectionPoint,
          spillPartitionBits);
    }
  } testSettings[] = {
      {"facebook::velox::exec::Driver::runInternal::addInput", 2},
      {"facebook::velox::exec::Driver::runInternal::getOutput", 2},
      {"facebook::velox::exec::Driver::runInternal::addInput", 3},
      {"facebook::velox::exec::Driver::runInternal::getOutput", 3}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const auto spillDirectory = TempDirectoryPath::create();
    auto queryCtx = core::QueryCtx::create(executor_.get());

    std::atomic_int numRound{0};
    SCOPED_TESTVALUE_SET(
        testData.spillInjectionPoint,
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "RowNumber") {
            return;
          }

          if (++numRound != 8) {
            return;
          }

          testingRunArbitration(op->pool(), 0);
          // We expect all the memory to be freed after the spill.
          ASSERT_EQ(op->pool()->usedBytes(), 40960);
        })));

    core::PlanNodeId rowNumberPlanNodeId;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->getPath())
            .config(core::QueryConfig::kSpillEnabled, true)
            .config(core::QueryConfig::kRowNumberSpillEnabled, true)
            .config(
                core::QueryConfig::kSpillNumPartitionBits,
                testData.spillPartitionBits)
            .queryCtx(queryCtx)
            .plan(
                PlanBuilder()
                    .values(vectors)
                    .rowNumber({"c0"})
                    .capturePlanNodeId(rowNumberPlanNodeId)
                    .planNode())
            .assertResults(
                "SELECT *, row_number() over (partition by c0) FROM tmp");
    auto taskStats = toPlanStats(task->taskStats());
    auto& planStats = taskStats.at(rowNumberPlanNodeId);
    ASSERT_GT(planStats.spilledBytes, 0);
    ASSERT_EQ(
        planStats.spilledPartitions,
        (static_cast<uint32_t>(1) << testData.spillPartitionBits) * 2);
    ASSERT_GT(planStats.spilledFiles, 0);
    ASSERT_GT(planStats.spilledRows, 0);

    task.reset();
    waitForAllTasksToBeDeleted();
  }
}

DEBUG_ONLY_TEST_F(RowNumberTest, recursiveSpill) {
  std::vector<RowVectorPtr> vectors = createVectors(32, rowType_, fuzzerOpts_);
  createDuckDbTable(vectors);

  struct {
    int32_t numSpills;
    int32_t maxSpillLevel;

    std::string debugString() const {
      return fmt::format(
          "numSpills {}, maxSpillLevel {}", numSpills, maxSpillLevel);
    }
  } testSettings[] = {{2, 3}, {8, 4}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const auto spillDirectory = TempDirectoryPath::create();
    auto queryCtx = core::QueryCtx::create(executor_.get());

    std::atomic_int numSpills{0};
    std::atomic_int numInputs{0};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::addInput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "RowNumber") {
            return;
          }

          if (++numInputs != 5) {
            return;
          }

          ++numSpills;
          testingRunArbitration(op->pool(), 0);
        })));

    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::getOutput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "RowNumber") {
            return;
          }

          if (!op->testingNoMoreInput()) {
            return;
          }

          if (numSpills++ >= testData.numSpills) {
            return;
          }

          testingRunArbitration(op->pool(), 0);
        })));

    core::PlanNodeId rowNumberPlanNodeId;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->getPath())
            .config(core::QueryConfig::kSpillEnabled, true)
            .config(core::QueryConfig::kRowNumberSpillEnabled, true)
            .config(
                core::QueryConfig::kMaxSpillLevel, testData.maxSpillLevel - 1)
            .queryCtx(queryCtx)
            .plan(
                PlanBuilder()
                    .values(vectors)
                    .rowNumber({"c0"})
                    .capturePlanNodeId(rowNumberPlanNodeId)
                    .planNode())
            .assertResults(
                "SELECT *, row_number() over (partition by c0) FROM tmp");
    auto taskStats = toPlanStats(task->taskStats());
    auto& planStats = taskStats.at(rowNumberPlanNodeId);
    ASSERT_GT(planStats.spilledBytes, 0);
    ASSERT_EQ(
        planStats.spilledPartitions,
        8 * 2 * std::min(testData.numSpills, testData.maxSpillLevel));
    ASSERT_GT(planStats.spilledFiles, 0);
    ASSERT_GT(planStats.spilledRows, 0);

    auto runTimeStats =
        task->taskStats().pipelineStats.back().operatorStats.at(1).runtimeStats;
    if (testData.numSpills > testData.maxSpillLevel) {
      ASSERT_GT(runTimeStats["exceededMaxSpillLevel"].sum, 0);
    } else {
      ASSERT_EQ(runTimeStats.count("exceededMaxSpillLevel"), 0);
    }

    task.reset();
    waitForAllTasksToBeDeleted();
  }
}

TEST_F(RowNumberTest, spillWithYield) {
  std::vector<RowVectorPtr> vectors = createVectors(8, rowType_, fuzzerOpts_);
  createDuckDbTable(vectors);

  struct {
    uint32_t numSpills;
    uint32_t cpuTimeSliceLimitMs;

    std::string debugString() const {
      return fmt::format(
          "numSpills {}, cpuTimeSliceLimitMs {}",
          numSpills,
          cpuTimeSliceLimitMs);
    }
  } testSettings[] = {{2, 0}, {2, 10}, {3, 0}, {3, 10}, {8, 10}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    TestScopedSpillInjection scopedSpillInjection(
        100, ".*", testData.numSpills);
    const auto spillDirectory = TempDirectoryPath::create();
    auto queryCtx = core::QueryCtx::create(executor_.get());

    core::PlanNodeId rowNumberPlanNodeId;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->getPath())
            .config(core::QueryConfig::kSpillEnabled, true)
            .config(core::QueryConfig::kRowNumberSpillEnabled, true)
            .config(
                core::QueryConfig::kDriverCpuTimeSliceLimitMs,
                testData.cpuTimeSliceLimitMs)
            .queryCtx(queryCtx)
            .plan(
                PlanBuilder()
                    .values(vectors)
                    .rowNumber({"c0"})
                    .capturePlanNodeId(rowNumberPlanNodeId)
                    .planNode())
            .assertResults(
                "SELECT *, row_number() over (partition by c0) FROM tmp");
    auto taskStats = toPlanStats(task->taskStats());
    auto& planStats = taskStats.at(rowNumberPlanNodeId);
    ASSERT_GT(planStats.spilledBytes, 0);
    ASSERT_GT(planStats.spilledFiles, 0);
    ASSERT_GT(planStats.spilledRows, 0);

    task.reset();
    waitForAllTasksToBeDeleted();
  }
}

DEBUG_ONLY_TEST_F(RowNumberTest, rowNumberSpillFileCreateConfig) {
  auto vectors = createVectors(8, rowType_, fuzzerOpts_);
  createDuckDbTable(vectors);

  auto tempDirectory = TempDirectoryPath::create();

  std::atomic_bool rowNumberConfigVerified{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal::isBlocked",
      std::function<void(exec::Operator*)>([&](exec::Operator* op) {
        const auto* spillConfig = op->testingSpillConfig();
        if (spillConfig == nullptr) {
          return;
        }
        const auto& opType = op->operatorType();
        if (opType == "RowNumber") {
          ASSERT_EQ(spillConfig->fileCreateConfig, "test_row_number_config")
              << "Operator: " << opType;
          rowNumberConfigVerified = true;
        }
      }));

  TestScopedSpillInjection scopedSpillInjection(100);
  AssertQueryBuilder(duckDbQueryRunner_)
      .spillDirectory(tempDirectory->getPath())
      .config(core::QueryConfig::kSpillEnabled, true)
      .config(core::QueryConfig::kRowNumberSpillEnabled, true)
      .config(core::QueryConfig::kSpillFileCreateConfig, "test_default_config")
      .config(
          core::QueryConfig::kRowNumberSpillFileCreateConfig,
          "test_row_number_config")
      .plan(PlanBuilder().values(vectors).rowNumber({"c0"}).planNode())
      .assertResults("SELECT *, row_number() over (partition by c0) FROM tmp");

  ASSERT_TRUE(rowNumberConfigVerified.load());
}

namespace {

// Defines a minimal plan node for the backpressuring downstream used by the
// regression test below.
class BackpressureNode : public core::PlanNode {
 public:
  BackpressureNode(const core::PlanNodeId& id, core::PlanNodePtr input)
      : PlanNode(id), sources_{std::move(input)} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "Backpressure";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  std::vector<core::PlanNodePtr> sources_;
};

// Passes input through but refuses it most of the time, keeping the upstream
// RowNumber holding a buffered batch. Before the needsInput() fix this made the
// Driver re-feed RowNumber and overwrite its undrained input_.
class BackpressureOperator : public Operator {
 public:
  BackpressureOperator(
      DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const BackpressureNode>& node)
      : Operator(ctx, node->outputType(), id, node->id(), "Backpressure") {}

  bool needsInput() const override {
    if (noMoreInput_ || input_ != nullptr) {
      return false;
    }
    // Accept input only every 4th poll. The rejected polls are when the Driver
    // would re-feed (and, before the fix, overwrite) the upstream RowNumber.
    return (++polls_ % 4) == 0;
  }

  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }

  RowVectorPtr getOutput() override {
    return std::move(input_);
  }

  BlockingReason isBlocked(ContinueFuture* /* future */) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }

 private:
  mutable int32_t polls_{0};
};

class BackpressureNodeFactory : public Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<Operator> toOperator(
      DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override {
    if (auto bpNode = std::dynamic_pointer_cast<const BackpressureNode>(node)) {
      return std::make_unique<BackpressureOperator>(ctx, id, bpNode);
    }
    return nullptr;
  }

  std::optional<uint32_t> maxDrivers(const core::PlanNodePtr& node) override {
    if (std::dynamic_pointer_cast<const BackpressureNode>(node)) {
      // Single driver, mirroring the BATCH-mode RPC operator that exposed the
      // bug.
      return 1;
    }
    return std::nullopt;
  }
};

} // namespace

// Regression test: a window operator (RowNumber) feeding a backpressuring
// downstream must not drop rows. Before the fix, RowNumber::needsInput() did
// not check input_ == nullptr, so when the downstream returned
// needsInput()=false the Driver re-fed RowNumber and addInput() overwrote the
// undrained batch, silently dropping rows (seen in production with a BATCH-mode
// RPC operator).
TEST_F(RowNumberTest, noRowLossWithBackpressuringDownstream) {
  Operator::registerOperator(std::make_unique<BackpressureNodeFactory>());

  // Many small batches so the Driver repeatedly feeds RowNumber while the
  // downstream refuses input.
  const int32_t numBatches = 50;
  const int32_t rowsPerBatch = 100;
  std::vector<RowVectorPtr> batches;
  batches.reserve(numBatches);
  for (int32_t i = 0; i < numBatches; ++i) {
    batches.push_back(makeRowVector(
        {makeFlatVector<int64_t>(
             rowsPerBatch,
             [&](auto row) { return (i * rowsPerBatch + row) % 17; }),
         makeFlatVector<int64_t>(
             rowsPerBatch, [&](auto row) { return i * rowsPerBatch + row; })}));
  }
  const vector_size_t totalRows = numBatches * rowsPerBatch;

  auto plan =
      PlanBuilder()
          .values(batches)
          .rowNumber({"c0"})
          .addNode([](const std::string& id, core::PlanNodePtr input) {
            return std::make_shared<BackpressureNode>(id, std::move(input));
          })
          .planNode();

  // Single driver so RowNumber and the backpressuring downstream share a
  // pipeline, as with a BATCH-mode RPC operator (maxDrivers=1).
  auto result = AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool());
  EXPECT_EQ(result->size(), totalRows);
}

} // namespace facebook::velox::exec::test
