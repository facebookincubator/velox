/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/vectorindex/FaissLocalVectorIndex.h"
#include "velox/common/vectorindex/LocalVectorIndexAccelerator.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/Values.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <cudf/lists/lists_column_view.hpp>

#include <chrono>
#include <cmath>
#include <numeric>
#include <thread>

namespace facebook::velox::cudf_velox {
namespace {

using namespace common::vectorindex;
using exec::test::AssertQueryBuilder;
using exec::test::PlanBuilder;
constexpr int64_t kLargeId = int64_t{1} << 50;

// Let the stream-ordering test feed device vectors with distinct streams.
class DeviceValuesAdapter final : public OperatorAdapter {
 public:
  DeviceValuesAdapter() : OperatorAdapter("DeviceValues") {}

  bool canHandle(const exec::Operator* op) const override {
    return op->is<exec::Values>();
  }

  bool canRunOnGPU(
      const exec::Operator*,
      const core::PlanNodePtr& plan,
      exec::DriverCtx*) const override {
    const auto& values = static_cast<const core::ValuesNode&>(*plan).values();
    return !values.empty() &&
        dynamic_cast<const CudfVector*>(values.front().get()) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  bool keepOperator() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator*,
      const core::PlanNodePtr&,
      exec::DriverCtx*,
      int32_t) const override {
    return {};
  }
};

class VectorSearchJoinTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    auto& config = CudfConfig::getInstance();
    savedSearchEnabled_ = config.vectorSearchEnabled;
    savedMinSize_ = config.vectorSearchMinIndexSize;
    config.vectorSearchEnabled = true;
    config.vectorSearchMinIndexSize = 0;
    registerCudf();
    queryCtx_ = createQueryCtx(driverExecutor_.get());
  }

  void TearDown() override {
    queryCtx_.reset();
    unregisterCudf();
    auto& config = CudfConfig::getInstance();
    config.vectorSearchEnabled = savedSearchEnabled_;
    config.vectorSearchMinIndexSize = savedMinSize_;
    OperatorTestBase::TearDown();
  }

  static std::string pack(
      const std::vector<float>& values,
      const std::vector<int64_t>& ids,
      VectorIndexKind kind = VectorIndexKind::kFlat,
      LocalVectorIndexOptions options = {}) {
    std::string bytes;
    FaissLocalVectorIndex::build(
        kind,
        VectorIndexMetric::kCosine,
        options,
        Embeddings(values, 2),
        ids,
        [&](std::string_view chunk) { bytes.append(chunk); });
    return bytes;
  }

  RowVectorPtr buildRows() {
    return makeRowVector(
        {"metadata", "index_bytes"},
        {makeFlatVector<int64_t>({10, 20}),
         makeFlatVector<std::string>(
             {pack({1, 0, 0, 1}, {-2, kLargeId}), pack({0, 1}, {-2})},
             VARBINARY())});
  }

  core::PlanNodePtr plan(
      const std::vector<RowVectorPtr>& probe,
      const RowVectorPtr& build,
      core::JoinType type = core::JoinType::kInner,
      bool routed = false,
      bool parallel = false,
      int64_t k = 10) {
    auto ids = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(ids)
        .values(probe, parallel)
        .vectorSearchJoin(
            routed ? std::vector<std::string>{"route"}
                   : std::vector<std::string>{},
            routed ? std::vector<std::string>{"partition"}
                   : std::vector<std::string>{},
            PlanBuilder(ids).values({build}).planNode(),
            "embedding",
            "index_bytes",
            k,
            {"query", "metadata", "neighbor", "score"},
            "neighbor",
            "score",
            type)
        .planNode();
  }

  AssertQueryBuilder run(const core::PlanNodePtr& plan) {
    AssertQueryBuilder builder(plan);
    builder.queryCtx(queryCtx_);
    builder.config(core::QueryConfig::kPreferredOutputBatchBytes, "1");
    builder.config(CudfFromVelox::kGpuBatchSizeRows, "1");
    return builder;
  }

  static int64_t operatorCount(const exec::Task& task, std::string_view type) {
    int64_t result = 0;
    for (const auto& pipeline : task.taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        if (op.operatorType == type) {
          result += op.numDrivers;
        }
      }
    }
    return result;
  }

  static int64_t deviceFallbacks(const exec::Task& task) {
    int64_t result = 0;
    for (const auto& pipeline : task.taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        const auto metric = op.runtimeStats.find("vectorSearchDeviceFallback");
        if (metric != op.runtimeStats.end()) {
          result += metric->second.sum;
        }
      }
    }
    return result;
  }

  static void expectDeviceProbes(const exec::Task& task, int64_t count = 1) {
    EXPECT_EQ(operatorCount(task, "CudfVectorSearchJoinProbe"), count);
    EXPECT_EQ(deviceFallbacks(task), 0);
  }

  static std::shared_ptr<core::QueryCtx> createQueryCtx(
      folly::Executor* executor) {
    auto formats = std::make_shared<LocalVectorIndexFormats>();
    formats->insert(
        std::string(FaissLocalVectorIndex::kFormat),
        FaissLocalVectorIndex::format());
    auto ctx = core::QueryCtx::create(executor);
    ctx->setRegistry(LocalVectorIndexFormat::kRegistryKey, formats);
    return ctx;
  }

  std::unique_ptr<exec::TaskCursor> cursorFor(
      const core::PlanNodePtr& plan,
      bool serial = false) {
    exec::CursorParameters params;
    params.planNode = plan;
    params.queryCtx = serial ? createQueryCtx(nullptr) : queryCtx_;
    params.serialExecution = serial;
    params.queryConfigs.emplace(
        core::QueryConfig::kPreferredOutputBatchBytes, "1");
    return exec::TaskCursor::create(params);
  }

 private:
  bool savedSearchEnabled_{false};
  int64_t savedMinSize_{0};
  std::shared_ptr<core::QueryCtx> queryCtx_;
};

TEST_F(VectorSearchJoinTest, multipleIndexesAndLeftUnmatched) {
  auto input = makeRowVector(
      {"query", "embedding"},
      {makeFlatVector<int64_t>({7, 7}),
       makeArrayVector<float>({{2, 0}, {0, 1}})});
  auto unmatched = makeRowVector(
      {"query", "embedding"},
      {makeFlatVector<int64_t>({8}),
       BaseVector::createNullConstant(ARRAY(REAL()), 1, pool())});
  const auto task =
      run(plan({input, unmatched}, buildRows(), core::JoinType::kLeft))
          .assertResults(makeRowVector(
              {makeFlatVector<int64_t>({7, 7, 7, 7, 7, 7, 8}),
               makeNullableFlatVector<int64_t>(
                   {10, 10, 20, 10, 10, 20, std::nullopt}),
               makeNullableFlatVector<int64_t>(
                   {-2, kLargeId, -2, -2, kLargeId, -2, std::nullopt}),
               makeNullableFlatVector<float>(
                   {1, 0, 0, 0, 1, 1, std::nullopt})}));
  expectDeviceProbes(*task);
}

TEST_F(VectorSearchJoinTest, invalidEmbeddingOnlyWhenSearched) {
  auto input = makeRowVector(
      {"query", "embedding"},
      {makeFlatVector<int64_t>({7}), makeArrayVector<float>({{1}})});
  VELOX_ASSERT_THROW(
      run(plan({input}, buildRows())).copyResults(pool()),
      "Query embedding length must match the vector index's");
  auto withNull = makeArrayVector<float>({{1, 0}});
  withNull->elements()->setNull(1, true);
  VELOX_ASSERT_THROW(
      run(plan(
              {makeRowVector(
                  {"query", "embedding"},
                  {makeFlatVector<int64_t>({7}), withNull})},
              buildRows()))
          .copyResults(pool()),
      "Query embedding must not contain nulls");
  auto empty = makeRowVector(
      {"metadata", "index_bytes"},
      {makeFlatVector<int64_t>({}),
       makeFlatVector<std::string>({}, VARBINARY())});
  run(plan({input}, empty, core::JoinType::kLeft))
      .assertResults(makeRowVector(
          {makeFlatVector<int64_t>({7}),
           makeNullableFlatVector<int64_t>({std::nullopt}),
           makeNullableFlatVector<int64_t>({std::nullopt}),
           makeNullableFlatVector<float>({std::nullopt})}));
}

TEST_F(VectorSearchJoinTest, cancelWithPendingBatches) {
  auto input = makeRowVector(
      {"query", "embedding"},
      {makeFlatVector<int64_t>({7, 8}),
       makeArrayVector<float>({{2, 0}, {0, 1}})});
  auto cursor = cursorFor(plan({input}, buildRows()));
  ASSERT_TRUE(cursor->moveNext());
  ASSERT_EQ(cursor->current()->size(), 1);
  cursor->task()->requestCancel();
  EXPECT_ANY_THROW(cursor->moveNext());
}

TEST_F(VectorSearchJoinTest, waitsForDifferentProducerStream) {
  OperatorAdapterRegistry::getInstance().registerAdapterFront(
      std::make_unique<DeviceValuesAdapter>());
  const auto firstStream = cudfGlobalStreamPool().get_stream();
  const auto secondStream = cudfGlobalStreamPool().get_stream();
  ASSERT_TRUE(firstStream != secondStream);
  const auto deviceInput = [&](int64_t id, cuda::stream_ref stream) {
    auto rows = makeRowVector(
        {"query", "embedding"},
        {makeFlatVector<int64_t>({id}), makeArrayVector<float>({{1, 0}})});
    auto table = with_arrow::toCudfTable(rows, pool(), stream, get_output_mr());
    return std::make_shared<CudfVector>(
        pool(), rows->type(), rows->size(), std::move(table), stream);
  };
  auto first = deviceInput(7, firstStream);
  auto second = deviceInput(8, secondStream);
  auto build = makeRowVector(
      {"metadata", "index_bytes"},
      {makeFlatVector<int64_t>({10}),
       makeFlatVector<std::string>({pack({1, 0}, {-2})}, VARBINARY())});
  auto cursor = cursorFor(plan({first, second}, build), true);
  ASSERT_TRUE(cursor->moveNext());
  EXPECT_EQ(
      cursor->current()->childAt(3)->asFlatVector<float>()->valueAt(0), 1);

  // The second vector still contains the old embedding when its producer
  // queues a delayed write. The probe must consume the replacement.
  const auto embedding =
      cudf::lists_column_view(second->getTableView().column(1));
  auto* values = const_cast<float*>(embedding.child().data<float>());
  ASSERT_EQ(
      cudaLaunchHostFunc(
          secondStream.get(),
          [](void*) {
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
          },
          nullptr),
      cudaSuccess);
  ASSERT_EQ(
      cudaMemsetAsync(values, 0, 2 * sizeof(float), secondStream.get()),
      cudaSuccess);
  ASSERT_TRUE(cursor->moveNext());
  EXPECT_EQ(
      cursor->current()->childAt(3)->asFlatVector<float>()->valueAt(0), 0);
  EXPECT_FALSE(cursor->moveNext());
}

TEST_F(VectorSearchJoinTest, partialIvfResults) {
  unregisterCudf();
  CudfConfig::getInstance().vectorSearchMinIndexSize = 1'024;
  registerCudf();
  // The production builder caps nlist at one list per 39 embeddings.
  constexpr int32_t kRows = 84;
  std::vector<float> values(kRows * 2);
  std::vector<int64_t> ids(kRows);
  for (int32_t row = 0; row < kRows; ++row) {
    values[2 * row] = row < kRows / 2 ? 1.0f : -1.0f;
    ids[row] = row == 0 ? -2 : kLargeId + row;
  }
  auto input = makeRowVector(
      {"query", "embedding"},
      {makeFlatVector<int64_t>({7, 8}),
       makeArrayVector<float>({{2, 0}, {-1, 0}})});
  const auto expected = makeRowVector(
      {makeFlatVector<int64_t>(
           kRows, [](auto row) { return row < kRows / 2 ? 7 : 8; }),
       makeFlatVector<int64_t>(kRows, [](auto) { return 10; }),
       makeFlatVector<int64_t>(ids),
       makeFlatVector<float>(kRows, [](auto) { return 1.0f; })});
  for (const auto kind : {VectorIndexKind::kIvfFlat, VectorIndexKind::kIvfSq}) {
    SCOPED_TRACE(VectorIndexKindName::toName(kind));
    auto build = makeRowVector(
        {"metadata", "index_bytes"},
        {makeFlatVector<int64_t>({10}),
         makeFlatVector<std::string>(
             {pack(values, ids, kind, {.nlist = 2, .nprobe = 1})},
             VARBINARY())});
    const auto task =
        run(plan({input}, build, core::JoinType::kInner, false, false, kRows))
            .assertResults(expected);
    expectDeviceProbes(*task);
  }
}

TEST_F(VectorSearchJoinTest, multipleProbeDrivers) {
  constexpr vector_size_t kIndexRows = 256;
  constexpr vector_size_t kQueries = 128;
  constexpr int32_t kDrivers = 4;
  constexpr int64_t k = 7;
  constexpr float kAngleStep = 0.02f;
  std::vector<float> embeddings;
  std::vector<int64_t> ids(kIndexRows);
  std::iota(ids.begin(), ids.end(), kLargeId);
  for (vector_size_t row = 0; row < kIndexRows; ++row) {
    const float angle = row * kAngleStep;
    embeddings.insert(embeddings.end(), {std::cos(angle), std::sin(angle)});
  }
  std::vector<std::vector<float>> queries;
  for (vector_size_t row = 0; row < kQueries; ++row) {
    const float angle = (2 * row + 0.31f) * kAngleStep;
    queries.push_back({std::cos(angle), std::sin(angle)});
  }
  auto input = makeRowVector(
      {"query", "embedding"},
      {makeFlatVector<int64_t>(kQueries, [](auto row) { return row; }),
       makeArrayVector<float>(queries)});
  auto build = makeRowVector(
      {"metadata", "index_bytes"},
      {makeConstant<int64_t>(10, 1),
       makeFlatVector<std::string>(
           {pack(
               embeddings,
               ids,
               VectorIndexKind::kIvfFlat,
               {.nlist = 4, .nprobe = 4})},
           VARBINARY())});
  const auto runWithDrivers = [&](bool parallel, int32_t drivers) {
    auto builder =
        run(plan({input}, build, core::JoinType::kInner, false, parallel, k));
    builder.maxDrivers(drivers)
        .config(core::QueryConfig::kPreferredOutputBatchBytes, 1 << 20)
        .config(CudfFromVelox::kGpuBatchSizeRows, kQueries);
    return builder;
  };
  unregisterCudf();
  const auto cpu = runWithDrivers(false, 1).copyResults(pool());
  registerCudf();
  ASSERT_EQ(cpu->size(), kQueries * k);
  std::shared_ptr<exec::Task> serialTask;
  const auto serial = runWithDrivers(false, 1).copyResults(pool(), serialTask);
  ASSERT_TRUE(exec::test::assertEqualResults({cpu}, {serial}));
  expectDeviceProbes(*serialTask);
  const auto parallelTask =
      runWithDrivers(true, kDrivers)
          .assertResults(std::vector<RowVectorPtr>(kDrivers, serial));
  expectDeviceProbes(*parallelTask, kDrivers);
}

TEST_F(VectorSearchJoinTest, smallIndexesStayOnGpuWithDeviceProbe) {
  constexpr int32_t kMinIndexSize = 1'024;
  unregisterCudf();
  CudfConfig::getInstance().vectorSearchMinIndexSize = kMinIndexSize;
  registerCudf();
  std::vector<float> embeddings;
  std::vector<int64_t> ids(kMinIndexSize);
  std::iota(ids.begin(), ids.end(), kLargeId);
  embeddings.insert(embeddings.end(), {1, 0});
  for (int32_t row = 1; row < kMinIndexSize; ++row) {
    embeddings.insert(embeddings.end(), {0, 1});
  }
  const auto small = pack({1, 0}, {-2});
  const auto large = pack(embeddings, ids);
  // Opening the same small index for host searches still respects the cutoff.
  const FaissLocalVectorIndex hostIndex(small, VectorIndexMetric::kCosine, {});
  EXPECT_FALSE(hostIndex.accelerated());
  auto build = makeRowVector(
      {"metadata", "index_bytes"},
      {makeFlatVector<int64_t>({10, 20}),
       makeFlatVector<std::string>({small, large}, VARBINARY())});
  auto input = makeRowVector(
      {"query", "embedding"},
      {makeFlatVector<int64_t>({7}), makeArrayVector<float>({{2, 0}})});
  const auto task =
      run(plan({input}, build, core::JoinType::kInner, false, false, 1))
          .assertResults(makeRowVector(
              {makeFlatVector<int64_t>({7, 7}),
               makeFlatVector<int64_t>({10, 20}),
               makeFlatVector<int64_t>({-2, kLargeId}),
               makeFlatVector<float>({1, 1})}));
  expectDeviceProbes(*task);
}

TEST_F(VectorSearchJoinTest, smallHnswUsesGpuProbe) {
  auto input = makeRowVector(
      {"query", "embedding"},
      {makeFlatVector<int64_t>({7, 8}),
       makeArrayVector<float>({{2, 1}, {1, -2}})});
  const auto bytes = pack(
      {1, 0, 0, 1, -1, 0, 0, -1},
      {-2, kLargeId, 21, 22},
      VectorIndexKind::kHnswSq);
  EXPECT_FALSE(FaissLocalVectorIndex(bytes, VectorIndexMetric::kCosine, {})
                   .accelerated());
  auto build = makeRowVector(
      {"metadata", "index_bytes"},
      {makeConstant<int64_t>(10, 1),
       makeFlatVector<std::string>({bytes}, VARBINARY())});
  const auto node =
      plan({input}, build, core::JoinType::kInner, false, false, 3);
  unregisterCudf();
  const auto expected = run(node).copyResults(pool());
  registerCudf();
  ASSERT_EQ(expected->size(), 6);
  const auto task = run(node).assertResults(expected);
  expectDeviceProbes(*task);
}

TEST_F(VectorSearchJoinTest, fallbackBeforeInput) {
  auto input = makeRowVector(
      {"query", "embedding"},
      {makeFlatVector<int64_t>({7}), makeArrayVector<float>({{2, 0}})});
  auto build = makeRowVector(
      {"metadata", "index_bytes"},
      {makeConstant<int64_t>(10, 1),
       makeFlatVector<std::string>(
           {pack(
               {1, 0, 0, 1, -1, 0, 0, -1, 0.6f, 0.8f},
               {-2, kLargeId, 21, 22, 23},
               VectorIndexKind::kHnswSq)},
           VARBINARY())});
  const auto node =
      plan({input}, build, core::JoinType::kInner, false, false, 2);
  unregisterCudf();
  const auto expected = run(node).copyResults(pool());
  registerCudf();
  ASSERT_EQ(expected->size(), 2);
  const auto task = run(node).assertResults(expected);
  EXPECT_EQ(operatorCount(*task, "CudfVectorSearchJoinProbe"), 1);
  EXPECT_EQ(deviceFallbacks(*task), 1);
}

TEST_F(VectorSearchJoinTest, routedDuplicateKeysAndUnmatchedQueries) {
  const auto indexes = buildRows();
  auto build = makeRowVector(
      {"partition", "metadata", "index_bytes"},
      {makeFlatVector<int64_t>({1, 1}),
       indexes->childAt(0),
       indexes->childAt(1)});
  auto input = makeRowVector(
      {"query", "route", "embedding"},
      {makeFlatVector<int64_t>({7, 8, 9}),
       makeNullableFlatVector<int64_t>({1, 99, std::nullopt}),
       makeArrayVector<float>({{2, 0}, {3}, {4}})});
  const auto task =
      run(plan({input}, build, core::JoinType::kLeft, true))
          .assertResults(makeRowVector(
              {makeFlatVector<int64_t>({7, 7, 7, 8, 9}),
               makeNullableFlatVector<int64_t>(
                   {10, 10, 20, std::nullopt, std::nullopt}),
               makeNullableFlatVector<int64_t>(
                   {-2, kLargeId, -2, std::nullopt, std::nullopt}),
               makeNullableFlatVector<float>(
                   {1, 0, 0, std::nullopt, std::nullopt})}));
  expectDeviceProbes(*task);
}

TEST_F(VectorSearchJoinTest, zeroAndNonFiniteEmbeddings) {
  auto input = makeRowVector(
      {"query", "embedding"},
      {makeFlatVector<int64_t>({7, 8, 9}),
       makeArrayVector<float>(
           {{0, 0},
            {std::numeric_limits<float>::quiet_NaN(), 1},
            {std::numeric_limits<float>::infinity(), 1}})});
  const auto node = plan({input}, buildRows(), core::JoinType::kLeft);
  unregisterCudf();
  const auto expected = run(node).copyResults(pool());
  registerCudf();
  const auto task = run(node).assertResults(expected);
  expectDeviceProbes(*task);
}

TEST_F(VectorSearchJoinTest, singleKeyTypeEligibility) {
  const auto indexes = buildRows();
  const auto check = [&](const VectorPtr& route,
                         const VectorPtr& partition,
                         bool expectGpu) {
    auto build = makeRowVector(
        {"partition", "metadata", "index_bytes"},
        {partition, indexes->childAt(0), indexes->childAt(1)});
    auto input = makeRowVector(
        {"query", "route", "embedding"},
        {makeFlatVector<int64_t>({7}),
         route,
         makeArrayVector<float>({{2, 0}})});
    const auto task = run(plan({input}, build, core::JoinType::kInner, true))
                          .assertResults(makeRowVector(
                              {makeFlatVector<int64_t>({7, 7}),
                               makeFlatVector<int64_t>({10, 10}),
                               makeFlatVector<int64_t>({-2, kLargeId}),
                               makeFlatVector<float>({1, 0})}));
    expectDeviceProbes(*task, expectGpu ? 1 : 0);
    EXPECT_EQ(operatorCount(*task, "VectorSearchJoinProbe"), expectGpu ? 0 : 1);
  };
  check(
      makeFlatVector<std::string>({"x"}),
      makeFlatVector<std::string>({"x", "y"}),
      true);
  check(makeFlatVector<float>({1}), makeFlatVector<float>({1, 2}), false);
}

TEST_F(VectorSearchJoinTest, vectorSearchFlagControlsJoinsAndRerank) {
  auto input = makeRowVector(
      {"query", "ordinal", "ds", "codec_id", "embedding"},
      {makeFlatVector<int64_t>({7, 7}),
       makeFlatVector<int64_t>({0, 1}),
       makeFlatVector<std::string>({"a", "b"}),
       makeFlatVector<int64_t>({10, 20}),
       makeArrayVector<float>({{2, 0}, {0, 3}})});
  auto centroids = makeRowVector(
      {"c_ds", "c_codec_id", "codec"},
      {makeFlatVector<std::string>({"a", "b"}),
       makeFlatVector<int64_t>({10, 20}),
       makeFlatVector<std::string>(
           {pack({1, 0, 0, 1}, {1, 2}), pack({0, 1, 1, 0}, {1, 2})},
           VARBINARY())});
  auto build = makeRowVector(
      {"i_ds", "i_codec_id", "i_cluster_id", "metadata", "index_bytes"},
      {makeFlatVector<std::string>({"a", "b", "b", "a"}),
       makeFlatVector<int64_t>({10, 20, 10, 10}),
       makeFlatVector<int64_t>({1, 1, 1, 2}),
       makeFlatVector<int64_t>({100, 200, 300, 400}),
       makeFlatVector<std::string>(
           {pack({1, 0}, {-2}),
            pack({0, 1}, {kLargeId}),
            pack({1, 0}, {30}),
            pack({1, 0}, {40})},
           VARBINARY())});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  const auto node =
      PlanBuilder(ids)
          .values({input})
          .vectorSearchJoin(
              {"ds", "codec_id"},
              {"c_ds", "c_codec_id"},
              PlanBuilder(ids).values({centroids}).planNode(),
              "embedding",
              "codec",
              1,
              {"query", "ordinal", "ds", "codec_id", "embedding", "cluster_id"},
              "cluster_id")
          .vectorSearchJoin(
              {"ds", "codec_id", "cluster_id"},
              {"i_ds", "i_codec_id", "i_cluster_id"},
              PlanBuilder(ids).values({build}).planNode(),
              "embedding",
              "index_bytes",
              10,
              {"query",
               "ordinal",
               "cluster_id",
               "metadata",
               "neighbor",
               "score"},
              "neighbor",
              "score")
          .topNRowNumber({"ordinal"}, {"score DESC"}, 10, false)
          .planNode();
  const auto expected = makeRowVector(
      {makeFlatVector<int64_t>({7, 7}),
       makeFlatVector<int64_t>({0, 1}),
       makeFlatVector<int64_t>({1, 1}),
       makeFlatVector<int64_t>({100, 200}),
       makeFlatVector<int64_t>({-2, kLargeId}),
       makeFlatVector<float>({1, 1})});
  for (const bool enabled : {false, true}) {
    SCOPED_TRACE(enabled);
    unregisterCudf();
    CudfConfig::getInstance().initialize(
        {{CudfConfig::kCudfVectorSearchEnabled, enabled ? "true" : "false"}});
    registerCudf();
    EXPECT_EQ(localVectorIndexAccelerator() != nullptr, enabled);
    const auto task = run(node).assertResults(expected);
    expectDeviceProbes(*task, enabled ? 2 : 0);
    EXPECT_EQ(operatorCount(*task, "VectorSearchJoinProbe"), enabled ? 0 : 2);
    EXPECT_EQ(operatorCount(*task, "CudfTopNRowNumber"), enabled ? 1 : 0);
    EXPECT_EQ(operatorCount(*task, "TopNRowNumber"), enabled ? 0 : 1);
    EXPECT_EQ(operatorCount(*task, "CudfFromVelox"), enabled ? 1 : 0);
    EXPECT_EQ(operatorCount(*task, "CudfToVelox"), enabled ? 1 : 0);
  }
}

TEST_F(VectorSearchJoinTest, multiKeyLeftJoin) {
  auto input = makeRowVector(
      {"query", "ds", "route", "embedding"},
      {makeFlatVector<int64_t>({0, 1, 2, 3, 4}),
       makeNullableFlatVector<std::string>({"a", "a", std::nullopt, "b", "a"}),
       makeNullableFlatVector<int32_t>({1, 2, 1, 2, std::nullopt}),
       // Independently matching columns are insufficient. Bad embeddings on
       // unmatched rows must not be validated or searched.
       makeArrayVector<float>({{1, 0}, {2}, {3}, {0, 1}, {4}})});
  auto build = makeRowVector(
      {"i_ds", "i_route", "index_bytes"},
      {makeNullableFlatVector<std::string>({"a", "a", "b", std::nullopt, "a"}),
       makeNullableFlatVector<int32_t>({1, 1, 2, 1, std::nullopt}),
       makeFlatVector<std::string>(
           {pack({1, 0}, {10}),
            pack({1, 0}, {20}),
            pack({0, 1}, {30}),
            pack({1, 0}, {40}),
            pack({1, 0}, {50})},
           VARBINARY())});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  const auto node = PlanBuilder(ids)
                        .values({input})
                        .vectorSearchJoin(
                            {"ds", "route"},
                            {"i_ds", "i_route"},
                            PlanBuilder(ids).values({build}).planNode(),
                            "embedding",
                            "index_bytes",
                            1,
                            {"query", "neighbor", "score"},
                            "neighbor",
                            "score",
                            core::JoinType::kLeft)
                        .planNode();
  const auto expected = makeRowVector(
      {makeFlatVector<int64_t>({0, 0, 1, 2, 3, 4}),
       makeNullableFlatVector<int64_t>(
           {10, 20, std::nullopt, std::nullopt, 30, std::nullopt}),
       makeNullableFlatVector<float>(
           {1, 1, std::nullopt, std::nullopt, 1, std::nullopt})});
  for (const int32_t bytes : {1, 65'536}) {
    const auto task =
        run(node)
            .config(core::QueryConfig::kPreferredOutputBatchBytes, bytes)
            .assertResults(expected);
    expectDeviceProbes(*task);
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox
