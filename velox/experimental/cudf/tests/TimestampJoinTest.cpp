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

// Reproducer for a bug where cudf's HashJoin crashes when probe and build
// sides have timestamp columns at different resolutions (MICROSECONDS vs
// NANOSECONDS). The fix normalizes timestamps before calling
// cudf's filter APIs.
//
// Strategy: register two CudfHiveConnector instances with different
// timestamp_type configs. Write Parquet files normally. Scan them through
// different connectors so each side gets a different timestamp resolution.
// Then run a filtered hash join.

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::connector;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::exec::test;

namespace {

// Connector IDs for the two timestamp resolutions.
const std::string kMicroConnectorId = "test-cudf-hive-micro";
const std::string kNanoConnectorId = "test-cudf-hive-nano";

class TimestampJoinTest : public CudfHiveConnectorTestBase {
 protected:
  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();

    registerTimestampConnector(
        kMicroConnectorId,
        static_cast<int>(cudf::type_id::TIMESTAMP_MICROSECONDS));
    registerTimestampConnector(
        kNanoConnectorId,
        static_cast<int>(cudf::type_id::TIMESTAMP_NANOSECONDS));

    // Probe table: columns t_key (INTEGER) and t_ts (TIMESTAMP).
    probeType_ = ROW({{"t_key", INTEGER()}, {"t_ts", TIMESTAMP()}});
    // Build table: columns u_key (INTEGER) and u_ts (TIMESTAMP).
    buildType_ = ROW({{"u_key", INTEGER()}, {"u_ts", TIMESTAMP()}});

    constexpr int32_t kNumRows{100};

    // Populate probe data. Timestamp seconds = row * 2.
    auto probeKeys = makeFlatVector<int32_t>(
        kNumRows, [](vector_size_t row) { return row % 11; });
    auto probeTimestamps = makeFlatVector<Timestamp>(
        kNumRows, [](vector_size_t row) { return Timestamp(row * 2, 0); });
    probeData_ = makeRowVector({"t_key", "t_ts"}, {probeKeys, probeTimestamps});

    // Populate build data. Timestamp seconds = row * 3.
    auto buildKeys = makeFlatVector<int32_t>(
        kNumRows, [](vector_size_t row) { return row % 11; });
    auto buildTimestamps = makeFlatVector<Timestamp>(
        kNumRows, [](vector_size_t row) { return Timestamp(row * 3, 0); });
    buildData_ = makeRowVector({"u_key", "u_ts"}, {buildKeys, buildTimestamps});

    // Write Parquet files.
    probeFile_ = TempFilePath::create();
    buildFile_ = TempFilePath::create();
    writeToFile(probeFile_->getPath(), probeData_);
    writeToFile(buildFile_->getPath(), buildData_);

    // Register DuckDB tables for reference queries.
    createDuckDbTable("t", {probeData_});
    createDuckDbTable("u", {buildData_});
  }

  void TearDown() override {
    probeData_.reset();
    buildData_.reset();
    ConnectorRegistry::global().erase(kMicroConnectorId);
    ConnectorRegistry::global().erase(kNanoConnectorId);
    CudfHiveConnectorTestBase::TearDown();
  }

  // Register a CudfHiveConnector with the given connector ID and timestamp
  // type (as an integer cudf::type_id value).
  void registerTimestampConnector(
      const std::string& connectorId,
      int timestampTypeId) {
    cudf_velox::connector::hive::CudfHiveConnectorFactory factory;
    auto config = std::unordered_map<std::string, std::string>{
        {cudf_velox::connector::hive::CudfHiveConfig::kTimestampType,
         std::to_string(timestampTypeId)},
    };
    auto connector = factory.newConnector(
        connectorId,
        std::make_shared<facebook::velox::config::ConfigBase>(
            std::move(config)),
        ioExecutor_.get());
    ConnectorRegistry::global().insert(connector->connectorId(), connector);
  }

  // Build a table-scan plan node that reads through the given connector.
  PlanBuilder scanThrough(
      const RowTypePtr& outputType,
      const std::string& connectorId,
      std::shared_ptr<PlanNodeIdGenerator> idGenerator) {
    return PlanBuilder(idGenerator, pool_.get())
        .startTableScan()
        .outputType(outputType)
        .connectorId(connectorId)
        .endTableScan();
  }

  // Build a CudfHiveConnectorSplit for the given file and connector.
  std::shared_ptr<cudf_velox::connector::hive::CudfHiveConnectorSplit>
  makeSplit(const std::string& filePath, const std::string& connectorId) {
    return cudf_velox::connector::hive::CudfHiveConnectorSplitBuilder(filePath)
        .connectorId(connectorId)
        .build();
  }

  RowTypePtr probeType_;
  RowTypePtr buildType_;
  RowVectorPtr probeData_;
  RowVectorPtr buildData_;
  std::shared_ptr<TempFilePath> probeFile_;
  std::shared_ptr<TempFilePath> buildFile_;
};

// Inner join with filter on mismatched timestamp resolutions.
// Probe side reads timestamps as MICROSECONDS, build side as NANOSECONDS.
// Without the normalization fix this crashes in cudf's filter evaluation.
TEST_F(TimestampJoinTest, innerJoinFilterWithMismatchedTimestamps) {
  auto idGenerator = std::make_shared<PlanNodeIdGenerator>();

  PlanNodeId probeScanId;
  PlanNodeId buildScanId;

  auto plan = scanThrough(probeType_, kMicroConnectorId, idGenerator)
                  .capturePlanNodeId(probeScanId)
                  .hashJoin(
                      {"t_key"},
                      {"u_key"},
                      scanThrough(buildType_, kNanoConnectorId, idGenerator)
                          .capturePlanNodeId(buildScanId)
                          .planNode(),
                      "t_ts < u_ts",
                      {"t_key", "t_ts", "u_ts"},
                      JoinType::kInner)
                  .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .split(probeScanId, makeSplit(probeFile_->getPath(), kMicroConnectorId))
      .split(buildScanId, makeSplit(buildFile_->getPath(), kNanoConnectorId))
      .assertResults(
          "SELECT t.t_key, t.t_ts, u.u_ts "
          "FROM t, u "
          "WHERE t.t_key = u.u_key AND t.t_ts < u.u_ts");
}

// Left semi filter join with mismatched timestamp resolutions.
TEST_F(TimestampJoinTest, leftSemiFilterJoinWithMismatchedTimestamps) {
  auto idGenerator = std::make_shared<PlanNodeIdGenerator>();

  PlanNodeId probeScanId;
  PlanNodeId buildScanId;

  auto plan = scanThrough(probeType_, kMicroConnectorId, idGenerator)
                  .capturePlanNodeId(probeScanId)
                  .hashJoin(
                      {"t_key"},
                      {"u_key"},
                      scanThrough(buildType_, kNanoConnectorId, idGenerator)
                          .capturePlanNodeId(buildScanId)
                          .planNode(),
                      "t_ts < u_ts",
                      {"t_key", "t_ts"},
                      JoinType::kLeftSemiFilter)
                  .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .split(probeScanId, makeSplit(probeFile_->getPath(), kMicroConnectorId))
      .split(buildScanId, makeSplit(buildFile_->getPath(), kNanoConnectorId))
      .assertResults(
          "SELECT t.t_key, t.t_ts "
          "FROM t "
          "WHERE EXISTS ("
          "  SELECT 1 FROM u WHERE t.t_key = u.u_key AND t.t_ts < u.u_ts"
          ")");
}

// Anti join with mismatched timestamp resolutions.
TEST_F(TimestampJoinTest, antiJoinWithMismatchedTimestamps) {
  auto idGenerator = std::make_shared<PlanNodeIdGenerator>();

  PlanNodeId probeScanId;
  PlanNodeId buildScanId;

  auto plan = scanThrough(probeType_, kMicroConnectorId, idGenerator)
                  .capturePlanNodeId(probeScanId)
                  .hashJoin(
                      {"t_key"},
                      {"u_key"},
                      scanThrough(buildType_, kNanoConnectorId, idGenerator)
                          .capturePlanNodeId(buildScanId)
                          .planNode(),
                      "t_ts < u_ts",
                      {"t_key", "t_ts"},
                      JoinType::kAnti)
                  .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .split(probeScanId, makeSplit(probeFile_->getPath(), kMicroConnectorId))
      .split(buildScanId, makeSplit(buildFile_->getPath(), kNanoConnectorId))
      .assertResults(
          "SELECT t.t_key, t.t_ts "
          "FROM t "
          "WHERE NOT EXISTS ("
          "  SELECT 1 FROM u WHERE t.t_key = u.u_key AND t.t_ts < u.u_ts"
          ")");
}

// Null-aware left semi project join with filter and mismatched timestamp
// resolutions.  This exercises the `accumulateIndeterminate` lambda inside
// `leftSemiProjectJoin`, which creates synthetic cross-product pairs and
// calls `filter_join_indices` on them.  Null keys on the probe side force
// entry into the "Type B" indeterminate path.
TEST_F(TimestampJoinTest, nullAwareLeftSemiProjectWithMismatchedTimestamps) {
  // Create data with null keys to trigger the indeterminate path.
  constexpr int32_t kNumRows{20};

  // Probe: some rows have null keys.
  auto probeKeys = makeFlatVector<int32_t>(
      kNumRows,
      [](vector_size_t row) { return row % 5; },
      [](vector_size_t row) { return row % 7 == 0; });
  auto probeTimestamps = makeFlatVector<Timestamp>(
      kNumRows, [](vector_size_t row) { return Timestamp(row * 2, 0); });
  auto probeWithNulls =
      makeRowVector({"t_key", "t_ts"}, {probeKeys, probeTimestamps});

  // Build: some rows have null keys.
  auto buildKeys = makeFlatVector<int32_t>(
      kNumRows,
      [](vector_size_t row) { return row % 5; },
      [](vector_size_t row) { return row % 6 == 0; });
  auto buildTimestamps = makeFlatVector<Timestamp>(
      kNumRows, [](vector_size_t row) { return Timestamp(row * 3, 0); });
  auto buildWithNulls =
      makeRowVector({"u_key", "u_ts"}, {buildKeys, buildTimestamps});

  // Write to separate Parquet files.
  auto probeNullFile = TempFilePath::create();
  auto buildNullFile = TempFilePath::create();
  writeToFile(probeNullFile->getPath(), probeWithNulls);
  writeToFile(buildNullFile->getPath(), buildWithNulls);

  // Register DuckDB tables for the reference query.
  createDuckDbTable("t_null", {probeWithNulls});
  createDuckDbTable("u_null", {buildWithNulls});

  auto idGenerator = std::make_shared<PlanNodeIdGenerator>();

  PlanNodeId probeScanId;
  PlanNodeId buildScanId;

  // kLeftSemiProject + nullAware=true + filter on timestamp columns.
  // The output is all probe columns plus a boolean "match" column.
  auto plan = scanThrough(probeType_, kMicroConnectorId, idGenerator)
                  .capturePlanNodeId(probeScanId)
                  .hashJoin(
                      {"t_key"},
                      {"u_key"},
                      scanThrough(buildType_, kNanoConnectorId, idGenerator)
                          .capturePlanNodeId(buildScanId)
                          .planNode(),
                      "t_ts < u_ts",
                      {"t_key", "t_ts", "match"},
                      JoinType::kLeftSemiProject,
                      /*nullAware=*/true)
                  .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .split(
          probeScanId, makeSplit(probeNullFile->getPath(), kMicroConnectorId))
      .split(buildScanId, makeSplit(buildNullFile->getPath(), kNanoConnectorId))
      .assertResults(
          "SELECT t_null.t_key, t_null.t_ts, "
          "  t_null.t_key IN ("
          "    SELECT u_null.u_key FROM u_null"
          "    WHERE t_null.t_ts < u_null.u_ts"
          "  ) AS match "
          "FROM t_null");
}

// Verifies correct join results when timestamps exceed the nanosecond overflow
// boundary (~year 2262).  Probe side reads as SECONDS, build side as
// NANOSECONDS.  Without overflow-safe normalization, the normalization step
// casts seconds to nanoseconds (×10^9) which overflows int64 for values beyond
// ~9.2×10^9 seconds, producing wrong filter comparisons.
//
// The test writes Parquet files directly via cuDF (bypassing Velox's Arrow
// bridge which also can't handle >2262 timestamps) to create a file with
// TIMESTAMP_SECONDS containing year-2300 values.
TEST_F(TimestampJoinTest, overflowSafeNormalization) {
  const std::string kSecConnectorId = "test-cudf-hive-sec";
  registerTimestampConnector(
      kSecConnectorId, static_cast<int>(cudf::type_id::TIMESTAMP_SECONDS));

  constexpr int64_t kFarFuture = 10'413'792'000; // ~2300-01-01 in seconds
  constexpr int32_t kNumRows = 10;

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // Write probe Parquet directly with cuDF: INT32 key + TIMESTAMP_SECONDS.
  auto probeFile = TempFilePath::create();
  {
    std::vector<int32_t> keys(kNumRows);
    std::vector<int64_t> ticks(kNumRows);
    for (int i = 0; i < kNumRows; ++i) {
      keys[i] = i % 5;
      ticks[i] = kFarFuture + i * 86'400;
    }
    auto keyCol = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, kNumRows);
    auto tsCol = cudf::make_timestamp_column(
        cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, kNumRows);
    CUDF_CUDA_TRY(cudaMemcpy(
        keyCol->mutable_view().head(),
        keys.data(),
        kNumRows * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CUDF_CUDA_TRY(cudaMemcpy(
        tsCol->mutable_view().head(),
        ticks.data(),
        kNumRows * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(keyCol));
    cols.push_back(std::move(tsCol));
    auto table = std::make_unique<cudf::table>(std::move(cols));

    auto metadata = cudf::io::table_input_metadata(table->view());
    metadata.column_metadata[0].set_name("t_key");
    metadata.column_metadata[1].set_name("t_ts");
    auto options = cudf::io::parquet_writer_options::builder(
                       cudf::io::sink_info(probeFile->getPath()), table->view())
                       .metadata(metadata)
                       .build();
    cudf::io::write_parquet(options, stream);
    stream.synchronize();
  }

  // Write build Parquet with cuDF: INT32 key + TIMESTAMP_NANOSECONDS (near
  // epoch, fits fine in nanos).
  auto buildFile = TempFilePath::create();
  {
    std::vector<int32_t> keys(kNumRows);
    std::vector<int64_t> ticks(kNumRows);
    for (int i = 0; i < kNumRows; ++i) {
      keys[i] = i % 5;
      // Small values near epoch: i * 86400 * 10^9 (fits in int64 for small i).
      ticks[i] = static_cast<int64_t>(i) * 86'400 * 1'000'000'000LL;
    }
    auto keyCol = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, kNumRows);
    auto tsCol = cudf::make_timestamp_column(
        cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}, kNumRows);
    CUDF_CUDA_TRY(cudaMemcpy(
        keyCol->mutable_view().head(),
        keys.data(),
        kNumRows * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CUDF_CUDA_TRY(cudaMemcpy(
        tsCol->mutable_view().head(),
        ticks.data(),
        kNumRows * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(keyCol));
    cols.push_back(std::move(tsCol));
    auto table = std::make_unique<cudf::table>(std::move(cols));

    auto metadata = cudf::io::table_input_metadata(table->view());
    metadata.column_metadata[0].set_name("u_key");
    metadata.column_metadata[1].set_name("u_ts");
    auto options = cudf::io::parquet_writer_options::builder(
                       cudf::io::sink_info(buildFile->getPath()), table->view())
                       .metadata(metadata)
                       .build();
    cudf::io::write_parquet(options, stream);
    stream.synchronize();
  }

  auto idGenerator = std::make_shared<PlanNodeIdGenerator>();

  PlanNodeId probeScanId;
  PlanNodeId buildScanId;

  // Filter: "t_ts > u_ts".  All probe timestamps are year 2300+ and all build
  // timestamps are near epoch, so every key-matching pair should pass.
  // With 10 rows per side and keys 0..4 (2 rows per key per side), we expect
  // 5 keys × 2 probe × 2 build = 20 result rows.
  auto plan = scanThrough(probeType_, kSecConnectorId, idGenerator)
                  .capturePlanNodeId(probeScanId)
                  .hashJoin(
                      {"t_key"},
                      {"u_key"},
                      scanThrough(buildType_, kNanoConnectorId, idGenerator)
                          .capturePlanNodeId(buildScanId)
                          .planNode(),
                      "t_ts > u_ts",
                      {"t_key"},
                      JoinType::kInner)
                  .planNode();

  auto result =
      AssertQueryBuilder(plan)
          .split(probeScanId, makeSplit(probeFile->getPath(), kSecConnectorId))
          .split(buildScanId, makeSplit(buildFile->getPath(), kNanoConnectorId))
          .copyResults(pool_.get());

  // Every key-matching pair passes the filter, so we expect 20 rows.
  EXPECT_EQ(result->size(), 20);

  ConnectorRegistry::global().erase(kSecConnectorId);
}

// ---------------------------------------------------------------------------
// Path B (filteredOutput) tests.
//
// Path B is taken when useAstFilter_ = false, which happens when the filter
// contains a non-AST-supported sub-expression (like CASE WHEN / switch) that
// references columns from both probe and build sides.  In this path,
// cudf::compute_column() is called on the concatenated joined table.  Its
// JIT engine compiles with C++ chrono types and implicitly converts mismatched
// timestamp resolutions using std::common_type (promoting to the finest
// resolution).  This conversion multiplies ticks by the resolution ratio and
// can overflow int64 for large values (e.g., SECONDS × 10^9 for nanos).
//
// The CASE WHEN filter below forces Path B:
//   "CASE WHEN t_key >= 0 THEN t_ts > u_ts ELSE true END"
// Because the CASE/WHEN (switch) expression is not supported in cuDF AST and
// references both probe (t_key, t_ts) and build (u_ts) columns.
// ---------------------------------------------------------------------------

// Test 1: Baseline — both sides use the same SECONDS precision.
// No type conversion occurs.  Verifies Path B produces correct results
// when timestamp resolutions match.
TEST_F(TimestampJoinTest, filteredOutputBaseline) {
  const std::string kSecConnectorId = "test-cudf-hive-sec";
  registerTimestampConnector(
      kSecConnectorId, static_cast<int>(cudf::type_id::TIMESTAMP_SECONDS));

  constexpr int64_t kFarFuture = 10'413'792'000; // ~2300-01-01 in seconds
  constexpr int32_t kNumRows = 10;

  auto stream = cudf::get_default_stream();

  // Probe: TIMESTAMP_SECONDS, year 2300+ values.
  auto probeFile = TempFilePath::create();
  {
    std::vector<int32_t> keys(kNumRows);
    std::vector<int64_t> ticks(kNumRows);
    for (int i = 0; i < kNumRows; ++i) {
      keys[i] = i % 5;
      ticks[i] = kFarFuture + i * 86'400;
    }
    auto keyCol = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, kNumRows);
    auto tsCol = cudf::make_timestamp_column(
        cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, kNumRows);
    CUDF_CUDA_TRY(cudaMemcpy(
        keyCol->mutable_view().head(),
        keys.data(),
        kNumRows * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CUDF_CUDA_TRY(cudaMemcpy(
        tsCol->mutable_view().head(),
        ticks.data(),
        kNumRows * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(keyCol));
    cols.push_back(std::move(tsCol));
    auto table = std::make_unique<cudf::table>(std::move(cols));

    auto metadata = cudf::io::table_input_metadata(table->view());
    metadata.column_metadata[0].set_name("t_key");
    metadata.column_metadata[1].set_name("t_ts");
    auto options = cudf::io::parquet_writer_options::builder(
                       cudf::io::sink_info(probeFile->getPath()), table->view())
                       .metadata(metadata)
                       .build();
    cudf::io::write_parquet(options, stream);
    stream.synchronize();
  }

  // Build: also TIMESTAMP_SECONDS, near-epoch values (days 1-10).
  auto buildFile = TempFilePath::create();
  {
    std::vector<int32_t> keys(kNumRows);
    std::vector<int64_t> ticks(kNumRows);
    for (int i = 0; i < kNumRows; ++i) {
      keys[i] = i % 5;
      ticks[i] = (i + 1) * 86'400;
    }
    auto keyCol = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, kNumRows);
    auto tsCol = cudf::make_timestamp_column(
        cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, kNumRows);
    CUDF_CUDA_TRY(cudaMemcpy(
        keyCol->mutable_view().head(),
        keys.data(),
        kNumRows * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CUDF_CUDA_TRY(cudaMemcpy(
        tsCol->mutable_view().head(),
        ticks.data(),
        kNumRows * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(keyCol));
    cols.push_back(std::move(tsCol));
    auto table = std::make_unique<cudf::table>(std::move(cols));

    auto metadata = cudf::io::table_input_metadata(table->view());
    metadata.column_metadata[0].set_name("u_key");
    metadata.column_metadata[1].set_name("u_ts");
    auto options = cudf::io::parquet_writer_options::builder(
                       cudf::io::sink_info(buildFile->getPath()), table->view())
                       .metadata(metadata)
                       .build();
    cudf::io::write_parquet(options, stream);
    stream.synchronize();
  }

  auto idGenerator = std::make_shared<PlanNodeIdGenerator>();
  PlanNodeId probeScanId;
  PlanNodeId buildScanId;

  auto plan = scanThrough(probeType_, kSecConnectorId, idGenerator)
                  .capturePlanNodeId(probeScanId)
                  .hashJoin(
                      {"t_key"},
                      {"u_key"},
                      scanThrough(buildType_, kSecConnectorId, idGenerator)
                          .capturePlanNodeId(buildScanId)
                          .planNode(),
                      "CASE WHEN t_key >= 0 THEN t_ts > u_ts ELSE true END",
                      {"t_key"},
                      JoinType::kInner)
                  .planNode();

  auto result =
      AssertQueryBuilder(plan)
          .split(probeScanId, makeSplit(probeFile->getPath(), kSecConnectorId))
          .split(buildScanId, makeSplit(buildFile->getPath(), kSecConnectorId))
          .copyResults(pool_.get());

  // 5 keys × 2 probe × 2 build = 20 pairs.  Year 2300 > days 1-10, all pass.
  EXPECT_EQ(result->size(), 20);

  ConnectorRegistry::global().erase(kSecConnectorId);
}

// Test 2: Mismatched precision (SECONDS vs NANOSECONDS) without overflow.
//
// Probe values are small (100–1000 seconds) so the JIT engine's implicit
// SECONDS → NANOSECONDS conversion (×10^9) stays within int64 range and
// produces correct comparisons.
//
// Data design:
//   Probe (SECONDS): key=i%5, ts = 100 + i*100  → 100s, 200s, ..., 1000s
//   Build (NANOSECONDS): key=i%5, ts = (400+i*50)*10^9 → 400s, 450s, ..., 850s
//
// After equi-join (5 keys × 2 probe × 2 build = 20 pairs), the filter
// "t_ts > u_ts" with correct conversion yields 8 rows:
//   Key 0: (600s>400s)=1 pass.  Key 1: (700s>450s)=1.  Key 2:
//   (800s>500s,750s)=2. Key 3: (900s>550s,800s)=2.  Key 4: (1000s>600s,850s)=2.
//   Total=8.
//
// Without any conversion (raw int64 tick comparison), all probe ticks
// (max 1000) are far smaller than all build nanos ticks (min 400×10^9),
// so "t_ts > u_ts" would always be FALSE → 0 rows.
//
// This test PASSES because the JIT engine correctly handles the conversion
// for non-overflowing values.
TEST_F(TimestampJoinTest, filteredOutputMismatchNoOverflow) {
  const std::string kSecConnectorId = "test-cudf-hive-sec";
  registerTimestampConnector(
      kSecConnectorId, static_cast<int>(cudf::type_id::TIMESTAMP_SECONDS));

  constexpr int32_t kNumRows = 10;

  auto stream = cudf::get_default_stream();

  // Probe: TIMESTAMP_SECONDS with small values (100–1000 seconds).
  auto probeFile = TempFilePath::create();
  {
    std::vector<int32_t> keys(kNumRows);
    std::vector<int64_t> ticks(kNumRows);
    for (int i = 0; i < kNumRows; ++i) {
      keys[i] = i % 5;
      ticks[i] = 100 + i * 100; // 100, 200, ..., 1000
    }
    auto keyCol = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, kNumRows);
    auto tsCol = cudf::make_timestamp_column(
        cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, kNumRows);
    CUDF_CUDA_TRY(cudaMemcpy(
        keyCol->mutable_view().head(),
        keys.data(),
        kNumRows * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CUDF_CUDA_TRY(cudaMemcpy(
        tsCol->mutable_view().head(),
        ticks.data(),
        kNumRows * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(keyCol));
    cols.push_back(std::move(tsCol));
    auto table = std::make_unique<cudf::table>(std::move(cols));

    auto metadata = cudf::io::table_input_metadata(table->view());
    metadata.column_metadata[0].set_name("t_key");
    metadata.column_metadata[1].set_name("t_ts");
    auto options = cudf::io::parquet_writer_options::builder(
                       cudf::io::sink_info(probeFile->getPath()), table->view())
                       .metadata(metadata)
                       .build();
    cudf::io::write_parquet(options, stream);
    stream.synchronize();
  }

  // Build: TIMESTAMP_NANOSECONDS with values 400–850 seconds (stored as nanos).
  auto buildFile = TempFilePath::create();
  {
    std::vector<int32_t> keys(kNumRows);
    std::vector<int64_t> ticks(kNumRows);
    for (int i = 0; i < kNumRows; ++i) {
      keys[i] = i % 5;
      ticks[i] = (400LL + i * 50) * 1'000'000'000LL; // 400s..850s in nanos
    }
    auto keyCol = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, kNumRows);
    auto tsCol = cudf::make_timestamp_column(
        cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}, kNumRows);
    CUDF_CUDA_TRY(cudaMemcpy(
        keyCol->mutable_view().head(),
        keys.data(),
        kNumRows * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CUDF_CUDA_TRY(cudaMemcpy(
        tsCol->mutable_view().head(),
        ticks.data(),
        kNumRows * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(keyCol));
    cols.push_back(std::move(tsCol));
    auto table = std::make_unique<cudf::table>(std::move(cols));

    auto metadata = cudf::io::table_input_metadata(table->view());
    metadata.column_metadata[0].set_name("u_key");
    metadata.column_metadata[1].set_name("u_ts");
    auto options = cudf::io::parquet_writer_options::builder(
                       cudf::io::sink_info(buildFile->getPath()), table->view())
                       .metadata(metadata)
                       .build();
    cudf::io::write_parquet(options, stream);
    stream.synchronize();
  }

  auto idGenerator = std::make_shared<PlanNodeIdGenerator>();
  PlanNodeId probeScanId;
  PlanNodeId buildScanId;

  auto plan = scanThrough(probeType_, kSecConnectorId, idGenerator)
                  .capturePlanNodeId(probeScanId)
                  .hashJoin(
                      {"t_key"},
                      {"u_key"},
                      scanThrough(buildType_, kNanoConnectorId, idGenerator)
                          .capturePlanNodeId(buildScanId)
                          .planNode(),
                      "CASE WHEN t_key >= 0 THEN t_ts > u_ts ELSE true END",
                      {"t_key"},
                      JoinType::kInner)
                  .planNode();

  auto result =
      AssertQueryBuilder(plan)
          .split(probeScanId, makeSplit(probeFile->getPath(), kSecConnectorId))
          .split(buildScanId, makeSplit(buildFile->getPath(), kNanoConnectorId))
          .copyResults(pool_.get());

  // With correct conversion: 8 rows pass the filter (see comment above).
  // Without conversion (raw ticks): 0 rows would pass.
  EXPECT_EQ(result->size(), 8);

  ConnectorRegistry::global().erase(kSecConnectorId);
}

// Test 3: Mismatched precision (SECONDS vs NANOSECONDS) WITH overflow.
//
// Probe values span the nanosecond overflow boundary (~9.22×10^9 seconds,
// approximately year 2262).  The JIT engine's implicit SECONDS → NANOSECONDS
// conversion (×10^9) overflows int64 for probe values above this boundary,
// wrapping to negative and corrupting comparisons.
//
// Data design:
//   Probe (SECONDS): key=i%5, ts = 8×10^9 + i*5×10^8
//     → 8.0e9, 8.5e9, 9.0e9, 9.5e9, 10.0e9, 10.5e9, 11.0e9, 11.5e9, 12.0e9, 12.5e9
//     Rows 0-2 (≤9.0e9) are below the overflow boundary (9,223,372,036s).
//     Rows 3-9 (≥9.5e9) exceed it and overflow when multiplied by 10^9.
//   Build (NANOSECONDS): key=i%5, ts = (i+1) * 86400 * 10^9
//     → 1 day, 2 days, ..., 10 days from epoch (all small, positive).
//
// After equi-join (5 keys × 2 probe × 2 build = 20 pairs), expected results:
//   Y = 20 (correct): All probe timestamps (years 2223–2366) > build (days
//   1-10). X = 6  (overflow): Only non-overflowing rows pass (keys 0,1,2 × 2
//   build = 6).
//                       Overflowing rows wrap negative → comparison FALSE.
//   Z = 0  (raw ticks): Probe ticks (max 12.5e9) << build nanos (min 86.4e12),
//                        so raw comparison is always FALSE.
//
// Without overflow-safe normalization in filteredOutput, this test FAILS.
TEST_F(TimestampJoinTest, filteredOutputOverflow) {
  const std::string kSecConnectorId = "test-cudf-hive-sec";
  registerTimestampConnector(
      kSecConnectorId, static_cast<int>(cudf::type_id::TIMESTAMP_SECONDS));

  constexpr int64_t kBase = 8'000'000'000LL;
  constexpr int64_t kStep = 500'000'000LL;
  constexpr int32_t kNumRows = 10;

  auto stream = cudf::get_default_stream();

  // Probe: TIMESTAMP_SECONDS spanning the overflow boundary.
  auto probeFile = TempFilePath::create();
  {
    std::vector<int32_t> keys(kNumRows);
    std::vector<int64_t> ticks(kNumRows);
    for (int i = 0; i < kNumRows; ++i) {
      keys[i] = i % 5;
      ticks[i] = kBase + i * kStep;
    }
    auto keyCol = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, kNumRows);
    auto tsCol = cudf::make_timestamp_column(
        cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, kNumRows);
    CUDF_CUDA_TRY(cudaMemcpy(
        keyCol->mutable_view().head(),
        keys.data(),
        kNumRows * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CUDF_CUDA_TRY(cudaMemcpy(
        tsCol->mutable_view().head(),
        ticks.data(),
        kNumRows * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(keyCol));
    cols.push_back(std::move(tsCol));
    auto table = std::make_unique<cudf::table>(std::move(cols));

    auto metadata = cudf::io::table_input_metadata(table->view());
    metadata.column_metadata[0].set_name("t_key");
    metadata.column_metadata[1].set_name("t_ts");
    auto options = cudf::io::parquet_writer_options::builder(
                       cudf::io::sink_info(probeFile->getPath()), table->view())
                       .metadata(metadata)
                       .build();
    cudf::io::write_parquet(options, stream);
    stream.synchronize();
  }

  // Build: TIMESTAMP_NANOSECONDS near epoch (1-10 days).
  auto buildFile = TempFilePath::create();
  {
    std::vector<int32_t> keys(kNumRows);
    std::vector<int64_t> ticks(kNumRows);
    for (int i = 0; i < kNumRows; ++i) {
      keys[i] = i % 5;
      ticks[i] = static_cast<int64_t>(i + 1) * 86'400LL * 1'000'000'000LL;
    }
    auto keyCol = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, kNumRows);
    auto tsCol = cudf::make_timestamp_column(
        cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}, kNumRows);
    CUDF_CUDA_TRY(cudaMemcpy(
        keyCol->mutable_view().head(),
        keys.data(),
        kNumRows * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CUDF_CUDA_TRY(cudaMemcpy(
        tsCol->mutable_view().head(),
        ticks.data(),
        kNumRows * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(keyCol));
    cols.push_back(std::move(tsCol));
    auto table = std::make_unique<cudf::table>(std::move(cols));

    auto metadata = cudf::io::table_input_metadata(table->view());
    metadata.column_metadata[0].set_name("u_key");
    metadata.column_metadata[1].set_name("u_ts");
    auto options = cudf::io::parquet_writer_options::builder(
                       cudf::io::sink_info(buildFile->getPath()), table->view())
                       .metadata(metadata)
                       .build();
    cudf::io::write_parquet(options, stream);
    stream.synchronize();
  }

  auto idGenerator = std::make_shared<PlanNodeIdGenerator>();
  PlanNodeId probeScanId;
  PlanNodeId buildScanId;

  auto plan = scanThrough(probeType_, kSecConnectorId, idGenerator)
                  .capturePlanNodeId(probeScanId)
                  .hashJoin(
                      {"t_key"},
                      {"u_key"},
                      scanThrough(buildType_, kNanoConnectorId, idGenerator)
                          .capturePlanNodeId(buildScanId)
                          .planNode(),
                      "CASE WHEN t_key >= 0 THEN t_ts > u_ts ELSE true END",
                      {"t_key"},
                      JoinType::kInner)
                  .planNode();

  auto result =
      AssertQueryBuilder(plan)
          .split(probeScanId, makeSplit(probeFile->getPath(), kSecConnectorId))
          .split(buildScanId, makeSplit(buildFile->getPath(), kNanoConnectorId))
          .copyResults(pool_.get());

  // Correct answer: 20 rows (all probe years 2223-2366 > build days 1-10).
  // With overflow (current behavior): 6 rows (only non-overflowing rows pass).
  // With raw tick comparison: 0 rows (probe ticks << build nanos ticks).
  EXPECT_EQ(result->size(), 20);

  ConnectorRegistry::global().erase(kSecConnectorId);
}

// Both sides use the same nano connector. Verifies no regression when
// timestamp resolutions already match.
TEST_F(TimestampJoinTest, noMismatchNoOp) {
  auto idGenerator = std::make_shared<PlanNodeIdGenerator>();

  PlanNodeId probeScanId;
  PlanNodeId buildScanId;

  auto plan = scanThrough(probeType_, kNanoConnectorId, idGenerator)
                  .capturePlanNodeId(probeScanId)
                  .hashJoin(
                      {"t_key"},
                      {"u_key"},
                      scanThrough(buildType_, kNanoConnectorId, idGenerator)
                          .capturePlanNodeId(buildScanId)
                          .planNode(),
                      "t_ts < u_ts",
                      {"t_key", "t_ts", "u_ts"},
                      JoinType::kInner)
                  .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .split(probeScanId, makeSplit(probeFile_->getPath(), kNanoConnectorId))
      .split(buildScanId, makeSplit(buildFile_->getPath(), kNanoConnectorId))
      .assertResults(
          "SELECT t.t_key, t.t_ts, u.u_ts "
          "FROM t, u "
          "WHERE t.t_key = u.u_key AND t.t_ts < u.u_ts");
}

} // namespace
