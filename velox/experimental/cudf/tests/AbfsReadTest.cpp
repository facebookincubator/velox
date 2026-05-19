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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/tests/CudfAbfsTest.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/storage_adapters/abfs/tests/AzuriteServer.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox::exec::test;

namespace {

constexpr const char* kCudfHiveConnectorId{"test-cudf-hive"};
constexpr int64_t kIntParquetRows{10};

class AbfsReadTest : public CudfAbfsTest, public ::test::VectorTestBase {
 protected:
  std::string sourceFilePath() const {
    return test::getDataFilePath(
        "velox/experimental/cudf/tests",
        "../../../dwio/parquet/tests/examples/int.parquet");
  }

  core::PlanNodePtr tableScanNode() {
    auto rowType = ROW({"int", "bigint"}, {INTEGER(), BIGINT()});
    auto tableHandle =
        std::make_shared<facebook::velox::connector::hive::HiveTableHandle>(
            kCudfHiveConnectorId,
            "int_table",
            common::SubfieldFilters{},
            nullptr);
    return PlanBuilder(pool())
        .startTableScan()
        .tableHandle(tableHandle)
        .outputType(rowType)
        .endTableScan()
        .planNode();
  }

  std::shared_ptr<facebook::velox::connector::hive::HiveConnectorSplit>
  makeSplit(const std::string& abfsPath) const {
    return facebook::velox::connector::hive::HiveConnectorSplitBuilder(abfsPath)
        .connectorId(kCudfHiveConnectorId)
        .fileFormat(dwio::common::FileFormat::PARQUET)
        .build();
  }

  RowVectorPtr expectedVector() {
    return makeRowVector(
        {makeFlatVector<int32_t>(
             kIntParquetRows, [](auto row) { return row + 100; }),
         makeFlatVector<int64_t>(
             kIntParquetRows, [](auto row) { return row + 1000; })});
  }
};

} // namespace

// Reads a Parquet file from an ABFS path
TEST_F(AbfsReadTest, readIntParquet) {
  const auto abfsFilePath = uploadFile(sourceFilePath());
  auto plan = tableScanNode();
  auto split = makeSplit(abfsFilePath);

  auto actual = AssertQueryBuilder(plan).split(split).copyResults(pool());

  assertEqualResults({expectedVector()}, {actual});
}

// An invalid ABFS split throws instead of falling back to the KvikIO data
// source
TEST_F(AbfsReadTest, nonExistentBlob) {
  const auto missingPath = azuriteServer_->URI() + "does_not_exist.parquet";
  auto plan = tableScanNode();
  auto split = makeSplit(missingPath);

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).split(split).copyResults(pool()),
      "Failed to generate file handle cache for ABFS path");
}

// Disabling buffered input for an ABFS split throws instead of falling back to
// KvikIO data source
TEST_F(AbfsReadTest, bufferedInputDisabledForAbfs) {
  const auto abfsFilePath = uploadFile(sourceFilePath());
  auto plan = tableScanNode();
  auto split = makeSplit(abfsFilePath);

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan)
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              cudf_velox::connector::hive::CudfHiveConfig::
                  kUseBufferedInputSession,
              "false")
          .split(split)
          .copyResults(pool()),
      "ABFS paths require buffered input data source");
}

// Multiple independent queries reading the same ABFS blob in parallel must
// all succeed and produce identical results
TEST_F(AbfsReadTest, concurrentSplits) {
  const auto abfsFilePath = uploadFile(sourceFilePath());
  auto plan = tableScanNode();
  const auto expected = expectedVector();

  constexpr int kNumThreads{8};
  std::atomic<bool> startThreads{false};
  std::vector<std::thread> threads;
  std::vector<RowVectorPtr> results(kNumThreads);
  threads.reserve(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&, i] {
      while (!startThreads.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      auto split = makeSplit(abfsFilePath);
      results[i] = AssertQueryBuilder(plan).split(split).copyResults(pool());
    });
  }
  startThreads.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  for (int i = 0; i < kNumThreads; ++i) {
    ASSERT_NE(results[i], nullptr) << "thread " << i << " produced no result";
    assertEqualResults({expected}, {results[i]});
  }
}
