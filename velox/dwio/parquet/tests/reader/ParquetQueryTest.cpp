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

#include <folly/init/Init.h>
#include <vector>

#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/duckdb_reader/ParquetReader.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::parquet;

static const int kNumDrivers = 4;

class ParquetQueryTest : public testing::Test {
 protected:
  // Setup a DuckDB instance for the entire suite and load TPC-H data with scale
  // factor 0.01.
  static void SetUpTestSuite() {
    if (duckDb_ == nullptr) {
      duckDb_ = std::make_shared<DuckDbQueryRunner>();
    }
    functions::prestosql::registerAllScalarFunctions();
    parse::registerTypeResolver();
    filesystems::registerLocalFileSystem();
    registerParquetReaderFactory(parquet::ParquetReaderType::NATIVE);

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(kHiveConnectorId, nullptr);
    connector::registerConnector(hiveConnector);
    tempDirectory_ = exec::test::TempDirectoryPath::create();
  }

  std::string getExampleFilePath(const std::string& fileName) {
    return facebook::velox::test::getDataFilePath(
        "velox/dwio/parquet/tests/reader", "../examples/" + fileName);
  }

  std::shared_ptr<Task> assertQuery(
      const core::PlanNodePtr& planNode,
      std::unordered_map<core::PlanNodeId, std::vector<std::string>> dataFiles,
      const std::string& duckQuery,
      std::optional<std::vector<uint32_t>> sortingKeys) const {
    bool noMoreSplits = false;
    constexpr int kNumSplits = 10;
    auto addSplits = [&](exec::Task* task) {
      if (!noMoreSplits) {
        for (const auto& entry : dataFiles) {
          for (const auto& path : entry.second) {
            auto const splits = HiveConnectorTestBase::makeHiveConnectorSplits(
                path, kNumSplits, dwio::common::FileFormat::PARQUET);
            for (const auto& split : splits) {
              task->addSplit(entry.first, exec::Split(split));
            }
          }
          task->noMoreSplits(entry.first);
        }
      }
      noMoreSplits = true;
    };
    CursorParameters params;
    params.maxDrivers = kNumDrivers;
    params.planNode = planNode;
    return exec::test::assertQuery(
        params, addSplits, duckQuery, *duckDb_, sortingKeys);
  }

  static std::shared_ptr<DuckDbQueryRunner> duckDb_;
  static std::shared_ptr<exec::test::TempDirectoryPath> tempDirectory_;
};

std::shared_ptr<DuckDbQueryRunner> ParquetQueryTest::duckDb_ = nullptr;
std::shared_ptr<exec::test::TempDirectoryPath>
    ParquetQueryTest::tempDirectory_ = nullptr;

TEST_F(ParquetQueryTest, simpleSelectFilter) {
  duckDb_->execute(fmt::format(
      "create table store as select * from read_parquet('{}')",
      getExampleFilePath("store.snappy.parquet")));

  auto result = duckDb_->execute("select * from store");
  result->Print();
  auto fieldType = ROW({{"s_store_sk", BIGINT()}, {"s_state", VARCHAR()}});
  const std::string filePath(getExampleFilePath("store.snappy.parquet"));
  core::PlanNodeId planNodeId;
  auto plan = PlanBuilder()
                  .tableScan("store", fieldType, {}, {})
                  .capturePlanNodeId(planNodeId)
                  .planNode();
  std::unordered_map<core::PlanNodeId, std::vector<std::string>> dataFiles{
      {planNodeId, {filePath}}};
  assertQuery(plan, dataFiles, "select s_store_sk from store", {});
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}