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

#include "velox/dwio/parquet/tests/ParquetTpchTestBase.h"

namespace facebook::velox::exec::test {

void ParquetTpchTestBase::SetUpTestSuite() {
  if (duckDb_ == nullptr) {
    duckDb_ = std::make_shared<DuckDbQueryRunner>();
  }
  tempDirectory_ = TempDirectoryPath::create();

  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();

  parse::registerTypeResolver();
  filesystems::registerLocalFileSystem();
  velox::parquet::registerParquetReaderFactory();

  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(kHiveConnectorId, nullptr);
  connector::registerConnector(hiveConnector);

  std::shared_ptr<memory::MemoryPool> rootPool{
      memory::defaultMemoryManager().addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool{rootPool->addLeafChild("leaf")};
  auto tpchConnector =
      connector::getConnectorFactory(
          connector::tpch::TpchConnectorFactory::kTpchConnectorName)
          ->newConnector(kTpchConnectorId, nullptr);
  connector::registerConnector(tpchConnector);
  saveTpchTablesAsParquet(pool.get());
  tpchBuilder_.initialize(tempDirectory_->path);
}

void ParquetTpchTestBase::TearDownTestSuite() {
  connector::unregisterConnector(kHiveConnectorId);
  connector::unregisterConnector(kTpchConnectorId);
  velox::parquet::unregisterParquetReaderFactory();
}

void ParquetTpchTestBase::saveTpchTablesAsParquet(memory::MemoryPool* pool) {
  for (const auto& table : tpch::tables) {
    auto tableName = toTableName(table);
    auto tableDirectory = fmt::format("{}/{}", tempDirectory_->path, tableName);
    auto tableSchema = tpch::getTableSchema(table);
    auto names = tableSchema->names();
    auto plan =
        PlanBuilder().tableScan(table, std::move(names), 0.01).planNode();
    auto split =
        exec::Split(std::make_shared<connector::tpch::TpchConnectorSplit>(
            kTpchConnectorId, 1, 0));

    auto rows = AssertQueryBuilder(plan).splits({split}).copyResults(pool);
    duckDb_->createTable(tableName.data(), {rows});

    plan = PlanBuilder()
               .values({rows})
               .tableWrite(tableDirectory, dwio::common::FileFormat::PARQUET)
               .planNode();

    AssertQueryBuilder(plan).copyResults(pool);
  }
}

void ParquetTpchTestBase::assertQuery(
    int queryId,
    const std::optional<std::vector<uint32_t>>& sortingKeys) {
  auto tpchPlan = tpchBuilder_.getQueryPlan(queryId);
  auto duckDbSql = tpch::getQuery(queryId);
  assertQuery(tpchPlan, duckDbSql, sortingKeys);
}

std::shared_ptr<Task> ParquetTpchTestBase::assertQuery(
    const TpchPlan& tpchPlan,
    const std::string& duckQuery,
    const std::optional<std::vector<uint32_t>>& sortingKeys) {
  bool noMoreSplits = false;
  constexpr int kNumSplits = 10;
  constexpr int kNumDrivers = 4;
  auto addSplits = [&](Task* task) {
    if (!noMoreSplits) {
      for (const auto& entry : tpchPlan.dataFiles) {
        for (const auto& path : entry.second) {
          auto const splits = HiveConnectorTestBase::makeHiveConnectorSplits(
              path, kNumSplits, tpchPlan.dataFileFormat);
          for (const auto& split : splits) {
            task->addSplit(entry.first, Split(split));
          }
        }
        task->noMoreSplits(entry.first);
      }
    }
    noMoreSplits = true;
  };
  CursorParameters params;
  params.maxDrivers = kNumDrivers;
  params.planNode = tpchPlan.plan;
  return exec::test::assertQuery(
      params, addSplits, duckQuery, *duckDb_, sortingKeys);
}

std::shared_ptr<DuckDbQueryRunner> ParquetTpchTestBase::duckDb_ = nullptr;
std::shared_ptr<TempDirectoryPath> ParquetTpchTestBase::tempDirectory_ =
    nullptr;
TpchQueryBuilder ParquetTpchTestBase::tpchBuilder_ =
    TpchQueryBuilder(dwio::common::FileFormat::PARQUET);

} // namespace facebook::velox::exec::test
