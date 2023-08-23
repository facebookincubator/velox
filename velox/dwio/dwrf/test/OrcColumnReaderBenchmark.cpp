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

#include <folly/Benchmark.h>
#include <folly/Varint.h>
#include <folly/init/Init.h>

#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwrf;
using exec::test::HiveConnectorTestBase;

static std::shared_ptr<folly::Executor> intTaskEnv() {
  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(exec::test::kHiveConnectorId, nullptr);
  connector::registerConnector(hiveConnector);
  filesystems::registerLocalFileSystem();
  dwrf::registerOrcReaderFactory();
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  return std::make_shared<folly::CPUThreadPoolExecutor>(
      std::thread::hardware_concurrency());
}

static std::shared_ptr<const RowType> getOutputType(
    const std::shared_ptr<memory::MemoryPool>& pool,
    const std::string& filePath,
    const std::vector<std::string>& projections) {
  ReaderOptions readerOpts(pool.get());
  readerOpts.setFileFormat(FileFormat::ORC);
  auto reader = DwrfReader::create(
      std::make_unique<BufferedInput>(
          std::make_shared<LocalReadFile>(filePath),
          readerOpts.getMemoryPool()),
      readerOpts);
  auto rowType = reader->rowType();
  if (projections.empty()) {
    return rowType;
  }
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto& projection : projections) {
    names.emplace_back(projection);
    types.emplace_back(rowType->findChild(projection));
  }
  return ROW(std::move(names), std::move(types));
}

static std::shared_ptr<exec::Task> initQueryTask(
    const std::string& filePath,
    const std::vector<std::string>& projections,
    const std::string& filter,
    bool filterPushdown) {
  static auto executor = intTaskEnv();
  static auto pool = memory::addDefaultLeafMemoryPool();
  auto outputType = getOutputType(pool, filePath, projections);
  exec::test::PlanBuilder builder(pool.get());
  parse::ParseOptions options;
  options.parseDecimalAsDouble = false;
  builder.setParseOptions(options);
  core::PlanNodeId scanNode;
  if (!filter.empty()) { // has Filter
    if (filterPushdown) {
      builder.tableScan(outputType, {filter}).capturePlanNodeId(scanNode);
    } else {
      builder.tableScan(outputType).capturePlanNodeId(scanNode).filter(filter);
    }
  }
  if (!projections.empty()) { // has Project
    builder.project(projections);
  }
  auto planFragment = builder.planFragment();
  auto queryCtx = std::make_shared<core::QueryCtx>(executor.get());
  auto queryTask = exec::Task::create("QueryTask", planFragment, 0, queryCtx);
  auto connectorSplit = std::make_shared<connector::hive::HiveConnectorSplit>(
      exec::test::kHiveConnectorId,
      "file:" + filePath,
      dwio::common::FileFormat::ORC);
  queryTask->addSplit(scanNode, exec::Split{connectorSplit});
  queryTask->noMoreSplits(scanNode);
  return queryTask;
}

BENCHMARK(decimalWithFilter) {
  folly::BenchmarkSuspender suspender;
  std::string filePath = test::getDataFilePath(
      "velox/dwio/dwrf/test", "examples/short_decimal.orc");
  auto task = initQueryTask(
      filePath,
      {"b"},
      "b BETWEEN CAST(427531 as DECIMAL(10,2)) AND CAST(428493 as DECIMAL(10,2))",
      false);
  suspender.dismiss();
  RowVectorPtr result = task->next();
  folly::doNotOptimizeAway(result);
}

BENCHMARK_RELATIVE(decimalWithFilterPushdown) {
  folly::BenchmarkSuspender suspender;
  std::string filePath = test::getDataFilePath(
      "velox/dwio/dwrf/test", "examples/short_decimal.orc");
  auto task = initQueryTask(
      filePath,
      {"b"},
      "b BETWEEN CAST(427531 as DECIMAL(10,2)) AND CAST(428493 as DECIMAL(10,2))",
      true);
  suspender.dismiss();
  RowVectorPtr result = task->next();
  folly::doNotOptimizeAway(result);
}

int32_t main(int32_t argc, char* argv[]) {
  folly::init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}

/*
CPU model name: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
Core(s): 64
Memory(GB): 256
============================================================================
[...]wrf/test/OrcColumnReaderBenchmark.cpp     relative  time/iter   iters/s
============================================================================
decimalWithFilter                                         199.95us     5.00K
decimalWithFilterPushdown                       130.55%   153.16us     6.53K
============================================================================
*/
