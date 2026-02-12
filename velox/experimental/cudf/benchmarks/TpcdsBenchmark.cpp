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

#include "velox/experimental/cudf/benchmarks/TpcdsBenchmark.h"

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/exec/PartitionFunction.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

#include <iostream>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

DEFINE_string(
    data_path,
    "",
    "Root path of TPC-DS data. Data layout must follow Hive-style partitioning. "
    "Example layout for '--data_path=/data/tpcds_sf100'\n"
    "       /data/tpcds_sf100/store_sales\n"
    "       /data/tpcds_sf100/customer\n"
    "       /data/tpcds_sf100/date_dim\n"
    "       /data/tpcds_sf100/store\n"
    "       ...\n"
    "Each directory contains data files (e.g. parquet) for the table.");

namespace {
static bool notEmpty(const char* /*flagName*/, const std::string& value) {
  return !value.empty();
}
} // namespace

DEFINE_validator(data_path, &notEmpty);

DEFINE_string(
    plan_path,
    "",
    "Directory containing Velox plan JSON files (Q1.json, Q2.json, ...). "
    "These are typically dumped from a Presto worker's plan-dump-dir.");

DEFINE_validator(plan_path, &notEmpty);

DEFINE_int32(
    run_query_verbose,
    -1,
    "Run a given query and print execution statistics");

void TpcdsBenchmark::initQueryBuilder() {
  queryBuilder_ =
      std::make_unique<TpcdsQueryBuilder>(toFileFormat(FLAGS_data_format));
  queryBuilder_->initialize(FLAGS_data_path);
}

void TpcdsBenchmark::initialize() {
  QueryBenchmarkBase::initialize();

  // TPC-DS plans from Presto use "presto.default." prefix for functions.
  functions::prestosql::registerAllScalarFunctions("presto.default.");
  aggregate::prestosql::registerAllAggregateFunctions("presto.default.");

  // Register serialization/deserialization for plan nodes.
  Type::registerSerDe();
  common::Filter::registerSerDe();
  connector::hive::HiveConnector::registerSerDe();
  core::PlanNode::registerSerDe();
  core::ITypedExpr::registerSerDe();
  exec::registerPartitionFunctionSerDe();

  // Presto-dumped plans use connector ID "hive", while the base class
  // registers under kHiveConnectorId ("test-hive"). Register a properly
  // configured connector under "hive" so both the plan and splits match.
  const std::string prestoConnectorId = "hive";
  if (!connector::hasConnector(prestoConnectorId)) {
    auto properties = makeConnectorProperties();
    connector::hive::HiveConnectorFactory factory;
    auto hiveConnector =
        factory.newConnector(prestoConnectorId, properties, ioExecutor_.get());
    connector::registerConnector(hiveConnector);
  }

  planDir_ = FLAGS_plan_path;
  pool_ = memory::memoryManager()->addLeafPool("TpcdsBenchmark");

  initQueryBuilder();
}

void TpcdsBenchmark::shutdown() {
  if (queryBuilder_) {
    queryBuilder_->shutdown();
    queryBuilder_.reset();
  }
  // Unregister the "hive" connector we registered in initialize().
  const std::string prestoConnectorId = "hive";
  if (connector::hasConnector(prestoConnectorId)) {
    connector::unregisterConnector(prestoConnectorId);
  }
  pool_.reset();
  QueryBenchmarkBase::shutdown();
}

std::vector<std::shared_ptr<connector::ConnectorSplit>>
TpcdsBenchmark::listSplits(
    const std::string& path,
    int32_t numSplitsPerFile,
    const exec::test::TpchPlan& plan) {
  // The base class creates splits with the default (empty) connector ID,
  // which doesn't match the plan's connector ID (e.g. "hive" from Presto).
  // Re-create splits with the correct connector ID from the query builder.
  auto baseSplits =
      QueryBenchmarkBase::listSplits(path, numSplitsPerFile, plan);

  const auto& cid =
      queryBuilder_ ? queryBuilder_->connectorId() : std::string();
  if (cid.empty()) {
    return baseSplits;
  }

  // Rebuild each split with the plan's connector ID.
  std::vector<std::shared_ptr<connector::ConnectorSplit>> result;
  result.reserve(baseSplits.size());
  for (const auto& baseSplit : baseSplits) {
    auto hiveSplit =
        std::dynamic_pointer_cast<connector::hive::HiveConnectorSplit>(
            baseSplit);
    if (hiveSplit) {
      result.push_back(
          connector::hive::HiveConnectorSplitBuilder(hiveSplit->filePath)
              .connectorId(cid)
              .fileFormat(hiveSplit->fileFormat)
              .start(hiveSplit->start)
              .length(hiveSplit->length)
              .build());
    } else {
      result.push_back(baseSplit);
    }
  }
  return result;
}

void TpcdsBenchmark::runQuery(int32_t queryId) {
  auto veloxPlan = queryBuilder_->getQueryPlan(queryId, planDir_, pool_.get());
  // Convert VeloxPlan to TpchPlan (structurally identical) for
  // QueryBenchmarkBase::run().
  TpchPlan tpchPlan;
  tpchPlan.plan = std::move(veloxPlan.plan);
  tpchPlan.dataFiles = std::move(veloxPlan.dataFiles);
  tpchPlan.dataFileFormat = veloxPlan.dataFileFormat;
  run(tpchPlan, queryConfigs_);
}

void TpcdsBenchmark::runMain(
    std::ostream& out,
    facebook::velox::RunStats& runStats) {
  if (FLAGS_run_query_verbose == -1) {
    folly::runBenchmarks();
  } else {
    auto veloxPlan = queryBuilder_->getQueryPlan(
        FLAGS_run_query_verbose, planDir_, pool_.get());
    TpchPlan tpchPlan;
    tpchPlan.plan = std::move(veloxPlan.plan);
    tpchPlan.dataFiles = std::move(veloxPlan.dataFiles);
    tpchPlan.dataFileFormat = veloxPlan.dataFileFormat;

    auto [cursor, actualResults] = run(tpchPlan, queryConfigs_);
    if (!cursor) {
      LOG(ERROR) << "Query terminated with error. Exiting";
      exit(1);
    }
    auto task = cursor->task();
    ensureTaskCompletion(task.get());
    if (FLAGS_include_results) {
      printResults(actualResults, out);
      out << std::endl;
    }
    const auto stats = task->taskStats();
    int64_t rawInputBytes = 0;
    for (auto& pipeline : stats.pipelineStats) {
      auto& first = pipeline.operatorStats[0];
      if (first.operatorType == "TableScan") {
        rawInputBytes += first.rawInputBytes;
      }
    }
    runStats.rawInputBytes = rawInputBytes;
    out << fmt::format(
               "Execution time: {}",
               facebook::velox::succinctMillis(
                   stats.executionEndTimeMs - stats.executionStartTimeMs))
        << std::endl;
    out << fmt::format(
               "Splits total: {}, finished: {}",
               stats.numTotalSplits,
               stats.numFinishedSplits)
        << std::endl;
    out << printPlanWithStats(*tpchPlan.plan, stats, FLAGS_include_custom_stats)
        << std::endl;
  }
}

std::unique_ptr<TpcdsBenchmark> tpcdsBenchmark;

// TPC-DS has 99 queries. Define BENCHMARK macros for each.
// Only a subset may have plan JSONs available; missing plans will cause
// a runtime error for that specific query.
BENCHMARK(tpcds_q1) {
  tpcdsBenchmark->runQuery(1);
}
BENCHMARK(tpcds_q2) {
  tpcdsBenchmark->runQuery(2);
}
BENCHMARK(tpcds_q3) {
  tpcdsBenchmark->runQuery(3);
}
BENCHMARK(tpcds_q4) {
  tpcdsBenchmark->runQuery(4);
}
BENCHMARK(tpcds_q5) {
  tpcdsBenchmark->runQuery(5);
}
BENCHMARK(tpcds_q6) {
  tpcdsBenchmark->runQuery(6);
}
BENCHMARK(tpcds_q7) {
  tpcdsBenchmark->runQuery(7);
}
BENCHMARK(tpcds_q8) {
  tpcdsBenchmark->runQuery(8);
}
BENCHMARK(tpcds_q9) {
  tpcdsBenchmark->runQuery(9);
}
BENCHMARK(tpcds_q10) {
  tpcdsBenchmark->runQuery(10);
}
BENCHMARK(tpcds_q11) {
  tpcdsBenchmark->runQuery(11);
}
BENCHMARK(tpcds_q12) {
  tpcdsBenchmark->runQuery(12);
}
BENCHMARK(tpcds_q13) {
  tpcdsBenchmark->runQuery(13);
}
BENCHMARK(tpcds_q14) {
  tpcdsBenchmark->runQuery(14);
}
BENCHMARK(tpcds_q15) {
  tpcdsBenchmark->runQuery(15);
}
BENCHMARK(tpcds_q16) {
  tpcdsBenchmark->runQuery(16);
}
BENCHMARK(tpcds_q17) {
  tpcdsBenchmark->runQuery(17);
}
BENCHMARK(tpcds_q18) {
  tpcdsBenchmark->runQuery(18);
}
BENCHMARK(tpcds_q19) {
  tpcdsBenchmark->runQuery(19);
}
BENCHMARK(tpcds_q20) {
  tpcdsBenchmark->runQuery(20);
}
BENCHMARK(tpcds_q21) {
  tpcdsBenchmark->runQuery(21);
}
BENCHMARK(tpcds_q22) {
  tpcdsBenchmark->runQuery(22);
}
BENCHMARK(tpcds_q23) {
  tpcdsBenchmark->runQuery(23);
}
BENCHMARK(tpcds_q24) {
  tpcdsBenchmark->runQuery(24);
}
BENCHMARK(tpcds_q25) {
  tpcdsBenchmark->runQuery(25);
}
BENCHMARK(tpcds_q26) {
  tpcdsBenchmark->runQuery(26);
}
BENCHMARK(tpcds_q27) {
  tpcdsBenchmark->runQuery(27);
}
BENCHMARK(tpcds_q28) {
  tpcdsBenchmark->runQuery(28);
}
BENCHMARK(tpcds_q29) {
  tpcdsBenchmark->runQuery(29);
}
BENCHMARK(tpcds_q30) {
  tpcdsBenchmark->runQuery(30);
}
BENCHMARK(tpcds_q31) {
  tpcdsBenchmark->runQuery(31);
}
BENCHMARK(tpcds_q32) {
  tpcdsBenchmark->runQuery(32);
}
BENCHMARK(tpcds_q33) {
  tpcdsBenchmark->runQuery(33);
}
BENCHMARK(tpcds_q34) {
  tpcdsBenchmark->runQuery(34);
}
BENCHMARK(tpcds_q35) {
  tpcdsBenchmark->runQuery(35);
}
BENCHMARK(tpcds_q36) {
  tpcdsBenchmark->runQuery(36);
}
BENCHMARK(tpcds_q37) {
  tpcdsBenchmark->runQuery(37);
}
BENCHMARK(tpcds_q38) {
  tpcdsBenchmark->runQuery(38);
}
BENCHMARK(tpcds_q39) {
  tpcdsBenchmark->runQuery(39);
}
BENCHMARK(tpcds_q40) {
  tpcdsBenchmark->runQuery(40);
}
BENCHMARK(tpcds_q41) {
  tpcdsBenchmark->runQuery(41);
}
BENCHMARK(tpcds_q42) {
  tpcdsBenchmark->runQuery(42);
}
BENCHMARK(tpcds_q43) {
  tpcdsBenchmark->runQuery(43);
}
BENCHMARK(tpcds_q44) {
  tpcdsBenchmark->runQuery(44);
}
BENCHMARK(tpcds_q45) {
  tpcdsBenchmark->runQuery(45);
}
BENCHMARK(tpcds_q46) {
  tpcdsBenchmark->runQuery(46);
}
BENCHMARK(tpcds_q47) {
  tpcdsBenchmark->runQuery(47);
}
BENCHMARK(tpcds_q48) {
  tpcdsBenchmark->runQuery(48);
}
BENCHMARK(tpcds_q49) {
  tpcdsBenchmark->runQuery(49);
}
BENCHMARK(tpcds_q50) {
  tpcdsBenchmark->runQuery(50);
}
BENCHMARK(tpcds_q51) {
  tpcdsBenchmark->runQuery(51);
}
BENCHMARK(tpcds_q52) {
  tpcdsBenchmark->runQuery(52);
}
BENCHMARK(tpcds_q53) {
  tpcdsBenchmark->runQuery(53);
}
BENCHMARK(tpcds_q54) {
  tpcdsBenchmark->runQuery(54);
}
BENCHMARK(tpcds_q55) {
  tpcdsBenchmark->runQuery(55);
}
BENCHMARK(tpcds_q56) {
  tpcdsBenchmark->runQuery(56);
}
BENCHMARK(tpcds_q57) {
  tpcdsBenchmark->runQuery(57);
}
BENCHMARK(tpcds_q58) {
  tpcdsBenchmark->runQuery(58);
}
BENCHMARK(tpcds_q59) {
  tpcdsBenchmark->runQuery(59);
}
BENCHMARK(tpcds_q60) {
  tpcdsBenchmark->runQuery(60);
}
BENCHMARK(tpcds_q61) {
  tpcdsBenchmark->runQuery(61);
}
BENCHMARK(tpcds_q62) {
  tpcdsBenchmark->runQuery(62);
}
BENCHMARK(tpcds_q63) {
  tpcdsBenchmark->runQuery(63);
}
BENCHMARK(tpcds_q64) {
  tpcdsBenchmark->runQuery(64);
}
BENCHMARK(tpcds_q65) {
  tpcdsBenchmark->runQuery(65);
}
BENCHMARK(tpcds_q66) {
  tpcdsBenchmark->runQuery(66);
}
BENCHMARK(tpcds_q67) {
  tpcdsBenchmark->runQuery(67);
}
BENCHMARK(tpcds_q68) {
  tpcdsBenchmark->runQuery(68);
}
BENCHMARK(tpcds_q69) {
  tpcdsBenchmark->runQuery(69);
}
BENCHMARK(tpcds_q70) {
  tpcdsBenchmark->runQuery(70);
}
BENCHMARK(tpcds_q71) {
  tpcdsBenchmark->runQuery(71);
}
BENCHMARK(tpcds_q72) {
  tpcdsBenchmark->runQuery(72);
}
BENCHMARK(tpcds_q73) {
  tpcdsBenchmark->runQuery(73);
}
BENCHMARK(tpcds_q74) {
  tpcdsBenchmark->runQuery(74);
}
BENCHMARK(tpcds_q75) {
  tpcdsBenchmark->runQuery(75);
}
BENCHMARK(tpcds_q76) {
  tpcdsBenchmark->runQuery(76);
}
BENCHMARK(tpcds_q77) {
  tpcdsBenchmark->runQuery(77);
}
BENCHMARK(tpcds_q78) {
  tpcdsBenchmark->runQuery(78);
}
BENCHMARK(tpcds_q79) {
  tpcdsBenchmark->runQuery(79);
}
BENCHMARK(tpcds_q80) {
  tpcdsBenchmark->runQuery(80);
}
BENCHMARK(tpcds_q81) {
  tpcdsBenchmark->runQuery(81);
}
BENCHMARK(tpcds_q82) {
  tpcdsBenchmark->runQuery(82);
}
BENCHMARK(tpcds_q83) {
  tpcdsBenchmark->runQuery(83);
}
BENCHMARK(tpcds_q84) {
  tpcdsBenchmark->runQuery(84);
}
BENCHMARK(tpcds_q85) {
  tpcdsBenchmark->runQuery(85);
}
BENCHMARK(tpcds_q86) {
  tpcdsBenchmark->runQuery(86);
}
BENCHMARK(tpcds_q87) {
  tpcdsBenchmark->runQuery(87);
}
BENCHMARK(tpcds_q88) {
  tpcdsBenchmark->runQuery(88);
}
BENCHMARK(tpcds_q89) {
  tpcdsBenchmark->runQuery(89);
}
BENCHMARK(tpcds_q90) {
  tpcdsBenchmark->runQuery(90);
}
BENCHMARK(tpcds_q91) {
  tpcdsBenchmark->runQuery(91);
}
BENCHMARK(tpcds_q92) {
  tpcdsBenchmark->runQuery(92);
}
BENCHMARK(tpcds_q93) {
  tpcdsBenchmark->runQuery(93);
}
BENCHMARK(tpcds_q94) {
  tpcdsBenchmark->runQuery(94);
}
BENCHMARK(tpcds_q95) {
  tpcdsBenchmark->runQuery(95);
}
BENCHMARK(tpcds_q96) {
  tpcdsBenchmark->runQuery(96);
}
BENCHMARK(tpcds_q97) {
  tpcdsBenchmark->runQuery(97);
}
BENCHMARK(tpcds_q98) {
  tpcdsBenchmark->runQuery(98);
}
BENCHMARK(tpcds_q99) {
  tpcdsBenchmark->runQuery(99);
}

void tpcdsBenchmarkMain() {
  VELOX_CHECK_NOT_NULL(tpcdsBenchmark);
  tpcdsBenchmark->initialize();
  if (FLAGS_test_flags_file.empty()) {
    RunStats ignore;
    tpcdsBenchmark->runMain(std::cout, ignore);
  } else {
    tpcdsBenchmark->runAllCombinations();
  }
  tpcdsBenchmark->shutdown();
}
