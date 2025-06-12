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
#pragma once

#include <gflags/gflags.h>

#include "velox/benchmarks/QueryBenchmarkBase.h"

DECLARE_int32(run_query_verbose);
DECLARE_int32(io_meter_column_pct);

class TpchBenchmark : public facebook::velox::QueryBenchmarkBase {
 public:
  void initialize() override {
    QueryBenchmarkBase::initialize();
    initQueryBuilder();
  }

  void shutdown() override {
    QueryBenchmarkBase::shutdown();
    queryBuilder.reset();
  }

  void runMain(std::ostream& out, facebook::velox::RunStats& runStats)
      override {
    if (FLAGS_run_query_verbose == -1 && FLAGS_io_meter_column_pct == 0) {
      folly::runBenchmarks();
    } else {
      const auto queryPlan = FLAGS_io_meter_column_pct > 0
          ? queryBuilder->getIoMeterPlan(FLAGS_io_meter_column_pct)
          : queryBuilder->getQueryPlan(FLAGS_run_query_verbose);
      auto [cursor, actualResults] = run(queryPlan);
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
      out << printPlanWithStats(
                 *queryPlan.plan, stats, FLAGS_include_custom_stats)
          << std::endl;
    }
  }

  void runQuery(int32_t queryId) {
    const auto planContext = queryBuilder->getQueryPlan(queryId);
    run(planContext);
  }

  void initQueryBuilder();

 private:
  std::shared_ptr<facebook::velox::exec::test::TpchQueryBuilder> queryBuilder;
};

extern std::unique_ptr<TpchBenchmark> benchmark;

void tpchBenchmarkMain();
