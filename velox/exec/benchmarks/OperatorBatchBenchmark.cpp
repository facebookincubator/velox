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
#include <folly/init/Init.h>
#include <map>
#include <utility>

#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using exec::test::PlanBuilder;

namespace {
constexpr int64_t kNumRows = 1 << 20;
constexpr int32_t kInputBatchRows = 1024;
enum class Workload {
  kAggregation,
  kFilterProject,
  kHashJoin,
};

struct BenchmarkCase {
  core::PlanNodePtr plan;
  Workload workload;
  int32_t numGroups;
  int32_t maxOutputRows;
  int64_t expectedRows;
  bool trackCpuUsage;
};

// Measure complete filter, hash join, and grouped aggregation queries with
// controlled output batch sizes. Each case processes 1,048,576 in-memory rows
// using one driver. The 65,536-group aggregation cases share their input and
// plan, changing the maximum output size from 1,024 to 64 rows to increase the
// number of output batches from 64 to 1,024 while returning identical results.
// Each Folly iteration includes cursor creation, hash table construction,
// output consumption, and cursor destruction. Input and plan construction and
// result verification run before measurement. Folly reports time per query
// using its default minimum-trial estimator.
class OperatorBatchBenchmark : public facebook::velox::test::VectorTestBase {
 public:
  OperatorBatchBenchmark() {
    for (const auto numGroups : {4096, 65536, 1048576}) {
      std::vector<RowVectorPtr> data;
      for (int64_t offset = 0; offset < kNumRows; offset += kInputBatchRows) {
        data.push_back(makeRowVector(
            {"k", "v"},
            {makeFlatVector<int64_t>(
                 kInputBatchRows,
                 [=](auto i) { return (offset + i) % numGroups; }),
             makeFlatVector<int64_t>(
                 kInputBatchRows, [=](auto i) { return offset + i; })}));
      }
      auto aggregation = PlanBuilder()
                             .values(data)
                             .singleAggregation({"k"}, {"sum(v)"})
                             .planNode();
      const auto name = "aggregation_" + std::to_string(numGroups) +
          "_groups_1024_output_rows";
      const BenchmarkCase testCase{
          aggregation,
          Workload::kAggregation,
          numGroups,
          1024,
          numGroups,
          true};
      cases_[name] = testCase;
      if (numGroups == 65536) {
        auto smallBatches = testCase;
        smallBatches.maxOutputRows = 64;
        const std::string smallName = "aggregation_65536_groups_64_output_rows";
        cases_[smallName] = smallBatches;
        auto cpuTrackingDisabled = testCase;
        cpuTrackingDisabled.trackCpuUsage = false;
        cases_[name + "_cpu_tracking_disabled"] = cpuTrackingDisabled;
        smallBatches.trackCpuUsage = false;
        cases_[smallName + "_cpu_tracking_disabled"] = smallBatches;
      }
      if (numGroups == 4096) {
        auto filter = PlanBuilder()
                          .values(data)
                          .filter("v % 2 = 0")
                          .project({"v + 1"})
                          .planNode();
        cases_["filter_project_1024_output_rows"] = {
            filter, Workload::kFilterProject, 0, 1024, kNumRows / 2, true};
        auto planNodeIds = std::make_shared<core::PlanNodeIdGenerator>();
        auto buildData = makeRowVector(
            {"b"},
            {makeFlatVector<int64_t>(numGroups, [](auto i) { return i; })});
        auto build = PlanBuilder(planNodeIds).values({buildData}).planNode();
        auto join = PlanBuilder(planNodeIds)
                        .values(data)
                        .hashJoin({"k"}, {"b"}, build, "", {"v", "b"})
                        .planNode();
        cases_["hash_join_4096_build_keys_1024_output_rows"] = {
            join, Workload::kHashJoin, 0, 1024, kNumRows, true};
      }
    }
  }

  void addBenchmarks() {
    for (const auto& [name, testCase] : cases_) {
      addBenchmark(name, testCase);
    }
  }

 private:
  std::pair<int64_t, int64_t> run(const BenchmarkCase& testCase, bool verify)
      const {
    CursorParameters params;
    params.planNode = testCase.plan;
    params.maxDrivers = 1;
    params.serialExecution = true;
    params.copyResult = false;
    params.queryConfigs = {
        {core::QueryConfig::kMaxOutputBatchRows,
         std::to_string(testCase.maxOutputRows)},
        {core::QueryConfig::kPreferredOutputBatchRows, "1024"},
        {core::QueryConfig::kOperatorTrackCpuUsage,
         testCase.trackCpuUsage ? "true" : "false"}};
    int64_t rows = 0;
    int64_t batches = 0;
    std::vector<bool> seen(verify ? testCase.expectedRows : 0);
    auto cursor = TaskCursor::create(params);
    while (cursor->moveNext()) {
      const auto& batch = cursor->current();
      rows += batch->size();
      ++batches;
      if (verify) {
        // Separate preflight only. Validate every key/value and reject
        // duplicates.
        DecodedVector first(*batch->childAt(0));
        DecodedVector second;
        if (batch->childrenSize() == 2) {
          second.decode(*batch->childAt(1));
        }
        for (int32_t i = 0; i < batch->size(); ++i) {
          VELOX_CHECK(!first.isNullAt(i));
          auto key = first.valueAt<int64_t>(i);
          if (testCase.workload == Workload::kFilterProject) {
            VELOX_CHECK_EQ(key % 2, 1);
            key /= 2;
          } else {
            VELOX_CHECK(!second.isNullAt(i));
            if (testCase.workload == Workload::kHashJoin) {
              VELOX_CHECK_EQ(second.valueAt<int64_t>(i), key % 4096);
            } else {
              const int64_t rowsPerGroup = kNumRows / testCase.numGroups;
              VELOX_CHECK_EQ(
                  second.valueAt<int64_t>(i),
                  rowsPerGroup * key +
                      testCase.numGroups * rowsPerGroup * (rowsPerGroup - 1) /
                          2);
            }
          }
          VELOX_CHECK(key >= 0 && key < testCase.expectedRows && !seen.at(key));
          seen[key] = true;
        }
      }
    }
    return {rows, batches};
  }

  void addBenchmark(const std::string& name, const BenchmarkCase& testCase) {
    const auto expectedRows = testCase.expectedRows;
    const auto expectedBatches = testCase.numGroups
        ? (testCase.numGroups + testCase.maxOutputRows - 1) /
            testCase.maxOutputRows
        : kNumRows / kInputBatchRows;
    // Verify payloads and output shape before Folly starts measuring.
    const auto verified = run(testCase, true);
    VELOX_CHECK_EQ(verified.first, expectedRows);
    VELOX_CHECK_EQ(verified.second, expectedBatches);
    folly::addBenchmark(
        __FILE__,
        name,
        [this, test = &testCase, expectedRows, expectedBatches](
            unsigned iterations) {
          for (unsigned iteration = 0; iteration < iterations; ++iteration) {
            const auto result = run(*test, false);
            BENCHMARK_SUSPEND {
              VELOX_CHECK_EQ(result.first, expectedRows);
              VELOX_CHECK_EQ(result.second, expectedBatches);
            }
          }
          return iterations;
        });
  }

  // Own the plans until runBenchmarks() returns; registered callbacks borrow
  // these cases so they do not retain vectors after the memory pool is
  // destroyed.
  std::map<std::string, BenchmarkCase> cases_;
};
} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize({});
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  OperatorBatchBenchmark benchmark;
  benchmark.addBenchmarks();
  folly::runBenchmarks();
}
