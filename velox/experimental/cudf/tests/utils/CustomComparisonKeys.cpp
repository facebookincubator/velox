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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/CustomComparisonKeys.h"

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox::test_utils {
namespace {

// The driver adapter reads the flag when cuDF is registered.
void setAllowCpuFallback(bool allow) {
  unregisterCudf();
  CudfConfig::getInstance().allowCpuFallback = allow;
  registerCudf();
}

} // namespace

CustomComparisonKeys::CustomComparisonKeys(memory::MemoryPool* pool)
    : pool_{pool},
      maker_{pool},
      previousAllowCpuFallback_{CudfConfig::getInstance().allowCpuFallback} {
  setAllowCpuFallback(true);
}

CustomComparisonKeys::~CustomComparisonKeys() {
  setAllowCpuFallback(previousAllowCpuFallback_);
}

namespace {

constexpr std::string_view kLosAngeles{"America/Los_Angeles"};
constexpr std::string_view kUtc{"UTC"};

// Packs 'second' seconds past the epoch with the key of 'zone'.
int64_t instant(int64_t second, std::string_view zone) {
  return pack(second * 1'000, tz::getTimeZoneID(zone));
}

} // namespace

RowVectorPtr CustomComparisonKeys::makeRows(int32_t numInstants) {
  const vector_size_t numRows = 2 * numInstants;
  return maker_.rowVector(
      {"g", "k", "id"},
      {maker_.flatVector<int64_t>(
           numRows, [](auto row) { return 1 + row / 2; }),
       maker_.flatVector<int64_t>(
           numRows,
           [](auto row) {
             const int64_t number = 1 + row / 2;
             const bool utc = (row % 2 == 0) == (number % 2 == 0);
             return instant(number, utc ? kUtc : kLosAngeles);
           },
           nullptr,
           TIMESTAMP_WITH_TIME_ZONE()),
       maker_.flatVector<int64_t>(numRows, [](auto row) { return row; })});
}

RowVectorPtr CustomComparisonKeys::makeRowsInZone(
    const std::vector<std::string>& names,
    std::string_view zone,
    const std::vector<int64_t>& seconds,
    const std::vector<int64_t>& values) {
  std::vector<int64_t> keys;
  for (const auto second : seconds) {
    keys.push_back(instant(second, zone));
  }
  return maker_.rowVector(
      names,
      {maker_.flatVector(keys, TIMESTAMP_WITH_TIME_ZONE()),
       maker_.flatVector(values)});
}

namespace {

// Operator types of a task in pipeline order and each result row as text. A
// TIMESTAMP WITH TIME ZONE value prints with its zone, so the two encodings of
// one instant read differently where the type's own comparison would call
// them equal.
struct Run {
  std::vector<std::string> operatorTypes;
  std::vector<std::string> rows;
};

// Runs 'plan' with cuDF enabled or disabled over 'maxDrivers' drivers.
Run run(
    const core::PlanNodePtr& plan,
    bool cudfEnabled,
    int32_t maxDrivers,
    memory::MemoryPool* pool) {
  std::shared_ptr<exec::Task> task;
  auto result =
      exec::test::AssertQueryBuilder(plan)
          .config(CudfConfig::kCudfEnabled, cudfEnabled ? "true" : "false")
          .maxDrivers(maxDrivers)
          .copyResults(pool, task);
  Run summary;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& operatorStats : pipeline.operatorStats) {
      summary.operatorTypes.push_back(operatorStats.operatorType);
    }
  }
  for (vector_size_t row = 0; row < result->size(); ++row) {
    std::string text;
    for (const auto& child : result->children()) {
      text += (text.empty() ? "" : ", ") + child->toString(row);
    }
    summary.rows.push_back(text);
  }
  return summary;
}

// Expects 'cpuOperator' to have run in place of any cuDF operator.
void expectCpuOperators(const Run& gpu, std::string_view cpuOperator) {
  EXPECT_THAT(gpu.operatorTypes, testing::Contains(std::string(cpuOperator)));
  EXPECT_THAT(
      gpu.operatorTypes,
      testing::Each(testing::Not(testing::StartsWith("Cudf"))));
}

} // namespace

void CustomComparisonKeys::assertFallsBackToCpu(
    const core::PlanNodePtr& plan,
    std::string_view cpuOperator,
    int32_t maxDrivers) {
  const auto gpu = run(plan, true, maxDrivers, pool_);
  expectCpuOperators(gpu, cpuOperator);
  EXPECT_THAT(
      gpu.rows,
      testing::UnorderedElementsAreArray(
          run(plan, false, maxDrivers, pool_).rows));
}

void CustomComparisonKeys::assertFallsBackToCpuInOrder(
    const core::PlanNodePtr& plan,
    std::string_view cpuOperator) {
  const auto gpu = run(plan, true, 1, pool_);
  expectCpuOperators(gpu, cpuOperator);
  EXPECT_THAT(
      gpu.rows, testing::ElementsAreArray(run(plan, false, 1, pool_).rows));
}

void CustomComparisonKeys::assertRunsOnGpu(
    const core::PlanNodePtr& plan,
    std::string_view cudfOperator) {
  const auto gpu = run(plan, true, 1, pool_);
  EXPECT_THAT(
      gpu.operatorTypes,
      testing::Contains(testing::StartsWith(std::string(cudfOperator))));
  EXPECT_THAT(
      gpu.rows,
      testing::UnorderedElementsAreArray(run(plan, false, 1, pool_).rows));
}

} // namespace facebook::velox::cudf_velox::test_utils
