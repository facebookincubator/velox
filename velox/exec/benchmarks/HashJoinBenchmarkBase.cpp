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

#include "velox/exec/benchmarks/HashJoinBenchmarkBase.h"

#include <fmt/format.h>
#include <folly/Benchmark.h>

#include <algorithm>

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

namespace facebook::velox::exec::test {
namespace {

// Returns the non-key columns 'joinType' allows in the output, plus 'match' for
// a semi project join.
std::vector<std::string> outputLayout(
    core::JoinType joinType,
    const RowTypePtr& probeType,
    const RowTypePtr& buildType) {
  std::vector<std::string> layout;
  const auto addNonKeys = [&](const RowTypePtr& type) {
    layout.insert(layout.end(), type->names().begin() + 1, type->names().end());
  };
  if (!core::isRightSemiFilterJoin(joinType) &&
      !core::isRightSemiProjectJoin(joinType) &&
      !core::isRightAntiJoin(joinType)) {
    addNonKeys(probeType);
  }
  if (!core::isProbeOnlyJoin(joinType)) {
    addNonKeys(buildType);
  }
  if (core::isLeftSemiProjectJoin(joinType) ||
      core::isRightSemiProjectJoin(joinType)) {
    layout.emplace_back("match");
  }
  return layout;
}

} // namespace

HashJoinBenchmarkBase::HashJoinBenchmarkBase(vector_size_t batchRows)
    : batchRows_{batchRows} {
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
}

std::vector<RowVectorPtr> HashJoinBenchmarkBase::makeBatches(
    const std::vector<std::string>& names,
    int64_t numRows,
    const std::function<int64_t(int64_t row)>& key) {
  VELOX_CHECK(names.size() == 1 || names.size() == 2);

  std::vector<RowVectorPtr> batches;
  for (int64_t start = 0; start < numRows; start += batchRows_) {
    const auto size = static_cast<vector_size_t>(
        std::min<int64_t>(batchRows_, numRows - start));
    std::vector<VectorPtr> columns{makeFlatVector<int64_t>(
        size, [&](vector_size_t row) { return key(start + row); })};
    if (names.size() == 2) {
      columns.push_back(makeFlatVector<int64_t>(size, [&](vector_size_t row) {
        return start + row;
      }));
    }
    batches.push_back(makeRowVector(names, columns));
  }
  return batches;
}

void HashJoinBenchmarkBase::addBenchmark(
    std::string_view name,
    const Join& join,
    const std::vector<RowVectorPtr>& probe,
    const std::vector<RowVectorPtr>& build,
    std::source_location location) {
  PlanBuilder plan(pool());
  plan.values(probe);
  const auto probeType = plan.planNode()->outputType();
  const auto buildNode = plan.newPlan().values(build).planNode();
  plan.hashJoin(
          {"t0"},
          {"u0"},
          buildNode,
          join.filter,
          outputLayout(join.joinType, probeType, buildNode->outputType()),
          join.joinType)
      .singleAggregation({}, {"count(1)"});

  auto title =
      fmt::format("{} {}", core::JoinTypeName::toName(join.joinType), name);
  if (!join.filter.empty()) {
    title += fmt::format(", {}", join.filter);
  }

  plans_.push_back(plan.planNode());
  folly::addBenchmark(
      location.file_name(),
      std::move(title),
      [this,
       index = plans_.size() - 1,
       configs = join.configs,
       expectedNumOutputRows =
           join.expectedNumOutputRows](folly::UserCounters& counters) {
        const auto result = AssertQueryBuilder(plans_[index])
                                .serialExecution(true)
                                .configs(configs)
                                .copyResults(pool());
        VELOX_CHECK_EQ(result->size(), 1);
        const auto numOutputRows =
            result->childAt(0)->as<SimpleVector<int64_t>>()->valueAt(0);
        if (expectedNumOutputRows.has_value()) {
          VELOX_CHECK_EQ(
              numOutputRows,
              expectedNumOutputRows.value(),
              "Unexpected join output row count");
        }
        counters["outputRows"] = numOutputRows;
        return 1;
      });
}

} // namespace facebook::velox::exec::test
