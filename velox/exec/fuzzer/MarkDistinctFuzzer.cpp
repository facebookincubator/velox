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

#include "velox/exec/fuzzer/MarkDistinctFuzzer.h"

#include <utility>

#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/exec/fuzzer/SpillFuzzerBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec {
using namespace facebook::velox::common::testutil;
namespace {

class MarkDistinctFuzzer : public SpillFuzzerBase {
 public:
  explicit MarkDistinctFuzzer(
      size_t initialSeed,
      std::unique_ptr<test::ReferenceQueryRunner>);

 private:
  void runSingleIteration() override;

  std::pair<std::vector<std::string>, std::vector<TypePtr>>
  generatePartitionKeys();

  std::vector<RowVectorPtr> generateInput(
      const std::vector<std::string>& keyNames,
      const std::vector<TypePtr>& keyTypes);

  // Makes the query plan: Values -> MarkDistinct -> Aggregation.
  // MarkDistinct marks distinct rows for (groupKey, distinctKey) combinations,
  // then aggregation uses the mask to compute count(distinct distinctKey).
  static PlanWithSplits makeDefaultPlan(
      const std::string& groupKey,
      const std::string& distinctKey,
      const std::vector<RowVectorPtr>& input);

  static PlanWithSplits makePlanWithTableScan(
      const RowTypePtr& type,
      const std::string& groupKey,
      const std::string& distinctKey,
      const std::vector<Split>& splits);

  void addPlansWithTableScan(
      const std::string& tableDir,
      const std::string& groupKey,
      const std::string& distinctKey,
      const std::vector<RowVectorPtr>& input,
      std::vector<PlanWithSplits>& altPlans);
};

MarkDistinctFuzzer::MarkDistinctFuzzer(
    size_t initialSeed,
    std::unique_ptr<test::ReferenceQueryRunner> referenceQueryRunner)
    : SpillFuzzerBase(initialSeed, std::move(referenceQueryRunner)) {
  vectorFuzzer_.getMutableOptions().timestampPrecision =
      fuzzer::FuzzerTimestampPrecision::kMilliSeconds;
}

std::pair<std::vector<std::string>, std::vector<TypePtr>>
MarkDistinctFuzzer::generatePartitionKeys() {
  // Generate 1-3 keys for grouping/distinct.
  const auto numKeys = randInt(1, 3);
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto i = 0; i < numKeys; ++i) {
    names.push_back(fmt::format("c{}", i));
    types.push_back(vectorFuzzer_.randType(/*maxDepth=*/1));
  }
  return std::make_pair(names, types);
}

std::vector<RowVectorPtr> MarkDistinctFuzzer::generateInput(
    const std::vector<std::string>& keyNames,
    const std::vector<TypePtr>& keyTypes) {
  std::vector<std::string> names = keyNames;
  std::vector<TypePtr> types = keyTypes;

  // Add up to 2 payload columns.
  const auto numPayload = randInt(0, 2);
  for (auto i = 0; i < numPayload; ++i) {
    names.push_back(fmt::format("c{}", i + keyNames.size()));
    types.push_back(vectorFuzzer_.randType(/*maxDepth=*/2));
  }

  const auto inputType = ROW(std::move(names), std::move(types));
  std::vector<RowVectorPtr> input;
  input.reserve(FLAGS_num_batches);
  for (auto i = 0; i < FLAGS_num_batches; ++i) {
    input.push_back(vectorFuzzer_.fuzzInputRow(inputType));
  }

  return input;
}

PlanWithSplits MarkDistinctFuzzer::makeDefaultPlan(
    const std::string& groupKey,
    const std::string& distinctKey,
    const std::vector<RowVectorPtr>& input) {
  // Build plan: Values -> MarkDistinct -> Aggregation
  // This is equivalent to: SELECT groupKey, count(DISTINCT distinctKey)
  //                         FROM input GROUP BY groupKey
  auto plan = test::PlanBuilder()
                  .values(input)
                  .markDistinct("distinct_marker", {groupKey, distinctKey})
                  .singleAggregation(
                      {groupKey},
                      {"count(\"" + distinctKey + "\")"},
                      {"distinct_marker"})
                  .planNode();
  return PlanWithSplits{std::move(plan)};
}

PlanWithSplits MarkDistinctFuzzer::makePlanWithTableScan(
    const RowTypePtr& type,
    const std::string& groupKey,
    const std::string& distinctKey,
    const std::vector<Split>& splits) {
  auto plan = test::PlanBuilder()
                  .tableScan(type)
                  .markDistinct("distinct_marker", {groupKey, distinctKey})
                  .singleAggregation(
                      {groupKey},
                      {"count(\"" + distinctKey + "\")"},
                      {"distinct_marker"})
                  .planNode();
  return PlanWithSplits{plan, splits};
}

void MarkDistinctFuzzer::addPlansWithTableScan(
    const std::string& tableDir,
    const std::string& groupKey,
    const std::string& distinctKey,
    const std::vector<RowVectorPtr>& input,
    std::vector<PlanWithSplits>& altPlans) {
  VELOX_CHECK(!tableDir.empty());

  if (!isTableScanSupported(input[0]->type())) {
    return;
  }

  const std::vector<Split> inputSplits = test::makeSplits(
      input, fmt::format("{}/mark_distinct", tableDir), writerPool_);
  altPlans.push_back(makePlanWithTableScan(
      asRowType(input[0]->type()), groupKey, distinctKey, inputSplits));
}

void MarkDistinctFuzzer::runSingleIteration() {
  const auto [keyNames, keyTypes] = generatePartitionKeys();

  // We need at least 2 keys: one for GROUP BY and one for DISTINCT.
  // If only 1 key was generated, add another one.
  std::vector<std::string> allNames = keyNames;
  std::vector<TypePtr> allTypes = keyTypes;
  if (allNames.size() < 2) {
    allNames.push_back(fmt::format("c{}", allNames.size()));
    allTypes.push_back(vectorFuzzer_.randType(/*maxDepth=*/1));
  }

  const auto input = generateInput(allNames, allTypes);
  test::logVectors(input);

  // Use first key as group-by key, second as distinct key.
  const auto& groupKey = allNames[0];
  const auto& distinctKey = allNames[1];

  auto defaultPlan = makeDefaultPlan(groupKey, distinctKey, input);
  const auto expected = execute(defaultPlan, /*injectSpill=*/false, false);

  // Validate against DuckDB using an equivalent plan without MarkDistinct.
  // DuckDB cannot translate MarkDistinctNode to SQL, so we build a reference
  // plan that uses count(DISTINCT ...) directly.
  if (expected != nullptr) {
    auto referencePlan =
        test::PlanBuilder()
            .values(input)
            .singleAggregation(
                {groupKey},
                {fmt::format("count(DISTINCT \"{}\")", distinctKey)})
            .planNode();
    validateExpectedResults(referencePlan, input, expected);
  }

  std::vector<PlanWithSplits> altPlans;
  altPlans.push_back(std::move(defaultPlan));

  const auto tableScanDir = TempDirectoryPath::create();
  addPlansWithTableScan(
      tableScanDir->getPath(), groupKey, distinctKey, input, altPlans);

  for (auto i = 0; i < altPlans.size(); ++i) {
    testPlan(
        altPlans[i],
        i,
        expected,
        "core::QueryConfig::kMarkDistinctSpillEnabled");
  }
}

} // namespace

void markDistinctFuzzer(
    size_t seed,
    std::unique_ptr<test::ReferenceQueryRunner> referenceQueryRunner) {
  MarkDistinctFuzzer(seed, std::move(referenceQueryRunner)).run();
}
} // namespace facebook::velox::exec
