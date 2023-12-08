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
#include "velox/benchmarks/AggregationBenchmarkBuilder.h"

#include <folly/Benchmark.h>
#include <string>

#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox {

namespace {

core::PlanNodePtr getAggregationPlan(
    PlanType planType,
    const std::vector<RowVectorPtr>& input,
    const AggregationPair& aggregationPair,
    memory::MemoryPool* pool) {
  auto plan = exec::test::PlanBuilder(pool).values(input);
  switch (planType) {
    case kSingle:
      plan.singleAggregation(
          aggregationPair.groupingKeys, aggregationPair.aggregations);
      break;
    case kPartialFinal:
      plan.partialAggregation(
              aggregationPair.groupingKeys, aggregationPair.aggregations)
          .finalAggregation();
      break;
    case kPartialIntermediateFinal:
      plan.partialAggregation(
              aggregationPair.groupingKeys, aggregationPair.aggregations)
          .intermediateAggregation()
          .finalAggregation();
      break;
    default:
      VELOX_UNREACHABLE("Not a valid plan type.");
  }
  return plan.planNode();
}

} // namespace

void AggregationBenchmarkBuilder::ensureInputVectors() {
  for (auto& [_, benchmarkSet] : benchmarkSets_) {
    VectorFuzzer fuzzer(benchmarkSet.fuzzerOptions_, pool_.get());
    benchmarkSet.inputRowVectors_.insert(
        benchmarkSet.inputRowVectors_.end(),
        benchmarkSet.iterations_,
        std::dynamic_pointer_cast<RowVector>(
            fuzzer.fuzzFlat(benchmarkSet.inputType_)));
  }
}

void AggregationBenchmarkBuilder::registerBenchmarks() {
  ensureInputVectors();

  for (auto& [setName, benchmarkSet] : benchmarkSets_) {
    for (auto& [aggregationPairName, aggregationPair] :
         benchmarkSet.aggregations_) {
      auto name = fmt::format("{}##{}", setName, aggregationPairName);
      auto& inputVectors = benchmarkSet.inputRowVectors_;
      auto planType = benchmarkSet.planType_;
      auto& aggregationPairLocal = aggregationPair;

      folly::addBenchmark(
          __FILE__,
          name,
          [this, &inputVectors, &aggregationPairLocal, planType]() {
            int cnt = 0;
            folly::BenchmarkSuspender suspender;

            auto plan = getAggregationPlan(
                planType, inputVectors, aggregationPairLocal, pool_.get());

            std::shared_ptr<folly::Executor> executor{
                std::make_shared<folly::CPUThreadPoolExecutor>(
                    std::thread::hardware_concurrency())};
            auto task = exec::Task::create(
                "aggregation benchmark task",
                core::PlanFragment{
                    plan, core::ExecutionStrategy::kUngrouped, 1, {}},
                0,
                std::make_shared<core::QueryCtx>(executor.get()));

            suspender.dismiss();

            auto result = task->next();
            while (result != nullptr) {
              cnt += result->size();
              result = task->next();
            }
            folly::doNotOptimizeAway(cnt);
            return 1;
          });
    }
    BENCHMARK_DRAW_LINE();
    BENCHMARK_DRAW_LINE();
  }
}

} // namespace facebook::velox
