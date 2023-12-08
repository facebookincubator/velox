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
#include "velox/benchmarks/AggregationBenchmarkBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  aggregate::prestosql::registerAllAggregateFunctions();

  AggregationBenchmarkBuilder benchmarkBuilder;
  auto inputType = ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), DOUBLE()});

  benchmarkBuilder.addBenchmarkSet("avg_single", inputType)
      .addAggregations("integer", {"c0"}, {"avg(c1)"})
      .addAggregations("double", {"c0"}, {"avg(c2)"})
      .withPlanType(kSingle)
      .withIterations(10);

  benchmarkBuilder.addBenchmarkSet("avg_partial_final", inputType)
      .addAggregations("integer", {"c0"}, {"avg(c1)"})
      .addAggregations("double", {"c0"}, {"avg(c2)"})
      .withPlanType(kPartialFinal)
      .withIterations(10);

  benchmarkBuilder.addBenchmarkSet("avg_partial_intermediate_final", inputType)
      .addAggregations("integer", {"c0"}, {"avg(c1)"})
      .addAggregations("double", {"c0"}, {"avg(c2)"})
      .withPlanType(kPartialIntermediateFinal)
      .withIterations(10);

  benchmarkBuilder.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
