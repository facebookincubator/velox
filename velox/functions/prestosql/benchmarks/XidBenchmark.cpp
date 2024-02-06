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

#include "velox/benchmarks/ExpressionBenchmarkBuilder.h"
#include "velox/functions/facebook/prestosql/Register.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};

  // Register the functions we want to benchmark
  functions::prestosql::registerAllScalarFacebookFunctions();

  // Set up the benchmarking objects
  ExpressionBenchmarkBuilder benchmarkBuilder;
  auto* pool = benchmarkBuilder.pool();
  auto& vm = benchmarkBuilder.vectorMaker();

  // Generate the input data
  VectorFuzzer::Options options;
  VectorFuzzer fuzzer(options, pool);
  auto rowVector = vm.rowVector({
      fuzzer.fuzzFlat(VARCHAR()), // c0
      fuzzer.fuzzFlat(VARCHAR()), // c1
      fuzzer.fuzzFlat(VARCHAR()), // c2
      fuzzer.fuzzFlat(VARCHAR()), // c3
      fuzzer.fuzzFlat(VARCHAR()), // c4
      fuzzer.fuzzFlat(VARCHAR()), // c5
      fuzzer.fuzzFlat(VARCHAR()), // c6
  });

  // Add the benchmarks
  benchmarkBuilder.addBenchmarkSet("xid_construct", rowVector)
      .addExpression("construct_small", "fb_xid_construct(c0, c1)")
      .addExpression(
          "construct_big", "fb_xid_construct(c0, c1, c2, c3, c4, c5, c6)");

  // Register and run them!
  benchmarkBuilder.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
