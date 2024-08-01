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
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/ArrayFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions;

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize({});

  functions::prestosql::registerArrayFunctions();

  ExpressionBenchmarkBuilder benchmarkBuilder;
  auto inputType = ARRAY(ARRAY(BIGINT()));
  const vector_size_t vectorSize = 1000;
  auto vectorMaker = benchmarkBuilder.vectorMaker();
  auto nestedArrays = vectorMaker.arrayVector<int64_t>(
      vectorSize,
      [](auto /*row*/) { return 10; },
      [](auto row) { return row; });

  benchmarkBuilder
      .addBenchmarkSet(
          "array_distinct_complex",
          vectorMaker.rowVector({"nested_array"}, {nestedArrays}))
      .addExpression("array", "array_distinct(nested_array)");

  benchmarkBuilder.registerBenchmarks();

  folly::runBenchmarks();
  return 0;
}
