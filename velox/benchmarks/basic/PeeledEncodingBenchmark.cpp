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

// Benchmarks expression evaluation over dictionary vectors whose base is much
// larger than the selected outer batch.

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/benchmarks/ExpressionBenchmarkBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook;
using namespace facebook::velox;

namespace {

RowVectorPtr makeDictionaryInput(
    ExpressionBenchmarkBuilder& builder,
    vector_size_t baseSize,
    vector_size_t batchSize) {
  auto& vectorMaker = builder.vectorMaker();
  auto lhs = vectorMaker.flatVector<int64_t>(
      baseSize, [](auto row) { return static_cast<int64_t>(row); });
  auto rhs = vectorMaker.flatVector<int64_t>(
      baseSize, [](auto row) { return static_cast<int64_t>(row * 3); });
  auto indices = test::makeIndices(
      batchSize,
      [baseSize](auto row) { return (row * 251) % baseSize; },
      lhs->pool());

  return vectorMaker.rowVector(
      {"a", "b"},
      {BaseVector::wrapInDictionary(nullptr, indices, batchSize, lhs),
       BaseVector::wrapInDictionary(nullptr, indices, batchSize, rhs)});
}

void addBenchmark(
    ExpressionBenchmarkBuilder& builder,
    vector_size_t baseSize,
    vector_size_t batchSize) {
  builder
      .addBenchmarkSet(
          fmt::format("base{}_batch{}", baseSize, batchSize),
          makeDictionaryInput(builder, baseSize, batchSize))
      .addExpression("plus", "a + b")
      .withIterations(10);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::prestosql::registerAllScalarFunctions();

  ExpressionBenchmarkBuilder benchmarkBuilder;
  constexpr vector_size_t kBaseSize = 1 << 20;
  addBenchmark(benchmarkBuilder, kBaseSize, 1 << 10);
  addBenchmark(benchmarkBuilder, kBaseSize, 1 << 12);
  addBenchmark(benchmarkBuilder, kBaseSize, 1 << 18);

  benchmarkBuilder.testBenchmarks();
  benchmarkBuilder.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
