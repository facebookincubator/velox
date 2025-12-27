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
#include "velox/functions/sparksql/registration/Register.h"

using namespace facebook;

using namespace facebook::velox;

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::sparksql::registerFunctions("");

  ExpressionBenchmarkBuilder benchmarkBuilder;
  const vector_size_t vectorSize = 1000;
  auto vectorMaker = benchmarkBuilder.vectorMaker();
  auto decimalInput = vectorMaker.flatVector<int64_t>(
      vectorSize, [&](auto j) { return 12345 * j; }, nullptr, DECIMAL(9, 2));
  auto shortDecimalInput = vectorMaker.flatVector<int64_t>(
      vectorSize,
      [&](auto j) { return 123456789 * j; },
      nullptr,
      DECIMAL(18, 6));
  auto smallDecimalInput = vectorMaker.flatVector<int64_t>(
      vectorSize,
      [&](auto j) { return 123456789 * j; },
      nullptr,
      DECIMAL(18, 17));
  auto longDecimalInput = vectorMaker.flatVector<int128_t>(
      vectorSize,
      [&](auto j) {
        return facebook::velox::HugeInt::build(12345 * j, 56789 * j + 12345);
      },
      nullptr,
      DECIMAL(38, 16));

  benchmarkBuilder
      .addBenchmarkSet(
          "cast_decimal_as_varchar",
          vectorMaker.rowVector(
              {"decimal", "short_decimal", "small_decimal", "long_decimal"},
              {
                  decimalInput,
                  shortDecimalInput,
                  smallDecimalInput,
                  longDecimalInput,
              }))
      .addExpression(
          "cast_decimal_to_inline_string", "cast (decimal as varchar)")
      .addExpression("cast_short_decimal", "cast (short_decimal as varchar)")
      .addExpression("cast_small_decimal", "cast (small_decimal as varchar)")
      .addExpression("cast_long_decimal", "cast (long_decimal as varchar)")
      .withIterations(100)
      .disableTesting();

  benchmarkBuilder.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
