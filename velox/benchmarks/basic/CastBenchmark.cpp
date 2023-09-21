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

using namespace facebook;

using namespace facebook::velox;

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  ExpressionBenchmarkBuilder benchmarkBuilder;
  const vector_size_t vectorSize = 1000;
  auto vectorMaker = benchmarkBuilder.vectorMaker();
  auto invalidInput = vectorMaker.flatVector<facebook::velox::StringView>({""});
  auto validInput = vectorMaker.flatVector<facebook::velox::StringView>({""});
  auto nanInput = vectorMaker.flatVector<facebook::velox::StringView>({""});
  auto decimalInput = vectorMaker.flatVector<int64_t>(
      vectorSize, [&](auto j) { return 12345 * j; }, nullptr, DECIMAL(9, 2));
  auto shortDecimalInput = vectorMaker.flatVector<int64_t>(
      vectorSize,
      [&](auto j) { return 123456789 * j; },
      nullptr,
      DECIMAL(18, 6));
  auto longDecimalInput = vectorMaker.flatVector<int128_t>(
      vectorSize,
      [&](auto j) {
        return facebook::velox::HugeInt::build(12345 * j, 56789 * j + 12345);
      },
      nullptr,
      DECIMAL(38, 16));

  invalidInput->resize(vectorSize);
  validInput->resize(vectorSize);
  nanInput->resize(vectorSize);

  for (int i = 0; i < vectorSize; i++) {
    nanInput->set(i, "$"_sv);
    invalidInput->set(i, StringView::makeInline(std::string("")));
    validInput->set(i, StringView::makeInline(std::to_string(i)));
  }

  benchmarkBuilder
      .addBenchmarkSet(
          "cast",
          vectorMaker.rowVector(
              {"valid",
               "empty",
               "nan",
               "decimal",
               "short_decimal",
               "long_decimal"},
              {validInput,
               invalidInput,
               nanInput,
               decimalInput,
               shortDecimalInput,
               longDecimalInput}))
      .addExpression("try_cast_invalid_empty_input", "try_cast (empty as int) ")
      .addExpression(
          "tryexpr_cast_invalid_empty_input", "try (cast (empty as int))")
      .addExpression("try_cast_invalid_nan", "try_cast (nan as int)")
      .addExpression("tryexpr_cast_invalid_nan", "try (cast (nan as int))")
      .addExpression("try_cast_valid", "try_cast (valid as int)")
      .addExpression("tryexpr_cast_valid", "try (cast (valid as int))")
      .addExpression("cast_valid", "cast(valid as int)")
      .addExpression(
          "cast_decimal_to_inline_string", "cast (decimal as varchar)")
      .addExpression("cast_short_decimal", "cast (short_decimal as varchar)")
      .addExpression("cast_long_decimal", "cast (long_decimal as varchar)")
      .withIterations(100)
      .disableTesting();

  benchmarkBuilder.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
