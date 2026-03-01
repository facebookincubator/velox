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
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});

  ExpressionBenchmarkBuilder benchmarkBuilder;
  const vector_size_t vectorSize = 300'000'000;
  auto vectorMaker = benchmarkBuilder.vectorMaker();

  auto tinyIntInputNullable = vectorMaker.flatVector<int8_t>(
      vectorSize,
      [](auto j) { return j % std::numeric_limits<int8_t>::max(); },
      [](auto j) { return j % 5 == 0; });
  auto tinyIntInput = vectorMaker.flatVector<int8_t>(
      vectorSize,
      [](auto j) { return j % std::numeric_limits<int8_t>::max(); },
      nullptr);

  auto smallIntInputNullable = vectorMaker.flatVector<int16_t>(
      vectorSize,
      [](auto j) { return j % std::numeric_limits<int16_t>::max(); },
      [](auto j) { return j % 5 == 0; });
  auto smallIntInput = vectorMaker.flatVector<int16_t>(
      vectorSize,
      [](auto j) { return j % std::numeric_limits<int16_t>::max(); },
      nullptr);

  auto integerInputNullable = vectorMaker.flatVector<int32_t>(
      vectorSize,
      [](auto j) { return j * 2 + 1; },
      [](auto j) { return j % 5 == 0; });
  auto integerInput = vectorMaker.flatVector<int32_t>(
      vectorSize, [](auto j) { return j * 2 + 1; }, nullptr);

  auto bigintInputNullable = vectorMaker.flatVector<int64_t>(
      vectorSize,
      [](auto j) { return j * 2 + 1; },
      [](auto j) { return j % 5 == 0; });
  auto bigintInput = vectorMaker.flatVector<int64_t>(
      vectorSize, [](auto j) { return j * 2 + 1; }, nullptr);

  auto realInputNullable = vectorMaker.flatVector<float>(
      vectorSize,
      [](auto j) { return j * 9.9999; },
      [](auto j) { return j % 5 == 0; });
  auto realInput = vectorMaker.flatVector<float>(
      vectorSize, [](auto j) { return j * 9.9999; }, nullptr);

  auto hugeintInputNullable = vectorMaker.flatVector<int128_t>(
      vectorSize,
      [](auto j) { return j * 999 + 1; },
      [](auto j) { return j % 5 == 0; });
  auto hugeintInput = vectorMaker.flatVector<int128_t>(
      vectorSize, [](auto j) { return j * 999 + 1; }, nullptr);

  benchmarkBuilder
      .addBenchmarkSet(
          "numeric_upcast",
          vectorMaker.rowVector(
              {
                  "tinyint_column_nullable",
                  "tinyint_column",
                  "smallint_column_nullable",
                  "smallint_column",
                  "integer_column_nullable",
                  "integer_column",
                  "bigint_column_nullable",
                  "bigint_column",
                  "real_column_nullable",
                  "real_column",
                  "hugeint_column_nullable",
                  "hugeint_column",
              },
              {
                  tinyIntInputNullable,
                  tinyIntInput,
                  smallIntInputNullable,
                  smallIntInput,
                  integerInputNullable,
                  integerInput,
                  bigintInputNullable,
                  bigintInput,
                  realInputNullable,
                  realInput,
                  hugeintInputNullable,
                  hugeintInput,
              }))
      // Cast from tinyint.
      .addExpression(
          "cast_tinyint_nullable_as_smallint",
          "cast(tinyint_column_nullable as smallint)")
      .addExpression(
          "cast_tinyint_as_smallint", "cast(tinyint_column as smallint)")

      .addExpression(
          "cast_tinyint_nullable_as_integer",
          "cast(tinyint_column_nullable as integer)")
      .addExpression(
          "cast_tinyint_as_integer", "cast(tinyint_column as integer)")

      .addExpression(
          "cast_tinyint_nullable_as_bigint",
          "cast(tinyint_column_nullable as bigint)")
      .addExpression("cast_tinyint_as_bigint", "cast(tinyint_column as bigint)")

      .addExpression(
          "cast_tinyint_nullable_as_real",
          "cast(tinyint_column_nullable as real)")
      .addExpression("cast_tinyint_as_real", "cast(tinyint_column as real)")

      .addExpression(
          "cast_tinyint_nullable_as_double",
          "cast(tinyint_column_nullable as double)")
      .addExpression("cast_tinyint_as_double", "cast(tinyint_column as double)")

      .addExpression(
          "cast_tinyint_nullable_as_hugeint",
          "cast(tinyint_column_nullable as hugeint)")
      .addExpression(
          "cast_tinyint_as_hugeint", "cast(tinyint_column as hugeint)")

      // Cast from smallint.
      .addExpression(
          "cast_smallint_nullable_as_integer",
          "cast(smallint_column_nullable as integer)")
      .addExpression(
          "cast_smallint_as_integer", "cast(smallint_column as integer)")

      .addExpression(
          "cast_smallint_nullable_as_bigint",
          "cast(smallint_column_nullable as bigint)")
      .addExpression(
          "cast_smallint_as_bigint", "cast(smallint_column as bigint)")

      .addExpression(
          "cast_smallint_nullable_as_real",
          "cast(smallint_column_nullable as real)")
      .addExpression("cast_smallint_as_real", "cast(smallint_column as real)")

      .addExpression(
          "cast_smallint_nullable_as_double",
          "cast(smallint_column_nullable as double)")
      .addExpression(
          "cast_smallint_as_double", "cast(smallint_column as double)")

      .addExpression(
          "cast_smallint_nullable_as_hugeint",
          "cast(smallint_column_nullable as hugeint)")
      .addExpression(
          "cast_smallint_as_hugeint", "cast(smallint_column as hugeint)")

      // Cast from integer.
      .addExpression(
          "cast_integer_nullable_as_bigint",
          "cast(integer_column_nullable as bigint)")
      .addExpression("cast_integer_as_bigint", "cast(integer_column as bigint)")

      .addExpression(
          "cast_integer_nullable_as_real",
          "cast(integer_column_nullable as real)")
      .addExpression("cast_integer_as_real", "cast(integer_column as real)")

      .addExpression(
          "cast_integer_nullable_as_double",
          "cast(integer_column_nullable as double)")
      .addExpression("cast_integer_as_double", "cast(integer_column as double)")

      .addExpression(
          "cast_integer_nullable_as_hugeint",
          "cast(integer_column_nullable as hugeint)")
      .addExpression(
          "cast_integer_as_hugeint", "cast(integer_column as hugeint) ")

      // Cast from bigint.
      .addExpression(
          "cast_bigint_nullable_as_real",
          "cast(bigint_column_nullable as real)")
      .addExpression("cast_bigint_as_real", "cast(bigint_column as real)")

      .addExpression(
          "cast_bigint_nullable_as_double",
          "cast(bigint_column_nullable as double)")
      .addExpression("cast_bigint_as_double", "cast(bigint_column as double)")

      .addExpression(
          "cast_bigint_nullable_as_hugeint",
          "cast(bigint_column_nullable as hugeint)")
      .addExpression("cast_bigint_as_hugeint", "cast(bigint_column as hugeint)")

      // Cast from hugeint.
      .addExpression(
          "cast_hugeint_nullable_as_real",
          "cast(hugeint_column_nullable as real)")
      .addExpression("cast_hugeint_as_real", "cast(hugeint_column as real)")

      .addExpression(
          "cast_hugeint_nullable_as_double",
          "cast(hugeint_column_nullable as double)")
      .addExpression("cast_hugeint_as_double", "cast(hugeint_column as double)")

      // Cast from real.
      .addExpression(
          "cast_real_nullable_as_double",
          "cast(real_column_nullable as double)")
      .addExpression("cast_real_as_double", "cast(real_column as double)")
      .withIterations(100)
      .disableTesting();

  benchmarkBuilder.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
