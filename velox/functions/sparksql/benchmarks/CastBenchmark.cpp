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
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/functions/sparksql/registration/Register.h"

using namespace facebook;

using namespace facebook::velox;

namespace {

core::TypedExprPtr makeCastExpr(
    const TypePtr& fromType,
    const std::string& fieldName,
    const TypePtr& toType) {
  auto input =
      std::make_shared<const core::FieldAccessTypedExpr>(fromType, fieldName);
  return std::make_shared<const core::CastTypedExpr>(
      toType, std::move(input), false);
}

// Add a benchmark for a cast expression. The benchmark will evaluate the cast
// expression on the input row vector for a number of iterations.
// This is used when 'DuckSqlExpressionsParser' cannot be used to parse the
// expression, e.g. when the input type is TIMESTAMP_UTC.
void addTypedCastBenchmark(
    const std::string& name,
    const RowVectorPtr& input,
    core::TypedExprPtr castExpr,
    core::ExecCtx& execCtx,
    int32_t iterations) {
  auto exprSet = std::make_shared<exec::ExprSet>(
      std::vector<core::TypedExprPtr>{std::move(castExpr)}, &execCtx);
  folly::addBenchmark(__FILE__, name, [input, exprSet, &execCtx, iterations]() {
    exec::EvalCtx evalCtx(&execCtx, exprSet.get(), input.get());
    SelectivityVector rows(input->size());
    std::vector<VectorPtr> results(1);

    int64_t count = 0;
    for (auto i = 0; i < iterations; ++i) {
      exprSet->eval(rows, evalCtx, results);
      BaseVector::flattenVector(results[0]);
      results[0]->prepareForReuse();
      count += results[0]->size();
    }
    folly::doNotOptimizeAway(count);
    return 1;
  });
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});

  ExpressionBenchmarkBuilder benchmarkBuilder;
  // 'ExpressionBenchmarkBuilder' registers the default special forms through
  // 'FunctionBenchmarkBase'. Register SparkSQL functions afterward so the
  // benchmarks use Spark cast implementations.
  functions::sparksql::registerFunctions("");
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

  auto timestampInput = vectorMaker.rowVector(
      {"timestamp", "timestamp_utc"},
      {vectorMaker.flatVector<Timestamp>(
           vectorSize,
           [](auto j) {
             return Timestamp(1'600'000'000 + j, j % 1'000'000'000);
           },
           nullptr,
           TIMESTAMP()),
       vectorMaker.flatVector<Timestamp>(
           vectorSize,
           [](auto j) {
             return Timestamp(1'600'000'000 + j, j % 1'000'000'000);
           },
           nullptr,
           TIMESTAMP_UTC())});

  const std::string setName = "spark cast";
  const int iterations = 100;
  benchmarkBuilder
      .addBenchmarkSet(
          setName,
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
      .withIterations(iterations)
      .disableTesting();

  auto queryCtx = core::QueryCtx::create();
  core::ExecCtx execCtx(benchmarkBuilder.pool(), queryCtx.get());
  addTypedCastBenchmark(
      setName + "##cast_timestamp_as_timestamp_utc",
      timestampInput,
      makeCastExpr(TIMESTAMP(), "timestamp", TIMESTAMP_UTC()),
      execCtx,
      iterations);
  addTypedCastBenchmark(
      setName + "##cast_timestamp_utc_as_timestamp",
      timestampInput,
      makeCastExpr(TIMESTAMP_UTC(), "timestamp_utc", TIMESTAMP()),
      execCtx,
      iterations);

  benchmarkBuilder.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
