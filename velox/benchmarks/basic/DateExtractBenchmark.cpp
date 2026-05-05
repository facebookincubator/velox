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
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/sparksql/registration/Register.h"

using namespace facebook;
using namespace facebook::velox;

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::prestosql::registerDateTimeFunctions("");
  functions::sparksql::registerFunctions("spark_");

  ExpressionBenchmarkBuilder benchmarkBuilder;
  VectorFuzzer::Options options;
  options.vectorSize = 1024;
  options.nullRatio = 0;
  auto* pool = benchmarkBuilder.pool();
  VectorFuzzer fuzzer(options, pool);
  auto vectorMaker = benchmarkBuilder.vectorMaker();

  // Each set runs a different extraction expression on the same fuzzed input,
  // exercising Timestamp::epochToCalendarUtc through getDateTime in
  // velox/functions/lib/TimeUtils.h.

  // DATE inputs (int32 days since epoch).
  benchmarkBuilder
      .addBenchmarkSet(
          "year_date", vectorMaker.rowVector({fuzzer.fuzz(DATE())}))
      .addExpression("year", "year(c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet(
          "month_date", vectorMaker.rowVector({fuzzer.fuzz(DATE())}))
      .addExpression("month", "month(c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet("day_date", vectorMaker.rowVector({fuzzer.fuzz(DATE())}))
      .addExpression("day", "day(c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet(
          "quarter_date", vectorMaker.rowVector({fuzzer.fuzz(DATE())}))
      .addExpression("quarter", "quarter(c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet(
          "day_of_year_date", vectorMaker.rowVector({fuzzer.fuzz(DATE())}))
      .addExpression("day_of_year", "day_of_year(c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet(
          "last_day_of_month_date",
          vectorMaker.rowVector({fuzzer.fuzz(DATE())}))
      .addExpression("last_day_of_month", "last_day_of_month(c0)")
      .disableTesting();

  // date_trunc on DATE inputs. Each unit hits a different code path
  // in truncateDate (kWeek is pure int math, kMonth/kQuarter/kYear
  // round-trip through the Neri-Schneider primitives in FastDate.h).
  // Use a bounded year range (1970..~2100) so the benchmark measures
  // the realistic in-range work rather than the rate of out-of-range
  // throws that an unbounded fuzz of int32 days would produce.
  auto dateTruncInput = vectorMaker.flatVector<int32_t>(
      1024,
      [](auto i) { return static_cast<int32_t>(i * 47); },
      nullptr,
      DATE());
  benchmarkBuilder
      .addBenchmarkSet(
          "date_trunc_year_date", vectorMaker.rowVector({dateTruncInput}))
      .addExpression("date_trunc_year", "date_trunc('year', c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet(
          "date_trunc_quarter_date", vectorMaker.rowVector({dateTruncInput}))
      .addExpression("date_trunc_quarter", "date_trunc('quarter', c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet(
          "date_trunc_month_date", vectorMaker.rowVector({dateTruncInput}))
      .addExpression("date_trunc_month", "date_trunc('month', c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet(
          "date_trunc_week_date", vectorMaker.rowVector({dateTruncInput}))
      .addExpression("date_trunc_week", "date_trunc('week', c0)")
      .disableTesting();

  // date_trunc on TIMESTAMP inputs for the date-level units. Same
  // recipe as the DATE benches with a bounded range (~130 years from
  // epoch) so the measurement reflects the in-range path rather than
  // the rate of out-of-range throws an unbounded fuzz would produce.
  auto timestampTruncInput = vectorMaker.flatVector<Timestamp>(
      1024, [](auto i) { return Timestamp(i * 86400 * 47, 0); });
  benchmarkBuilder
      .addBenchmarkSet(
          "date_trunc_year_timestamp",
          vectorMaker.rowVector({timestampTruncInput}))
      .addExpression("date_trunc_year", "date_trunc('year', c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet(
          "date_trunc_quarter_timestamp",
          vectorMaker.rowVector({timestampTruncInput}))
      .addExpression("date_trunc_quarter", "date_trunc('quarter', c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet(
          "date_trunc_month_timestamp",
          vectorMaker.rowVector({timestampTruncInput}))
      .addExpression("date_trunc_month", "date_trunc('month', c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet(
          "date_trunc_week_timestamp",
          vectorMaker.rowVector({timestampTruncInput}))
      .addExpression("date_trunc_week", "date_trunc('week', c0)")
      .disableTesting();

  // TIMESTAMP inputs (seconds + nanos).
  benchmarkBuilder
      .addBenchmarkSet(
          "year_timestamp", vectorMaker.rowVector({fuzzer.fuzz(TIMESTAMP())}))
      .addExpression("year", "year(c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet(
          "month_timestamp", vectorMaker.rowVector({fuzzer.fuzz(TIMESTAMP())}))
      .addExpression("month", "month(c0)")
      .disableTesting();

  benchmarkBuilder
      .addBenchmarkSet(
          "day_timestamp", vectorMaker.rowVector({fuzzer.fuzz(TIMESTAMP())}))
      .addExpression("day", "day(c0)")
      .disableTesting();

  // Inverse direction: (year, month, day) -> Date via Spark make_date.
  // Three INTEGER columns of small valid year/month/day values.
  VectorFuzzer::Options yearOpts = options;
  yearOpts.dataSpec.includeNaN = false;
  VectorFuzzer yearFuzzer(yearOpts, pool);
  benchmarkBuilder
      .addBenchmarkSet(
          "make_date",
          vectorMaker.rowVector(
              {"y", "m", "d"},
              {vectorMaker.flatVector<int32_t>(
                   1024, [](auto i) { return 1970 + (int)(i % 80); }),
               vectorMaker.flatVector<int32_t>(
                   1024, [](auto i) { return 1 + (int)(i % 12); }),
               vectorMaker.flatVector<int32_t>(
                   1024, [](auto i) { return 1 + (int)(i % 28); })}))
      .addExpression("make_date", "spark_make_date(y, m, d)")
      .disableTesting();

  benchmarkBuilder.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
