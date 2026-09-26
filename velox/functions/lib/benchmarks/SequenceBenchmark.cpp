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
#include "velox/functions/lib/benchmarks/SequenceBenchmark.h"

#include <folly/Benchmark.h>
#include <random>

#include "velox/functions/lib/DateTimeUtil.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"

namespace facebook::velox::functions::test {
namespace {
constexpr vector_size_t kBatchSize = 10'240;
constexpr int64_t kMillisPerDay = 86'400'000;

// Time evaluation and result allocation/release, excluding input generation,
// compilation and complete result validation.
template <typename T, typename Step, typename MakeStart, typename Advance>
void run(
    FunctionBenchmarkBase& benchmark,
    unsigned iterations,
    vector_size_t length,
    int32_t nullPercent,
    std::string_view encoding,
    const TypePtr& type,
    const TypePtr& stepType,
    MakeStart makeStart,
    Step stepUnit,
    Advance advance) {
  folly::BenchmarkSuspender suspender;
  std::mt19937 random(20260926);
  std::vector<T> starts(kBatchSize), stops(kBatchSize);
  std::vector<Step> steps(kBatchSize);
  std::vector<bool> nulls(kBatchSize);
  for (vector_size_t row = 0; row < kBatchSize; ++row) {
    starts[row] = makeStart(random() % 16);
    steps[row] = row % 2 == 0 ? stepUnit : -stepUnit;
    stops[row] = advance(starts[row], steps[row], length - 1);
    nulls[row] = random() % 100 < nullPercent;
  }
  std::vector<VectorPtr> arguments{
      benchmark.maker().flatVector<T>(
          kBatchSize,
          [&](auto row) { return starts[row]; },
          [&](auto row) { return nulls[row]; },
          type),
      benchmark.maker().flatVector(stops, type),
      benchmark.maker().flatVector(steps, stepType)};
  VectorPtr expected = benchmark.maker().arrayVector<T>(
      kBatchSize,
      [length](auto) { return length; },
      [&](auto row, auto index) {
        return advance(starts[row], steps[row], index);
      },
      [&](auto row) { return nulls[row]; },
      ARRAY(type));
  if (encoding == "dictionary") {
    auto indices =
        AlignedBuffer::allocate<vector_size_t>(kBatchSize, benchmark.pool());
    auto* rawIndices = indices->asMutable<vector_size_t>();
    for (vector_size_t row = 0; row < kBatchSize; ++row) {
      rawIndices[row] = kBatchSize - row - 1;
    }
    for (auto& argument : arguments) {
      argument =
          BaseVector::wrapInDictionary(nullptr, indices, kBatchSize, argument);
    }
    expected =
        BaseVector::wrapInDictionary(nullptr, indices, kBatchSize, expected);
  }
  auto input = benchmark.maker().rowVector(arguments);
  auto expression =
      benchmark.compileExpression("sequence(c0, c1, c2)", input->type());
  const SelectivityVector rows(kBatchSize);
  {
    auto result = benchmark.evaluate(expression, input, rows);
    for (vector_size_t row = 0; row < kBatchSize; ++row) {
      VELOX_CHECK(result->equalValueAt(expected.get(), row, row));
    }
  }
  suspender.dismiss();
  for (unsigned iteration = 0; iteration < iterations; ++iteration) {
    auto result = benchmark.evaluate(expression, input, rows);
    folly::doNotOptimizeAway(result);
  }
  suspender.rehire();
}

template <typename T>
void runInteger(
    FunctionBenchmarkBase& benchmark,
    unsigned iterations,
    vector_size_t length,
    int32_t nullPercent,
    std::string_view encoding) {
  run<T, T>(
      benchmark,
      iterations,
      length,
      nullPercent,
      encoding,
      CppToType<T>::create(),
      CppToType<T>::create(),
      [](auto value) { return static_cast<T>(value); },
      T{1},
      [](T start, T step, auto index) {
        return static_cast<T>(start + step * index);
      });
}

void runType(
    FunctionBenchmarkBase& benchmark,
    unsigned iterations,
    std::string_view type,
    vector_size_t length,
    int32_t nullPercent,
    std::string_view encoding) {
  if (type == "bigint") {
    runInteger<int64_t>(benchmark, iterations, length, nullPercent, encoding);
  } else if (type == "integer") {
    runInteger<int32_t>(benchmark, iterations, length, nullPercent, encoding);
  } else if (type == "smallint") {
    runInteger<int16_t>(benchmark, iterations, length, nullPercent, encoding);
  } else if (type == "tinyint") {
    runInteger<int8_t>(benchmark, iterations, length, nullPercent, encoding);
  } else if (type == "date_days") {
    run<int32_t, int64_t>(
        benchmark,
        iterations,
        length,
        nullPercent,
        encoding,
        DATE(),
        INTERVAL_DAY_TIME(),
        [](auto value) { return 18'262 + value; },
        kMillisPerDay,
        [](auto start, auto step, auto index) {
          return start + step / kMillisPerDay * index;
        });
  } else if (type == "date_months") {
    run<int32_t, int32_t>(
        benchmark,
        iterations,
        length,
        nullPercent,
        encoding,
        DATE(),
        INTERVAL_YEAR_MONTH(),
        [](auto value) { return 18'262 + value; },
        1,
        [](auto start, auto step, auto index) {
          return addToDate(start, DateTimeUnit::kMonth, step * index);
        });
  } else if (type == "timestamp_millis") {
    run<Timestamp, int64_t>(
        benchmark,
        iterations,
        length,
        nullPercent,
        encoding,
        TIMESTAMP(),
        INTERVAL_DAY_TIME(),
        [](auto value) {
          return Timestamp::fromMillis((18'262 + value) * kMillisPerDay);
        },
        1'000,
        [](auto start, auto step, auto index) {
          return Timestamp::fromMillis(start.toMillis() + step * index);
        });
  } else if (type == "timestamp_months") {
    run<Timestamp, int32_t>(
        benchmark,
        iterations,
        length,
        nullPercent,
        encoding,
        TIMESTAMP(),
        INTERVAL_YEAR_MONTH(),
        [](auto value) {
          return Timestamp::fromMillis((18'262 + value) * kMillisPerDay);
        },
        1,
        [](auto start, auto step, auto index) {
          return addToTimestamp(start, DateTimeUnit::kMonth, step * index);
        });
  } else {
    VELOX_UNREACHABLE("Unsupported sequence benchmark type: {}", type);
  }
}
} // namespace

void addSequenceBenchmarks(
    const char* file,
    FunctionBenchmarkBase& benchmark,
    std::string_view dialect,
    std::initializer_list<std::string_view> types) {
  for (const auto type : types) {
    for (const auto length : {1, 8, 64}) {
      for (const auto nullPercent : {0, 50}) {
        for (const std::string encoding : {"flat", "dictionary"}) {
          folly::addBenchmark(
              file,
              fmt::format(
                  "{}_sequence_{}_len{}_null{}_{}",
                  dialect,
                  type,
                  length,
                  nullPercent,
                  encoding),
              [&benchmark,
               type = std::string(type),
               length,
               nullPercent,
               encoding](unsigned iterations) {
                runType(
                    benchmark, iterations, type, length, nullPercent, encoding);
                return iterations;
              });
        }
      }
    }
  }
}
} // namespace facebook::velox::functions::test
