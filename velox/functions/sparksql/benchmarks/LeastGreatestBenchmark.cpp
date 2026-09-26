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
#include <optional>
#include <random>
#include <type_traits>

#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/sparksql/registration/Register.h"

using namespace facebook::velox;

namespace {

using functions::test::FunctionBenchmarkBase;
constexpr vector_size_t kBatchSize = 10'240;

// One iteration evaluates a batch and releases its result. Input generation,
// expression compilation and value validation are excluded from timing.
template <typename T>
void run(
    FunctionBenchmarkBase& benchmark,
    unsigned iterations,
    const std::string& function,
    int32_t numArgs,
    int32_t nullPercent,
    std::string_view encoding,
    int32_t stringLength) {
  folly::BenchmarkSuspender suspender;
  std::mt19937 random(20260926);
  std::vector<std::vector<std::optional<T>>> values(
      numArgs, std::vector<std::optional<T>>(kBatchSize));
  std::vector<VectorPtr> arguments;
  std::string expression = function + "(";
  for (int32_t argument = 0; argument < numArgs; ++argument) {
    for (auto& value : values[argument]) {
      if (random() % 100 < nullPercent) {
        continue;
      }
      if constexpr (std::is_same_v<T, std::string>) {
        std::string text(stringLength, 'a');
        for (auto& character : text) {
          character += random() % 26;
        }
        value = std::move(text);
      } else if constexpr (std::is_same_v<T, bool>) {
        value = random() % 2 != 0;
      } else {
        value = static_cast<T>(static_cast<int32_t>(random() % 2'001) - 1'000);
      }
    }
    arguments.push_back(benchmark.maker().flatVectorNullable(values[argument]));
    expression += fmt::format("{}c{}", argument == 0 ? "" : ",", argument);
  }
  expression += ")";

  if (encoding == "constant") {
    arguments[0] = BaseVector::wrapInConstant(kBatchSize, 17, arguments[0]);
  } else if (encoding == "dictionary") {
    auto indices =
        AlignedBuffer::allocate<vector_size_t>(kBatchSize, benchmark.pool());
    auto* rawIndices = indices->asMutable<vector_size_t>();
    for (vector_size_t row = 0; row < kBatchSize; ++row) {
      rawIndices[row] = kBatchSize - row - 1;
    }
    arguments[0] = BaseVector::wrapInDictionary(
        nullptr, indices, kBatchSize, arguments[0]);
  }

  std::vector<std::optional<T>> expected(kBatchSize);
  for (vector_size_t row = 0; row < kBatchSize; ++row) {
    for (int32_t argument = 0; argument < numArgs; ++argument) {
      const auto index = argument != 0 || encoding == "flat"
          ? row
          : (encoding == "constant" ? 17 : kBatchSize - row - 1);
      const auto& candidate = values[argument][index];
      if (candidate &&
          (!expected[row] ||
           (function == "least" ? *candidate < *expected[row]
                                : *candidate > *expected[row]))) {
        expected[row] = candidate;
      }
    }
  }
  auto expectedVector = benchmark.maker().flatVectorNullable(expected);
  auto input = benchmark.maker().rowVector(arguments);
  auto expressionSet = benchmark.compileExpression(expression, input->type());
  const SelectivityVector rows(kBatchSize);
  {
    auto result = benchmark.evaluate(expressionSet, input, rows);
    for (vector_size_t row = 0; row < kBatchSize; ++row) {
      VELOX_CHECK(result->equalValueAt(expectedVector.get(), row, row));
    }
  }

  suspender.dismiss();
  for (unsigned iteration = 0; iteration < iterations; ++iteration) {
    auto result = benchmark.evaluate(expressionSet, input, rows);
    folly::doNotOptimizeAway(result);
  }
  suspender.rehire();
}

template <typename T>
void addBenchmarks(
    FunctionBenchmarkBase& benchmark,
    std::string_view type,
    int32_t stringLength = 0) {
  for (const std::string function : {"least", "greatest"}) {
    for (const auto numArgs : {2, 4, 8}) {
      for (const auto nullPercent : {0, 50}) {
        for (const std::string encoding : {"flat", "constant", "dictionary"}) {
          folly::addBenchmark(
              __FILE__,
              fmt::format(
                  "spark_{}_{}_args{}_null{}_{}",
                  function,
                  type,
                  numArgs,
                  nullPercent,
                  encoding),
              [&benchmark,
               function,
               numArgs,
               nullPercent,
               encoding,
               stringLength](unsigned iterations) {
                run<T>(
                    benchmark,
                    iterations,
                    function,
                    numArgs,
                    nullPercent,
                    encoding,
                    stringLength);
                return iterations;
              });
        }
      }
    }
  }
}
} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::sparksql::registerFunctions("");
  FunctionBenchmarkBase benchmark;
  addBenchmarks<bool>(benchmark, "boolean");
  addBenchmarks<int64_t>(benchmark, "bigint");
  addBenchmarks<double>(benchmark, "double");
  addBenchmarks<std::string>(benchmark, "varchar8", 8);
  addBenchmarks<std::string>(benchmark, "varchar64", 64);
  folly::runBenchmarks();
  return 0;
}
