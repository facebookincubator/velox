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
#pragma once

#include <folly/Benchmark.h>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"

namespace facebook::velox::functions::test {

// One iteration evaluates 10,240 rows, including result allocation and
// release. Input generation, compilation and value validation are not timed.
// Length includes two spaces on trimmed rows (one at each end for trim).
// "all" trims every row, "half" trims odd rows, and "spaces" has no body.
inline void runTrimBenchmark(
    FunctionBenchmarkBase& benchmark,
    unsigned iterations,
    const std::string& function,
    vector_size_t length,
    std::string_view pattern) {
  folly::BenchmarkSuspender suspender;
  constexpr vector_size_t kBatchSize = 10'240;
  std::mt19937 random(20260912);
  const bool left = function != "rtrim";
  const bool right = function != "ltrim";
  const auto numLeadingSpaces = left ? (right ? 1 : 2) : 0;
  const auto numTrailingSpaces = right ? (left ? 1 : 2) : 0;
  std::vector<std::optional<std::string>> expected(kBatchSize);
  auto input = benchmark.maker().flatVector<std::string>(
      kBatchSize,
      [&](vector_size_t row) {
        const bool trim = pattern == "all" ||
            (pattern == "half" && row % 2 == 1) ||
            (pattern == "null50" && row % 4 == 3);
        std::string value(length, ' ');
        if (pattern != "spaces") {
          for (auto& character : value) {
            character = 'a' + random() % 26;
          }
        }
        if (trim) {
          value.replace(0, numLeadingSpaces, numLeadingSpaces, ' ');
          value.replace(
              length - numTrailingSpaces,
              numTrailingSpaces,
              numTrailingSpaces,
              ' ');
        }
        expected[row] = pattern == "spaces"
            ? ""
            : (trim ? value.substr(numLeadingSpaces, length - 2) : value);
        return value;
      },
      // NULL50 interleaves NULL, unchanged, NULL, trimmed rows.
      [&](vector_size_t row) { return pattern == "null50" && row % 2 == 0; });
  auto rowVector = benchmark.maker().rowVector({input});
  auto exprSet = benchmark.compileExpression(
      fmt::format("{}(c0)", function), rowVector->type());
  SelectivityVector rows(kBatchSize);
  int64_t expectedTotalLength = 0;
  {
    auto result = benchmark.evaluate(exprSet, rowVector, rows);
    const auto* strings = result->asChecked<FlatVector<StringView>>();
    for (vector_size_t row = 0; row < kBatchSize; ++row) {
      VELOX_CHECK_EQ(strings->isNullAt(row), !expected[row].has_value());
      if (expected[row]) {
        VELOX_CHECK_EQ(strings->valueAt(row), StringView(*expected[row]));
        expectedTotalLength += static_cast<int64_t>(expected[row]->size());
      } else {
        --expectedTotalLength;
      }
    }
  }

  suspender.dismiss();
  int64_t totalLength = 0;
  for (unsigned i = 0; i < iterations; ++i) {
    auto result = benchmark.evaluate(exprSet, rowVector, rows);
    const auto* strings = result->asChecked<FlatVector<StringView>>();
    for (vector_size_t row = 0; row < kBatchSize; ++row) {
      totalLength += strings->isNullAt(row)
          ? -1
          : static_cast<int64_t>(strings->valueAt(row).size());
    }
    folly::doNotOptimizeAway(totalLength);
  }
  suspender.rehire();
  VELOX_CHECK_EQ(totalLength, expectedTotalLength * iterations);
}

inline void addTrimBenchmarks(
    const char* file,
    FunctionBenchmarkBase& benchmark,
    const std::string& dialect) {
  for (const std::string function : {"rtrim", "ltrim", "trim"}) {
    const auto add = [&](vector_size_t length, const std::string& pattern) {
      folly::addBenchmark(
          file,
          fmt::format("trim_{}_{}_l{}_{}", dialect, function, length, pattern),
          [&benchmark, function, length, pattern](unsigned iterations) {
            runTrimBenchmark(benchmark, iterations, function, length, pattern);
            return iterations;
          });
    };
    for (auto length : {10, 12, 13, 64, 256}) {
      for (const auto* pattern : {"none", "all", "half"}) {
        add(length, pattern);
      }
    }
    add(64, "null50");
    add(0, "empty");
    add(64, "spaces");
  }
}

} // namespace facebook::velox::functions::test
