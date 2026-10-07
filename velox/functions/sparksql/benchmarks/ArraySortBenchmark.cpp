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

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/benchmarks/ExpressionBenchmarkBuilder.h"
#include "velox/common/base/VeloxException.h"
#include "velox/functions/sparksql/registration/Register.h"

namespace facebook::velox {
namespace {

constexpr vector_size_t kElementsPerBatch = 1 << 15;

enum class InputPattern {
  kAscending,
  kDescending,
  kRandom,
  kDuplicates,
};

std::string_view patternName(InputPattern pattern) {
  switch (pattern) {
    case InputPattern::kAscending:
      return "ascending";
    case InputPattern::kDescending:
      return "descending";
    case InputPattern::kRandom:
      return "random";
    case InputPattern::kDuplicates:
      return "duplicates";
  }
  VELOX_UNREACHABLE();
}

int32_t randomValue(vector_size_t row, vector_size_t position) {
  uint32_t value = static_cast<uint32_t>(row) * 0x9e3779b9U +
      static_cast<uint32_t>(position) * 0x85ebca6bU;
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  return static_cast<int32_t>(value & 0x7fffffffU);
}

RowVectorPtr makeInput(
    ExpressionBenchmarkBuilder& builder,
    vector_size_t arrayLength,
    InputPattern pattern) {
  const auto numRows =
      std::max<vector_size_t>(32, kElementsPerBatch / arrayLength);
  auto arrays = builder.vectorMaker().arrayVector<int32_t>(
      numRows,
      [=](vector_size_t /*row*/) { return arrayLength; },
      [=](vector_size_t row, vector_size_t position) {
        switch (pattern) {
          case InputPattern::kAscending:
            return static_cast<int32_t>(position);
          case InputPattern::kDescending:
            return static_cast<int32_t>(arrayLength - position);
          case InputPattern::kRandom: {
            const auto value = randomValue(row, position);
            return position % 2 == 0 ? value : -value;
          }
          case InputPattern::kDuplicates:
            return static_cast<int32_t>((row + position * 5) % 8) - 4;
        }
        VELOX_UNREACHABLE();
      });
  auto captures = builder.vectorMaker().flatVector<int32_t>(
      numRows,
      [](vector_size_t row) { return static_cast<int32_t>(row % 32) - 16; });
  return builder.vectorMaker().rowVector({arrays, captures});
}

void addDefaultOrderingBenchmarks(ExpressionBenchmarkBuilder& builder) {
  constexpr std::array<vector_size_t, 4> kArrayLengths = {8, 64, 256, 1024};
  constexpr std::array<InputPattern, 4> kPatterns = {
      InputPattern::kAscending,
      InputPattern::kDescending,
      InputPattern::kRandom,
      InputPattern::kDuplicates};

  for (const auto arrayLength : kArrayLengths) {
    for (const auto pattern : kPatterns) {
      builder
          .addBenchmarkSet(
              fmt::format(
                  "array_sort_default_{}_length{}",
                  patternName(pattern),
                  arrayLength),
              makeInput(builder, arrayLength, pattern))
          .withIterations(20)
          .addExpression("default", "array_sort(c0)");
    }
  }
}

void addComparatorBenchmarks(ExpressionBenchmarkBuilder& builder) {
  constexpr std::array<vector_size_t, 4> kArrayLengths = {8, 64, 256, 1024};

  for (const auto arrayLength : kArrayLengths) {
    builder
        .addBenchmarkSet(
            fmt::format("array_sort_comparators_length{}", arrayLength),
            makeInput(builder, arrayLength, InputPattern::kRandom))
        .withIterations(20)
        .disableTesting()
        .addExpression(
            "identityStableComparator",
            "array_sort(c0, (x, y) -> "
            "if(lessthan(x, y), (-10)::integer, "
            "if(greaterthan(x, y), 37::integer, 0::integer)))")
        .addExpression(
            "descending",
            "array_sort(c0, (x, y) -> "
            "if(greaterthan(x, y), (-10)::integer, "
            "if(lessthan(x, y), 37::integer, 0::integer)))")
        .addExpression(
            "absoluteValue",
            "array_sort(c0, (x, y) -> "
            "if(lessthan(abs(x), abs(y)), (-10)::integer, "
            "if(greaterthan(abs(x), abs(y)), 37::integer, 0::integer)))")
        .addExpression(
            "equalityFirst",
            "array_sort(c0, (x, y) -> "
            "if(equalto(abs(x), abs(y)), 0::integer, "
            "if(lessthan(abs(x), abs(y)), (-10)::integer, 37::integer)))")
        .addExpression(
            "caseWhenNormalized",
            "array_sort(c0, (x, y) -> "
            "if(lessthan(abs(x), abs(y)), (-1)::integer, "
            "if(greaterthan(abs(x), abs(y)), 1::integer, 0::integer)))")
        .addExpression(
            "capturedTransform",
            "array_sort(c0, (x, y) -> "
            "if(lessthan(greatest(x, c1), greatest(y, c1)), (-1)::integer, "
            "if(greaterthan(greatest(x, c1), greatest(y, c1)), "
            "1::integer, 0::integer)))");
  }
}

} // namespace
} // namespace facebook::velox

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  facebook::velox::functions::sparksql::registerFunctions("");

  facebook::velox::ExpressionBenchmarkBuilder builder;
  facebook::velox::addDefaultOrderingBenchmarks(builder);
  facebook::velox::addComparatorBenchmarks(builder);
  builder.registerBenchmarks();
  builder.testBenchmarks();
  folly::runBenchmarks();
  return 0;
}
