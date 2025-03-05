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

#include "folly/Benchmark.h"
#include "folly/Random.h"
#include "folly/init/Init.h"
#include "velox/type/Conversions.h"

std::vector<double> randSciDouble;
std::vector<float> randSciFloat;

namespace {
template <typename T>
std::string castScientificNotation(T value) {
  using namespace facebook::velox::util;
  using namespace facebook::velox;
  return Converter<TypeKind::VARCHAR, void, PrestoCastPolicy>::
      castScientificNotation(value);
}

template <typename T>
std::string castFloatFmt(T value) {
  return fmt::format("{}", value);
}

template <typename T>
std::string castFloatFmtExp(T value) {
  return fmt::format("{:E}", value);
}

template <typename T>
std::string castFloatFmtExpPrecision(T value) {
  return fmt::format(std::is_same_v<T, float> ? "{:.7E}" : "{:.16E}", value);
}
} // namespace

#define FLOAT_TO_STRING(TYPE, IMPL)                                          \
  do {                                                                       \
    for (auto it = randSci##TYPE.begin(); it != randSci##TYPE.end(); it++) { \
      folly::doNotOptimizeAway(IMPL(*it));                                   \
    }                                                                        \
  } while (false)

#define DEFINE_FLOAT_TO_SCIENTIFIC_BENCHMARKS(TYPE)  \
  BENCHMARK(TYPE##ToScientificNotation) {            \
    FLOAT_TO_STRING(TYPE, castScientificNotation);   \
  }                                                  \
  BENCHMARK_RELATIVE(TYPE##ToFmt) {                  \
    FLOAT_TO_STRING(TYPE, castFloatFmt);             \
  }                                                  \
  BENCHMARK_RELATIVE(TYPE##ToFmtExp) {               \
    FLOAT_TO_STRING(TYPE, castFloatFmtExp);          \
  }                                                  \
  BENCHMARK_RELATIVE(TYPE##ToFmtExpPrecision) {      \
    FLOAT_TO_STRING(TYPE, castFloatFmtExpPrecision); \
  }

DEFINE_FLOAT_TO_SCIENTIFIC_BENCHMARKS(Float)
DEFINE_FLOAT_TO_SCIENTIFIC_BENCHMARKS(Double)

template <typename T>
T genScientificNotationNumber() {
  // mantissa in (0.0, 100)
  T mantissa = folly::Random::randDouble(1e-37, 10.0);

  // exp in [7, maxExp)
  constexpr int maxExp = std::is_same_v<T, double> ? 308 : 38;
  int exponent;
  exponent = folly::Random::rand32() % (maxExp - 7) + 7;
  if (folly::Random::randBool(0.5)) {
    exponent = -exponent;
  }

  T value = mantissa * std::pow(10, exponent);
  return folly::Random::randBool(0.5) ? -value : value;
}

int32_t main(int32_t argc, char* argv[]) {
  folly::Init init{&argc, &argv};
  constexpr int32_t kNumValues = 1000000;

  randSciDouble.reserve(kNumValues);
  for (int i = 0; i < kNumValues; ++i) {
    randSciDouble.emplace_back(genScientificNotationNumber<double>());
  }

  randSciFloat.reserve(kNumValues);
  for (int i = 0; i < kNumValues; ++i) {
    randSciFloat.emplace_back(genScientificNotationNumber<float>());
  }

  folly::runBenchmarks();
}
