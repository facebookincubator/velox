/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "folly/Benchmark.h"
#include "folly/Random.h"
#include "folly/init/Init.h"
#include "velox/dwio/common/Lemire/BitPacking/bitpackinghelpers.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"

using namespace facebook::nimble;

namespace {

constexpr uint32_t kGroupSize = SimdForBitpackEncoding<uint32_t>::kGroupSize;
constexpr uint32_t kGroups = 8192;

template <typename T>
std::vector<std::array<T, kGroupSize>> makeResidualGroups(uint32_t bitWidth) {
  std::vector<std::array<T, kGroupSize>> groups(kGroups);
  const uint32_t mask =
      bitWidth == 32 ? ~uint32_t{0} : ((uint32_t{1} << bitWidth) - 1);
  for (auto& group : groups) {
    for (auto& value : group) {
      value = static_cast<T>(folly::Random::secureRand32() & mask);
    }
  }
  return groups;
}

template <typename T>
std::vector<uint32_t> packWithTemp(
    const std::vector<std::array<T, kGroupSize>>& groups,
    uint32_t bitWidth) {
  std::vector<uint32_t> packed(groups.size() * bitWidth);
  auto* out = packed.data();
  for (const auto& group : groups) {
    std::array<uint32_t, kGroupSize> widened{};
    for (uint32_t i = 0; i < kGroupSize; ++i) {
      widened[i] = static_cast<uint32_t>(group[i]);
    }
    facebook::velox::fastpforlib::fastpack(widened.data(), out, bitWidth);
    out += bitWidth;
  }
  return packed;
}

template <typename T>
std::vector<uint32_t> packNative(
    const std::vector<std::array<T, kGroupSize>>& groups,
    uint32_t bitWidth) {
  std::vector<uint32_t> packed(groups.size() * bitWidth);
  auto* out = packed.data();
  for (const auto& group : groups) {
    if constexpr (sizeof(T) == 1) {
      facebook::velox::fastpforlib::fastpack(
          reinterpret_cast<const uint8_t*>(group.data()),
          reinterpret_cast<uint8_t*>(out),
          bitWidth);
    } else {
      static_assert(sizeof(T) == 2);
      facebook::velox::fastpforlib::fastpack(
          reinterpret_cast<const uint16_t*>(group.data()),
          reinterpret_cast<uint16_t*>(out),
          bitWidth);
    }
    out += bitWidth;
  }
  return packed;
}

template <typename T>
void unpackWithTemp(
    const std::vector<uint32_t>& packed,
    std::array<T, kGroupSize>& output,
    uint32_t bitWidth) {
  const auto* in = packed.data();
  for (uint32_t group = 0; group < kGroups; ++group) {
    std::array<uint32_t, kGroupSize> temp{};
    facebook::velox::fastpforlib::fastunpack(in, temp.data(), bitWidth);
    for (uint32_t i = 0; i < kGroupSize; ++i) {
      output[i] = static_cast<T>(temp[i]);
    }
    folly::doNotOptimizeAway(output);
    in += bitWidth;
  }
}

template <typename T>
void unpackNative(
    const std::vector<uint32_t>& packed,
    std::array<T, kGroupSize>& output,
    uint32_t bitWidth) {
  const auto* in = packed.data();
  for (uint32_t group = 0; group < kGroups; ++group) {
    if constexpr (sizeof(T) == 1) {
      facebook::velox::fastpforlib::fastunpack(
          reinterpret_cast<const uint8_t*>(in),
          reinterpret_cast<uint8_t*>(output.data()),
          bitWidth);
    } else {
      static_assert(sizeof(T) == 2);
      facebook::velox::fastpforlib::fastunpack(
          reinterpret_cast<const uint16_t*>(in),
          reinterpret_cast<uint16_t*>(output.data()),
          bitWidth);
    }
    folly::doNotOptimizeAway(output);
    in += bitWidth;
  }
}

template <typename T>
void runUnpackTemp(uint32_t bitWidth, uint32_t iters) {
  std::vector<uint32_t> packed;
  BENCHMARK_SUSPEND {
    packed = packWithTemp(makeResidualGroups<T>(bitWidth), bitWidth);
  }
  std::array<T, kGroupSize> output{};
  while (iters--) {
    unpackWithTemp(packed, output, bitWidth);
  }
}

template <typename T>
void runUnpackNative(uint32_t bitWidth, uint32_t iters) {
  std::vector<uint32_t> packed;
  BENCHMARK_SUSPEND {
    packed = packNative(makeResidualGroups<T>(bitWidth), bitWidth);
  }
  std::array<T, kGroupSize> output{};
  while (iters--) {
    unpackNative(packed, output, bitWidth);
  }
}

template <typename T>
void runPackTemp(uint32_t bitWidth, uint32_t iters) {
  std::vector<std::array<T, kGroupSize>> groups;
  BENCHMARK_SUSPEND {
    groups = makeResidualGroups<T>(bitWidth);
  }
  while (iters--) {
    auto packed = packWithTemp(groups, bitWidth);
    folly::doNotOptimizeAway(packed);
  }
}

template <typename T>
void runPackNative(uint32_t bitWidth, uint32_t iters) {
  std::vector<std::array<T, kGroupSize>> groups;
  BENCHMARK_SUSPEND {
    groups = makeResidualGroups<T>(bitWidth);
  }
  while (iters--) {
    auto packed = packNative(groups, bitWidth);
    folly::doNotOptimizeAway(packed);
  }
}

#define SIMD_FOR_NARROW_BENCHMARKS(Type, Width)                                \
  BENCHMARK(SimdForBitpack_UnpackTemp_##Type##_##Width##bit, iters) {          \
    runUnpackTemp<Type>(Width, iters);                                         \
  }                                                                            \
  BENCHMARK_RELATIVE(                                                          \
      SimdForBitpack_UnpackNative_##Type##_##Width##bit, iters) {              \
    runUnpackNative<Type>(Width, iters);                                       \
  }                                                                            \
  BENCHMARK(SimdForBitpack_PackTemp_##Type##_##Width##bit, iters) {            \
    runPackTemp<Type>(Width, iters);                                           \
  }                                                                            \
  BENCHMARK_RELATIVE(SimdForBitpack_PackNative_##Type##_##Width##bit, iters) { \
    runPackNative<Type>(Width, iters);                                         \
  }

SIMD_FOR_NARROW_BENCHMARKS(uint8_t, 4)
SIMD_FOR_NARROW_BENCHMARKS(uint8_t, 8)
SIMD_FOR_NARROW_BENCHMARKS(uint16_t, 8)
SIMD_FOR_NARROW_BENCHMARKS(uint16_t, 12)
SIMD_FOR_NARROW_BENCHMARKS(uint16_t, 16)

} // namespace

int main(int argc, char** argv) {
  const folly::Init init{&argc, &argv};
  folly::runBenchmarks();
  return 0;
}
