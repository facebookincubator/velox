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
#include <folly/Random.h>
#include <folly/hash/detail/ChecksumDetail.h>
#include <array>
#include <iostream>
#include <random>
#include <vector>

#if defined(_MSC_VER) // Microsoft Visual Studio
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__) // GCC or Clang
#include <cpuid.h>
#endif

// Add the following definitions to allow Clion runs
DEFINE_bool(gtest_color, false, "");
DEFINE_string(gtest_filter, "*", "");

namespace {

bool isSSE42Supported() {
#if defined(_MSC_VER) // Microsoft Visual Studio
  int cpuInfo[4] = {0};
  __cpuid(cpuInfo, 1); // Query CPU features with EAX = 1
  return (cpuInfo[2] & (1 << 20)) != 0; // Check bit 20 in ECX

#elif defined(__GNUC__) || defined(__clang__) // GCC or Clang
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return false; // cpuid not supported
  }
  return (ecx & (1 << 20)) != 0; // Check bit 20 in ECX
#else
  return false;
#endif
}

std::vector<uint8_t> generateRandomData(size_t size) {
  std::vector<uint8_t> data(size);
  std::mt19937_64 rng(folly::Random::rand64());
  std::uniform_int_distribution<uint8_t> dist(0, 255);

  for (size_t i = 0; i < size; ++i) {
    data[i] = dist(rng);
  }

  return data;
}

void benchmarkCRC32Software(uint32_t dataSize) {
  std::vector<uint8_t> data;
  BENCHMARK_SUSPEND {
    data = generateRandomData(dataSize);
  }

  uint32_t checksum = folly::detail::crc32_sw(data.data(), data.size());
  folly::doNotOptimizeAway(checksum);
}

void benchmarkCRC32CSoftware(uint32_t dataSize) {
  std::vector<uint8_t> data;
  BENCHMARK_SUSPEND {
    data = generateRandomData(dataSize);
  }

  uint32_t checksum = folly::detail::crc32c_sw(data.data(), data.size());
  folly::doNotOptimizeAway(checksum);
}

void benchmarkCRC32Hardware(uint32_t dataSize) {
  std::vector<uint8_t> data;
  BENCHMARK_SUSPEND {
    data = generateRandomData(dataSize);
  }

  uint32_t checksum = folly::detail::crc32_hw(data.data(), data.size());
  folly::doNotOptimizeAway(checksum);
}

void benchmarkCRC32CHardware(uint32_t dataSize) {
  std::vector<uint8_t> data;
  BENCHMARK_SUSPEND {
    data = generateRandomData(dataSize);
  }

  uint32_t checksum = folly::detail::crc32c_hw(data.data(), data.size());
  folly::doNotOptimizeAway(checksum);
}

} // namespace

BENCHMARK(CRC32_SW) {
  benchmarkCRC32Software(1048576);
}

BENCHMARK(CRC32C_SW) {
  benchmarkCRC32CSoftware(1048576);
}

BENCHMARK(CRC32_HW) {
  if (isSSE42Supported()) {
    benchmarkCRC32Hardware(1048576);
  }
}

BENCHMARK(CRC32C_HW) {
  if (isSSE42Supported()) {
    benchmarkCRC32CHardware(1048576);
  }
}

int main(int argc, char** argv) {
  folly::runBenchmarks();

  if (!isSSE42Supported()) {
    std::cerr
        << "Hardware CRC32 not supported on this platform. Discard the results of CRC32_HW and CRC32C_HW."
        << std::endl;
  }

  return 0;
}
