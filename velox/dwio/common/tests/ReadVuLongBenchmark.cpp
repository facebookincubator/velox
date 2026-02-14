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

// Benchmark comparing old vs new readVuLong implementation.
// The "old" version uses UNLIKELY for the first byte termination check.
// The "new" version (in production IntDecoder.h) uses LIKELY and early return.

#include "folly/Benchmark.h"
#include "folly/Random.h"
#include "folly/Varint.h"
#include "folly/init/Init.h"
#include "velox/dwio/common/IntCodecCommon.h"

using namespace facebook::velox::dwio::common;

constexpr size_t kNumElements = 1000000;

// Encapsulates benchmark state to avoid global mutable variables.
struct BenchmarkState {
  std::vector<char> bufferSmall; // Values 0-127 (1-byte varints)
  std::vector<char> bufferMedium; // Values 0-16383 (1-2 byte varints)
  std::vector<char> bufferMixed; // Random uint32 (1-5 byte varints)

  size_t lenSmall{0};
  size_t lenMedium{0};
  size_t lenMixed{0};

  size_t numValuesSmall{0};
  size_t numValuesMedium{0};
  size_t numValuesMixed{0};

  static BenchmarkState& instance() {
    static BenchmarkState state;
    return state;
  }
};

// Helper to write a varint to buffer
size_t writeVulong(uint64_t val, char* buffer, size_t pos) {
  while (true) {
    if ((val & ~0x7f) == 0) {
      buffer[pos++] = static_cast<char>(val);
      return pos;
    }
    buffer[pos++] = static_cast<char>(0x80 | (val & BASE_128_MASK));
    val = (static_cast<uint64_t>(val) >> 7);
  }
}

// OLD implementation: uses UNLIKELY for first byte, no early return
// This is the original code before the optimization.
uint64_t readVuLongOld(const char*& bufferStart, const char* bufferEnd) {
  if (LIKELY(bufferEnd - bufferStart >= folly::kMaxVarintLength64)) {
    const char* p = bufferStart;
    uint64_t val;
    do {
      int64_t b;
      b = *p++;
      val = (b & 0x7f);
      if (UNLIKELY(b >= 0)) { // OLD: UNLIKELY here
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 7;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 14;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 21;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 28;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 35;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 42;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 49;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 56;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x01) << 63;
    } while (false);

    bufferStart = p;
    return val;
  }

  // Slow path
  uint64_t result = 0;
  uint64_t offset = 0;
  signed char ch;
  do {
    ch = *(bufferStart++);
    result |= (ch & BASE_128_MASK) << offset;
    offset += 7;
  } while (ch < 0);
  return result;
}

// NEW implementation: uses LIKELY for first byte with early return
// This matches the optimized code in IntDecoder.h
uint64_t readVuLongNew(const char*& bufferStart, const char* bufferEnd) {
  if (LIKELY(bufferEnd - bufferStart >= folly::kMaxVarintLength64)) {
    const char* p = bufferStart;
    uint64_t val;

    // Fast path for 1-byte varints (values 0-127), which are very common.
    // This avoids the do-while loop overhead for the most frequent case.
    int64_t b = *p++;
    val = (b & 0x7f);
    if (b >= 0) { // better without likely or unlikely here.
      bufferStart = p;
      return val;
    }

    // Multi-byte varint path
    do {
      b = *p++;
      val |= (b & 0x7f) << 7;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 14;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 21;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 28;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 35;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 42;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 49;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x7f) << 56;
      if (UNLIKELY(b >= 0)) {
        break;
      }
      b = *p++;
      val |= (b & 0x01) << 63;
    } while (false);

    bufferStart = p;
    return val;
  }

  // Slow path
  uint64_t result = 0;
  uint64_t offset = 0;
  signed char ch;
  do {
    ch = *(bufferStart++);
    result |= (ch & BASE_128_MASK) << offset;
    offset += 7;
  } while (ch < 0);
  return result;
}

// Benchmarks for small values (0-127, all 1-byte varints)
// This is where the optimization should show the biggest improvement.
BENCHMARK(small_values_old) {
  auto& state = BenchmarkState::instance();
  const char* p = state.bufferSmall.data();
  const char* end = p + state.lenSmall;
  for (size_t i = 0; i < state.numValuesSmall; ++i) {
    auto result = readVuLongOld(p, end);
    folly::doNotOptimizeAway(result);
  }
}

BENCHMARK_RELATIVE(small_values_new) {
  auto& state = BenchmarkState::instance();
  const char* p = state.bufferSmall.data();
  const char* end = p + state.lenSmall;
  for (size_t i = 0; i < state.numValuesSmall; ++i) {
    auto result = readVuLongNew(p, end);
    folly::doNotOptimizeAway(result);
  }
}

BENCHMARK_DRAW_LINE();

// Benchmarks for medium values (0-16383, mix of 1-2 byte varints)
BENCHMARK(medium_values_old) {
  auto& state = BenchmarkState::instance();
  const char* p = state.bufferMedium.data();
  const char* end = p + state.lenMedium;
  for (size_t i = 0; i < state.numValuesMedium; ++i) {
    auto result = readVuLongOld(p, end);
    folly::doNotOptimizeAway(result);
  }
}

BENCHMARK_RELATIVE(medium_values_new) {
  auto& state = BenchmarkState::instance();
  const char* p = state.bufferMedium.data();
  const char* end = p + state.lenMedium;
  for (size_t i = 0; i < state.numValuesMedium; ++i) {
    auto result = readVuLongNew(p, end);
    folly::doNotOptimizeAway(result);
  }
}

BENCHMARK_DRAW_LINE();

// Benchmarks for mixed/random uint32 values (1-5 byte varints)
BENCHMARK(mixed_values_old) {
  auto& state = BenchmarkState::instance();
  const char* p = state.bufferMixed.data();
  const char* end = p + state.lenMixed;
  for (size_t i = 0; i < state.numValuesMixed; ++i) {
    auto result = readVuLongOld(p, end);
    folly::doNotOptimizeAway(result);
  }
}

BENCHMARK_RELATIVE(mixed_values_new) {
  auto& state = BenchmarkState::instance();
  const char* p = state.bufferMixed.data();
  const char* end = p + state.lenMixed;
  for (size_t i = 0; i < state.numValuesMixed; ++i) {
    auto result = readVuLongNew(p, end);
    folly::doNotOptimizeAway(result);
  }
}

int32_t main(int32_t argc, char* argv[]) {
  folly::Init init{&argc, &argv};

  auto& state = BenchmarkState::instance();

  // Populate small values buffer (0-127, all 1-byte varints)
  state.bufferSmall.resize(kNumElements);
  size_t pos = 0;
  state.numValuesSmall = 500000;
  for (size_t i = 0; i < state.numValuesSmall; i++) {
    uint64_t val = folly::Random::rand32() % 128; // 0-127
    pos = writeVulong(val, state.bufferSmall.data(), pos);
  }
  state.lenSmall = pos;

  // Populate medium values buffer (0-16383, mix of 1-2 byte varints)
  state.bufferMedium.resize(kNumElements);
  pos = 0;
  state.numValuesMedium = 400000;
  for (size_t i = 0; i < state.numValuesMedium; i++) {
    uint64_t val = folly::Random::rand32() % 16384; // 0-16383
    pos = writeVulong(val, state.bufferMedium.data(), pos);
  }
  state.lenMedium = pos;

  // Populate mixed values buffer (random uint32, 1-5 byte varints)
  state.bufferMixed.resize(kNumElements);
  pos = 0;
  state.numValuesMixed = 200000;
  for (size_t i = 0; i < state.numValuesMixed; i++) {
    uint64_t val = folly::Random::rand32();
    pos = writeVulong(val, state.bufferMixed.data(), pos);
  }
  state.lenMixed = pos;

  folly::runBenchmarks();
  return 0;
}
