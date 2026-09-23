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
#include <cstring>
#include <memory>
#include <vector>

#include "folly/Benchmark.h"
#include "folly/init/Init.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"

using namespace ::facebook;

namespace {

constexpr uint64_t kNumElements = 1000 * 1000;
constexpr uint64_t kBaseline = 12345;
constexpr uint64_t kStartOffset = 37;

uint64_t maskForBitWidth(int bitWidth) {
  return bitWidth == 64 ? ~0ULL : ((1ULL << bitWidth) - 1);
}

std::vector<uint64_t> makeValues(int bitWidth) {
  const uint64_t mask = maskForBitWidth(bitWidth);
  const uint64_t baseline = bitWidth == 64 ? 0 : kBaseline;
  std::vector<uint64_t> values(kNumElements);
  for (uint64_t i = 0; i < kNumElements; ++i) {
    const uint64_t residual = (i * 1000003ULL) & mask;
    values[i] = residual + baseline;
  }
  return values;
}

void clearBuffer(char* _Nonnull buffer, uint64_t bufferBytes) {
  std::memset(buffer, 0, bufferBytes);
}

void observeWrittenBuffer(
    const char* buffer,
    uint64_t elementCount,
    int bitWidth) {
  const uint64_t writtenBytes =
      ((elementCount * static_cast<uint64_t>(bitWidth)) + 7) >> 3;
  folly::doNotOptimizeAway(buffer[writtenBytes - 1]);
}

#define FIXED_BIT_ARRAY_BENCHMARKS(bitWidth)                                   \
  BENCHMARK(BulkSet64WithBaseline_##bitWidth, iters) {                         \
    std::vector<uint64_t> values;                                              \
    std::unique_ptr<char[]> buffer;                                            \
    uint64_t bufferBytes;                                                      \
    BENCHMARK_SUSPEND {                                                        \
      values = makeValues(bitWidth);                                           \
      bufferBytes = nimble::FixedBitArray::bufferSize(kNumElements, bitWidth); \
      buffer = std::make_unique<char[]>(bufferBytes);                          \
    }                                                                          \
    nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);               \
    const uint64_t baseline = bitWidth == 64 ? 0 : kBaseline;                  \
    while (iters--) {                                                          \
      BENCHMARK_SUSPEND {                                                      \
        clearBuffer(buffer.get(), bufferBytes);                                \
      }                                                                        \
      fixedBitArray.bulkSetWithBaseline(                                       \
          0, kNumElements, values.data(), baseline);                           \
      observeWrittenBuffer(buffer.get(), kNumElements, bitWidth);              \
    }                                                                          \
  }                                                                            \
  BENCHMARK_RELATIVE(ScalarSetWithBaseline_##bitWidth, iters) {                \
    std::vector<uint64_t> values;                                              \
    std::unique_ptr<char[]> buffer;                                            \
    uint64_t bufferBytes;                                                      \
    BENCHMARK_SUSPEND {                                                        \
      values = makeValues(bitWidth);                                           \
      bufferBytes = nimble::FixedBitArray::bufferSize(kNumElements, bitWidth); \
      buffer = std::make_unique<char[]>(bufferBytes);                          \
    }                                                                          \
    nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);               \
    const uint64_t baseline = bitWidth == 64 ? 0 : kBaseline;                  \
    while (iters--) {                                                          \
      BENCHMARK_SUSPEND {                                                      \
        clearBuffer(buffer.get(), bufferBytes);                                \
      }                                                                        \
      const uint64_t* nextValue = values.data();                               \
      for (uint64_t i = 0; i < kNumElements; ++i) {                            \
        fixedBitArray.set(i, *nextValue - baseline);                           \
        ++nextValue;                                                           \
      }                                                                        \
      observeWrittenBuffer(buffer.get(), kNumElements, bitWidth);              \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BulkGet64WithBaseline_##bitWidth, iters) {                         \
    std::vector<uint64_t> values;                                              \
    std::unique_ptr<char[]> buffer;                                            \
    std::vector<uint64_t> output;                                              \
    BENCHMARK_SUSPEND {                                                        \
      values = makeValues(bitWidth);                                           \
      const uint64_t bufferBytes =                                             \
          nimble::FixedBitArray::bufferSize(kNumElements, bitWidth);           \
      buffer = std::make_unique<char[]>(bufferBytes);                          \
      clearBuffer(buffer.get(), bufferBytes);                                  \
      output.resize(kNumElements);                                             \
      nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);             \
      const uint64_t baseline = bitWidth == 64 ? 0 : kBaseline;                \
      fixedBitArray.bulkSetWithBaseline(                                       \
          0, kNumElements, values.data(), baseline);                           \
    }                                                                          \
    nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);               \
    const uint64_t baseline = bitWidth == 64 ? 0 : kBaseline;                  \
    while (iters--) {                                                          \
      fixedBitArray.bulkGetWithBaseline(                                       \
          0, kNumElements, output.data(), baseline);                           \
      folly::doNotOptimizeAway(*(output.data() + kNumElements - 1));           \
    }                                                                          \
  }                                                                            \
  BENCHMARK_RELATIVE(ScalarGetWithBaseline_##bitWidth, iters) {                \
    std::vector<uint64_t> values;                                              \
    std::unique_ptr<char[]> buffer;                                            \
    std::vector<uint64_t> output;                                              \
    BENCHMARK_SUSPEND {                                                        \
      values = makeValues(bitWidth);                                           \
      const uint64_t bufferBytes =                                             \
          nimble::FixedBitArray::bufferSize(kNumElements, bitWidth);           \
      buffer = std::make_unique<char[]>(bufferBytes);                          \
      clearBuffer(buffer.get(), bufferBytes);                                  \
      output.resize(kNumElements);                                             \
      nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);             \
      const uint64_t baseline = bitWidth == 64 ? 0 : kBaseline;                \
      fixedBitArray.bulkSetWithBaseline(                                       \
          0, kNumElements, values.data(), baseline);                           \
    }                                                                          \
    nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);               \
    const uint64_t baseline = bitWidth == 64 ? 0 : kBaseline;                  \
    while (iters--) {                                                          \
      uint64_t* nextOutput = output.data();                                    \
      for (uint64_t i = 0; i < kNumElements; ++i) {                            \
        *nextOutput = fixedBitArray.get(i) + baseline;                         \
        ++nextOutput;                                                          \
      }                                                                        \
      folly::doNotOptimizeAway(*(output.data() + kNumElements - 1));           \
    }                                                                          \
  }

#define FIXED_BIT_ARRAY_OFFSET_SET_BENCHMARKS(bitWidth)               \
  BENCHMARK(BulkSet64WithBaselineOffset_##bitWidth, iters) {          \
    std::vector<uint64_t> values;                                     \
    std::unique_ptr<char[]> buffer;                                   \
    uint64_t bufferBytes;                                             \
    BENCHMARK_SUSPEND {                                               \
      values = makeValues(bitWidth);                                  \
      bufferBytes = nimble::FixedBitArray::bufferSize(                \
          kStartOffset + kNumElements, bitWidth);                     \
      buffer = std::make_unique<char[]>(bufferBytes);                 \
    }                                                                 \
    nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);      \
    const uint64_t baseline = bitWidth == 64 ? 0 : kBaseline;         \
    while (iters--) {                                                 \
      BENCHMARK_SUSPEND {                                             \
        clearBuffer(buffer.get(), bufferBytes);                       \
      }                                                               \
      fixedBitArray.bulkSetWithBaseline(                              \
          kStartOffset, kNumElements, values.data(), baseline);       \
      observeWrittenBuffer(                                           \
          buffer.get(), kStartOffset + kNumElements, bitWidth);       \
    }                                                                 \
  }                                                                   \
  BENCHMARK_RELATIVE(ScalarSetWithBaselineOffset_##bitWidth, iters) { \
    std::vector<uint64_t> values;                                     \
    std::unique_ptr<char[]> buffer;                                   \
    uint64_t bufferBytes;                                             \
    BENCHMARK_SUSPEND {                                               \
      values = makeValues(bitWidth);                                  \
      bufferBytes = nimble::FixedBitArray::bufferSize(                \
          kStartOffset + kNumElements, bitWidth);                     \
      buffer = std::make_unique<char[]>(bufferBytes);                 \
    }                                                                 \
    nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);      \
    const uint64_t baseline = bitWidth == 64 ? 0 : kBaseline;         \
    while (iters--) {                                                 \
      BENCHMARK_SUSPEND {                                             \
        clearBuffer(buffer.get(), bufferBytes);                       \
      }                                                               \
      const uint64_t* nextValue = values.data();                      \
      for (uint64_t i = 0; i < kNumElements; ++i) {                   \
        fixedBitArray.set(kStartOffset + i, *nextValue - baseline);   \
        ++nextValue;                                                  \
      }                                                               \
      observeWrittenBuffer(                                           \
          buffer.get(), kStartOffset + kNumElements, bitWidth);       \
    }                                                                 \
  }

FIXED_BIT_ARRAY_BENCHMARKS(16)
FIXED_BIT_ARRAY_BENCHMARKS(32)
FIXED_BIT_ARRAY_BENCHMARKS(33)
FIXED_BIT_ARRAY_BENCHMARKS(40)
FIXED_BIT_ARRAY_BENCHMARKS(48)
FIXED_BIT_ARRAY_BENCHMARKS(56)
FIXED_BIT_ARRAY_BENCHMARKS(57)
FIXED_BIT_ARRAY_BENCHMARKS(58)
FIXED_BIT_ARRAY_BENCHMARKS(60)
FIXED_BIT_ARRAY_BENCHMARKS(64)

FIXED_BIT_ARRAY_OFFSET_SET_BENCHMARKS(32)
FIXED_BIT_ARRAY_OFFSET_SET_BENCHMARKS(40)
FIXED_BIT_ARRAY_OFFSET_SET_BENCHMARKS(48)
FIXED_BIT_ARRAY_OFFSET_SET_BENCHMARKS(56)
FIXED_BIT_ARRAY_OFFSET_SET_BENCHMARKS(64)

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
}
