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

// Micro-benchmark for the Parquet BooleanDecoder.
// Bypasses the full reader pipeline to isolate decoder performance.
//
// Decoders benchmarked:
//   - BooleanDecoder (PLAIN boolean, bit-by-bit)

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/parquet/reader/BooleanDecoder.h"

using namespace facebook::velox;
using namespace facebook::velox::parquet;

namespace {

// ---------------------------------------------------------------------------
// Encoder helpers -- produce encoded buffers to feed into decoders.
// ---------------------------------------------------------------------------

// Encode PLAIN booleans (LSB-first bit-packing within each byte).
std::vector<char> encodePlainBooleans(const std::vector<bool>& values) {
  int numBytes = (values.size() + 7) / 8;
  std::vector<char> buf(numBytes + 16, 0); // extra padding
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i]) {
      buf[i / 8] |= static_cast<char>(1 << (i % 8));
    }
  }
  return buf;
}

// Simple no-filter, dense, no-hook visitor for benchmarking.
// Collects decoded values into an output buffer. Only implements the members
// the decoders' scalar (signed char) path exercises.
template <typename T>
class BenchmarkVisitor {
 public:
  using DataType = T;
  static constexpr bool dense = true;

  BenchmarkVisitor(T* output, int32_t numRows)
      : output_(output), numRows_(numRows) {}

  bool allowNulls() const {
    return false;
  }

  int32_t start() {
    return 0;
  }

  int32_t checkAndSkipNulls(const uint64_t*, int32_t&, bool&) {
    return 0;
  }

  int32_t processNull(bool& atEnd) {
    atEnd = (++outputIdx_ >= numRows_);
    return 0;
  }

  int32_t process(T value, bool& atEnd) {
    output_[outputIdx_++] = value;
    atEnd = (outputIdx_ >= numRows_);
    return 0;
  }

 private:
  T* output_;
  int32_t numRows_;
  int32_t outputIdx_ = 0;
};

constexpr int kBenchNumValues = 100'000;

} // namespace

// ===========================================================================
// BooleanDecoder benchmark
//
// Baseline for the batch boolean decoding optimization.
// Measures per-bit decode overhead.
// ===========================================================================

BENCHMARK(BooleanDecoder_100k) {
  folly::BenchmarkSuspender suspender;

  std::vector<bool> values(kBenchNumValues);
  for (int i = 0; i < kBenchNumValues; ++i) {
    values[i] = (i % 3 != 0); // ~67% true
  }
  auto encoded = encodePlainBooleans(values);
  std::vector<int8_t> output(kBenchNumValues);
  auto visitor = BenchmarkVisitor<int8_t>(output.data(), kBenchNumValues);

  suspender.dismiss();

  BooleanDecoder decoder(encoded.data(), encoded.data() + encoded.size());
  decoder.readWithVisitor<false>(nullptr, visitor);

  suspender.rehire();

  // Consume every decoded value so the optimizer cannot discard the decode
  // work or the per-bit extraction. Excluded from the timing via rehire().
  int64_t checksum = 0;
  for (auto value : output) {
    checksum += value;
  }
  folly::doNotOptimizeAway(checksum);
}

BENCHMARK_DRAW_LINE();

// ===========================================================================
// Main
// ===========================================================================

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
