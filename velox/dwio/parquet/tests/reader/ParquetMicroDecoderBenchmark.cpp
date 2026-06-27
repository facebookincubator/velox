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

// Micro-benchmarks for individual Parquet decoder implementations.
// These bypass the full reader pipeline to isolate decoder performance.
//
// Decoders benchmarked:
//   - DeltaBpDecoder (DELTA_BINARY_PACKED)
//   - BooleanDecoder (PLAIN boolean, bit-by-bit)
//   - RleBpDataDecoder (RLE/bit-packed, used for dictionary indices)
//   - StringDecoder (PLAIN BYTE_ARRAY)
//   - DirectDecoder (PLAIN fixed-width)

#include <folly/Benchmark.h>
#include <folly/BenchmarkUtil.h>
#include <folly/Random.h>
#include <folly/init/Init.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/parquet/reader/BooleanDecoder.h"
#include "velox/dwio/parquet/reader/DeltaBpDecoder.h"
#include "velox/dwio/parquet/reader/RleBpDataDecoder.h"
#include "velox/dwio/parquet/reader/StringDecoder.h"

using namespace facebook::velox;
using namespace facebook::velox::parquet;

namespace {

// ---------------------------------------------------------------------------
// Encoder helpers -- produce encoded buffers to feed into decoders.
// ---------------------------------------------------------------------------

// Encode values as DELTA_BINARY_PACKED.
// Format: header (block_size, miniblock_count, total_count, first_value) +
// blocks of (min_delta + bitwidths + miniblocks of bit-packed deltas).
std::vector<char> encodeDeltaBinaryPacked(
    const std::vector<int64_t>& values) {
  // Use a simple encoding: block_size=128, miniblocks_per_block=4,
  // miniblock_size=32. All deltas stored with a single bit width per
  // miniblock.
  constexpr int kBlockSize = 128;
  constexpr int kMiniblocksPerBlock = 4;
  constexpr int kMiniblockSize = kBlockSize / kMiniblocksPerBlock;

  std::vector<char> buf;
  buf.reserve(values.size() * 10); // generous estimate

  auto appendVlq = [&](uint64_t v) {
    uint8_t tmp[folly::kMaxVarintLength64];
    auto len = folly::encodeVarint(v, tmp);
    buf.insert(buf.end(), tmp, tmp + len);
  };

  auto appendZigZagVlq = [&](int64_t v) {
    uint64_t zigzag = static_cast<uint64_t>((v << 1) ^ (v >> 63));
    appendVlq(zigzag);
  };

  // Header.
  appendVlq(kBlockSize);
  appendVlq(kMiniblocksPerBlock);
  appendVlq(values.size());
  appendZigZagVlq(values.empty() ? 0 : values[0]);

  // Encode blocks.
  for (size_t blockStart = 1; blockStart < values.size();
       blockStart += kBlockSize) {
    size_t blockEnd = std::min(blockStart + kBlockSize, values.size());

    // Compute deltas.
    std::vector<int64_t> deltas;
    deltas.reserve(blockEnd - blockStart);
    for (size_t i = blockStart; i < blockEnd; ++i) {
      deltas.push_back(values[i] - values[i - 1]);
    }

    // Find min delta.
    int64_t minDelta = *std::min_element(deltas.begin(), deltas.end());
    appendZigZagVlq(minDelta);

    // Compute residuals and max bit width per miniblock.
    std::vector<uint64_t> residuals(deltas.size());
    for (size_t i = 0; i < deltas.size(); ++i) {
      residuals[i] = static_cast<uint64_t>(deltas[i] - minDelta);
    }

    // Pad residuals to full block.
    while (residuals.size() < kBlockSize) {
      residuals.push_back(0);
    }

    // Per-miniblock bit widths.
    std::vector<uint8_t> bitWidths(kMiniblocksPerBlock);
    for (int mb = 0; mb < kMiniblocksPerBlock; ++mb) {
      uint64_t maxVal = 0;
      for (int j = 0; j < kMiniblockSize; ++j) {
        maxVal = std::max(maxVal, residuals[mb * kMiniblockSize + j]);
      }
      bitWidths[mb] = maxVal == 0 ? 0 : (64 - __builtin_clzll(maxVal));
    }

    // Write bit widths.
    for (auto bw : bitWidths) {
      buf.push_back(static_cast<char>(bw));
    }

    // Write miniblock data.
    for (int mb = 0; mb < kMiniblocksPerBlock; ++mb) {
      uint8_t bw = bitWidths[mb];
      if (bw == 0) {
        continue;
      }
      int totalBits = kMiniblockSize * bw;
      int totalBytes = (totalBits + 7) / 8;
      size_t startPos = buf.size();
      buf.resize(buf.size() + totalBytes, 0);

      // Pack bits.
      int bitPos = 0;
      for (int j = 0; j < kMiniblockSize; ++j) {
        uint64_t val = residuals[mb * kMiniblockSize + j];
        for (int b = 0; b < bw; ++b) {
          if (val & (1ULL << b)) {
            buf[startPos + bitPos / 8] |=
                static_cast<char>(1 << (bitPos % 8));
          }
          ++bitPos;
        }
      }
    }
  }

  // Trailing padding for SIMD safety.
  buf.resize(buf.size() + DeltaBpDecoder::kRequiredTrailingPadding, 0);
  return buf;
}

// Encode values as RLE/bit-packed (Parquet hybrid encoding).
std::vector<char> encodeRleBitPacked(
    const std::vector<int32_t>& values,
    uint8_t bitWidth) {
  std::vector<char> buf;
  // Encode all values as a single bit-packed run.
  // indicator = (num_groups << 1) | 1, where num_groups = ceil(count / 8).
  int numGroups = (values.size() + 7) / 8;
  uint8_t tmp[folly::kMaxVarintLength64];
  auto len = folly::encodeVarint((numGroups << 1) | 1, tmp);
  buf.insert(buf.end(), tmp, tmp + len);

  // Bit-pack the values.
  int totalBits = values.size() * bitWidth;
  int totalBytes = (totalBits + 7) / 8;
  size_t startPos = buf.size();
  buf.resize(buf.size() + totalBytes, 0);

  int bitPos = 0;
  for (auto val : values) {
    for (int b = 0; b < bitWidth; ++b) {
      if (val & (1 << b)) {
        buf[startPos + bitPos / 8] |= static_cast<char>(1 << (bitPos % 8));
      }
      ++bitPos;
    }
  }

  // Padding for safe reads.
  buf.resize(buf.size() + 16, 0);
  return buf;
}

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

// Encode PLAIN BYTE_ARRAY strings.
std::vector<char> encodePlainStrings(
    const std::vector<std::string_view>& values) {
  std::vector<char> buf;
  for (auto sv : values) {
    int32_t len = sv.size();
    buf.insert(
        buf.end(),
        reinterpret_cast<const char*>(&len),
        reinterpret_cast<const char*>(&len) + 4);
    buf.insert(buf.end(), sv.begin(), sv.end());
  }
  // simd::kPadding bytes for safe reads.
  buf.resize(buf.size() + 32, 0);
  return buf;
}

// Encode PLAIN fixed-width INT64 values.
std::vector<char> encodePlainInt64(const std::vector<int64_t>& values) {
  std::vector<char> buf(values.size() * sizeof(int64_t) + 32, 0);
  memcpy(buf.data(), values.data(), values.size() * sizeof(int64_t));
  return buf;
}

// Simple no-filter, dense, no-hook visitor for benchmarking.
// Collects decoded values into an output buffer.
template <typename T>
class BenchmarkVisitor {
 public:
  using DataType = T;
  using FilterType = facebook::velox::common::AlwaysTrue;
  using HookType = dwio::common::NoHook;
  static constexpr bool dense = true;
  static constexpr bool kHasFilter = false;
  static constexpr bool kHasHook = false;
  static constexpr bool kHasBulkPath = false;
  static constexpr bool kFilterOnly = false;

  BenchmarkVisitor(T* output, int32_t numRows)
      : output_(output), numRows_(numRows) {
    rows_.resize(numRows);
    std::iota(rows_.begin(), rows_.end(), 0);
  }

  const facebook::velox::common::AlwaysTrue& filter() const {
    return filter_;
  }

  bool allowNulls() const {
    return false;
  }

  int32_t start() {
    return 0;
  }

  int32_t numRows() const {
    return numRows_;
  }

  const int32_t* rows() const {
    return rows_.data();
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

  void setNumValues(int32_t) {}

  void addRowNumber(int32_t) {}

  // Stubs for fastPath compilation (never called due to kHasBulkPath=false).
  int32_t numValuesBias() const {
    return 0;
  }

  raw_vector<int32_t>& outerNonNullRows() {
    static raw_vector<int32_t> v;
    return v;
  }

  raw_vector<int32_t>& innerNonNullRows() {
    static raw_vector<int32_t> v;
    return v;
  }

  void setAllNull(int32_t) {}

  void setHasNulls() {}

  uint64_t* rawNulls(int32_t) {
    return nullptr;
  }

  void setRows(folly::Range<const int32_t*>) {}

  T* rawValues(int32_t) {
    return output_;
  }

  int32_t* outputRows(int32_t) {
    return nullptr;
  }

  bool atEnd() const {
    return outputIdx_ >= numRows_;
  }

  template <bool, bool, bool>
  void processRun(
      const T*,
      int32_t numInput,
      const int32_t*,
      int32_t*,
      T*,
      int32_t& numValues) {
    numValues += numInput;
  }

  template <bool, bool, bool>
  void processRle(
      int64_t,
      int64_t,
      int32_t numInput,
      int32_t,
      const int32_t*,
      int32_t*,
      T*,
      int32_t& numValues) {
    numValues += numInput;
  }

 private:
  T* output_;
  int32_t numRows_;
  int32_t outputIdx_ = 0;
  facebook::velox::common::AlwaysTrue filter_;
  std::vector<int32_t> rows_;
};

constexpr int kBenchNumValues = 100'000;

} // namespace

// ===========================================================================
// DeltaBpDecoder benchmarks
//
// Baseline for optimization 3.1 (SIMD prefix sum) and to measure different
// delta distributions (constant, small, large bit-widths).
// ===========================================================================

void benchDeltaBpSequential(uint32_t, int maxDelta) {
  folly::BenchmarkSuspender suspender;

  std::vector<int64_t> values(kBenchNumValues);
  values[0] = 1000;
  for (int i = 1; i < kBenchNumValues; ++i) {
    values[i] = values[i - 1] + (i % (maxDelta + 1));
  }

  auto encoded = encodeDeltaBinaryPacked(values);
  std::vector<int64_t> output(kBenchNumValues);

  suspender.dismiss();

  DeltaBpDecoder decoder(encoded.data());
  decoder.readValues<int64_t>(output.data(), kBenchNumValues);
  folly::doNotOptimizeAway(output[kBenchNumValues - 1]);
}

// Constant deltas (bitWidth=0 fast path).
BENCHMARK_NAMED_PARAM(benchDeltaBpSequential, ConstantDelta, 0)

// Small deltas (bitWidth ~3-4).
BENCHMARK_NAMED_PARAM(benchDeltaBpSequential, SmallDelta_max7, 7)

// Medium deltas (bitWidth ~8).
BENCHMARK_NAMED_PARAM(benchDeltaBpSequential, MediumDelta_max255, 255)

// Large deltas (bitWidth ~16).
BENCHMARK_NAMED_PARAM(benchDeltaBpSequential, LargeDelta_max65535, 65535)

BENCHMARK_DRAW_LINE();

// DeltaBpDecoder SKIP benchmarks -- measures the optimized skip path.
void benchDeltaBpSkip(uint32_t, int maxDelta) {
  folly::BenchmarkSuspender suspender;

  std::vector<int64_t> values(kBenchNumValues);
  values[0] = 1000;
  for (int i = 1; i < kBenchNumValues; ++i) {
    values[i] = values[i - 1] + (i % (maxDelta + 1));
  }

  auto encoded = encodeDeltaBinaryPacked(values);

  suspender.dismiss();

  // Skip all values (simulates sparse read that skips most of the page).
  DeltaBpDecoder decoder(encoded.data());
  decoder.skip(kBenchNumValues);
  folly::doNotOptimizeAway(decoder.bufferStart());
}

// Constant-delta skip (O(1) fast path).
BENCHMARK_NAMED_PARAM(benchDeltaBpSkip, Skip_ConstantDelta, 0)

// Small-delta skip (batched decode path).
BENCHMARK_NAMED_PARAM(benchDeltaBpSkip, Skip_SmallDelta_max7, 7)

// Large-delta skip (batched decode path, wider bit widths).
BENCHMARK_NAMED_PARAM(benchDeltaBpSkip, Skip_LargeDelta_max65535, 65535)

BENCHMARK_DRAW_LINE();

// ===========================================================================
// BooleanDecoder benchmark
//
// Baseline for optimization 2.2 (batch boolean decoding).
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
  folly::doNotOptimizeAway(output[0]);
}

// Compare with RLE boolean decoder (same data re-encoded).
BENCHMARK(RleBpDecoder_Boolean_100k) {
  folly::BenchmarkSuspender suspender;

  // Encode as RLE bit-packed with bitWidth=1.
  std::vector<int32_t> values(kBenchNumValues);
  for (int i = 0; i < kBenchNumValues; ++i) {
    values[i] = (i % 3 != 0) ? 1 : 0;
  }
  auto encoded = encodeRleBitPacked(values, 1);

  // RleBpDataDecoder needs: first 4 bytes = data length, then data.
  // Actually for the benchmark we just feed the encoded data directly.
  std::vector<int8_t> output(kBenchNumValues);
  auto visitor = BenchmarkVisitor<int8_t>(output.data(), kBenchNumValues);

  suspender.dismiss();

  RleBpDataDecoder decoder(
      encoded.data(), encoded.data() + encoded.size(), 1);
  decoder.readWithVisitor<false>(nullptr, visitor);
  folly::doNotOptimizeAway(output[0]);
}

BENCHMARK_DRAW_LINE();

// ===========================================================================
// StringDecoder benchmark
//
// Baseline for optimization 3.2 (batched string visitor path).
// ===========================================================================

void benchStringDecoder(uint32_t, int avgLen) {
  folly::BenchmarkSuspender suspender;

  constexpr int kNumStrings = 50'000;
  std::string padding(avgLen, 'x');
  std::vector<std::string> storage(kNumStrings);
  std::vector<std::string_view> values(kNumStrings);
  for (int i = 0; i < kNumStrings; ++i) {
    storage[i] = padding.substr(0, avgLen) + std::to_string(i);
    values[i] = storage[i];
  }
  auto encoded = encodePlainStrings(values);
  std::vector<folly::StringPiece> output(kNumStrings);

  suspender.dismiss();

  StringDecoder decoder(encoded.data(), encoded.data() + encoded.size());
  // Manual decode loop (StringDecoder's readWithVisitor is template-heavy).
  for (int i = 0; i < kNumStrings; ++i) {
    auto sv = std::string_view(
        encoded.data(), encoded.size()); // will be advanced by decoder
    folly::doNotOptimizeAway(decoder);
  }
  // Since StringDecoder::readWithVisitor requires a complex Visitor type,
  // we benchmark the skip path which exercises the length parsing hot path.
  StringDecoder decoder2(encoded.data(), encoded.data() + encoded.size());
  decoder2.skip(kNumStrings);
  folly::doNotOptimizeAway(decoder2);
}

BENCHMARK_NAMED_PARAM(benchStringDecoder, ShortStrings_8B, 8)
BENCHMARK_NAMED_PARAM(benchStringDecoder, MediumStrings_64B, 64)
BENCHMARK_NAMED_PARAM(benchStringDecoder, LongStrings_256B, 256)

BENCHMARK_DRAW_LINE();

// ===========================================================================
// RleBpDataDecoder benchmark (dictionary index decoding)
//
// Baseline for optimization 2.3 (SIMD dictionary lookups).
// Tests different bit widths to exercise different unpack kernels.
// ===========================================================================

void benchRleBpDecode(uint32_t, int bitWidth) {
  folly::BenchmarkSuspender suspender;

  int maxVal = (1 << bitWidth) - 1;
  std::vector<int32_t> values(kBenchNumValues);
  for (int i = 0; i < kBenchNumValues; ++i) {
    values[i] = i % (maxVal + 1);
  }
  auto encoded = encodeRleBitPacked(values, bitWidth);
  std::vector<int32_t> output(kBenchNumValues);
  auto visitor = BenchmarkVisitor<int32_t>(output.data(), kBenchNumValues);

  suspender.dismiss();

  RleBpDataDecoder decoder(
      encoded.data(), encoded.data() + encoded.size(), bitWidth);
  decoder.readWithVisitor<false>(nullptr, visitor);
  folly::doNotOptimizeAway(output[0]);
}

BENCHMARK_NAMED_PARAM(benchRleBpDecode, BitWidth_1, 1)
BENCHMARK_NAMED_PARAM(benchRleBpDecode, BitWidth_4, 4)
BENCHMARK_NAMED_PARAM(benchRleBpDecode, BitWidth_8, 8)
BENCHMARK_NAMED_PARAM(benchRleBpDecode, BitWidth_12, 12)
BENCHMARK_NAMED_PARAM(benchRleBpDecode, BitWidth_16, 16)
BENCHMARK_NAMED_PARAM(benchRleBpDecode, BitWidth_20, 20)

BENCHMARK_DRAW_LINE();

// ===========================================================================
// StringDecoder SKIP benchmark (variable vs fixed length)
// Uses manual timing to avoid folly BenchmarkSuspender measurement issues.
// ===========================================================================

// ===========================================================================
// Dictionary gather benchmark (prefetch effectiveness)
// Uses manual timing to avoid folly BenchmarkSuspender measurement issues.
// ===========================================================================

void runManualBenchmarks() {
  constexpr int kNumStrings = 100'000;
  constexpr int kStringSkipIters = 200;
  constexpr int kDictGatherIters = 200;
  constexpr int kNumValues = 100'000;

  printf("\n--- Manual micro-benchmarks (not affected by folly suspender) ---\n");

  // ---- String Skip: Variable-length ----
  {
    std::vector<std::string> storage(kNumStrings);
    std::vector<std::string_view> values(kNumStrings);
    for (int i = 0; i < kNumStrings; ++i) {
      storage[i] = std::string(16, 'x');
      values[i] = storage[i];
    }
    auto encoded = encodePlainStrings(values);

    // Warmup.
    for (int i = 0; i < 10; ++i) {
      StringDecoder d(encoded.data(), encoded.data() + encoded.size());
      d.skip(kNumStrings);
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kStringSkipIters; ++i) {
      StringDecoder d(encoded.data(), encoded.data() + encoded.size());
      d.skip(kNumStrings);
      folly::doNotOptimizeAway(d);
    }
    auto elapsed = std::chrono::steady_clock::now() - start;
    double usPerIter = std::chrono::duration<double, std::micro>(elapsed).count()
        / kStringSkipIters;
    printf("  StringSkip(VarLen_16B, 100K):    %8.1f us/iter  (%6.2f ns/value)\n",
        usPerIter, usPerIter * 1000.0 / kNumStrings);
  }

  // ---- String Skip: Fixed-length ----
  {
    constexpr int kFixedLen = 16;
    std::vector<char> encoded(
        static_cast<size_t>(kNumStrings) * kFixedLen, 'x');

    // Warmup.
    for (int i = 0; i < 10; ++i) {
      StringDecoder d(encoded.data(), encoded.data() + encoded.size(), kFixedLen);
      d.skip(kNumStrings);
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kStringSkipIters; ++i) {
      StringDecoder d(encoded.data(), encoded.data() + encoded.size(), kFixedLen);
      d.skip(kNumStrings);
      folly::doNotOptimizeAway(d);
    }
    auto elapsed = std::chrono::steady_clock::now() - start;
    double usPerIter = std::chrono::duration<double, std::micro>(elapsed).count()
        / kStringSkipIters;
    printf("  StringSkip(FixedLen_16B, 100K):  %8.1f us/iter  (%6.2f ns/value)\n",
        usPerIter, usPerIter * 1000.0 / kNumStrings);
  }

  printf("\n");

  // ---- Dictionary Gather: with and without prefetch ----
  auto runDictBench = [&](int dictSize, const char* label) {
    std::vector<int64_t> dictionary(dictSize);
    for (int i = 0; i < dictSize; ++i) {
      dictionary[i] = i * 7 + 42;
    }
    std::vector<int32_t> indices(kNumValues);
    for (int i = 0; i < kNumValues; ++i) {
      indices[i] = folly::Random::rand32(dictSize);
    }
    std::vector<int64_t> output(kNumValues);
    auto* dictPtr = dictionary.data();
    auto* idxPtr = indices.data();

    // With prefetch.
    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < kNumValues; ++j) {
        output[j] = dictPtr[idxPtr[j]];
      }
    }
    auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < kDictGatherIters; ++iter) {
      constexpr int kPF = 8;
      for (int i = 0; i < kNumValues; ++i) {
        if (i + kPF < kNumValues) {
          __builtin_prefetch(&dictPtr[idxPtr[i + kPF]], 0, 1);
        }
        output[i] = dictPtr[idxPtr[i]];
      }
      folly::doNotOptimizeAway(output[kNumValues / 2]);
    }
    auto elapsedPF = std::chrono::steady_clock::now() - start;
    double usPF = std::chrono::duration<double, std::micro>(elapsedPF).count()
        / kDictGatherIters;

    // Without prefetch.
    start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < kDictGatherIters; ++iter) {
      for (int i = 0; i < kNumValues; ++i) {
        output[i] = dictPtr[idxPtr[i]];
      }
      folly::doNotOptimizeAway(output[kNumValues / 2]);
    }
    auto elapsedNoPF = std::chrono::steady_clock::now() - start;
    double usNoPF = std::chrono::duration<double, std::micro>(elapsedNoPF).count()
        / kDictGatherIters;

    printf("  DictGather(%s, 100K):  prefetch=%7.1f us  no_prefetch=%7.1f us  speedup=%.2fx\n",
        label, usPF, usNoPF, usNoPF / usPF);
  };

  runDictBench(256, "256 entries / 2KB ");
  runDictBench(4096, "4K entries / 32KB ");
  runDictBench(65536, "64K entries / 512KB");
  printf("\n");
}

// ===========================================================================
// Main
// ===========================================================================

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  folly::runBenchmarks();
  runManualBenchmarks();
  return 0;
}
