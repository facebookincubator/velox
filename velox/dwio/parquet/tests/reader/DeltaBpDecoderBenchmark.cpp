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

// Micro-benchmarks for the Parquet DELTA_BINARY_PACKED decoder.
// Bypasses the full reader pipeline to isolate decoder performance.
//
// Three groups:
//   1. Skip vs decode-and-discard. Skipping is the optimization; decoding then
//      throwing the values away is the alternative it replaces, so the pair is
//      a fair in-binary A/B. Reported for constant-delta miniblocks (O(1) skip)
//      and variable-delta miniblocks (delta summation).
//   2. Sequential full decode. The no-regression guard for the common read
//      path; compare across a `main` vs branch build.
//   3. Dispatch: switch vs fold-expression selection of the per-bit-width
//      miniblock kernel. Answers how much the jump-table dispatch actually
//      buys, since dispatch runs once per 32-value miniblock.

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/parquet/reader/DeltaBpDecoder.h"

using namespace facebook::velox;
using namespace facebook::velox::parquet;

namespace {

// ---------------------------------------------------------------------------
// DELTA_BINARY_PACKED encoder -- produces spec-compliant pages from int64
// input, choosing the per-block min delta and per-miniblock bit width the way
// a real writer does.
// ---------------------------------------------------------------------------

void putUleb(std::vector<uint8_t>& buf, uint64_t value) {
  do {
    uint8_t byte = value & 0x7f;
    value >>= 7;
    if (value != 0) {
      byte |= 0x80;
    }
    buf.push_back(byte);
  } while (value != 0);
}

void putZigZag(std::vector<uint8_t>& buf, int64_t value) {
  putUleb(
      buf,
      (static_cast<uint64_t>(value) << 1) ^ static_cast<uint64_t>(value >> 63));
}

uint8_t bitsNeeded(uint64_t value) {
  return value == 0 ? 0 : static_cast<uint8_t>(64 - __builtin_clzll(value));
}

void appendPacked(
    std::vector<uint8_t>& buf,
    const std::vector<uint64_t>& values,
    uint8_t bitWidth) {
  if (bitWidth == 0) {
    return;
  }
  const size_t numBits = values.size() * bitWidth;
  std::vector<uint8_t> packed((numBits + 7) / 8, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint64_t value = values[i];
    const size_t bitPos = i * bitWidth;
    for (uint8_t b = 0; b < bitWidth; ++b) {
      if ((value >> b) & 1ULL) {
        packed[(bitPos + b) >> 3] |=
            static_cast<uint8_t>(1u << ((bitPos + b) & 7));
      }
    }
  }
  buf.insert(buf.end(), packed.begin(), packed.end());
}

// Encodes 'values' as a single DELTA_BINARY_PACKED page. The value count minus
// one must be a multiple of valuesPerBlock so every block is full.
std::vector<char> encodeDelta(
    const std::vector<int64_t>& values,
    uint32_t valuesPerMiniBlock = 32,
    uint32_t miniBlocksPerBlock = 4) {
  const uint32_t valuesPerBlock = valuesPerMiniBlock * miniBlocksPerBlock;
  std::vector<uint8_t> buf;
  putUleb(buf, valuesPerBlock);
  putUleb(buf, miniBlocksPerBlock);
  putUleb(buf, values.size());
  putZigZag(buf, values.empty() ? 0 : values[0]);

  const size_t numDeltas = values.empty() ? 0 : values.size() - 1;
  std::vector<int64_t> deltas(numDeltas);
  for (size_t i = 0; i < numDeltas; ++i) {
    deltas[i] = static_cast<int64_t>(
        static_cast<uint64_t>(values[i + 1]) -
        static_cast<uint64_t>(values[i]));
  }

  for (size_t base = 0; base < numDeltas; base += valuesPerBlock) {
    const size_t blockCount =
        std::min<size_t>(valuesPerBlock, numDeltas - base);
    int64_t minDelta = INT64_MAX;
    for (size_t i = 0; i < blockCount; ++i) {
      minDelta = std::min(minDelta, deltas[base + i]);
    }
    putZigZag(buf, minDelta);

    std::vector<uint8_t> widths(miniBlocksPerBlock, 0);
    for (uint32_t mb = 0; mb < miniBlocksPerBlock; ++mb) {
      uint64_t maxStored = 0;
      for (uint32_t j = 0; j < valuesPerMiniBlock; ++j) {
        const size_t idx = mb * valuesPerMiniBlock + j;
        if (idx < blockCount) {
          maxStored = std::max(
              maxStored, static_cast<uint64_t>(deltas[base + idx] - minDelta));
        }
      }
      widths[mb] = bitsNeeded(maxStored);
    }
    for (uint8_t w : widths) {
      buf.push_back(w);
    }
    for (uint32_t mb = 0; mb < miniBlocksPerBlock; ++mb) {
      std::vector<uint64_t> stored(valuesPerMiniBlock, 0);
      for (uint32_t j = 0; j < valuesPerMiniBlock; ++j) {
        const size_t idx = mb * valuesPerMiniBlock + j;
        stored[j] = idx < blockCount
            ? static_cast<uint64_t>(deltas[base + idx] - minDelta)
            : 0;
      }
      appendPacked(buf, stored, widths[mb]);
    }
  }
  // Trailing padding for the decoder's SIMD over-read.
  buf.insert(buf.end(), DeltaBpDecoder::kRequiredTrailingPadding + 8, 0);
  return std::vector<char>(buf.begin(), buf.end());
}

// (values - 1) is a multiple of 128 so all blocks are full.
constexpr int kBenchNumValues = 128 * 781 + 1; // 99'969, ~100k
constexpr int kTailValues = 64; // values read after a skip

// Constant delta: every miniblock has bit width 0 (the O(1) skip path).
const std::vector<char>& constantPage() {
  static const std::vector<char> page = [] {
    std::vector<int64_t> values(kBenchNumValues);
    for (int i = 0; i < kBenchNumValues; ++i) {
      values[i] = 1'000'000 + int64_t{7} * i;
    }
    return encodeDelta(values);
  }();
  return page;
}

// Variable delta: a spread of deltas so miniblocks use non-zero bit widths (the
// delta-summation skip path and the SIMD decode path).
const std::vector<char>& variablePage() {
  static const std::vector<char> page = [] {
    std::vector<int64_t> values(kBenchNumValues);
    int64_t acc = 0;
    for (int i = 0; i < kBenchNumValues; ++i) {
      values[i] = acc;
      acc += 1 + (i * 2654435761u) % 4096; // pseudo-random positive delta
    }
    return encodeDelta(values);
  }();
  return page;
}

int64_t sink = 0;

} // namespace

// ===========================================================================
// 1. Skip vs decode-and-discard
// ===========================================================================

BENCHMARK(Skip_ConstantDelta) {
  const auto& page = constantPage();
  DeltaBpDecoder decoder(page.data());
  decoder.skip(kBenchNumValues - kTailValues);
  int64_t out[kTailValues];
  decoder.readValues(out, kTailValues);
  sink += out[0];
  folly::doNotOptimizeAway(sink);
}

BENCHMARK_RELATIVE(DecodeDiscard_ConstantDelta) {
  const auto& page = constantPage();
  static std::vector<int64_t> out(kBenchNumValues);
  DeltaBpDecoder decoder(page.data());
  decoder.readValues(out.data(), kBenchNumValues);
  sink += out[kBenchNumValues - kTailValues];
  folly::doNotOptimizeAway(sink);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(Skip_VariableDelta) {
  const auto& page = variablePage();
  DeltaBpDecoder decoder(page.data());
  decoder.skip(kBenchNumValues - kTailValues);
  int64_t out[kTailValues];
  decoder.readValues(out, kTailValues);
  sink += out[0];
  folly::doNotOptimizeAway(sink);
}

BENCHMARK_RELATIVE(DecodeDiscard_VariableDelta) {
  const auto& page = variablePage();
  static std::vector<int64_t> out(kBenchNumValues);
  DeltaBpDecoder decoder(page.data());
  decoder.readValues(out.data(), kBenchNumValues);
  sink += out[kBenchNumValues - kTailValues];
  folly::doNotOptimizeAway(sink);
}

BENCHMARK_DRAW_LINE();

// ===========================================================================
// 2. Sequential full decode (no-regression guard)
// ===========================================================================

BENCHMARK(SequentialDecode_ConstantDelta) {
  const auto& page = constantPage();
  static std::vector<int64_t> out(kBenchNumValues);
  DeltaBpDecoder decoder(page.data());
  decoder.readValues(out.data(), kBenchNumValues);
  sink += out[kBenchNumValues - 1];
  folly::doNotOptimizeAway(sink);
}

BENCHMARK(SequentialDecode_VariableDelta) {
  const auto& page = variablePage();
  static std::vector<int64_t> out(kBenchNumValues);
  DeltaBpDecoder decoder(page.data());
  decoder.readValues(out.data(), kBenchNumValues);
  sink += out[kBenchNumValues - 1];
  folly::doNotOptimizeAway(sink);
}

BENCHMARK_DRAW_LINE();

// ===========================================================================
// 3. Dispatch: switch vs fold-expression
//
// A representative per-bit-width miniblock kernel (bit-unpack fused with the
// prefix sum) selected two ways. Both dispatchers pick the same kernel, so the
// difference is purely the selection mechanism. Dispatch runs once per 32-value
// miniblock, so this bounds how much the jump table can help.
// ===========================================================================

namespace {

template <int kBitWidth>
FOLLY_ALWAYS_INLINE void decodeKernel(
    const char* src,
    int32_t numValues,
    int64_t minDelta,
    int64_t& lastValue,
    int64_t* out) {
  constexpr uint64_t mask =
      (kBitWidth == 64) ? ~0ULL : ((1ULL << kBitWidth) - 1);
  const auto* source = reinterpret_cast<const uint64_t*>(src);
  uint64_t cumulative = static_cast<uint64_t>(lastValue);
  const uint64_t step = static_cast<uint64_t>(minDelta);
  for (int32_t i = 0; i < numValues; ++i) {
    const uint64_t value =
        bits::detail::loadBits<uint64_t>(
            source, static_cast<uint64_t>(i) * kBitWidth, kBitWidth) &
        mask;
    cumulative += step + value;
    out[i] = static_cast<int64_t>(cumulative);
  }
  lastValue = static_cast<int64_t>(cumulative);
}

// Jump-table selection (mirrors DeltaBpDecoder::dispatchSimdMiniBlock).
bool dispatchSwitch(
    uint32_t bitWidth,
    const char* src,
    int32_t numValues,
    int64_t minDelta,
    int64_t& lastValue,
    int64_t* out) {
  switch (bitWidth) {
#define CASE(W)                                                \
  case W:                                                      \
    decodeKernel<W>(src, numValues, minDelta, lastValue, out); \
    return true
    CASE(1);
    CASE(2);
    CASE(3);
    CASE(4);
    CASE(5);
    CASE(6);
    CASE(7);
    CASE(8);
    CASE(9);
    CASE(10);
    CASE(11);
    CASE(12);
    CASE(13);
    CASE(14);
    CASE(15);
    CASE(16);
    CASE(17);
    CASE(18);
    CASE(19);
    CASE(20);
    CASE(21);
    CASE(22);
    CASE(23);
    CASE(24);
    CASE(25);
    CASE(26);
    CASE(27);
    CASE(28);
    CASE(29);
    CASE(30);
    CASE(31);
    CASE(32);
#undef CASE
    default:
      return false;
  }
}

template <int... Is>
bool dispatchFoldImpl(
    std::integer_sequence<int, Is...>,
    uint32_t bitWidth,
    const char* src,
    int32_t numValues,
    int64_t minDelta,
    int64_t& lastValue,
    int64_t* out) {
  bool done = false;
  (void)((bitWidth == (Is + 1)
              ? (decodeKernel<Is + 1>(src, numValues, minDelta, lastValue, out),
                 done = true)
              : false) ||
         ...);
  return done;
}

// Fold-expression linear comparison chain over bit widths 1..32.
bool dispatchFold(
    uint32_t bitWidth,
    const char* src,
    int32_t numValues,
    int64_t minDelta,
    int64_t& lastValue,
    int64_t* out) {
  return dispatchFoldImpl(
      std::make_integer_sequence<int, 32>{},
      bitWidth,
      src,
      numValues,
      minDelta,
      lastValue,
      out);
}

constexpr int kMiniBlockValues = 32;
constexpr int kNumMiniBlocks = 4096;

// A pool of packed miniblocks with bit widths cycling 1..32, plus the width of
// each, so the dispatcher's branch is not trivially predictable.
struct DispatchFixture {
  std::vector<char> packed;
  std::vector<uint32_t> widths;
  std::vector<size_t> offsets;
};

const DispatchFixture& dispatchFixture() {
  static const DispatchFixture fixture = [] {
    DispatchFixture f;
    for (int m = 0; m < kNumMiniBlocks; ++m) {
      const uint32_t bitWidth = 1 + (m % 32);
      f.widths.push_back(bitWidth);
      f.offsets.push_back(f.packed.size());
      std::vector<uint64_t> vals(kMiniBlockValues);
      const uint64_t mask = (bitWidth == 64) ? ~0ULL : ((1ULL << bitWidth) - 1);
      for (int i = 0; i < kMiniBlockValues; ++i) {
        vals[i] = (static_cast<uint64_t>(i) * 2654435761u) & mask;
      }
      std::vector<uint8_t> tmp;
      appendPacked(tmp, vals, static_cast<uint8_t>(bitWidth));
      f.packed.insert(f.packed.end(), tmp.begin(), tmp.end());
    }
    f.packed.insert(f.packed.end(), 16, 0); // over-read padding
    return f;
  }();
  return fixture;
}

} // namespace

BENCHMARK(Dispatch_Switch) {
  const auto& f = dispatchFixture();
  int64_t last = 0;
  int64_t out[kMiniBlockValues];
  for (int m = 0; m < kNumMiniBlocks; ++m) {
    dispatchSwitch(
        f.widths[m],
        f.packed.data() + f.offsets[m],
        kMiniBlockValues,
        1,
        last,
        out);
  }
  sink += last;
  folly::doNotOptimizeAway(sink);
}

BENCHMARK_RELATIVE(Dispatch_Fold) {
  const auto& f = dispatchFixture();
  int64_t last = 0;
  int64_t out[kMiniBlockValues];
  for (int m = 0; m < kNumMiniBlocks; ++m) {
    dispatchFold(
        f.widths[m],
        f.packed.data() + f.offsets[m],
        kMiniBlockValues,
        1,
        last,
        out);
  }
  sink += last;
  folly::doNotOptimizeAway(sink);
}

// ===========================================================================
// Main
// ===========================================================================

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
