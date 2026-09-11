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

// Microbenchmark: isolates the raw decode step for BlockBitPacking.
// No Nimble encoding, no Velox reader pipeline, no filters — directly
// decodes packed bit arrays to measure pure decode cost.
// 8192 blocks x 1024 rows = ~8M values per iteration.
//
// Strategies compared:
//   A) FBA::get() per element (scalar, random access)
//   B) Lemire fullUnpack entire block, then pick
//   C) Lemire fastunpack per-group (32 values), pick within group
//
// For E2E benchmarks through the full Velox selective reader pipeline,
// see IntEncodingBenchmark.cpp.
//
// Run: buck2 run @mode/opt
//   fbcode//velox/dwio/nimble/encodings/benchmarks:block_bit_packing_benchmark
//
// Results (opt mode, 8192 blocks x 1024 rows = ~8M values):
//
// FBA time as % of Lemire decode-all-then-pick (lower = faster):
//
// uint8:
//   bw   1%sel   10%sel   25%sel
//    1     4%      35%      71%
//    4     6%      50%      94%
//    8     6%      49%      94%
//
// uint16:
//   bw   1%sel   10%sel   25%sel
//    1     4%      18%      40%
//    4     6%      21%      41%
//    8    13%      41%      71%
//   16    16%      51%      84%
//
// uint32:
//   bw   1%sel   10%sel   20%sel
//   16    22%      59%      68%
//   20     5%      26%      55%
//   25     8%      29%      39%
//   28    10%      39%      48%
//   32    21%      85%      80%
//
// uint64:
//   bw   1%sel   10%sel   20%sel
//   25     9%      38%      66%
//   40     6%      33%      48%
//   48     7%      49%      51%
//   60     3%      75%      37%
//
// Decision: use FBA per-element for all types at <5% selectivity
// (~50 rows per 1024-row block, 4.88%). Dense path always uses
// Lemire (fastest for full-block decode).

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "folly/Benchmark.h"
#include "velox/dwio/common/Lemire/BitPacking/bitpackinghelpers.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"

using namespace facebook::nimble;

namespace {

constexpr uint32_t kBlockSize = 1024;
constexpr uint8_t kBitWidth = 5;
constexpr uint32_t kNumBlocks = 8192;
constexpr uint32_t kTotalValues = kBlockSize * kNumBlocks;

struct PackedBlocks {
  std::vector<char> bits;
  uint32_t bytesPerBlock;
};

PackedBlocks packedBlocks;
std::vector<uint32_t> output(kTotalValues);

// Row sets for various selectivities (N rows out of 1024).
std::vector<uint32_t> sparseRows5; // 0.5%
std::vector<uint32_t> sparseRows10; // 1%
std::vector<uint32_t> sparseRows50; // 5%
std::vector<uint32_t> sparseRows100; // 10%
std::vector<uint32_t> sparseRows200; // 20%
std::vector<uint32_t> sparseRows205; // 20% (Section 3 — at kMaxFba boundary)
std::vector<uint32_t> sparseRows256; // 25%
std::vector<uint32_t> sparseRows307; // 30%
std::vector<uint32_t> sparseRows410; // 40%
std::vector<uint32_t> sparseRows500; // 49%
std::vector<uint32_t> sparseRows512; // 50%
std::vector<uint32_t> sparseRows614; // 60%
std::vector<uint32_t> sparseRows717; // 70%
std::vector<uint32_t> sparseRows800; // 78%
std::vector<uint32_t> sparseRows819; // 80%
std::vector<uint32_t> sparseRows922; // 90%
std::vector<uint32_t> sparseRows973; // 95%
std::vector<uint32_t> sparseRows1014; // 99%
std::vector<uint32_t> denseRows1024;

std::vector<uint32_t> makeSelectedRows(uint32_t count, uint32_t blockSize) {
  std::vector<uint32_t> all(blockSize);
  std::iota(all.begin(), all.end(), 0);
  std::mt19937 rng(42);
  std::shuffle(all.begin(), all.end(), rng);
  all.resize(count);
  std::sort(all.begin(), all.end());
  return all;
}

void setupData() {
  const auto blockBytes = FixedBitArray::bufferSize(kBlockSize, kBitWidth);
  packedBlocks.bytesPerBlock = blockBytes;
  packedBlocks.bits.resize(blockBytes * kNumBlocks, 0);
  std::mt19937 rng(123);
  uint32_t maxVal = (1u << kBitWidth) - 1;
  for (uint32_t b = 0; b < kNumBlocks; ++b) {
    FixedBitArray fba(packedBlocks.bits.data() + b * blockBytes, kBitWidth);
    for (uint32_t i = 0; i < kBlockSize; ++i) {
      fba.set(i, rng() % (maxVal + 1));
    }
  }

  sparseRows5 = makeSelectedRows(5, kBlockSize);
  sparseRows10 = makeSelectedRows(10, kBlockSize);
  sparseRows50 = makeSelectedRows(50, kBlockSize);
  sparseRows100 = makeSelectedRows(100, kBlockSize);
  sparseRows200 = makeSelectedRows(200, kBlockSize);
  sparseRows205 = makeSelectedRows(205, kBlockSize);
  sparseRows256 = makeSelectedRows(256, kBlockSize);
  sparseRows307 = makeSelectedRows(307, kBlockSize);
  sparseRows410 = makeSelectedRows(410, kBlockSize);
  sparseRows500 = makeSelectedRows(500, kBlockSize);
  sparseRows512 = makeSelectedRows(512, kBlockSize);
  sparseRows614 = makeSelectedRows(614, kBlockSize);
  sparseRows717 = makeSelectedRows(717, kBlockSize);
  sparseRows800 = makeSelectedRows(800, kBlockSize);
  sparseRows819 = makeSelectedRows(819, kBlockSize);
  sparseRows922 = makeSelectedRows(922, kBlockSize);
  sparseRows973 = makeSelectedRows(973, kBlockSize);
  sparseRows1014 = makeSelectedRows(1014, kBlockSize);
  denseRows1024.resize(kBlockSize);
  std::iota(denseRows1024.begin(), denseRows1024.end(), 0);
}

// Option A: FBA::get() per selected row.
void decodeFBA(const std::vector<uint32_t>& rows, uint32_t iters) {
  while (iters--) {
    const auto blockBytes = packedBlocks.bytesPerBlock;
    auto* out = output.data();
    for (uint32_t b = 0; b < kNumBlocks; ++b) {
      FixedBitArray fba{
          {packedBlocks.bits.data() + b * blockBytes,
           static_cast<size_t>(blockBytes)},
          kBitWidth};
      for (uint32_t i = 0; i < rows.size(); ++i) {
        out[i] = static_cast<uint32_t>(fba.get(rows[i]));
      }
      out += rows.size();
    }
  }
}

// Option B: fullUnpack entire block via Lemire, then pick rows.
// When rows.size() == kBlockSize (dense), decodes directly into output.
void decodeFullBlockThenPick(
    const std::vector<uint32_t>& rows,
    uint32_t iters) {
  constexpr uint32_t kGroupSize = 32;
  const bool isDense = rows.size() == kBlockSize;
  while (iters--) {
    const auto blockBytes = packedBlocks.bytesPerBlock;
    auto* out = output.data();
    for (uint32_t b = 0; b < kNumBlocks; ++b) {
      const auto* input = reinterpret_cast<const uint32_t*>(
          packedBlocks.bits.data() + b * blockBytes);
      if (isDense) {
        for (uint32_t r = 0; r < kBlockSize; r += kGroupSize) {
          facebook::velox::fastpforlib::fastunpack(
              input + r / kGroupSize * kBitWidth, out + r, kBitWidth);
        }
      } else {
        uint32_t decoded[kBlockSize];
        for (uint32_t r = 0; r < kBlockSize; r += kGroupSize) {
          facebook::velox::fastpforlib::fastunpack(
              input + r / kGroupSize * kBitWidth, decoded + r, kBitWidth);
        }
        for (uint32_t i = 0; i < rows.size(); ++i) {
          out[i] = decoded[rows[i]];
        }
      }
      out += rows.size();
    }
  }
}

// Option C: fastunpack per-group, pick within group.
void decodePerGroup(const std::vector<uint32_t>& rows, uint32_t iters) {
  constexpr uint32_t kGroupSize = 32;
  while (iters--) {
    const auto blockBytes = packedBlocks.bytesPerBlock;
    auto* out = output.data();
    for (uint32_t b = 0; b < kNumBlocks; ++b) {
      const auto* input = reinterpret_cast<const uint32_t*>(
          packedBlocks.bits.data() + b * blockBytes);
      uint32_t tmp[kGroupSize];
      uint32_t i = 0;
      while (i < rows.size()) {
        const auto groupIdx = rows[i] / kGroupSize;
        facebook::velox::fastpforlib::fastunpack(
            input + groupIdx * kBitWidth, tmp, kBitWidth);
        while (i < rows.size() && rows[i] / kGroupSize == groupIdx) {
          *out++ = tmp[rows[i] % kGroupSize];
          ++i;
        }
      }
    }
  }
}

} // namespace

// ============================================================================
// Section 1: uint32 sparse decode — FBA vs FullBlock vs PerGroup
// (~8M values across 8192 blocks)
// ============================================================================

#define BENCHMARK_SPARSE_U32(n)                     \
  BENCHMARK(FBA_sparse_##n, iters) {                \
    decodeFBA(sparseRows##n, iters);                \
  }                                                 \
  BENCHMARK_RELATIVE(FullBlock_sparse_##n, iters) { \
    decodeFullBlockThenPick(sparseRows##n, iters);  \
  }                                                 \
  BENCHMARK_RELATIVE(PerGroup_sparse_##n, iters) {  \
    decodePerGroup(sparseRows##n, iters);           \
  }                                                 \
  BENCHMARK_DRAW_LINE();

BENCHMARK_SPARSE_U32(5)
BENCHMARK_SPARSE_U32(50)
BENCHMARK_SPARSE_U32(100)
BENCHMARK_SPARSE_U32(200)
BENCHMARK_SPARSE_U32(500)
BENCHMARK_SPARSE_U32(800)

BENCHMARK(FBA_dense_1024, iters) {
  decodeFBA(denseRows1024, iters);
}
BENCHMARK_RELATIVE(FullBlock_dense_1024, iters) {
  decodeFullBlockThenPick(denseRows1024, iters);
}
BENCHMARK_RELATIVE(PerGroup_dense_1024, iters) {
  decodePerGroup(denseRows1024, iters);
}
BENCHMARK_DRAW_LINE();

// ============================================================================
// Section 2: uint8 decode — fastunpack_quarter vs fastunpack32 + narrow
//
// Option A: Lemire fastunpack_quarter — native 8-bit unpack, 8 values/call,
//   outputs directly to uint8_t*.
// Option B: Lemire fastunpack (32-bit) into uint32_t[32] temp buffer, then
//   narrowing loop to uint8_t.
//
// Results (Release mode, ~8M values):
//
//  bw  Option A (8-bit)  Option B (32-bit+narrow)   A vs B
//   1         2.24ms              3.36ms            A 1.5x faster
//   2         2.14ms              3.73ms            A 1.7x faster
//   3         2.68ms              3.55ms            A 1.3x faster
//   4         1.39ms              3.97ms            A 2.9x faster
//   5         3.02ms              3.60ms            A 1.2x faster
//   6         2.73ms              3.77ms            A 1.4x faster
//   7         2.59ms              3.79ms            A 1.5x faster
//   8         1.39ms              3.99ms            A 2.9x faster
//
// Conclusion: Option A wins 1.3-2.9x across all bit widths.
// ============================================================================

namespace {

constexpr uint32_t kUint8NumBlocks = 8192;
constexpr uint32_t kUint8TotalValues = kBlockSize * kUint8NumBlocks;

struct Uint8PackedData {
  std::vector<char> bits;
  uint32_t bytesPerBlock;
};

std::vector<uint8_t> uint8OutputA(kUint8TotalValues);
std::vector<uint8_t> uint8OutputB(kUint8TotalValues);

Uint8PackedData uint8PackedBw[9]; // index 1..8

Uint8PackedData packUint8Data(uint8_t bitWidth) {
  const auto blockBytes = FixedBitArray::bufferSize(kBlockSize, bitWidth);
  const auto totalBytes = blockBytes * kUint8NumBlocks;
  std::vector<char> bits(totalBytes, 0);
  std::mt19937 rng(42);
  const uint32_t maxVal = (1u << bitWidth) - 1;
  for (uint32_t b = 0; b < kUint8NumBlocks; ++b) {
    FixedBitArray fba(bits.data() + b * blockBytes, bitWidth);
    for (uint32_t i = 0; i < kBlockSize; ++i) {
      fba.set(i, rng() % (maxVal + 1));
    }
  }
  return {std::move(bits), static_cast<uint32_t>(blockBytes)};
}

void decodeQuarter(
    const Uint8PackedData& data,
    uint8_t bitWidth,
    uint32_t iters) {
  constexpr uint32_t kGroupSize = 8;
  while (iters--) {
    auto* out = uint8OutputA.data();
    for (uint32_t b = 0; b < kUint8NumBlocks; ++b) {
      const auto* input = reinterpret_cast<const uint8_t*>(
          data.bits.data() + b * data.bytesPerBlock);
      for (uint32_t r = 0; r < kBlockSize; r += kGroupSize) {
        facebook::velox::fastpforlib::internal::fastunpack_quarter(
            input, out + b * kBlockSize + r, bitWidth);
        input += bitWidth;
      }
    }
  }
}

void decode32Narrow(
    const Uint8PackedData& data,
    uint8_t bitWidth,
    uint32_t iters) {
  constexpr uint32_t kGroupSize = 32;
  uint32_t tmp[kGroupSize];
  while (iters--) {
    auto* out = uint8OutputB.data();
    for (uint32_t b = 0; b < kUint8NumBlocks; ++b) {
      const auto* input = reinterpret_cast<const uint32_t*>(
          data.bits.data() + b * data.bytesPerBlock);
      for (uint32_t r = 0; r < kBlockSize; r += kGroupSize) {
        facebook::velox::fastpforlib::fastunpack(input, tmp, bitWidth);
        for (uint32_t i = 0; i < kGroupSize; ++i) {
          out[b * kBlockSize + r + i] = static_cast<uint8_t>(tmp[i]);
        }
        input += bitWidth;
      }
    }
  }
}

void setupUint8Data() {
  for (uint8_t bw = 1; bw <= 8; ++bw) {
    uint8PackedBw[bw] = packUint8Data(bw);
  }
}

} // namespace

#define BENCHMARK_UINT8_BW(bw)                    \
  BENCHMARK(quarter_bw##bw, iters) {              \
    decodeQuarter(uint8PackedBw[bw], bw, iters);  \
  }                                               \
  BENCHMARK_RELATIVE(narrow32_bw##bw, iters) {    \
    decode32Narrow(uint8PackedBw[bw], bw, iters); \
  }                                               \
  BENCHMARK_DRAW_LINE();

BENCHMARK_UINT8_BW(1)
BENCHMARK_UINT8_BW(2)
BENCHMARK_UINT8_BW(3)
BENCHMARK_UINT8_BW(4)
BENCHMARK_UINT8_BW(5)
BENCHMARK_UINT8_BW(6)
BENCHMARK_UINT8_BW(7)
BENCHMARK_UINT8_BW(8)

// ============================================================================
// Section 3: Multi-type FBA vs Lemire — sparse and dense across types and
// bit widths. Reuses PackedBlocks, kNumBlocks, kBlockSize from Section 1.
// ============================================================================

namespace {

PackedBlocks packBlocks(uint8_t bw) {
  const auto blockBytes = FixedBitArray::bufferSize(kBlockSize, bw);
  PackedBlocks result;
  result.bytesPerBlock = blockBytes;
  result.bits.resize(blockBytes * kNumBlocks, 0);
  std::mt19937 rng(42);
  for (uint32_t b = 0; b < kNumBlocks; ++b) {
    FixedBitArray fba(result.bits.data() + b * blockBytes, bw);
    for (uint32_t i = 0; i < kBlockSize; ++i) {
      if (bw >= 64) {
        fba.set(i, rng());
      } else {
        fba.set(i, rng() % (1ULL << bw));
      }
    }
  }
  return result;
}

// Lemire decode — type-specialized. Dense decodes directly; sparse via temp.
template <typename T>
void lemireTyped(
    const PackedBlocks& data,
    uint8_t bw,
    const std::vector<uint32_t>& rows,
    uint32_t iters);

template <>
void lemireTyped<uint8_t>(
    const PackedBlocks& data,
    uint8_t bw,
    const std::vector<uint32_t>& rows,
    uint32_t iters) {
  constexpr uint32_t kGroup = 8;
  const bool isDense = rows.size() == kBlockSize;
  std::vector<uint8_t> out(rows.size() * kNumBlocks);
  while (iters--) {
    auto* dst = out.data();
    for (uint32_t b = 0; b < kNumBlocks; ++b) {
      const auto* in = reinterpret_cast<const uint8_t*>(
          data.bits.data() + b * data.bytesPerBlock);
      if (isDense) {
        for (uint32_t r = 0; r < kBlockSize; r += kGroup) {
          facebook::velox::fastpforlib::internal::fastunpack_quarter(
              in, dst + r, bw);
          in += bw;
        }
      } else {
        uint8_t tmp[kBlockSize];
        for (uint32_t r = 0; r < kBlockSize; r += kGroup) {
          facebook::velox::fastpforlib::internal::fastunpack_quarter(
              in, tmp + r, bw);
          in += bw;
        }
        for (uint32_t i = 0; i < rows.size(); ++i) {
          dst[i] = tmp[rows[i]];
        }
      }
      dst += rows.size();
    }
  }
}

template <>
void lemireTyped<uint16_t>(
    const PackedBlocks& data,
    uint8_t bw,
    const std::vector<uint32_t>& rows,
    uint32_t iters) {
  constexpr uint32_t kGroup = 16;
  const bool isDense = rows.size() == kBlockSize;
  std::vector<uint16_t> out(rows.size() * kNumBlocks);
  while (iters--) {
    auto* dst = out.data();
    for (uint32_t b = 0; b < kNumBlocks; ++b) {
      const auto* in = reinterpret_cast<const uint16_t*>(
          data.bits.data() + b * data.bytesPerBlock);
      if (isDense) {
        for (uint32_t r = 0; r < kBlockSize; r += kGroup) {
          facebook::velox::fastpforlib::internal::fastunpack_half(
              in, dst + r, bw);
          in += bw;
        }
      } else {
        uint16_t tmp[kBlockSize];
        for (uint32_t r = 0; r < kBlockSize; r += kGroup) {
          facebook::velox::fastpforlib::internal::fastunpack_half(
              in, tmp + r, bw);
          in += bw;
        }
        for (uint32_t i = 0; i < rows.size(); ++i) {
          dst[i] = tmp[rows[i]];
        }
      }
      dst += rows.size();
    }
  }
}

template <>
void lemireTyped<uint32_t>(
    const PackedBlocks& data,
    uint8_t bw,
    const std::vector<uint32_t>& rows,
    uint32_t iters) {
  constexpr uint32_t kGroup = 32;
  const bool isDense = rows.size() == kBlockSize;
  std::vector<uint32_t> out(rows.size() * kNumBlocks);
  while (iters--) {
    auto* dst = out.data();
    for (uint32_t b = 0; b < kNumBlocks; ++b) {
      const auto* in = reinterpret_cast<const uint32_t*>(
          data.bits.data() + b * data.bytesPerBlock);
      if (isDense) {
        for (uint32_t r = 0; r < kBlockSize; r += kGroup) {
          facebook::velox::fastpforlib::fastunpack(in, dst + r, bw);
          in += bw;
        }
      } else {
        uint32_t tmp[kBlockSize];
        for (uint32_t r = 0; r < kBlockSize; r += kGroup) {
          facebook::velox::fastpforlib::fastunpack(in, tmp + r, bw);
          in += bw;
        }
        for (uint32_t i = 0; i < rows.size(); ++i) {
          dst[i] = tmp[rows[i]];
        }
      }
      dst += rows.size();
    }
  }
}

template <>
void lemireTyped<uint64_t>(
    const PackedBlocks& data,
    uint8_t bw,
    const std::vector<uint32_t>& rows,
    uint32_t iters) {
  constexpr uint32_t kGroup = 32;
  const bool isDense = rows.size() == kBlockSize;
  std::vector<uint64_t> out(rows.size() * kNumBlocks);
  while (iters--) {
    auto* dst = out.data();
    for (uint32_t b = 0; b < kNumBlocks; ++b) {
      const auto* in = reinterpret_cast<const uint32_t*>(
          data.bits.data() + b * data.bytesPerBlock);
      if (isDense) {
        for (uint32_t r = 0; r < kBlockSize; r += kGroup) {
          facebook::velox::fastpforlib::fastunpack(in, dst + r, bw);
          in += bw;
        }
      } else {
        uint64_t tmp[kBlockSize];
        for (uint32_t r = 0; r < kBlockSize; r += kGroup) {
          facebook::velox::fastpforlib::fastunpack(in, tmp + r, bw);
          in += bw;
        }
        for (uint32_t i = 0; i < rows.size(); ++i) {
          dst[i] = tmp[rows[i]];
        }
      }
      dst += rows.size();
    }
  }
}

// FBA per-element decode — type-generic.
template <typename T>
void fbaTyped(
    const PackedBlocks& data,
    uint8_t bw,
    const std::vector<uint32_t>& rows,
    uint32_t iters) {
  std::vector<T> out(rows.size() * kNumBlocks);
  while (iters--) {
    auto* dst = out.data();
    for (uint32_t b = 0; b < kNumBlocks; ++b) {
      FixedBitArray fba{
          {data.bits.data() + b * data.bytesPerBlock,
           static_cast<size_t>(data.bytesPerBlock)},
          bw};
      for (uint32_t i = 0; i < rows.size(); ++i) {
        dst[i] = static_cast<T>(fba.get(rows[i]));
      }
      dst += rows.size();
    }
  }
}

PackedBlocks bwData[65]; // index 1..64

void setupMultiTypeData() {
  for (uint8_t bw = 1; bw <= 64; ++bw) {
    bwData[bw] = packBlocks(bw);
  }
}

} // namespace

// --- uint8: FBA vs Lemire ---

#define BM_U8_SPARSE(bw, sel, rows)                             \
  BENCHMARK(u8_bw##bw##_sel##sel##pct_fba, iters) {             \
    fbaTyped<uint8_t>(bwData[bw], bw, rows, iters);             \
  }                                                             \
  BENCHMARK_RELATIVE(u8_bw##bw##_sel##sel##pct_lemire, iters) { \
    lemireTyped<uint8_t>(bwData[bw], bw, rows, iters);          \
  }                                                             \
  BENCHMARK_DRAW_LINE();

#define BM_U8(bw)                     \
  BM_U8_SPARSE(bw, 1, sparseRows10)   \
  BM_U8_SPARSE(bw, 10, sparseRows100) \
  BM_U8_SPARSE(bw, 25, sparseRows256) \
  BM_U8_SPARSE(bw, 50, sparseRows512) \
  BM_U8_SPARSE(bw, 90, sparseRows922)

BM_U8(1)
BM_U8(4)
BM_U8(8)

// --- uint16: FBA vs Lemire at multiple selectivities ---

#define BM_U16_SPARSE(bw, sel, rows)                             \
  BENCHMARK(u16_bw##bw##_sel##sel##pct_fba, iters) {             \
    fbaTyped<uint16_t>(bwData[bw], bw, rows, iters);             \
  }                                                              \
  BENCHMARK_RELATIVE(u16_bw##bw##_sel##sel##pct_lemire, iters) { \
    lemireTyped<uint16_t>(bwData[bw], bw, rows, iters);          \
  }                                                              \
  BENCHMARK_DRAW_LINE();

#define BM_U16_DENSE(bw)                                         \
  BENCHMARK(u16_bw##bw##_dense_lemire, iters) {                  \
    lemireTyped<uint16_t>(bwData[bw], bw, denseRows1024, iters); \
  }                                                              \
  BENCHMARK_DRAW_LINE();

#define BM_U16_ALL(bw)                 \
  BM_U16_SPARSE(bw, 1, sparseRows10)   \
  BM_U16_SPARSE(bw, 10, sparseRows100) \
  BM_U16_SPARSE(bw, 25, sparseRows256) \
  BM_U16_SPARSE(bw, 50, sparseRows512) \
  BM_U16_SPARSE(bw, 70, sparseRows717) \
  BM_U16_SPARSE(bw, 90, sparseRows922) \
  BM_U16_DENSE(bw)

BM_U16_ALL(1)
BM_U16_ALL(4)
BM_U16_ALL(8)
BM_U16_ALL(16)

// --- uint32: FBA vs Lemire at multiple selectivities ---

#define BM_U32_SPARSE(bw, sel, rows)                             \
  BENCHMARK(u32_bw##bw##_sel##sel##pct_fba, iters) {             \
    fbaTyped<uint32_t>(bwData[bw], bw, rows, iters);             \
  }                                                              \
  BENCHMARK_RELATIVE(u32_bw##bw##_sel##sel##pct_lemire, iters) { \
    lemireTyped<uint32_t>(bwData[bw], bw, rows, iters);          \
  }                                                              \
  BENCHMARK_DRAW_LINE();

#define BM_U32_DENSE(bw)                                         \
  BENCHMARK(u32_bw##bw##_dense_lemire, iters) {                  \
    lemireTyped<uint32_t>(bwData[bw], bw, denseRows1024, iters); \
  }                                                              \
  BENCHMARK_DRAW_LINE();

#define BM_U32_ALL(bw)                  \
  BM_U32_SPARSE(bw, 1, sparseRows10)    \
  BM_U32_SPARSE(bw, 10, sparseRows100)  \
  BM_U32_SPARSE(bw, 20, sparseRows205)  \
  BM_U32_SPARSE(bw, 25, sparseRows256)  \
  BM_U32_SPARSE(bw, 30, sparseRows307)  \
  BM_U32_SPARSE(bw, 40, sparseRows410)  \
  BM_U32_SPARSE(bw, 50, sparseRows512)  \
  BM_U32_SPARSE(bw, 60, sparseRows614)  \
  BM_U32_SPARSE(bw, 70, sparseRows717)  \
  BM_U32_SPARSE(bw, 80, sparseRows819)  \
  BM_U32_SPARSE(bw, 90, sparseRows922)  \
  BM_U32_SPARSE(bw, 95, sparseRows973)  \
  BM_U32_SPARSE(bw, 99, sparseRows1014) \
  BM_U32_DENSE(bw)

BM_U32_ALL(16)
BM_U32_ALL(17)
BM_U32_ALL(18)
BM_U32_ALL(19)
BM_U32_ALL(20)
BM_U32_ALL(21)
BM_U32_ALL(22)
BM_U32_ALL(23)
BM_U32_ALL(24)
BM_U32_ALL(25)
BM_U32_ALL(26)
BM_U32_ALL(27)
BM_U32_ALL(28)
BM_U32_ALL(29)
BM_U32_ALL(30)
BM_U32_ALL(31)
BM_U32_ALL(32)

// --- uint64: FBA vs Lemire at multiple selectivities ---

#define BM_U64_SPARSE(bw, sel, rows)                             \
  BENCHMARK(u64_bw##bw##_sel##sel##pct_fba, iters) {             \
    fbaTyped<uint64_t>(bwData[bw], bw, rows, iters);             \
  }                                                              \
  BENCHMARK_RELATIVE(u64_bw##bw##_sel##sel##pct_lemire, iters) { \
    lemireTyped<uint64_t>(bwData[bw], bw, rows, iters);          \
  }                                                              \
  BENCHMARK_DRAW_LINE();

#define BM_U64_ALL(bw)                 \
  BM_U64_SPARSE(bw, 1, sparseRows10)   \
  BM_U64_SPARSE(bw, 10, sparseRows100) \
  BM_U64_SPARSE(bw, 20, sparseRows205) \
  BM_U64_SPARSE(bw, 25, sparseRows256) \
  BM_U64_SPARSE(bw, 30, sparseRows307) \
  BM_U64_SPARSE(bw, 40, sparseRows410) \
  BM_U64_SPARSE(bw, 50, sparseRows512) \
  BM_U64_SPARSE(bw, 60, sparseRows614) \
  BM_U64_SPARSE(bw, 70, sparseRows717) \
  BM_U64_SPARSE(bw, 80, sparseRows819) \
  BM_U64_SPARSE(bw, 90, sparseRows922) \
  BM_U64_SPARSE(bw, 95, sparseRows973) \
  BM_U64_SPARSE(bw, 99, sparseRows1014)

BM_U64_ALL(16)
BM_U64_ALL(17)
BM_U64_ALL(18)
BM_U64_ALL(19)
BM_U64_ALL(20)
BM_U64_ALL(21)
BM_U64_ALL(22)
BM_U64_ALL(23)
BM_U64_ALL(24)
BM_U64_ALL(25)
BM_U64_ALL(26)
BM_U64_ALL(27)
BM_U64_ALL(28)
BM_U64_ALL(29)
BM_U64_ALL(30)
BM_U64_ALL(31)
BM_U64_ALL(32)
BM_U64_ALL(33)
BM_U64_ALL(34)
BM_U64_ALL(35)
BM_U64_ALL(36)
BM_U64_ALL(37)
BM_U64_ALL(38)
BM_U64_ALL(39)
BM_U64_ALL(40)
BM_U64_ALL(41)
BM_U64_ALL(42)
BM_U64_ALL(43)
BM_U64_ALL(44)
BM_U64_ALL(45)
BM_U64_ALL(46)
BM_U64_ALL(47)
BM_U64_ALL(48)
BM_U64_ALL(49)
BM_U64_ALL(50)
BM_U64_ALL(51)
BM_U64_ALL(52)
BM_U64_ALL(53)
BM_U64_ALL(54)
BM_U64_ALL(55)
BM_U64_ALL(56)
BM_U64_ALL(57)
BM_U64_ALL(58)
BM_U64_ALL(59)
BM_U64_ALL(60)
BM_U64_ALL(61)
BM_U64_ALL(62)
BM_U64_ALL(63)
BM_U64_ALL(64)

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  setupData();
  setupUint8Data();
  setupMultiTypeData();
  folly::runBenchmarks();
  return 0;
}
