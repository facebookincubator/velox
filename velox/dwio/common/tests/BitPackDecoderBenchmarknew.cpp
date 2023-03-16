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

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/common/BitPackDecoder.h"
#include "velox/vector/TypeAliases.h"

#include <folly/Benchmark.h>
#include <folly/Random.h>
#include <folly/init/Init.h>
#include <fstream>
#include <iostream>

using namespace folly;
using namespace facebook::velox;

using RowSet = folly::Range<const facebook::velox::vector_size_t*>;

static const uint64_t kNumValues = 1024768 * 8;

// Array of bit packed representations of randomInts_u32. The array at index i
// is packed i bits wide and the values come from the low bits of
std::vector<std::vector<uint64_t>> bitPackedData;

std::vector<uint8_t> result8;
std::vector<uint16_t> result16;
std::vector<uint32_t> result32;

std::vector<int32_t> allRowNumbers;
std::vector<int32_t> oddRowNumbers;
RowSet allRows;
RowSet oddRows;

static size_t len_u32 = 0;
std::vector<uint32_t> randomInts_u32;
std::vector<uint64_t> randomInts_u32_result;

#define BYTES(numValues, bitWidth) (numValues * bitWidth + 7) / 8

template <typename T>
void unpackAVX512_new(uint8_t bitWidth, T* result) {
  const uint8_t* inputIter =
      reinterpret_cast<const uint8_t*>(bitPackedData[bitWidth].data());
  facebook::velox::dwio::common::unpackAVX512<T>(
      inputIter, BYTES(kNumValues, bitWidth), kNumValues, bitWidth, result);
}

template <typename T>
void veloxBitUnpack(uint8_t bitWidth, T* result) {
  const uint8_t* inputIter =
      reinterpret_cast<const uint8_t*>(bitPackedData[bitWidth].data());
  facebook::velox::dwio::common::unpackAVX2<T>(
      inputIter, BYTES(kNumValues, bitWidth), kNumValues, bitWidth, result);
}

template <typename T>
void legacyUnpackFast(RowSet rows, uint8_t bitWidth, T* result) {
  auto data = bitPackedData[bitWidth].data();
  auto numBytes = bits::roundUp((rows.back() + 1) * bitWidth, 8) / 8;
  auto end = reinterpret_cast<const char*>(data) + numBytes;
  facebook::velox::dwio::common::unpack(
      data,
      0,
      rows,
      0,
      bitWidth,
      end,
      reinterpret_cast<int32_t*>(result32.data()));
}

#define BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_8(width)                \
  BENCHMARK(velox_unpack_fullrows_switch_##width##_8) {               \
    veloxBitUnpack<uint8_t>(width, result8.data());                   \
  }                                                                   \
  BENCHMARK_RELATIVE(avx512_new_unpack_fullrows_switch_##width##_8) { \
    unpackAVX512_new<uint8_t>(width, result8.data());                 \
  }                                                                   \
  BENCHMARK_DRAW_LINE();

#define BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(width)                \
  BENCHMARK(velox_unpack_fullrows_switch_##width##_16) {               \
    veloxBitUnpack<uint16_t>(width, result16.data());                  \
  }                                                                    \
  BENCHMARK_RELATIVE(avx512_new_unpack_fullrows_switch_##width##_16) { \
    unpackAVX512_new<uint16_t>(width, result16.data());                \
  }                                                                    \
  BENCHMARK_DRAW_LINE();

#define BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(width)                \
  BENCHMARK(velox_unpack_fullrows_switch_##width##_32) {               \
    veloxBitUnpack<uint32_t>(width, result32.data());                  \
  }                                                                    \
  BENCHMARK_RELATIVE(avx512_new_unpack_fullrows_switch_##width##_32) { \
    unpackAVX512_new<uint32_t>(width, result32.data());                \
  }                                                                    \
  BENCHMARK_DRAW_LINE();

#define BENCHMARK_UNPACK_ODDROWS_CASE_8(width)                      \
  BENCHMARK(legacy_velox_unpack_fast_oddrows_##width##_8) {         \
    legacyUnpackFast<uint8_t>(oddRows, width, result8.data());      \
  }                                                                 \
  BENCHMARK_RELATIVE(avx512_unpack_oddrows_##width##_8) {           \
    unpackAVX512Selective<uint8_t>(oddRows, width, result8.data()); \
  }                                                                 \
  BENCHMARK_DRAW_LINE();

#define BENCHMARK_UNPACK_ODDROWS_CASE_16(width)                       \
  BENCHMARK(legacy_velox_unpack_fast_oddrows_##width##_16) {          \
    legacyUnpackFast<uint16_t>(oddRows, width, result16.data());      \
  }                                                                   \
  BENCHMARK_RELATIVE(avx512_unpack_oddrows_##width##_16) {            \
    unpackAVX512Selective<uint16_t>(oddRows, width, result16.data()); \
  }                                                                   \
  BENCHMARK_DRAW_LINE();

#define BENCHMARK_UNPACK_ODDROWS_CASE_32(width)                       \
  BENCHMARK(legacy_velox_unpack_fast_oddrows_##width##_32) {          \
    legacyUnpackFast<uint32_t>(oddRows, width, result32.data());      \
  }                                                                   \
  BENCHMARK_RELATIVE(avx512_unpack_oddrows_##width##_32) {            \
    unpackAVX512Selective<uint32_t>(oddRows, width, result32.data()); \
  }                                                                   \
  BENCHMARK_DRAW_LINE();

BENCHMARK(warmup) {
  unpackAVX512_new<uint8_t>(1, result8.data());
  veloxBitUnpack<uint8_t>(1, result8.data());
}

BENCHMARK_DRAW_LINE();

BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_8(1)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_8(2)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_8(3)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_8(4)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_8(5)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_8(6)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_8(7)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_8(8)

BENCHMARK_DRAW_LINE();

BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(1)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(2)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(3)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(4)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(5)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(6)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(7)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(8)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(9)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(10)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(11)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(12)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(13)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(14)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(15)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_16(16)

BENCHMARK_DRAW_LINE();

BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(1)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(2)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(3)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(4)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(5)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(6)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(7)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(8)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(9)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(10)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(11)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(13)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(15)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(17)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(19)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(21)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(24)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(28)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(30)
BENCHMARK_UNPACK_FULLROWS_SWITCH_CASE_32(32)

void populateBitPacked() {
  bitPackedData.resize(33);
  for (auto bitWidth = 1; bitWidth <= 32; ++bitWidth) {
    auto numWords = bits::roundUp(randomInts_u32.size() * bitWidth, 64) / 64;
    bitPackedData[bitWidth].resize(numWords);
    auto source = reinterpret_cast<uint64_t*>(randomInts_u32.data());
    auto destination =
        reinterpret_cast<uint64_t*>(bitPackedData[bitWidth].data());
    for (auto i = 0; i < randomInts_u32.size(); ++i) {
      bits::copyBits(source, i * 32, destination, i * bitWidth, bitWidth);
    }
  }

  allRowNumbers.resize(randomInts_u32.size());
  std::iota(allRowNumbers.begin(), allRowNumbers.end(), 0);

  oddRowNumbers.resize(randomInts_u32.size() / 3 + 1);
  for (auto i = 0; i < oddRowNumbers.size(); i++) {
    oddRowNumbers[i] = i * 3 + 1;
  }

  allRows = RowSet(allRowNumbers);
  oddRows = RowSet(oddRowNumbers);
}

int32_t main(int32_t argc, char* argv[]) {
  folly::init(&argc, &argv);

  // Populate uint32 buffer
  for (int32_t i = 0; i < kNumValues; i++) {
    auto randomInt = folly::Random::rand32();
    randomInts_u32.push_back(randomInt);
  }

  randomInts_u32_result.resize(randomInts_u32.size());

  populateBitPacked();

  result8.resize(randomInts_u32.size());
  result16.resize(randomInts_u32.size());
  result32.resize(randomInts_u32.size());

  folly::runBenchmarks();
  return 0;
}

/*
============================================================================
[...]/tests/BitPackDecoderBenchmarknew.cpp     relative  time/iter   iters/s
============================================================================
velox_unpack_fullrows_switch_1_8                          419.13us     2.39K
avx512_new_unpack_fullrows_switch_1_8           124.47%   336.73us     2.97K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_2_8                          441.24us     2.27K
avx512_new_unpack_fullrows_switch_2_8           117.59%   375.24us     2.66K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_3_8                          487.36us     2.05K
avx512_new_unpack_fullrows_switch_3_8            116.9%   416.89us     2.40K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_4_8                          510.54us     1.96K
avx512_new_unpack_fullrows_switch_4_8           114.42%   446.22us     2.24K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_5_8                          538.42us     1.86K
avx512_new_unpack_fullrows_switch_5_8           111.51%   482.84us     2.07K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_6_8                          585.22us     1.71K
avx512_new_unpack_fullrows_switch_6_8           111.56%   524.56us     1.91K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_7_8                          644.45us     1.55K
avx512_new_unpack_fullrows_switch_7_8           112.84%   571.13us     1.75K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_8_8                          635.43us     1.57K
avx512_new_unpack_fullrows_switch_8_8           129.49%   490.71us     2.04K
----------------------------------------------------------------------------
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_1_16                         997.13us     1.00K
avx512_new_unpack_fullrows_switch_1_16          143.05%   697.07us     1.43K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_2_16                           1.03ms    968.25
avx512_new_unpack_fullrows_switch_2_16          141.28%   731.02us     1.37K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_3_16                           1.08ms    928.70
avx512_new_unpack_fullrows_switch_3_16          134.65%   799.68us     1.25K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_4_16                           1.14ms    876.54
avx512_new_unpack_fullrows_switch_4_16          134.89%   845.80us     1.18K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_5_16                           1.30ms    767.20
avx512_new_unpack_fullrows_switch_5_16          143.13%   910.66us     1.10K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_6_16                           1.35ms    740.65
avx512_new_unpack_fullrows_switch_6_16          141.27%   955.73us     1.05K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_7_16                           1.46ms    686.16
avx512_new_unpack_fullrows_switch_7_16           138.3%     1.05ms    948.93
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_8_16                           1.64ms    610.13
avx512_new_unpack_fullrows_switch_8_16          122.72%     1.34ms    748.73
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_9_16                           1.94ms    515.81
avx512_new_unpack_fullrows_switch_9_16          156.38%     1.24ms    806.61
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_10_16                          1.97ms    507.97
avx512_new_unpack_fullrows_switch_10_16         145.24%     1.36ms    737.80
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_11_16                          2.36ms    423.80
avx512_new_unpack_fullrows_switch_11_16         162.22%     1.45ms    687.49
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_12_16                          2.19ms    455.73
avx512_new_unpack_fullrows_switch_12_16         149.27%     1.47ms    680.29
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_13_16                          2.71ms    368.83
avx512_new_unpack_fullrows_switch_13_16         168.16%     1.61ms    620.21
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_14_16                          2.68ms    373.53
avx512_new_unpack_fullrows_switch_14_16         156.04%     1.72ms    582.85
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_15_16                          2.70ms    370.77
avx512_new_unpack_fullrows_switch_15_16         147.53%     1.83ms    547.01
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_16_16                          1.11ms    901.19
avx512_new_unpack_fullrows_switch_16_16         100.96%     1.10ms    909.88
----------------------------------------------------------------------------
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_1_32                           3.42ms    292.39
avx512_new_unpack_fullrows_switch_1_32          149.23%     2.29ms    436.33
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_2_32                           3.43ms    291.90
avx512_new_unpack_fullrows_switch_2_32          148.95%     2.30ms    434.77
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_3_32                           3.44ms    290.71
avx512_new_unpack_fullrows_switch_3_32          142.38%     2.42ms    413.92
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_4_32                           3.52ms    283.73
avx512_new_unpack_fullrows_switch_4_32          140.76%     2.50ms    399.39
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_5_32                           3.59ms    278.93
avx512_new_unpack_fullrows_switch_5_32          124.85%     2.87ms    348.25
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_6_32                           3.79ms    264.19
avx512_new_unpack_fullrows_switch_6_32          131.74%     2.87ms    348.06
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_7_32                           3.96ms    252.82
avx512_new_unpack_fullrows_switch_7_32          129.24%     3.06ms    326.73
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_8_32                           4.15ms    240.98
avx512_new_unpack_fullrows_switch_8_32          144.71%     2.87ms    348.72
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_9_32                           3.99ms    250.50
avx512_new_unpack_fullrows_switch_9_32          129.85%     3.07ms    325.27
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_10_32                          4.08ms    245.20
avx512_new_unpack_fullrows_switch_10_32         128.89%     3.16ms    316.03
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_11_32                          4.23ms    236.67
avx512_new_unpack_fullrows_switch_11_32         127.25%     3.32ms    301.17
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_13_32                          4.47ms    223.66
avx512_new_unpack_fullrows_switch_13_32         121.95%     3.67ms    272.75
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_15_32                          4.66ms    214.54
avx512_new_unpack_fullrows_switch_15_32         119.02%     3.92ms    255.34
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_17_32                          5.22ms    191.41
avx512_new_unpack_fullrows_switch_17_32         129.89%     4.02ms    248.62
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_19_32                          5.22ms    191.69
avx512_new_unpack_fullrows_switch_19_32         120.75%     4.32ms    231.46
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_21_32                          5.63ms    177.67
avx512_new_unpack_fullrows_switch_21_32         123.26%     4.57ms    219.00
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_24_32                          5.90ms    169.62
avx512_new_unpack_fullrows_switch_24_32         122.39%     4.82ms    207.60
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_28_32                          6.35ms    157.42
avx512_new_unpack_fullrows_switch_28_32         117.49%     5.41ms    184.95
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_30_32                          6.40ms    156.19
avx512_new_unpack_fullrows_switch_30_32         113.29%     5.65ms    176.95
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_32_32                          3.41ms    292.83
avx512_new_unpack_fullrows_switch_32_32         101.94%     3.35ms    298.53
----------------------------------------------------------------------------
*/