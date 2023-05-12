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
velox_unpack_fullrows_switch_1_8                          404.06us     2.47K
avx512_new_unpack_fullrows_switch_1_8           121.02%   333.87us     3.00K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_2_8                          435.25us     2.30K
avx512_new_unpack_fullrows_switch_2_8           117.34%   370.94us     2.70K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_3_8                          460.43us     2.17K
avx512_new_unpack_fullrows_switch_3_8           112.17%   410.49us     2.44K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_4_8                          491.38us     2.04K
avx512_new_unpack_fullrows_switch_4_8           111.37%   441.22us     2.27K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_5_8                          513.82us     1.95K
avx512_new_unpack_fullrows_switch_5_8           108.46%   473.76us     2.11K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_6_8                          535.98us     1.87K
avx512_new_unpack_fullrows_switch_6_8            105.7%   507.07us     1.97K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_7_8                          566.35us     1.77K
avx512_new_unpack_fullrows_switch_7_8           104.41%   542.44us     1.84K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_8_8                          588.18us     1.70K
avx512_new_unpack_fullrows_switch_8_8           106.21%   553.76us     1.81K
----------------------------------------------------------------------------
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_1_16                         741.30us     1.35K
avx512_new_unpack_fullrows_switch_1_16          116.03%   638.87us     1.57K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_2_16                         754.73us     1.32K
avx512_new_unpack_fullrows_switch_2_16          112.44%   671.22us     1.49K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_3_16                         798.92us     1.25K
avx512_new_unpack_fullrows_switch_3_16          112.72%   708.74us     1.41K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_4_16                         826.81us     1.21K
avx512_new_unpack_fullrows_switch_4_16          111.65%   740.56us     1.35K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_5_16                         856.92us     1.17K
avx512_new_unpack_fullrows_switch_5_16          110.29%   776.95us     1.29K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_6_16                         886.25us     1.13K
avx512_new_unpack_fullrows_switch_6_16          109.19%   811.63us     1.23K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_7_16                           1.06ms    947.71
avx512_new_unpack_fullrows_switch_7_16          124.14%   849.99us     1.18K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_8_16                         985.99us     1.01K
avx512_new_unpack_fullrows_switch_8_16          109.04%   904.25us     1.11K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_9_16                           1.36ms    736.46
avx512_new_unpack_fullrows_switch_9_16          146.04%   929.79us     1.08K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_10_16                          1.20ms    833.48
avx512_new_unpack_fullrows_switch_10_16         123.66%   970.26us     1.03K
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_11_16                          1.60ms    624.14
avx512_new_unpack_fullrows_switch_11_16         158.88%     1.01ms    991.60
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_12_16                          1.28ms    779.54
avx512_new_unpack_fullrows_switch_12_16         122.34%     1.05ms    953.68
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_13_16                          1.87ms    535.54
avx512_new_unpack_fullrows_switch_13_16         171.32%     1.09ms    917.46
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_14_16                          1.69ms    589.99
avx512_new_unpack_fullrows_switch_14_16         150.03%     1.13ms    885.16
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_15_16                          1.48ms    674.56
avx512_new_unpack_fullrows_switch_15_16         125.67%     1.18ms    847.73
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_16_16                        959.37us     1.04K
avx512_new_unpack_fullrows_switch_16_16         100.03%   959.11us     1.04K
----------------------------------------------------------------------------
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_1_32                           1.77ms    564.97
avx512_new_unpack_fullrows_switch_1_32          132.26%     1.34ms    747.21
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_2_32                           1.81ms    552.36
avx512_new_unpack_fullrows_switch_2_32          131.41%     1.38ms    725.87
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_3_32                           1.86ms    536.52
avx512_new_unpack_fullrows_switch_3_32          130.76%     1.43ms    701.53
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_4_32                           1.91ms    523.60
avx512_new_unpack_fullrows_switch_4_32          129.92%     1.47ms    680.26
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_5_32                           1.96ms    509.08
avx512_new_unpack_fullrows_switch_5_32          125.82%     1.56ms    640.54
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_6_32                           2.02ms    494.05
avx512_new_unpack_fullrows_switch_6_32          124.66%     1.62ms    615.88
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_7_32                           2.09ms    478.68
avx512_new_unpack_fullrows_switch_7_32          121.69%     1.72ms    582.52
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_8_32                           2.15ms    464.74
avx512_new_unpack_fullrows_switch_8_32          128.88%     1.67ms    598.94
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_9_32                           2.05ms    488.40
avx512_new_unpack_fullrows_switch_9_32          117.06%     1.75ms    571.74
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_10_32                          2.11ms    474.41
avx512_new_unpack_fullrows_switch_10_32         116.39%     1.81ms    552.19
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_11_32                          2.23ms    448.61
avx512_new_unpack_fullrows_switch_11_32         118.79%     1.88ms    532.92
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_13_32                          2.34ms    426.80
avx512_new_unpack_fullrows_switch_13_32         114.72%     2.04ms    489.60
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_15_32                          2.58ms    387.42
avx512_new_unpack_fullrows_switch_15_32         118.68%     2.17ms    459.78
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_17_32                          2.92ms    342.05
avx512_new_unpack_fullrows_switch_17_32         129.62%     2.26ms    443.36
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_19_32                          3.03ms    330.48
avx512_new_unpack_fullrows_switch_19_32         125.36%     2.41ms    414.30
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_21_32                          3.13ms    319.70
avx512_new_unpack_fullrows_switch_21_32         123.02%     2.54ms    393.29
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_24_32                          3.23ms    309.13
avx512_new_unpack_fullrows_switch_24_32         117.26%     2.76ms    362.49
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_28_32                          3.62ms    276.24
avx512_new_unpack_fullrows_switch_28_32         118.76%     3.05ms    328.06
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_30_32                          3.71ms    269.51
avx512_new_unpack_fullrows_switch_30_32          115.5%     3.21ms    311.28
----------------------------------------------------------------------------
velox_unpack_fullrows_switch_32_32                          3.41ms    292.83
avx512_new_unpack_fullrows_switch_32_32         101.94%     3.35ms    298.53
----------------------------------------------------------------------------
*/