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

#include "velox/dwio/common/BitPackDecoder.h"
#include "velox/common/base/Nulls.h"
#include "velox/dwio/parquet/reader/RleBpDataDecoder.h"

#include <folly/Random.h>
#include <gflags/gflags.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstring>

DECLARE_bool(bmi2); // NOLINT

using namespace facebook::velox::dwio::common;
using namespace facebook::velox;

class BitPackDecoderTest : public testing::Test {
 protected:
  void SetUp() {
    for (int32_t i = 0; i < 100003; i++) {
      auto randomInt = folly::Random::rand64();
      randomInts_.push_back(randomInt);
    }
    populateBitPackedData();
    allRowNumbers_.resize(randomInts_.size());
    std::iota(allRowNumbers_.begin(), allRowNumbers_.end(), 0);
    oddRowNumbers_.resize(randomInts_.size() / 2);
    for (auto i = 0; i < oddRowNumbers_.size(); i++) {
      oddRowNumbers_[i] = i * 2 + 1;
    }
    allRows_ = RowSet(allRowNumbers_);
    oddRows_ = RowSet(oddRowNumbers_);
  }

  void populateBitPackedData() {
    bitPackedData_.resize(33);
    for (auto bitWidth = 1; bitWidth <= 32; ++bitWidth) {
      auto numWords = bits::divRoundUp(randomInts_.size() * bitWidth, 64);
      bitPackedData_[bitWidth].resize(numWords);
      auto source = randomInts_.data();
      auto destination =
          reinterpret_cast<uint64_t*>(bitPackedData_[bitWidth].data());
      for (auto i = 0; i < randomInts_.size(); ++i) {
        bits::copyBits(
            source,
            i * sizeof(*source) * 8,
            destination,
            i * bitWidth,
            bitWidth);
      }
    }
  }

  template <typename T, typename U>
  void checkDecodeResult(
      const T* reference,
      RowSet rows,
      int8_t bitWidth,
      const U* result) {
    uint64_t mask = bits::lowMask(bitWidth);
    for (auto i = 0; i < rows.size(); ++i) {
      uint64_t original = reference[rows[i]] & mask;
      ASSERT_EQ(original, result[i])
          << " at " << i << " with bitWidth " << bitWidth;
    }
  }

  template <typename T>
  void testUnpack(uint8_t width, RowSet rows) {
    std::vector<T> result(rows.size());
    int32_t start = 0;

    int32_t batch = 1;
    // Read the encoding in progressively larger batches, each time 3x more than
    // previous.
    auto bits = bitPackedData_[width].data();
    do {
      auto row = rows[start];
      int32_t bit = row * width;
      auto byteOffset = bit / 8;
      auto bitOffset = bit & 7;
      auto numRows = std::min<int32_t>(start + batch, rows.size()) - start;
      auto bitsPointer = reinterpret_cast<const uint64_t*>(
          reinterpret_cast<const char*>(bits) + byteOffset);

      // end is the first unaddressable address after the bit packed data. We
      // set this to be the byte of the last bit field to exercise the safe
      // path.
      auto end = reinterpret_cast<const char*>(bitsPointer) +
          (((start + rows[numRows - 1] - rows[start]) * width) / 8);
      unpack(
          bitsPointer,
          bitOffset,
          RowSet(&rows[start], numRows),
          rows[start],
          width,
          end,
          result.data() + start);
      start += batch;
      batch *= 3;
    } while (start < rows.size());
    checkDecodeResult(randomInts_.data(), rows, width, result.data());
  }

  uint32_t bytes(uint64_t numValues, uint8_t bitWidth) {
    return (numValues * bitWidth + 7) / 8;
  }

  // Tests
  template <typename T>
  void testUnpack(uint8_t bitWidth) {
    auto numValues = randomInts_.size();
    std::vector<T> result(numValues);

    const uint8_t* inputIter =
        reinterpret_cast<const uint8_t*>(bitPackedData_[bitWidth].data());
    T* outputIter = reinterpret_cast<T*>(result.data());
    facebook::velox::dwio::common::unpack<T>(
        inputIter, bytes(numValues, bitWidth), numValues, bitWidth, outputIter);

    checkDecodeResult(randomInts_.data(), allRows_, bitWidth, result.data());
  }

  std::vector<uint64_t> randomInts_;

  // All indices into 'randomInts_'.
  std::vector<int32_t> allRowNumbers_;

  // Indices into odd positions in 'randomInts_'.
  std::vector<int32_t> oddRowNumbers_;

  // Array of bit packed representations of randomInts_. The array at index i
  // is packed i bits wide and the values come from the low bits of
  std::vector<std::vector<uint64_t>> bitPackedData_;
  RowSet allRows_;
  RowSet oddRows_;
};

// Parameterized fixture that exercises both FLAGS_bmi2=true (PDEP path) and
// FLAGS_bmi2=false (shift+mask fallback) on every CI run.
class BitPackDecoderBmi2Test : public BitPackDecoderTest,
                               public testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    savedBmi2_ = FLAGS_bmi2;
    FLAGS_bmi2 = GetParam(); // NOLINT
    BitPackDecoderTest::SetUp();
  }

  void TearDown() override {
    FLAGS_bmi2 = savedBmi2_; // NOLINT
  }

 private:
  bool savedBmi2_;
};

TEST_F(BitPackDecoderTest, allWidths) {
  for (auto width = 0; width < bitPackedData_.size() - 1; ++width) {
    testUnpack<int32_t>(width, allRows_);
    testUnpack<int64_t>(width, allRows_);
    testUnpack<int32_t>(width, oddRows_);
    testUnpack<int64_t>(width, oddRows_);
  }
}

TEST_P(BitPackDecoderBmi2Test, uint8AllRows) {
  for (auto width = 1; width <= 8; ++width) {
    testUnpack<uint8_t>(width);
  }
}

TEST_P(BitPackDecoderBmi2Test, uint16AllRows) {
  for (auto width = 1; width <= 16; ++width) {
    testUnpack<uint16_t>(width);
  }
}

TEST_P(BitPackDecoderBmi2Test, uint32AllRows) {
  for (auto width = 1; width <= 32; ++width) {
    testUnpack<uint32_t>(width);
  }
}

// Edge case: input buffer is exactly the minimum size with no padding.
// Verifies that the fast paths correctly fall through to unpackNaive when
// fewer than sizeof(uint64_t) bytes remain, avoiding out-of-bounds reads.
TEST_P(BitPackDecoderBmi2Test, smallBufferNoPadding) {
  // Test uint8_t: 8 values at bitWidth=1 -> exactly 1 byte of packed data.
  {
    // Pack 8 known bit values: alternating 0,1,0,1,0,1,0,1 = 0xAA
    uint8_t packed = 0xAA;
    const uint8_t* input = &packed;
    uint8_t output[8] = {};
    uint8_t* outputPtr = output;
    uint64_t bufLen = 1; // Exactly 1 byte, no padding.
    facebook::velox::dwio::common::unpack<uint8_t>(
        input, bufLen, 8, 1, outputPtr);
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(output[i], (0xAA >> i) & 1) << "uint8 at index " << i;
    }
  }

  // Test uint8_t: 8 values at bitWidth=3 -> exactly 3 bytes of packed data.
  {
    // Values: 0,1,2,3,4,5,6,7 packed at 3 bits each = 24 bits = 3 bytes.
    // Bit layout: 000 001 010 011 100 101 110 111
    // LSB first:  0b11_101_100_011_010_001_000 spread across 3 bytes.
    uint8_t packed[3];
    // Pack manually: value[i] = i, each 3 bits wide, LSB first.
    uint64_t bits = 0;
    for (int i = 0; i < 8; ++i) {
      bits |= (uint64_t)i << (i * 3);
    }
    std::memcpy(packed, &bits, 3);

    const uint8_t* input = packed;
    uint8_t output[8] = {};
    uint8_t* outputPtr = output;
    facebook::velox::dwio::common::unpack<uint8_t>(input, 3, 8, 3, outputPtr);
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(output[i], i) << "uint8 bw3 at index " << i;
    }
  }

  // Test uint32_t: 4 values at bitWidth=2 -> exactly 1 byte of packed data.
  {
    // Values: 3,2,1,0 packed at 2 bits each = 8 bits = 1 byte.
    // Bit layout LSB first: 11 10 01 00 = 0b00011011 = 0x1B
    uint8_t packed = 0x1B;
    const uint8_t* input = &packed;
    uint32_t output[4] = {};
    uint32_t* outputPtr = output;
    facebook::velox::dwio::common::unpack<uint32_t>(input, 1, 4, 2, outputPtr);
    EXPECT_EQ(output[0], 3);
    EXPECT_EQ(output[1], 2);
    EXPECT_EQ(output[2], 1);
    EXPECT_EQ(output[3], 0);
  }

  // Test uint32_t: 8 values at bitWidth=4 -> exactly 4 bytes of packed data.
  {
    // Values: 0..7 packed at 4 bits each = 32 bits = 4 bytes.
    uint8_t packed[4];
    uint32_t bits = 0;
    for (int i = 0; i < 8; ++i) {
      bits |= (uint32_t)i << (i * 4);
    }
    std::memcpy(packed, &bits, 4);

    const uint8_t* input = packed;
    uint32_t output[8] = {};
    uint32_t* outputPtr = output;
    facebook::velox::dwio::common::unpack<uint32_t>(input, 4, 8, 4, outputPtr);
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(output[i], i) << "uint32 bw4 at index " << i;
    }
  }

  // Test uint16_t: 16 values at bitWidth=1 -> exactly 2 bytes of packed data.
  {
    // Values: alternating 0,1 packed at 1 bit each = 16 bits = 2 bytes.
    uint8_t packed[2] = {0xAA, 0xAA}; // 01010101 01010101
    const uint8_t* input = packed;
    uint16_t output[16] = {};
    uint16_t* outputPtr = output;
    facebook::velox::dwio::common::unpack<uint16_t>(input, 2, 16, 1, outputPtr);
    for (int i = 0; i < 16; ++i) {
      EXPECT_EQ(output[i], (i % 2 == 0) ? 0 : 1) << "uint16 bw1 at index " << i;
    }
  }

  // Test uint16_t: 16 values at bitWidth=2 -> exactly 4 bytes of packed data.
  {
    // Values: 0,1,2,3 repeating packed at 2 bits each = 32 bits = 4 bytes.
    uint8_t packed[4];
    uint32_t bits = 0;
    for (int i = 0; i < 16; ++i) {
      bits |= (uint32_t)(i % 4) << (i * 2);
    }
    std::memcpy(packed, &bits, 4);

    const uint8_t* input = packed;
    uint16_t output[16] = {};
    uint16_t* outputPtr = output;
    facebook::velox::dwio::common::unpack<uint16_t>(input, 4, 16, 2, outputPtr);
    for (int i = 0; i < 16; ++i) {
      EXPECT_EQ(output[i], i % 4) << "uint16 bw2 at index " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Bmi2,
    BitPackDecoderBmi2Test,
    testing::Values(true, false),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "bmi2_on" : "bmi2_off";
    });

// Verifies that unpack correctly advances both the input and result pointers
// when the naive fallback handles all values (buffer too small for fast path).
TEST_F(BitPackDecoderTest, naiveFallbackAdvancesPointers) {
  // First byte: 0,1,0,1,0,1,0,1.
  // Second byte: 1,1,1,1,0,0,0,0.
  const uint8_t packed[] = {0xAA, 0x0F};
  const uint8_t* input = packed;
  const uint8_t* inputEnd = packed + sizeof(packed);

  uint8_t output[16] = {};
  uint8_t* result = output;

  facebook::velox::dwio::common::unpack<uint8_t>(
      input, inputEnd - input, 8, 1, result);

  EXPECT_EQ(input, packed + 1);
  EXPECT_EQ(result, output + 8);

  facebook::velox::dwio::common::unpack<uint8_t>(
      input, inputEnd - input, 8, 1, result);

  EXPECT_EQ(input, inputEnd);
  EXPECT_EQ(result, output + 16);

  const uint8_t expected[] = {0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0};
  EXPECT_THAT(output, ::testing::ElementsAreArray(expected));
}
