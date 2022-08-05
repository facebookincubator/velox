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

#include "velox/dwio/common/BitUnpacking.h"

#include <arrow/util/rle_encoding.h> // @manual
#include <gtest/gtest.h>

#include <random>

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;

template <typename T>
class BitUnpackingTest {
 public:
  BitUnpackingTest() {
    inputBuffer.resize(kNumValues * 4, 0);
    outputBuffer.resize(kNumValues * 4, 0);
    expectedOutputBuffer.resize(kNumValues * 4, 0);
  }

  void populateInputBuffer(
      uint8_t bitWidth,
      uint8_t* inputBuf,
      uint32_t inputBytes) {
    auto gen = std::bind(
        std::uniform_int_distribution<uint32_t>(0, (1UL << bitWidth) - 1),
        std::default_random_engine());
    arrow::bit_util::BitWriter bitWriter(inputBuf, inputBytes);
    for (auto j = 0; j < kNumValues; j++) {
      uint32_t val = gen();
      bitWriter.PutValue(val, bitWidth);
    }
    bitWriter.Flush(true);
  }

  uint32_t bytes(uint8_t bitWidth) {
    return (kNumValues * bitWidth + 7) / 8;
  }

  void testUnpack(uint8_t bitWidth) {
    populateInputBuffer(bitWidth, inputBuffer.data(), bytes(bitWidth));

    const uint8_t* inputIter = inputBuffer.data();
    T* outputIter = outputBuffer.data();
    facebook::velox::dwio::common::unpack(
        bitWidth, inputIter, bytes(bitWidth), kNumValues, outputIter);

    inputIter = inputBuffer.data();
    T* expectedOutputIter = expectedOutputBuffer.data();
    arrow::bit_util::BitReader bitReader(inputIter, bytes(bitWidth));
    bitReader.GetBatch(bitWidth, expectedOutputIter, kNumValues);

    for (int i = 0; i < kNumValues; i++) {
      if (outputBuffer[i] != expectedOutputBuffer[i]) {
        break;
      }
      ASSERT_EQ(outputBuffer[i], expectedOutputBuffer[i]);
    }
  }
  // multiple of 8
  static const uint32_t kNumValues = 1024;

  std::vector<uint8_t> inputBuffer;
  std::vector<T> outputBuffer;
  std::vector<T> expectedOutputBuffer;
};

TEST(BitUnpackingTest, uint8) {
  BitUnpackingTest<uint8_t> test;
  test.testUnpack(1);
  test.testUnpack(2);
  test.testUnpack(3);
  test.testUnpack(4);
  test.testUnpack(5);
  test.testUnpack(6);
  test.testUnpack(7);
  test.testUnpack(8);
}

TEST(BitUnpackingTest, uint16) {
  BitUnpackingTest<uint16_t> test;
  test.testUnpack(1);
  test.testUnpack(2);
  test.testUnpack(3);
  test.testUnpack(4);
  test.testUnpack(5);
  test.testUnpack(6);
  test.testUnpack(7);
  test.testUnpack(8);
  test.testUnpack(9);
  test.testUnpack(10);
  test.testUnpack(11);
  test.testUnpack(12);
  test.testUnpack(13);
  test.testUnpack(14);
  test.testUnpack(15);
  test.testUnpack(16);
}

TEST(BitUnpackingTest, uint32) {
  BitUnpackingTest<uint32_t> test;
  test.testUnpack(1);
  test.testUnpack(2);
  test.testUnpack(3);
  test.testUnpack(4);
  test.testUnpack(5);
  test.testUnpack(6);
  test.testUnpack(7);
  test.testUnpack(8);
  test.testUnpack(9);
  test.testUnpack(10);
  test.testUnpack(11);
  test.testUnpack(12);
  test.testUnpack(13);
  test.testUnpack(14);
  test.testUnpack(15);
  test.testUnpack(16);
  test.testUnpack(17);
  test.testUnpack(18);
  test.testUnpack(19);
  test.testUnpack(20);
  test.testUnpack(21);
  test.testUnpack(22);
  test.testUnpack(23);
  test.testUnpack(24);
  test.testUnpack(25);
  test.testUnpack(26);
  test.testUnpack(27);
  test.testUnpack(28);
  test.testUnpack(29);
  test.testUnpack(30);
  test.testUnpack(31);
  test.testUnpack(32);
}
