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

#include <arrow/util/rle_encoding.h> // @manual
#include <gtest/gtest.h>

#include <random>
#include <arrow/buffer.h>
#include <arrow/status.h>
#include <arrow/result.h>

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;


template <typename T>
class BooleanDecoderTest {
 public:
  BooleanDecoderTest() {
    inputValues_.resize(numValues_, 0);
    outputValues_.resize(numValues_, 0);
    encodedValues_.resize(numValues_ * 4, 0);
  }

  BooleanDecoderTest(uint32_t numValues) : numValues_(numValues) {
    VELOX_CHECK(numValues % 8 == 0);

    inputValues_.resize(numValues_, 0);
    outputValues_.resize(numValues_, 0);
    encodedValues_.resize(numValues_ * 4, 0);
  }

  void testDecodeSuppliedData(std::vector<T> inputValues, uint8_t bitWidth) {
    numValues_ = inputValues.size();
    inputValues_ = inputValues;
    bitWidth_ = bitWidth;

    encodeInputValues();
    testDecode();
  }

 private:
  void testDecode() {
    T* output = outputValues_.data();
    arrow::util::RleDecoder arrowDecoder(
        reinterpret_cast<uint8_t*>(encodedValues_.data()),
        bytes(bitWidth_),
        bitWidth_);
    int numOfDecodedElements = arrowDecoder.GetBatch(output, numValues_);
    T* expectedOutput = inputValues_.data();
    for (int i = 0; i < numValues_; i++) {
      ASSERT_EQ(output[i], expectedOutput[i]);
    }
  }

  void encodeInputValues() {
    arrow::util::RleEncoder arrowEncoder(
        reinterpret_cast<uint8_t*>(encodedValues_.data()),
        bytes(bitWidth_),
        bitWidth_);
    for (auto i = 0; i < numValues_; i++) {
      arrowEncoder.Put(inputValues_[i]);
    }
    arrowEncoder.Flush();
  }

  uint32_t bytes(uint8_t bitWidth) {
    return (numValues_/(64/bitWidth_))*8 + bitWidth_ + arrow::util::RleEncoder::MinBufferSize(bitWidth_);
  }

  // multiple of 8
  uint32_t numValues_ = 1024;
  std::vector<T> inputValues_;
  std::vector<T> outputValues_;
  std::vector<uint8_t> encodedValues_;
  uint8_t bitWidth_;
};

TEST(BooleanDecoderTest, allOnes) {
  std::vector<uint8_t> allOnesVector(1024, 1);
  BooleanDecoderTest<uint8_t> test;
  test.testDecodeSuppliedData(allOnesVector, 1);
}

TEST(BooleanDecoderTest, allZeros) {
  std::vector<uint8_t> allZerosVector(1024, 0);
  BooleanDecoderTest<uint8_t> test;
  test.testDecodeSuppliedData(allZerosVector, 1);
}

TEST(BooleanDecoderTest, withNulls) {
  // 0, 1, and 2 represent false, true, and null respectively
  std::vector<uint8_t> zeroOneNullsVector(520);
  for(int i = 0; i < zeroOneNullsVector.size(); i++) {
    zeroOneNullsVector[i] =  std::rand() % 2;
  }
  zeroOneNullsVector[0] = 2;
  zeroOneNullsVector[20] = 2;
  BooleanDecoderTest<uint8_t> test;
  test.testDecodeSuppliedData(zeroOneNullsVector, 2);
}

TEST(BooleanDecoderTest, zeroAndOnes) {
  std::vector<uint8_t> zeroOneVector(520);
  for(int i = 0; i < zeroOneVector.size(); i++) {
    zeroOneVector[i] =  std::rand() % 2;
  }
  BooleanDecoderTest<uint8_t> test;
  test.testDecodeSuppliedData(zeroOneVector, 1);
}