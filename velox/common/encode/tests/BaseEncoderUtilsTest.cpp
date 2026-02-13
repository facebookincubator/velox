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

#include "velox/common/encode/BaseEncoderUtils.h"

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::encoding {

class BaseEncoderUtilsTest : public ::testing::Test {};

TEST_F(BaseEncoderUtilsTest, isPadded) {
  EXPECT_TRUE(BaseEncoderUtils::isPadded("ABC="));
  EXPECT_TRUE(BaseEncoderUtils::isPadded("AB=="));
  EXPECT_FALSE(BaseEncoderUtils::isPadded("ABC"));
  EXPECT_FALSE(BaseEncoderUtils::isPadded(""));
}

TEST_F(BaseEncoderUtilsTest, numPadding) {
  EXPECT_EQ(0, BaseEncoderUtils::numPadding("ABC"));
  EXPECT_EQ(1, BaseEncoderUtils::numPadding("ABC="));
  EXPECT_EQ(2, BaseEncoderUtils::numPadding("AB=="));
  EXPECT_EQ(0, BaseEncoderUtils::numPadding(""));
}

TEST_F(BaseEncoderUtilsTest, reverseLookup) {
  // Build a simple reverse index for testing.
  std::array<uint8_t, 256> reverseIndex{};
  reverseIndex.fill(255);
  reverseIndex['A'] = 0;
  reverseIndex['B'] = 1;
  reverseIndex['C'] = 2;

  EXPECT_EQ(0, BaseEncoderUtils::reverseLookup('A', reverseIndex, 64).value());
  EXPECT_EQ(1, BaseEncoderUtils::reverseLookup('B', reverseIndex, 64).value());
  EXPECT_EQ(2, BaseEncoderUtils::reverseLookup('C', reverseIndex, 64).value());

  // Invalid character should return an error.
  EXPECT_EQ(
      BaseEncoderUtils::reverseLookup('Z', reverseIndex, 64).error(),
      Status::UserError(
          "decode() - invalid input string: invalid character 'Z'"));
}

TEST_F(BaseEncoderUtilsTest, calculateEncodedSizeBase64) {
  // Base64: binaryBlockByteSize=3, encodedBlockByteSize=4.
  EXPECT_EQ(0, BaseEncoderUtils::calculateEncodedSize(0, true, 3, 4));
  EXPECT_EQ(4, BaseEncoderUtils::calculateEncodedSize(1, true, 3, 4));
  EXPECT_EQ(4, BaseEncoderUtils::calculateEncodedSize(2, true, 3, 4));
  EXPECT_EQ(4, BaseEncoderUtils::calculateEncodedSize(3, true, 3, 4));
  EXPECT_EQ(8, BaseEncoderUtils::calculateEncodedSize(4, true, 3, 4));
  EXPECT_EQ(8, BaseEncoderUtils::calculateEncodedSize(6, true, 3, 4));

  // Without padding.
  EXPECT_EQ(2, BaseEncoderUtils::calculateEncodedSize(1, false, 3, 4));
  EXPECT_EQ(3, BaseEncoderUtils::calculateEncodedSize(2, false, 3, 4));
  EXPECT_EQ(4, BaseEncoderUtils::calculateEncodedSize(3, false, 3, 4));
  EXPECT_EQ(6, BaseEncoderUtils::calculateEncodedSize(4, false, 3, 4));
}

TEST_F(BaseEncoderUtilsTest, calculateEncodedSizeBase32) {
  // Base32: binaryBlockByteSize=5, encodedBlockByteSize=8.
  EXPECT_EQ(0, BaseEncoderUtils::calculateEncodedSize(0, true, 5, 8));
  EXPECT_EQ(8, BaseEncoderUtils::calculateEncodedSize(1, true, 5, 8));
  EXPECT_EQ(8, BaseEncoderUtils::calculateEncodedSize(5, true, 5, 8));
  EXPECT_EQ(16, BaseEncoderUtils::calculateEncodedSize(6, true, 5, 8));
  EXPECT_EQ(16, BaseEncoderUtils::calculateEncodedSize(10, true, 5, 8));

  // Without padding.
  EXPECT_EQ(2, BaseEncoderUtils::calculateEncodedSize(1, false, 5, 8));
  EXPECT_EQ(4, BaseEncoderUtils::calculateEncodedSize(2, false, 5, 8));
  EXPECT_EQ(5, BaseEncoderUtils::calculateEncodedSize(3, false, 5, 8));
  EXPECT_EQ(7, BaseEncoderUtils::calculateEncodedSize(4, false, 5, 8));
  EXPECT_EQ(8, BaseEncoderUtils::calculateEncodedSize(5, false, 5, 8));
}

TEST_F(BaseEncoderUtilsTest, calculateDecodedSizeBase64) {
  // Base64: binaryBlockByteSize=3, encodedBlockByteSize=4.

  // Empty input.
  size_t inputSize = 0;
  EXPECT_EQ(
      0, BaseEncoderUtils::calculateDecodedSize("", inputSize, 3, 4).value());

  // Padded input: "SGVsbG8sIFdvcmxkIQ==" (20 chars, 2 padding).
  inputSize = 20;
  EXPECT_EQ(
      13,
      BaseEncoderUtils::calculateDecodedSize(
          "SGVsbG8sIFdvcmxkIQ==", inputSize, 3, 4)
          .value());
  EXPECT_EQ(18, inputSize);

  // Unpadded input: "SGVsbG8sIFdvcmxkIQ" (18 chars).
  inputSize = 18;
  EXPECT_EQ(
      13,
      BaseEncoderUtils::calculateDecodedSize(
          "SGVsbG8sIFdvcmxkIQ", inputSize, 3, 4)
          .value());
  EXPECT_EQ(18, inputSize);

  // Invalid padded input: length not a multiple of encodedBlockByteSize.
  inputSize = 21;
  EXPECT_EQ(
      Status::UserError("decode() - invalid input string length."),
      BaseEncoderUtils::calculateDecodedSize(
          "SGVsbG8sIFdvcmxkIQ===", inputSize, 3, 4)
          .error());

  // Single extra byte (invalid for Base64 unpadded).
  inputSize = 1;
  EXPECT_EQ(
      Status::UserError("decode() - invalid input string length."),
      BaseEncoderUtils::calculateDecodedSize("A", inputSize, 3, 4).error());
}

TEST_F(BaseEncoderUtilsTest, calculateDecodedSizeBase32) {
  // Base32: binaryBlockByteSize=5, encodedBlockByteSize=8.

  // Empty input.
  size_t inputSize = 0;
  EXPECT_EQ(
      0, BaseEncoderUtils::calculateDecodedSize("", inputSize, 5, 8).value());

  // Padded input: 8 chars with 6 padding (single byte encoded).
  inputSize = 8;
  EXPECT_EQ(
      1,
      BaseEncoderUtils::calculateDecodedSize("ME======", inputSize, 5, 8)
          .value());
  EXPECT_EQ(2, inputSize);

  // Padded input: 8 chars with 1 padding (4 bytes encoded).
  inputSize = 8;
  EXPECT_EQ(
      4,
      BaseEncoderUtils::calculateDecodedSize("MFRGIZL=", inputSize, 5, 8)
          .value());
  EXPECT_EQ(7, inputSize);

  // Unpadded input: 2 chars (single byte).
  inputSize = 2;
  EXPECT_EQ(
      1, BaseEncoderUtils::calculateDecodedSize("ME", inputSize, 5, 8).value());
  EXPECT_EQ(2, inputSize);

  // Invalid padded input: length not a multiple of 8.
  inputSize = 9;
  EXPECT_EQ(
      Status::UserError("decode() - invalid input string length."),
      BaseEncoderUtils::calculateDecodedSize("ME======X", inputSize, 5, 8)
          .error());

  // Single extra byte (invalid for Base32 unpadded).
  inputSize = 1;
  EXPECT_EQ(
      Status::UserError("decode() - invalid input string length."),
      BaseEncoderUtils::calculateDecodedSize("M", inputSize, 5, 8).error());
}

} // namespace facebook::velox::encoding
