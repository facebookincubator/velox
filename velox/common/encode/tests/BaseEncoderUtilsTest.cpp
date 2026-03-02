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
namespace {

void testCalculateDecodedSize(
    const std::string& input,
    size_t initialInputSize,
    const CodecBlockSizes& blockSizes,
    size_t expectedDecodedSize,
    size_t expectedInputSize) {
  size_t inputSize = initialInputSize;
  EXPECT_EQ(
      expectedDecodedSize,
      BaseEncoderUtils::calculateDecodedSize(input, inputSize, blockSizes)
          .value());
  EXPECT_EQ(expectedInputSize, inputSize);
}

void testCalculateDecodedSizeError(
    const std::string& input,
    size_t initialInputSize,
    const CodecBlockSizes& blockSizes) {
  size_t inputSize = initialInputSize;
  EXPECT_EQ(
      Status::UserError("decode() - invalid input string length."),
      BaseEncoderUtils::calculateDecodedSize(input, inputSize, blockSizes)
          .error());
}

TEST(BaseEncoderUtilsTest, isPadded) {
  EXPECT_TRUE(BaseEncoderUtils::isPadded("ABC="));
  EXPECT_TRUE(BaseEncoderUtils::isPadded("AB=="));
  EXPECT_FALSE(BaseEncoderUtils::isPadded("ABC"));
  EXPECT_FALSE(BaseEncoderUtils::isPadded(""));
}

TEST(BaseEncoderUtilsTest, numPadding) {
  EXPECT_EQ(0, BaseEncoderUtils::numPadding("ABC"));
  EXPECT_EQ(1, BaseEncoderUtils::numPadding("ABC="));
  EXPECT_EQ(2, BaseEncoderUtils::numPadding("AB=="));
  EXPECT_EQ(0, BaseEncoderUtils::numPadding(""));
}

TEST(BaseEncoderUtilsTest, reverseLookup) {
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

TEST(BaseEncoderUtilsTest, calculateEncodedSizeBase64) {
  EXPECT_EQ(
      0, BaseEncoderUtils::calculateEncodedSize(0, true, kBase64BlockSizes));
  EXPECT_EQ(
      4, BaseEncoderUtils::calculateEncodedSize(1, true, kBase64BlockSizes));
  EXPECT_EQ(
      4, BaseEncoderUtils::calculateEncodedSize(2, true, kBase64BlockSizes));
  EXPECT_EQ(
      4, BaseEncoderUtils::calculateEncodedSize(3, true, kBase64BlockSizes));
  EXPECT_EQ(
      8, BaseEncoderUtils::calculateEncodedSize(4, true, kBase64BlockSizes));
  EXPECT_EQ(
      8, BaseEncoderUtils::calculateEncodedSize(6, true, kBase64BlockSizes));

  // Without padding.
  EXPECT_EQ(
      2, BaseEncoderUtils::calculateEncodedSize(1, false, kBase64BlockSizes));
  EXPECT_EQ(
      3, BaseEncoderUtils::calculateEncodedSize(2, false, kBase64BlockSizes));
  EXPECT_EQ(
      4, BaseEncoderUtils::calculateEncodedSize(3, false, kBase64BlockSizes));
  EXPECT_EQ(
      6, BaseEncoderUtils::calculateEncodedSize(4, false, kBase64BlockSizes));
}

TEST(BaseEncoderUtilsTest, calculateEncodedSizeBase32) {
  EXPECT_EQ(
      0, BaseEncoderUtils::calculateEncodedSize(0, true, kBase32BlockSizes));
  EXPECT_EQ(
      8, BaseEncoderUtils::calculateEncodedSize(1, true, kBase32BlockSizes));
  EXPECT_EQ(
      8, BaseEncoderUtils::calculateEncodedSize(5, true, kBase32BlockSizes));
  EXPECT_EQ(
      16, BaseEncoderUtils::calculateEncodedSize(6, true, kBase32BlockSizes));
  EXPECT_EQ(
      16, BaseEncoderUtils::calculateEncodedSize(10, true, kBase32BlockSizes));

  // Without padding.
  EXPECT_EQ(
      2, BaseEncoderUtils::calculateEncodedSize(1, false, kBase32BlockSizes));
  EXPECT_EQ(
      4, BaseEncoderUtils::calculateEncodedSize(2, false, kBase32BlockSizes));
  EXPECT_EQ(
      5, BaseEncoderUtils::calculateEncodedSize(3, false, kBase32BlockSizes));
  EXPECT_EQ(
      7, BaseEncoderUtils::calculateEncodedSize(4, false, kBase32BlockSizes));
  EXPECT_EQ(
      8, BaseEncoderUtils::calculateEncodedSize(5, false, kBase32BlockSizes));
}

TEST(BaseEncoderUtilsTest, calculateDecodedSizeBase64) {
  // Empty input.
  testCalculateDecodedSize("", 0, kBase64BlockSizes, 0, 0);

  // Padded input: "SGVsbG8sIFdvcmxkIQ==" (20 chars, 2 padding).
  testCalculateDecodedSize(
      "SGVsbG8sIFdvcmxkIQ==", 20, kBase64BlockSizes, 13, 18);

  // Unpadded input: "SGVsbG8sIFdvcmxkIQ" (18 chars).
  testCalculateDecodedSize("SGVsbG8sIFdvcmxkIQ", 18, kBase64BlockSizes, 13, 18);

  // Invalid padded input: length not a multiple of encodedBlockByteSize.
  testCalculateDecodedSizeError("SGVsbG8sIFdvcmxkIQ===", 21, kBase64BlockSizes);

  // Single extra byte (invalid for Base64 unpadded).
  testCalculateDecodedSizeError("A", 1, kBase64BlockSizes);
}

TEST(BaseEncoderUtilsTest, calculateDecodedSizeBase32) {
  // Empty input.
  testCalculateDecodedSize("", 0, kBase32BlockSizes, 0, 0);

  // Padded input: 8 chars with 6 padding (single byte encoded).
  testCalculateDecodedSize("ME======", 8, kBase32BlockSizes, 1, 2);

  // Padded input: 8 chars with 1 padding (4 bytes encoded).
  testCalculateDecodedSize("MFRGIZL=", 8, kBase32BlockSizes, 4, 7);

  // Unpadded input: 2 chars (single byte).
  testCalculateDecodedSize("ME", 2, kBase32BlockSizes, 1, 2);

  // Invalid padded input: length not a multiple of 8.
  testCalculateDecodedSizeError("ME======X", 9, kBase32BlockSizes);

  // Single extra byte (invalid for Base32 unpadded).
  testCalculateDecodedSizeError("M", 1, kBase32BlockSizes);
}

} // namespace
} // namespace facebook::velox::encoding
