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

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/encode/EncoderUtils.h"

namespace facebook::velox::encoding {
class EncoderUtilsTest : public ::testing::Test {
 protected:
  const int binaryBlockByteSize = 3;
  const int encodedBlockByteSize = 4;
};

TEST_F(EncoderUtilsTest, isPadded) {
  EXPECT_TRUE(isPadded("ABC="));
  EXPECT_FALSE(isPadded("ABC"));
}

TEST_F(EncoderUtilsTest, numPadding) {
  EXPECT_EQ(0, numPadding("ABC"));
  EXPECT_EQ(1, numPadding("ABC="));
  EXPECT_EQ(2, numPadding("AB=="));
}

TEST_F(EncoderUtilsTest, CalculateDecodedSizeTest) {
  size_t inputSize = 8;
  size_t decodedSize = 0;

  EXPECT_EQ(
      calculateDecodedSize(
          "abcdabcd",
          inputSize,
          decodedSize,
          binaryBlockByteSize,
          encodedBlockByteSize),
      Status::OK());
  EXPECT_EQ(decodedSize, 6);

  inputSize = 8;
  EXPECT_EQ(
      calculateDecodedSize(
          "abcdab==",
          inputSize,
          decodedSize,
          binaryBlockByteSize,
          encodedBlockByteSize),
      Status::OK());
  EXPECT_EQ(decodedSize, 4);

  EXPECT_EQ(
      calculateDecodedSize(
          "abcdab=",
          inputSize,
          decodedSize,
          binaryBlockByteSize,
          encodedBlockByteSize),
      Status::UserError("decode() - invalid input string length."));
}

TEST_F(EncoderUtilsTest, CalculateEncodedSizeTest) {
  EXPECT_EQ(
      calculateEncodedSize(3, true, binaryBlockByteSize, encodedBlockByteSize),
      4);
  EXPECT_EQ(
      calculateEncodedSize(3, false, binaryBlockByteSize, encodedBlockByteSize),
      4);
  EXPECT_EQ(
      calculateEncodedSize(6, true, binaryBlockByteSize, encodedBlockByteSize),
      8);
  EXPECT_EQ(
      calculateEncodedSize(0, true, binaryBlockByteSize, encodedBlockByteSize),
      0);
}

TEST_F(EncoderUtilsTest, Base64ReverseLookupTest) {
  std::array<uint8_t, 256> reverseIndex{};
  reverseIndex.fill(255);
  reverseIndex['A'] = 0;
  reverseIndex['B'] = 1;
  uint8_t reverseLookupValue = 0;

  EXPECT_EQ(
      base64ReverseLookup('A', reverseIndex, reverseLookupValue), Status::OK());
  EXPECT_EQ(reverseLookupValue, 0);

  EXPECT_EQ(
      base64ReverseLookup('B', reverseIndex, reverseLookupValue), Status::OK());
  EXPECT_EQ(reverseLookupValue, 1);

  EXPECT_EQ(
      base64ReverseLookup('Z', reverseIndex, reverseLookupValue),
      Status::UserError("decode() - contains invalid character 'Z'"));
}

TEST_F(EncoderUtilsTest, CheckForwardIndexTest) {
  constexpr std::string_view charset =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::array<uint8_t, 64> reverseIndex{};

  for (size_t i = 0; i < charset.size(); ++i) {
    reverseIndex[static_cast<uint8_t>(charset[i])] = i;
  }

  EXPECT_TRUE(checkForwardIndex(63, charset, reverseIndex));
}

TEST_F(EncoderUtilsTest, FindCharacterInCharsetTest) {
  constexpr std::string_view charset =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  EXPECT_TRUE(findCharacterInCharset(charset, 0, 'A'));
  EXPECT_FALSE(findCharacterInCharset(charset, 0, 'z'));
}

TEST_F(EncoderUtilsTest, CheckReverseIndexTest) {
  constexpr std::string_view charset =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::array<uint8_t, 256> reverseIndex{};

  reverseIndex.fill(255);
  for (size_t i = 0; i < charset.size(); ++i) {
    reverseIndex[static_cast<uint8_t>(charset[i])] = i;
  }

  EXPECT_TRUE(checkReverseIndex(255, charset, reverseIndex));
}

} // namespace facebook::velox::encoding
