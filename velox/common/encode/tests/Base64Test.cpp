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

#include "velox/common/encode/Base64.h"

#include <gtest/gtest.h>
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::encoding {

class Base64Test : public ::testing::Test {};

TEST_F(Base64Test, fromBase64) {
  EXPECT_EQ(
      "Hello, World!",
      Base64::decode(folly::StringPiece("SGVsbG8sIFdvcmxkIQ==")));
  EXPECT_EQ(
      "Base64 encoding is fun.",
      Base64::decode(folly::StringPiece("QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4=")));
  EXPECT_EQ(
      "Simple text", Base64::decode(folly::StringPiece("U2ltcGxlIHRleHQ=")));
  EXPECT_EQ(
      "1234567890", Base64::decode(folly::StringPiece("MTIzNDU2Nzg5MA==")));

  // Check encoded strings without padding
  EXPECT_EQ(
      "Hello, World!",
      Base64::decode(folly::StringPiece("SGVsbG8sIFdvcmxkIQ")));
  EXPECT_EQ(
      "Base64 encoding is fun.",
      Base64::decode(folly::StringPiece("QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4")));
  EXPECT_EQ(
      "Simple text", Base64::decode(folly::StringPiece("U2ltcGxlIHRleHQ")));
  EXPECT_EQ("1234567890", Base64::decode(folly::StringPiece("MTIzNDU2Nzg5MA")));
}

TEST_F(Base64Test, calculateDecodedSizeProperSize) {
  size_t encoded_size{0};

  encoded_size = 20;
  EXPECT_EQ(
      13, Base64::calculateDecodedSize("SGVsbG8sIFdvcmxkIQ==", encoded_size));
  EXPECT_EQ(18, encoded_size);

  encoded_size = 18;
  EXPECT_EQ(
      13, Base64::calculateDecodedSize("SGVsbG8sIFdvcmxkIQ", encoded_size));
  EXPECT_EQ(18, encoded_size);

  encoded_size = 21;
  EXPECT_THROW(
      Base64::calculateDecodedSize("SGVsbG8sIFdvcmxkIQ==", encoded_size),
      VeloxUserError);

  encoded_size = 32;
  EXPECT_EQ(
      23,
      Base64::calculateDecodedSize(
          "QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4=", encoded_size));
  EXPECT_EQ(31, encoded_size);

  encoded_size = 31;
  EXPECT_EQ(
      23,
      Base64::calculateDecodedSize(
          "QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4", encoded_size));
  EXPECT_EQ(31, encoded_size);

  encoded_size = 16;
  EXPECT_EQ(10, Base64::calculateDecodedSize("MTIzNDU2Nzg5MA==", encoded_size));
  EXPECT_EQ(14, encoded_size);

  encoded_size = 14;
  EXPECT_EQ(10, Base64::calculateDecodedSize("MTIzNDU2Nzg5MA", encoded_size));
  EXPECT_EQ(14, encoded_size);
}

TEST_F(Base64Test, ChecksPadding) {
  EXPECT_TRUE(Base64::isPadded("ABC=", 4));
  EXPECT_FALSE(Base64::isPadded("ABC", 3));
}

TEST_F(Base64Test, CountsPaddingCorrectly) {
  EXPECT_EQ(0, Base64::numPadding("ABC", 3));
  EXPECT_EQ(1, Base64::numPadding("ABC=", 4));
  EXPECT_EQ(2, Base64::numPadding("AB==", 4));
}

constexpr Charset testCharset = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};

constexpr ReverseIndex testReverseIndex = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 62,  255,
    255, 255, 63,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  255, 255,
    255, 255, 255, 255, 255, 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
    10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
    25,  255, 255, 255, 255, 255, 255, 26,  27,  28,  29,  30,  31,  32,  33,
    34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
    49,  50,  51,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255};

constexpr ReverseIndex testWrongReverseIndex = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 62,  255,
    62,  255, 63,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  255, 255,
    255, 255, 255, 255, 255, 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
    10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
    25,  255, 255, 255, 255, 63,  255, 26,  27,  28,  29,  30,  31,  32,  33,
    34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
    49,  50,  51,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255};

TEST_F(Base64Test, HandlesLookupAndExceptions) {
  EXPECT_NO_THROW(Base64::base64ReverseLookup('A', testReverseIndex));
  EXPECT_THROW(
      Base64::base64ReverseLookup('=', testReverseIndex), VeloxUserError);
}

TEST_F(Base64Test, ValidatesCharsetWithReverseIndex) {
  EXPECT_TRUE(checkForwardIndex(63, testCharset, testReverseIndex));
}

TEST_F(Base64Test, ValidatesReverseIndexWithCharset) {
  EXPECT_TRUE(checkReverseIndex(255, testCharset, testReverseIndex));
  EXPECT_FALSE(checkReverseIndex(255, testCharset, testWrongReverseIndex));
}

TEST_F(Base64Test, CharacterDoesNotExist) {
  Charset charset = {'A', 'B', 'C', 'D', 'E', 'F'};
  EXPECT_TRUE(findCharacterInCharSet(charset, charset.size(), 'C'));
  EXPECT_FALSE(findCharacterInCharSet(charset, charset.size(), 'Z'));
  EXPECT_TRUE(findCharacterInCharSet(charset, charset.size(), 'A'));
}

TEST_F(Base64Test, EmptyCharset) {
  Charset emptyCharset;
  EXPECT_FALSE(findCharacterInCharSet(emptyCharset, emptyCharset.size(), 'A'));
}

TEST_F(Base64Test, CheckForwardIndex_PartialValid) {
  // Test partial index range
  EXPECT_TRUE(checkForwardIndex(10, testCharset, testReverseIndex));
  EXPECT_TRUE(checkForwardIndex(20, testCharset, testReverseIndex));
  EXPECT_TRUE(checkForwardIndex(30, testCharset, testReverseIndex));
}

TEST_F(Base64Test, CheckForwardIndex_CorruptedReverseIndex) {
  // Corrupting reverse index
  ReverseIndex corruptedReverseIndex = testReverseIndex;
  corruptedReverseIndex['A'] = 255;
  EXPECT_FALSE(checkForwardIndex(63, testCharset, corruptedReverseIndex));
}

TEST_F(Base64Test, CheckReverseIndex_PartialValid) {
  // Test partial index range
  EXPECT_TRUE(checkReverseIndex(10, testCharset, testReverseIndex));
  EXPECT_TRUE(checkReverseIndex(20, testCharset, testReverseIndex));
  EXPECT_TRUE(checkReverseIndex(30, testCharset, testReverseIndex));
}

TEST_F(Base64Test, CheckReverseIndex_CorruptedCharset) {
  // Corrupting charset
  Charset corruptedCharset = testCharset;
  corruptedCharset[10] = '@';
  EXPECT_FALSE(checkReverseIndex(64, corruptedCharset, testReverseIndex));
}

} // namespace facebook::velox::encoding
