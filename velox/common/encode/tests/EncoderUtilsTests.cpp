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
class EncoderUtilsTest : public ::testing::Test {};

TEST_F(EncoderUtilsTest, isPadded) {
  EXPECT_TRUE(isPadded("ABC=", 4));
  EXPECT_FALSE(isPadded("ABC", 3));
}

TEST_F(EncoderUtilsTest, numPadding) {
  EXPECT_EQ(0, numPadding("ABC", 3));
  EXPECT_EQ(1, numPadding("ABC=", 4));
  EXPECT_EQ(2, numPadding("AB==", 4));
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

constexpr const Charset testWrongCharset = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '_'};

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

int base64 = 64;

TEST_F(EncoderUtilsTest, baseReverseLookup) {
  EXPECT_NO_THROW(baseReverseLookup(base64, 'A', testReverseIndex));
  EXPECT_THROW(
      baseReverseLookup(base64, '=', testReverseIndex), VeloxUserError);
}

TEST_F(EncoderUtilsTest, checkForwardIndex) {
  EXPECT_TRUE(checkForwardIndex(63, testCharset, testReverseIndex));
}

TEST_F(EncoderUtilsTest, checkReverseIndex) {
  EXPECT_TRUE(checkReverseIndex(255, base64, testCharset, testReverseIndex));
}

TEST_F(EncoderUtilsTest, HandlesLookupAndExceptions) {
  EXPECT_NO_THROW(baseReverseLookup(base64, 'A', testReverseIndex));
  EXPECT_THROW(
      baseReverseLookup(base64, '=', testReverseIndex), VeloxUserError);
}

TEST_F(EncoderUtilsTest, ValidatesCharsetWithReverseIndex) {
  EXPECT_TRUE(checkForwardIndex(63, testCharset, testReverseIndex));
}

TEST_F(EncoderUtilsTest, ValidatesReverseIndexWithCharset) {
  EXPECT_TRUE(checkReverseIndex(
      sizeof(testCharset) - 1, base64, testCharset, testReverseIndex));
  EXPECT_FALSE(checkReverseIndex(
      sizeof(testWrongCharset) - 1,
      base64,
      testWrongCharset,
      testWrongReverseIndex));
}

TEST_F(EncoderUtilsTest, CharacterDoesNotExist) {
  Charset charset = {'A', 'B', 'C', 'D', 'E', 'F'};
  EXPECT_TRUE(findCharacterInCharSet(charset, charset.size(), 1, 'C'));
  EXPECT_FALSE(findCharacterInCharSet(charset, charset.size(), 1, 'A'));
  EXPECT_TRUE(findCharacterInCharSet(charset, charset.size(), 0, 'A'));
}

TEST_F(EncoderUtilsTest, EmptyCharset) {
  Charset emptyCharset;
  EXPECT_FALSE(
      findCharacterInCharSet(emptyCharset, base64, emptyCharset.size(), 'A'));
}

TEST_F(EncoderUtilsTest, CheckForwardIndex_PartialValid) {
  // Test partial index range
  EXPECT_TRUE(checkForwardIndex(10, testCharset, testReverseIndex));
  EXPECT_TRUE(checkForwardIndex(20, testCharset, testReverseIndex));
  EXPECT_TRUE(checkForwardIndex(30, testCharset, testReverseIndex));
}

TEST_F(EncoderUtilsTest, CheckForwardIndex_CorruptedReverseIndex) {
  // Corrupting reverse index
  ReverseIndex corruptedReverseIndex = testReverseIndex;
  corruptedReverseIndex['A'] = 255;
  EXPECT_FALSE(checkForwardIndex(63, testCharset, corruptedReverseIndex));
}

TEST_F(EncoderUtilsTest, CheckReverseIndex_PartialValid) {
  // Test partial index range
  EXPECT_TRUE(checkReverseIndex(10, base64, testCharset, testReverseIndex));
  EXPECT_TRUE(checkReverseIndex(20, base64, testCharset, testReverseIndex));
  EXPECT_TRUE(checkReverseIndex(30, base64, testCharset, testReverseIndex));
}

TEST_F(EncoderUtilsTest, CheckReverseIndex_CorruptedCharset) {
  // Corrupting charset
  Charset corruptedCharset = testCharset;
  corruptedCharset[10] = '@';
  EXPECT_FALSE(
      checkReverseIndex(64, base64, corruptedCharset, testReverseIndex));
  EXPECT_TRUE(checkReverseIndex(255, base64, testCharset, testReverseIndex));
}

} // namespace facebook::velox::encoding
