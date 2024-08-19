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
#include "velox/common/base/Status.h"
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::encoding {

class Base64Test : public ::testing::Test {
 protected:
  void checkDecodedSize(
      const std::string& encodedString,
      size_t expectedEncodedSize,
      size_t expectedDecodedSize) {
    size_t encodedSize = expectedEncodedSize;
    size_t decodedSize = 0;
    EXPECT_EQ(
        Status::OK(),
        Base64::calculateDecodedSize(encodedString, encodedSize, decodedSize));
    EXPECT_EQ(expectedEncodedSize, encodedSize);
    EXPECT_EQ(expectedDecodedSize, decodedSize);
  }
};

TEST_F(Base64Test, fromBase64) {
  EXPECT_EQ("Hello, World!", Base64::decode("SGVsbG8sIFdvcmxkIQ=="));
  EXPECT_EQ(
      "Base64 encoding is fun.",
      Base64::decode("QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4="));
  EXPECT_EQ("Simple text", Base64::decode("U2ltcGxlIHRleHQ="));
  EXPECT_EQ("1234567890", Base64::decode("MTIzNDU2Nzg5MA=="));

  // Check encoded strings without padding
  EXPECT_EQ("Hello, World!", Base64::decode("SGVsbG8sIFdvcmxkIQ"));
  EXPECT_EQ(
      "Base64 encoding is fun.",
      Base64::decode("QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4"));
  EXPECT_EQ("Simple text", Base64::decode("U2ltcGxlIHRleHQ"));
  EXPECT_EQ("1234567890", Base64::decode("MTIzNDU2Nzg5MA"));
}

TEST_F(Base64Test, calculateDecodedSizeProperSize) {
  checkDecodedSize("SGVsbG8sIFdvcmxkIQ==", 18, 13);
  checkDecodedSize("SGVsbG8sIFdvcmxkIQ", 18, 13);
  checkDecodedSize("QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4=", 31, 23);
  checkDecodedSize("QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4", 31, 23);
  checkDecodedSize("MTIzNDU2Nzg5MA==", 14, 10);
  checkDecodedSize("MTIzNDU2Nzg5MA", 14, 10);
}

TEST_F(Base64Test, calculateDecodedSizeImproperSize) {
  size_t encodedSize{21};
  size_t decodedSize;

  EXPECT_EQ(
      Status::UserError(
          "Base64::decode() - invalid input string: string length is not a multiple of 4."),
      Base64::calculateDecodedSize(
          "SGVsbG8sIFdvcmxkIQ===", encodedSize, decodedSize));
}

TEST_F(Base64Test, isPadded) {
  EXPECT_TRUE(Base64::isPadded("ABC=", 4));
  EXPECT_FALSE(Base64::isPadded("ABC", 3));
}

TEST_F(Base64Test, numPadding) {
  EXPECT_EQ(0, Base64::numPadding("ABC", 3));
  EXPECT_EQ(1, Base64::numPadding("ABC=", 4));
  EXPECT_EQ(2, Base64::numPadding("AB==", 4));
}

TEST_F(Base64Test, testDecodeImpl) {
  constexpr const Base64::ReverseIndex reverseTable = {
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

  auto testDecode = [&](const std::string_view input,
                        char* output1,
                        size_t outputSize,
                        Status expectedStatus) {
    EXPECT_EQ(
        Base64::decodeImpl(
            input, input.size(), output1, outputSize, reverseTable),
        expectedStatus);
  };

  // Predefine buffer sizes and reuse.
  char output1[20] = {};
  char output2[11] = {};
  char output3[1] = {};
  char output4[1] = {};
  char output5[1] = {};

  // Invalid characters in the input string
  testDecode(
      "SGVsbG8gd29ybGQ$",
      output1,
      sizeof(output1),
      Status::UserError(
          "Base64::decode() - invalid input string: contains invalid characters."));

  // All characters are padding characters
  testDecode("====", output1, sizeof(output1), Status::OK());

  // Invalid input size
  testDecode(
      "S",
      output1,
      sizeof(output1),
      Status::UserError(
          "Base64::decode() - invalid input string: string length cannot be 1 more than a multiple of 4."));

  // Valid input without padding characters
  testDecode("SGVsbG8gd29ybGQ", output1, sizeof(output1), Status::OK());
  EXPECT_STREQ(output1, "Hello world");

  // Valid input with padding characters
  testDecode("SGVsbG8gd29ybGQ=", output2, sizeof(output2), Status::OK());
  EXPECT_STREQ(output2, "Hello world");

  // Empty input string
  testDecode("", output3, sizeof(output3), Status::OK());
  EXPECT_STREQ(output3, "");

  // Invalid input size
  testDecode(
      "SGVsbG8gd29ybGQ===",
      output1,
      sizeof(output1),
      Status::UserError(
          "Base64::decode() - invalid input string: string length is not a multiple of 4."));

  // whiltespaces in the input string
  testDecode(
      " SGVsb G8gd2 9ybGQ= ",
      output1,
      sizeof(output1),
      Status::UserError(
          "Base64::decode() - invalid input string: contains invalid characters."));

  // insufficient buffer size
  testDecode(
      " SGVsb G8gd2 9ybGQ= ",
      output5,
      sizeof(output4),
      Status::UserError(
          "Base64::decode() - invalid output string: output string is too small."));
}

} // namespace facebook::velox::encoding
