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
  // Lambda function to reduce repetition in test cases
  auto checkBase64Decode = [](const std::string& expected,
                              const std::string& encoded) {
    EXPECT_EQ(expected, Base64::decode(folly::StringPiece(encoded)));
  };

  // Check encoded strings with padding
  checkBase64Decode("Hello, World!", "SGVsbG8sIFdvcmxkIQ==");
  checkBase64Decode(
      "Base64 encoding is fun.", "QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4=");
  checkBase64Decode("Simple text", "U2ltcGxlIHRleHQ=");
  checkBase64Decode("1234567890", "MTIzNDU2Nzg5MA==");

  // Check encoded strings without padding
  checkBase64Decode("Hello, World!", "SGVsbG8sIFdvcmxkIQ");
  checkBase64Decode(
      "Base64 encoding is fun.", "QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4");
  checkBase64Decode("Simple text", "U2ltcGxlIHRleHQ");
  checkBase64Decode("1234567890", "MTIzNDU2Nzg5MA");
}

TEST_F(Base64Test, calculateDecodedSize) {
  auto checkDecodedSize = [](std::string_view encodedString,
                             size_t initialEncodedSize,
                             size_t expectedEncodedSize,
                             size_t expectedDecodedSize,
                             Status expectedStatus = Status::OK()) {
    size_t encoded_size = initialEncodedSize;
    size_t decoded_size = 0;
    Status status =
        Base64::calculateDecodedSize(encodedString, encoded_size, decoded_size);

    if (expectedStatus.ok()) {
      EXPECT_EQ(Status::OK(), status);
      EXPECT_EQ(expectedEncodedSize, encoded_size);
      EXPECT_EQ(expectedDecodedSize, decoded_size);
    } else {
      EXPECT_EQ(expectedStatus, status);
    }
  };

  // Using the lambda to reduce repetitive code
  checkDecodedSize("SGVsbG8sIFdvcmxkIQ==", 20, 18, 13);
  checkDecodedSize("SGVsbG8sIFdvcmxkIQ", 18, 18, 13);
  checkDecodedSize(
      "SGVsbG8sIFdvcmxkIQ===",
      21,
      0,
      0,
      Status::UserError(
          "Base64::decode() - invalid input string: string length is not a multiple of 4."));
  checkDecodedSize("QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4=", 32, 31, 23);
  checkDecodedSize("QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4", 31, 31, 23);
  checkDecodedSize("MTIzNDU2Nzg5MA==", 16, 14, 10);
  checkDecodedSize("MTIzNDU2Nzg5MA", 14, 14, 10);
}

TEST_F(Base64Test, isPadded) {
  EXPECT_TRUE(Base64::isPadded("ABC="));
  EXPECT_FALSE(Base64::isPadded("ABC"));
}

TEST_F(Base64Test, numPadding) {
  EXPECT_EQ(0, Base64::numPadding("ABC"));
  EXPECT_EQ(1, Base64::numPadding("ABC="));
  EXPECT_EQ(2, Base64::numPadding("AB=="));
}
} // namespace facebook::velox::encoding
