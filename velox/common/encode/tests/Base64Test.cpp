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

#include <common/encode/EncoderUtils.h>
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
  size_t encodedSize = 20;
  EXPECT_EQ(
      13,
      calculateDecodedSize("SGVsbG8sIFdvcmxkIQ==", encodedSize, 3, 4).value());
  EXPECT_EQ(18, encodedSize);

  encodedSize = 18;
  EXPECT_EQ(
      13,
      calculateDecodedSize("SGVsbG8sIFdvcmxkIQ", encodedSize, 3, 4).value());
  EXPECT_EQ(18, encodedSize);

  encodedSize = 21;
  EXPECT_EQ(
      Status::UserError("decode() - invalid input string length."),
      calculateDecodedSize("SGVsbG8sIFdvcmxkIQ===", encodedSize, 3, 4).error());

  encodedSize = 32;
  EXPECT_EQ(
      23,
      calculateDecodedSize(
          "QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4=", encodedSize, 3, 4)
          .value());
  EXPECT_EQ(31, encodedSize);

  encodedSize = 31;
  EXPECT_EQ(
      23,
      calculateDecodedSize("QmFzZTY0IGVuY29kaW5nIGlzIGZ1bi4", encodedSize, 3, 4)
          .value());
  EXPECT_EQ(31, encodedSize);

  encodedSize = 16;
  EXPECT_EQ(
      10, calculateDecodedSize("MTIzNDU2Nzg5MA==", encodedSize, 3, 4).value());
  EXPECT_EQ(14, encodedSize);

  encodedSize = 14;
  EXPECT_EQ(
      10, calculateDecodedSize("MTIzNDU2Nzg5MA", encodedSize, 3, 4).value());
  EXPECT_EQ(14, encodedSize);
}

} // namespace facebook::velox::encoding
