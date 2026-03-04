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

#include "velox/common/encode/Base32.h"

#include <gtest/gtest.h>

namespace facebook::velox::encoding {
namespace {

TEST(Base32Test, encode) {
  EXPECT_EQ("", Base32::encode(""));
  EXPECT_EQ("ME======", Base32::encode("a"));
  EXPECT_EQ("MFRA====", Base32::encode("ab"));
  EXPECT_EQ("MFRGG===", Base32::encode("abc"));
  EXPECT_EQ("MFRGGZA=", Base32::encode("abcd"));
  EXPECT_EQ("MFRGGZDF", Base32::encode("abcde"));
  EXPECT_EQ("MFRGGZDFMY======", Base32::encode("abcdef"));
  EXPECT_EQ("NBSWY3DPEB3W64TMMQ======", Base32::encode("hello world"));
}

TEST(Base32Test, encodeToBuffer) {
  std::string input = "abc";
  auto encodedSize = Base32::calculateEncodedSize(input.size());
  EXPECT_EQ(8, encodedSize);
  std::string output(encodedSize, '\0');
  Base32::encode(input.data(), input.size(), output.data());
  EXPECT_EQ("MFRGG===", output);
}

TEST(Base32Test, calculateEncodedSize) {
  // With padding (default).
  EXPECT_EQ(0, Base32::calculateEncodedSize(0));
  EXPECT_EQ(8, Base32::calculateEncodedSize(1));
  EXPECT_EQ(8, Base32::calculateEncodedSize(2));
  EXPECT_EQ(8, Base32::calculateEncodedSize(3));
  EXPECT_EQ(8, Base32::calculateEncodedSize(4));
  EXPECT_EQ(8, Base32::calculateEncodedSize(5));
  EXPECT_EQ(16, Base32::calculateEncodedSize(6));
  EXPECT_EQ(16, Base32::calculateEncodedSize(10));
  EXPECT_EQ(24, Base32::calculateEncodedSize(11));

  // Without padding.
  EXPECT_EQ(0, Base32::calculateEncodedSize(0, false));
  EXPECT_EQ(2, Base32::calculateEncodedSize(1, false));
  EXPECT_EQ(4, Base32::calculateEncodedSize(2, false));
  EXPECT_EQ(5, Base32::calculateEncodedSize(3, false));
  EXPECT_EQ(7, Base32::calculateEncodedSize(4, false));
  EXPECT_EQ(8, Base32::calculateEncodedSize(5, false));
  EXPECT_EQ(10, Base32::calculateEncodedSize(6, false));
}

TEST(Base32Test, roundTrip) {
  auto testRoundTrip = [](const std::string& input) {
    auto encoded = Base32::encode(input);
    auto decodedSize =
        Base32::calculateDecodedSize(encoded.data(), encoded.size());
    ASSERT_FALSE(decodedSize.hasError()) << "input: " << input;
    std::string decoded(decodedSize.value(), '\0');
    auto status = Base32::decode(
        encoded.data(), encoded.size(), decoded.data(), decoded.size());
    ASSERT_TRUE(status.ok()) << "input: " << input;
    EXPECT_EQ(input, decoded) << "input: " << input;
  };

  testRoundTrip("");
  testRoundTrip("a");
  testRoundTrip("ab");
  testRoundTrip("abc");
  testRoundTrip("abcd");
  testRoundTrip("abcde");
  testRoundTrip("hello world");
  testRoundTrip("Hello World from Velox!");
  testRoundTrip(std::string(256, '\xff'));
}

} // namespace
} // namespace facebook::velox::encoding
