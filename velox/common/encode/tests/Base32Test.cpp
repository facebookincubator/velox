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

TEST(Base32Test, encode) {
  // RFC 4648 test vectors.
  EXPECT_EQ(Base32::encode(""), "");
  EXPECT_EQ(Base32::encode("f"), "MY======");
  EXPECT_EQ(Base32::encode("fo"), "MZXQ====");
  EXPECT_EQ(Base32::encode("foo"), "MZXW6===");
  EXPECT_EQ(Base32::encode("foob"), "MZXW6YQ=");
  EXPECT_EQ(Base32::encode("fooba"), "MZXW6YTB");
  EXPECT_EQ(Base32::encode("foobar"), "MZXW6YTBOI======");
}

TEST(Base32Test, roundTrip) {
  // Encodes 'value', decodes the result, and verifies it matches the original.
  auto verify = [](std::string_view value) {
    auto encoded = Base32::encode(value);

    auto size = Base32::calculateDecodedSize(encoded.data(), encoded.size());
    ASSERT_FALSE(size.hasError());

    std::string decoded(size.value(), '\0');
    auto status = Base32::decode(
        encoded.data(), encoded.size(), decoded.data(), decoded.size());
    ASSERT_TRUE(status.ok());

    EXPECT_EQ(decoded, value);
  };

  for (const auto& value :
       {"", "a", "ab", "abc", "abcd", "abcde", "hello world", "red_value"}) {
    verify(value);
  }
}

} // namespace facebook::velox::encoding
