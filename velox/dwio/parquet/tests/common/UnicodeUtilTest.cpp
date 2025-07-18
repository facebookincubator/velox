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

#include "velox/dwio/parquet/common/UnicodeUtil.h"
#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::parquet;

class UnicodeUtilTest : public testing::Test {};

TEST_F(UnicodeUtilTest, isCharHighSurrogate) {
  // High surrogate range is 0xD800 to 0xDBFF.
  EXPECT_TRUE(UnicodeUtil::isCharHighSurrogate(0xD800));
  EXPECT_TRUE(UnicodeUtil::isCharHighSurrogate(0xD900));
  EXPECT_TRUE(UnicodeUtil::isCharHighSurrogate(0xDBFF));

  // Not high surrogates.
  EXPECT_FALSE(UnicodeUtil::isCharHighSurrogate(0xDC00)); // Low surrogate.
  EXPECT_FALSE(UnicodeUtil::isCharHighSurrogate(0x0041)); // Regular ASCII.
  EXPECT_FALSE(UnicodeUtil::isCharHighSurrogate(0x20AC)); // Euro symbol.
}

TEST_F(UnicodeUtilTest, truncateString) {
  // ASCII string.
  std::string ascii = "Hello, world!";
  EXPECT_EQ(UnicodeUtil::truncateString(ascii, 5), "Hello");
  EXPECT_EQ(UnicodeUtil::truncateString(ascii, 13), ascii);
  EXPECT_EQ(UnicodeUtil::truncateString(ascii, 20), ascii);

  // String with multi-bytes characters.
  std::string unicode = "Hello, 世界!";
  EXPECT_EQ(UnicodeUtil::truncateString(unicode, 7), "Hello, ");
  EXPECT_EQ(UnicodeUtil::truncateString(unicode, 8), "Hello, 世");
  EXPECT_EQ(UnicodeUtil::truncateString(unicode, 9), "Hello, 世界");
  EXPECT_EQ(UnicodeUtil::truncateString(unicode, 10), "Hello, 世界!");
  EXPECT_EQ(UnicodeUtil::truncateString(unicode, 20), unicode);

  // String with emoji (surrogate pairs).
  std::string emoji = "Hello 🌍!";
  EXPECT_EQ(UnicodeUtil::truncateString(emoji, 6), "Hello ");
  EXPECT_EQ(UnicodeUtil::truncateString(emoji, 7), "Hello 🌍");
  EXPECT_EQ(UnicodeUtil::truncateString(emoji, 8), "Hello 🌍!");
  EXPECT_EQ(UnicodeUtil::truncateString(emoji, 10), emoji);
}

TEST_F(UnicodeUtilTest, truncateStringMin) {
  std::optional<std::string> nullInput = std::nullopt;
  EXPECT_EQ(UnicodeUtil::truncateStringMin(nullInput, 5), std::nullopt);
  std::optional<std::string> ascii = "Hello, world!";
  EXPECT_EQ(UnicodeUtil::truncateStringMin(ascii, 5), "Hello");
  std::optional<std::string> unicode = "Hello, 世界!";
  EXPECT_EQ(UnicodeUtil::truncateStringMin(unicode, 8), "Hello, 世");
}

TEST_F(UnicodeUtilTest, truncateStringMax) {
  std::optional<std::string> nullInput = std::nullopt;
  EXPECT_EQ(UnicodeUtil::truncateStringMax(nullInput, 5), std::nullopt);

  std::optional<std::string> ascii = "Hello, world!";
  auto result = UnicodeUtil::truncateStringMax(ascii, 5);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "Hellp"); // 'o' -> 'p'.

  ascii = "Customer#000001500";
  result = UnicodeUtil::truncateStringMax(ascii, 16);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "Customer#0000016");

  std::optional<std::string> unicode = "Hello, 世界!";
  result = UnicodeUtil::truncateStringMax(unicode, 8);
  EXPECT_TRUE(result.has_value());

  // No truncation needed.
  std::optional<std::string> shortString = "Hi";
  result = UnicodeUtil::truncateStringMax(shortString, 5);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "Hi");

  // Last character is already at maximum value.
  std::optional<std::string> maxChar =
      "Hello\U0010FFFF"; // Maximum Unicode code point.
  result = UnicodeUtil::truncateStringMax(maxChar, 6);
  EXPECT_EQ(result.value(), maxChar.value());
}
