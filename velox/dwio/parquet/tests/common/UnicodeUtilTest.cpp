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

using namespace facebook::velox::parquet;

class UnicodeUtilTest : public testing::Test {};

TEST_F(UnicodeUtilTest, truncateStringMin) {
  // ASCII string.
  std::string ascii = "Hello, world!";
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(ascii.c_str(), ascii.length(), 0), "");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(ascii.c_str(), ascii.length(), 1), "H");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(ascii.c_str(), ascii.length(), 5),
      "Hello");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(ascii.c_str(), ascii.length(), 13), ascii);
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(ascii.c_str(), ascii.length(), 20), ascii);

  // String with multi-bytes characters.
  std::string unicode = "Hello, 世界!";
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(unicode.c_str(), unicode.length(), 7),
      "Hello, ");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(unicode.c_str(), unicode.length(), 8),
      "Hello, 世");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(unicode.c_str(), unicode.length(), 9),
      "Hello, 世界");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(unicode.c_str(), unicode.length(), 10),
      unicode);
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(unicode.c_str(), unicode.length(), 20),
      unicode);

  // String with emoji (surrogate pairs).
  std::string emoji = "Hello 🌍!";
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(emoji.c_str(), emoji.length(), 6),
      "Hello ");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(emoji.c_str(), emoji.length(), 7),
      "Hello 🌍");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(emoji.c_str(), emoji.length(), 8), emoji);
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(emoji.c_str(), emoji.length(), 10), emoji);

  std::string empty = "";
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(empty.c_str(), empty.length(), 0), "");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(empty.c_str(), empty.length(), 5), "");

  std::string mixed = "café世界🌍";
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(mixed.c_str(), mixed.length(), 3), "caf");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(mixed.c_str(), mixed.length(), 4), "café");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(mixed.c_str(), mixed.length(), 5),
      "café世");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(mixed.c_str(), mixed.length(), 6),
      "café世界");
  EXPECT_EQ(
      UnicodeUtil::truncateStringMin(mixed.c_str(), mixed.length(), 7), mixed);
}

TEST_F(UnicodeUtilTest, truncateStringMax) {
  std::string ascii = "Hello, world!";
  auto result =
      UnicodeUtil::truncateStringMax(ascii.c_str(), ascii.length(), 0);
  EXPECT_EQ(result, "");
  result = UnicodeUtil::truncateStringMax(ascii.c_str(), ascii.length(), 5);
  EXPECT_EQ(result, "Hellp"); // 'o' -> 'p'.
  result = UnicodeUtil::truncateStringMax(
      ascii.c_str(), ascii.length(), ascii.length());
  EXPECT_EQ(result, ascii);

  ascii = "Customer#000001500";
  result = UnicodeUtil::truncateStringMax(ascii.c_str(), ascii.length(), 16);
  EXPECT_EQ(result, "Customer#0000016"); //. '5' -> '6'.

  std::string unicode = "Hello, 世界!";
  result = UnicodeUtil::truncateStringMax(unicode.c_str(), unicode.length(), 8);
  EXPECT_EQ(result, "Hello, 丗");

  // No truncation needed.
  std::string shortString = "Hi";
  result = UnicodeUtil::truncateStringMax(
      shortString.c_str(), shortString.length(), 2);
  EXPECT_EQ(result, shortString);
  result = UnicodeUtil::truncateStringMax(
      shortString.c_str(), shortString.length(), 20);
  EXPECT_EQ(result, shortString);

  // Last character is already at maximum value.
  std::string maxChar =
      "Hello\U0010FFFF"; // U0010FFFF is maximum Unicode code point.
  result = UnicodeUtil::truncateStringMax(maxChar.c_str(), maxChar.length(), 6);
  EXPECT_EQ(result, maxChar);

  std::string empty = "";
  result = UnicodeUtil::truncateStringMax(empty.c_str(), empty.length(), 0);
  EXPECT_EQ(result, "");
  result = UnicodeUtil::truncateStringMax(empty.c_str(), empty.length(), 5);
  EXPECT_EQ(result, "");

  std::string single = "a";
  result = UnicodeUtil::truncateStringMax(single.c_str(), single.length(), 1);
  EXPECT_EQ(result, "a");

  std::string zChar = "zz";
  result = UnicodeUtil::truncateStringMax(zChar.c_str(), zChar.length(), 1);
  EXPECT_EQ(result, "{"); // 'z' -> '{'.

  // Emoji increment test
  std::string emojiTest = "🌍!!";
  result =
      UnicodeUtil::truncateStringMax(emojiTest.c_str(), emojiTest.length(), 1);
  EXPECT_EQ(result, "\U0001F30E"); // U1F30D (🌍) -> U1F30E.

  std::string multiByteTest = "café+";
  result = UnicodeUtil::truncateStringMax(
      multiByteTest.c_str(), multiByteTest.length(), 3);
  EXPECT_EQ(result, "cag"); // 'f' -> 'g'.
  result = UnicodeUtil::truncateStringMax(
      multiByteTest.c_str(), multiByteTest.length(), 4);
  EXPECT_EQ(result, "cafê"); // 'é' -> 'ê'.
}
