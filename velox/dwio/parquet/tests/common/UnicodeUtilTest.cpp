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
#include <variant>

using namespace facebook::velox::parquet;

class UnicodeUtilTest : public testing::Test {};

namespace {
// Helper to safely cast string length to int32_t for function calls
inline int32_t len(const std::string& s) {
  return static_cast<int32_t>(s.length());
}
} // namespace

TEST_F(UnicodeUtilTest, truncateStringMinLength) {
  const auto truncated = [](const std::string& input, int32_t numCodePoints) {
    return std::string_view{
        input.data(),
        static_cast<size_t>(UnicodeUtil::truncateStringMinLength(
            input.c_str(), len(input), numCodePoints))};
  };

  // ASCII string.
  const std::string ascii = "Hello, world!";
  EXPECT_EQ(truncated(ascii, 0), "");
  EXPECT_EQ(truncated(ascii, 1), "H");
  EXPECT_EQ(truncated(ascii, 5), "Hello");
  EXPECT_EQ(truncated(ascii, 13), ascii);
  EXPECT_EQ(truncated(ascii, 20), ascii);

  // String with multi-bytes characters.
  const std::string unicode = "Hello, 世界!";
  EXPECT_EQ(truncated(unicode, 7), "Hello, ");
  EXPECT_EQ(truncated(unicode, 8), "Hello, 世");
  EXPECT_EQ(truncated(unicode, 9), "Hello, 世界");
  EXPECT_EQ(truncated(unicode, 10), unicode);
  EXPECT_EQ(truncated(unicode, 20), unicode);

  // String with emoji. In UTF-16, characters outside the Basic Multilingual
  // Plane, such as 🌍 (U+1F30D), are represented as surrogate pairs: two
  // 16-bit code units that together encode one Unicode code point. That means
  // "Hello 🌍!" occupies 6 code units for "Hello ", 2 for 🌍, and 1 for "!".
  // These expectations verify truncation respects character boundaries and
  // never splits the surrogate pair for the emoji.
  const std::string emoji = "Hello 🌍!";
  EXPECT_EQ(truncated(emoji, 6), "Hello ");
  EXPECT_EQ(truncated(emoji, 7), "Hello 🌍");
  EXPECT_EQ(truncated(emoji, 8), emoji);
  EXPECT_EQ(truncated(emoji, 10), emoji);

  const std::string empty = "";
  EXPECT_EQ(truncated(empty, 0), "");
  EXPECT_EQ(truncated(empty, 5), "");

  const std::string mixed = "café世界🌍";
  EXPECT_EQ(truncated(mixed, 3), "caf");
  EXPECT_EQ(truncated(mixed, 4), "café");
  EXPECT_EQ(truncated(mixed, 5), "café世");
  EXPECT_EQ(truncated(mixed, 6), "café世界");
  EXPECT_EQ(truncated(mixed, 7), mixed);
}

TEST_F(UnicodeUtilTest, truncateStringUpper) {
  const auto asStringView = [](const auto& result) -> std::string_view {
    return std::visit(
        [](const auto& value) -> std::string_view { return value; }, result);
  };

  const std::string ascii = "Hello, world!";
  auto result = UnicodeUtil::truncateStringUpper(ascii.c_str(), len(ascii), 0);
  EXPECT_EQ(asStringView(result), "");
  result = UnicodeUtil::truncateStringUpper(ascii.c_str(), len(ascii), 5);
  EXPECT_EQ(asStringView(result), "Hellp"); // 'o' -> 'p'.
  result =
      UnicodeUtil::truncateStringUpper(ascii.c_str(), len(ascii), len(ascii));
  EXPECT_EQ(asStringView(result), ascii);

  const std::string ascii2 = "Customer#000001500";
  result = UnicodeUtil::truncateStringUpper(ascii2.c_str(), len(ascii2), 16);
  EXPECT_EQ(asStringView(result), "Customer#0000016"); // '5' -> '6'.

  const std::string unicode = "Hello, 世界!";
  result = UnicodeUtil::truncateStringUpper(unicode.c_str(), len(unicode), 8);
  EXPECT_EQ(asStringView(result), "Hello, 丗");

  // No truncation needed.
  const std::string shortString = "Hi";
  result = UnicodeUtil::truncateStringUpper(
      shortString.c_str(), len(shortString), 2);
  EXPECT_EQ(asStringView(result), shortString);
  result = UnicodeUtil::truncateStringUpper(
      shortString.c_str(), len(shortString), 20);
  EXPECT_EQ(asStringView(result), shortString);

  // Last character is already at maximum value.
  const std::string maxChar =
      "Hello\U0010FFFF"; // U0010FFFF is maximum Unicode code point.
  result = UnicodeUtil::truncateStringUpper(maxChar.c_str(), len(maxChar), 6);
  EXPECT_EQ(asStringView(result), maxChar);

  const std::string empty = "";
  result = UnicodeUtil::truncateStringUpper(empty.c_str(), len(empty), 0);
  EXPECT_EQ(asStringView(result), "");
  result = UnicodeUtil::truncateStringUpper(empty.c_str(), len(empty), 5);
  EXPECT_EQ(asStringView(result), "");

  const std::string single = "a";
  result = UnicodeUtil::truncateStringUpper(single.c_str(), len(single), 1);
  EXPECT_EQ(asStringView(result), "a");

  const std::string zChar = "zz";
  result = UnicodeUtil::truncateStringUpper(zChar.c_str(), len(zChar), 1);
  EXPECT_EQ(asStringView(result), "{"); // 'z' -> '{'.

  // Emoji increment test
  const std::string emojiTest = "🌍!!";
  result =
      UnicodeUtil::truncateStringUpper(emojiTest.c_str(), len(emojiTest), 1);
  EXPECT_EQ(asStringView(result), "\U0001F30E"); // U1F30D (🌍) -> U1F30E.

  const std::string multiByteTest = "café+";
  result = UnicodeUtil::truncateStringUpper(
      multiByteTest.c_str(), len(multiByteTest), 3);
  EXPECT_EQ(asStringView(result), "cag"); // 'f' -> 'g'.
  result = UnicodeUtil::truncateStringUpper(
      multiByteTest.c_str(), len(multiByteTest), 4);
  EXPECT_EQ(asStringView(result), "cafê"); // 'é' -> 'ê'.

  // Test case where truncation happens but no code point can be incremented.
  // All retained code points are U+10FFFF (maximum Unicode code point).
  // Falls back to returning the truncated string.
  const std::string noUpperBound = "\U0010FFFF\U0010FFFFabc";
  result = UnicodeUtil::truncateStringUpper(
      noUpperBound.c_str(), len(noUpperBound), 2);
  EXPECT_EQ(
      asStringView(result),
      "\U0010FFFF\U0010FFFF"); // Fallback to truncated string.

  // Test surrogate boundary case: U+D7FF should increment to U+E000.
  const std::string surrogateBoundary = "\U0000D7FFabc";
  result = UnicodeUtil::truncateStringUpper(
      surrogateBoundary.c_str(), len(surrogateBoundary), 1);
  EXPECT_EQ(
      asStringView(result),
      "\U0000E000"); // U+D7FF -> U+E000 (skips surrogate range).
}
