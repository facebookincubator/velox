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
#include <gmock/gmock.h>
#include <string_view>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using ::testing::ElementsAre;
using ::testing::IsEmpty;

namespace facebook::velox::functions::sparksql::test {
namespace {

class EncodeTest : public SparkFunctionBaseTest {
 protected:
  std::optional<std::string> encode(
      const std::optional<std::string>& input,
      const std::optional<std::string>& charset) {
    return evaluateOnce<std::string>("encode(c0, c1)", input, charset);
  }

  // Encodes and returns the raw output bytes, asserting a value was produced.
  std::vector<uint8_t> encodeBytes(
      const std::optional<std::string>& input,
      const std::optional<std::string>& charset) {
    auto result = encode(input, charset);
    EXPECT_TRUE(result.has_value());
    if (!result.has_value()) {
      return {};
    }
    return toBytes(*result);
  }

  static std::vector<uint8_t> toBytes(std::string_view s) {
    return std::vector<uint8_t>(s.begin(), s.end());
  }
};

TEST_F(EncodeTest, utf8) {
  EXPECT_EQ(encode("hello", "UTF-8"), "hello");
  EXPECT_EQ(encode("Spark SQL", "UTF-8"), "Spark SQL");
  EXPECT_EQ(encode("", "UTF-8"), "");
  // Canonical name and Java aliases, all case-insensitive.
  for (const auto* name : {"utf-8", "UTF8", "utf8"}) {
    SCOPED_TRACE(name);
    EXPECT_EQ(encode("abc", name), "abc");
  }
}

TEST_F(EncodeTest, usAscii) {
  EXPECT_EQ(encode("hello", "US-ASCII"), "hello");
  EXPECT_EQ(encode("", "US-ASCII"), "");
  // Canonical name and Java aliases, all case-insensitive.
  for (const auto* name : {"us-ascii", "ASCII", "ascii", "US_ASCII"}) {
    SCOPED_TRACE(name);
    EXPECT_EQ(encode("abc", name), "abc");
  }
}

TEST_F(EncodeTest, usAsciiReplacement) {
  // Non-ASCII codepoints should be replaced with '?' (0x3F).
  // U+00E9 (e-acute) is 0xC3 0xA9 in UTF-8, above 0x7F in US-ASCII.
  EXPECT_EQ(encode("\xC3\xA9", "US-ASCII"), "?");

  // Mixed ASCII and non-ASCII.
  EXPECT_EQ(
      encode(
          "a\xC3\xA9"
          "b",
          "US-ASCII"),
      "a?b");
}

TEST_F(EncodeTest, iso8859) {
  EXPECT_EQ(encode("hello", "ISO-8859-1"), "hello");
  EXPECT_EQ(encode("", "ISO-8859-1"), "");
  // Canonical name and Java aliases, all case-insensitive.
  for (const auto* name :
       {"iso-8859-1",
        "LATIN1",
        "latin1",
        "ISO8859_1",
        "ISO_8859_1",
        "ISO8859-1"}) {
    SCOPED_TRACE(name);
    EXPECT_EQ(encode("abc", name), "abc");
  }
}

TEST_F(EncodeTest, iso8859Passthrough) {
  // U+00E9 (e-acute) should pass through as 0xE9 in ISO-8859-1.
  EXPECT_THAT(encodeBytes("\xC3\xA9", "ISO-8859-1"), ElementsAre(0xE9));
}

TEST_F(EncodeTest, iso8859Replacement) {
  // Codepoints > 0xFF should be replaced with '?' (0x3F).
  // U+0100 (Latin A with macron) = 0xC4 0x80 in UTF-8, above 0xFF.
  EXPECT_EQ(encode("\xC4\x80", "ISO-8859-1"), "?");
}

TEST_F(EncodeTest, utf16be) {
  // 'A' in UTF-16BE is 0x00 0x41.
  EXPECT_THAT(encodeBytes("A", "UTF-16BE"), ElementsAre(0x00, 0x41));
  // Aliases produce the same result.
  for (const auto* name : {"UTF16BE", "UnicodeBigUnmarked"}) {
    SCOPED_TRACE(name);
    EXPECT_THAT(encodeBytes("A", name), ElementsAre(0x00, 0x41));
  }
  // Empty input yields empty output (no BOM for the unmarked variant).
  EXPECT_THAT(encodeBytes("", "UTF-16BE"), IsEmpty());
}

TEST_F(EncodeTest, utf16le) {
  // 'A' in UTF-16LE is 0x41 0x00.
  EXPECT_THAT(encodeBytes("A", "UTF-16LE"), ElementsAre(0x41, 0x00));
  // Aliases produce the same result.
  for (const auto* name : {"UTF16LE", "UnicodeLittleUnmarked"}) {
    SCOPED_TRACE(name);
    EXPECT_THAT(encodeBytes("A", name), ElementsAre(0x41, 0x00));
  }
  // Empty input yields empty output (no BOM for the unmarked variant).
  EXPECT_THAT(encodeBytes("", "UTF-16LE"), IsEmpty());
}

TEST_F(EncodeTest, unicodeLittleWithBom) {
  // Java's "UnicodeLittle" charset is UTF-16LE with a little-endian BOM,
  // matching "A".getBytes("UnicodeLittle") = FF FE 41 00.
  EXPECT_THAT(
      encodeBytes("A", "UnicodeLittle"), ElementsAre(0xFF, 0xFE, 0x41, 0x00));
  // Empty input yields no BOM, matching "".getBytes("UnicodeLittle").
  EXPECT_THAT(encodeBytes("", "UnicodeLittle"), IsEmpty());
}

TEST_F(EncodeTest, utf16) {
  // UTF-16 produces a big-endian BOM (0xFE 0xFF) followed by big-endian data,
  // matching Java/Spark behavior. BOM + 'A' in UTF-16BE = FE FF 00 41.
  EXPECT_THAT(encodeBytes("A", "UTF-16"), ElementsAre(0xFE, 0xFF, 0x00, 0x41));
  // Alias.
  EXPECT_THAT(encodeBytes("A", "UTF16"), ElementsAre(0xFE, 0xFF, 0x00, 0x41));
  // Empty string short-circuits before the encoder runs, returning empty
  // bytes with no BOM (matching Spark's Encode.encode()).
  EXPECT_THAT(encodeBytes("", "UTF-16"), IsEmpty());
}

TEST_F(EncodeTest, supplementaryCodepoint) {
  // U+1F600 (grinning face) is F0 9F 98 80 in UTF-8.
  // In UTF-16BE it is a surrogate pair D83D DE00.
  std::string emoji = "\xF0\x9F\x98\x80";
  EXPECT_THAT(
      encodeBytes(emoji, "UTF-16BE"), ElementsAre(0xD8, 0x3D, 0xDE, 0x00));
  // In UTF-16LE: 3D D8 00 DE.
  EXPECT_THAT(
      encodeBytes(emoji, "UTF-16LE"), ElementsAre(0x3D, 0xD8, 0x00, 0xDE));
}

TEST_F(EncodeTest, multibyte) {
  // Multi-byte UTF-8 character encoded to UTF-16BE.
  // U+00E9 (e-acute) is 0xC3 0xA9 in UTF-8, 0x00 0xE9 in UTF-16BE.
  EXPECT_THAT(encodeBytes("\xC3\xA9", "UTF-16BE"), ElementsAre(0x00, 0xE9));
}

TEST_F(EncodeTest, nullInputs) {
  EXPECT_EQ(encode(std::nullopt, "UTF-8"), std::nullopt);
  EXPECT_EQ(encode("hello", std::nullopt), std::nullopt);
  EXPECT_EQ(encode(std::nullopt, std::nullopt), std::nullopt);
}

TEST_F(EncodeTest, unsupportedCharset) {
  VELOX_ASSERT_USER_THROW(
      encode("hello", "INVALID-CHARSET"),
      "encode: unsupported charset 'INVALID-CHARSET'");
  // The unsupported-charset error is raised in call(), not initialize(), so it
  // is catchable by TRY even when the charset is a constant.
  EXPECT_EQ(
      evaluateOnce<std::string>(
          "try(encode(c0, c1))",
          std::optional<std::string>("hello"),
          std::optional<std::string>("INVALID-CHARSET")),
      std::nullopt);
}

TEST_F(EncodeTest, malformedUtf8) {
  // Truncated 2-byte sequence: 0xC3 alone. Should produce U+FFFD.
  // In UTF-8: U+FFFD = 0xEF 0xBF 0xBD.
  EXPECT_THAT(encodeBytes("\xC3", "UTF-8"), ElementsAre(0xEF, 0xBF, 0xBD));

  // Invalid continuation byte. Use an explicit length so the embedded NUL is
  // preserved (a bare "\xC3\x00" literal would truncate at the NUL).
  // 0xC3 is invalid (bad continuation) → U+FFFD, then 0x00 is valid ASCII.
  EXPECT_THAT(
      encodeBytes(std::string("\xC3\x00", 2), "UTF-8"),
      ElementsAre(0xEF, 0xBF, 0xBD, 0x00));

  // Invalid start byte 0xFF.
  EXPECT_THAT(encodeBytes("\xFF", "UTF-8"), ElementsAre(0xEF, 0xBF, 0xBD));
}

TEST_F(EncodeTest, realReplacementCharInInput) {
  // A genuine U+FFFD (EF BF BD) in the input is valid UTF-8 and must be copied
  // through unchanged (identity), not confused with a decode-time replacement.
  EXPECT_THAT(
      encodeBytes(std::string("\xEF\xBF\xBD", 3), "UTF-8"),
      ElementsAre(0xEF, 0xBF, 0xBD));

  // Real U+FFFD surrounded by ASCII, still an identity copy.
  EXPECT_THAT(
      encodeBytes(
          std::string(
              "a\xEF\xBF\xBD"
              "b",
              5),
          "UTF-8"),
      ElementsAre('a', 0xEF, 0xBF, 0xBD, 'b'));
}

TEST_F(EncodeTest, perRowCharset) {
  // Drive the non-constant charset path in call() with a different charset per
  // row so resolveCharset() runs per row rather than once in initialize().
  auto inputs = makeFlatVector<StringView>({"AB", "AB", "AB"});
  auto charsets = makeFlatVector<StringView>({"UTF-8", "US-ASCII", "UTF-16BE"});
  auto result = evaluate<SimpleVector<StringView>>(
      "encode(c0, c1)", makeRowVector({inputs, charsets}));
  auto utf8 = result->valueAt(0);
  auto ascii = result->valueAt(1);
  auto utf16be = result->valueAt(2);
  // UTF-8: "AB" -> 41 42.
  EXPECT_THAT(toBytes(std::string_view(utf8)), ElementsAre(0x41, 0x42));
  // US-ASCII: "AB" -> 41 42.
  EXPECT_THAT(toBytes(std::string_view(ascii)), ElementsAre(0x41, 0x42));
  // UTF-16BE: "AB" -> 00 41 00 42 (2 bytes per char, no BOM for BE).
  EXPECT_THAT(
      toBytes(std::string_view(utf16be)), ElementsAre(0x00, 0x41, 0x00, 0x42));
}

TEST_F(EncodeTest, charsetNameLengthBoundary) {
  // A 24-char (== kMaxCharsetLen) unsupported name must be handled without
  // overrunning the stack buffer, and rejected as unsupported.
  VELOX_ASSERT_USER_THROW(
      encode("hello", "ABCDEFGHIJKLMNOPQRSTUVWX"),
      "encode: unsupported charset 'ABCDEFGHIJKLMNOPQRSTUVWX'");
  // A 25-char (> kMaxCharsetLen) name is rejected early, also unsupported.
  VELOX_ASSERT_USER_THROW(
      encode("hello", "ABCDEFGHIJKLMNOPQRSTUVWXY"),
      "encode: unsupported charset 'ABCDEFGHIJKLMNOPQRSTUVWXY'");
}

TEST_F(EncodeTest, batch) {
  // Test with multiple rows using flat vectors.
  auto inputs = makeFlatVector<StringView>({"hello", "world", "abc"});
  auto charsets = makeFlatVector<StringView>({"UTF-8", "UTF-8", "UTF-8"});
  auto result = evaluate<SimpleVector<StringView>>(
      "encode(c0, c1)", makeRowVector({inputs, charsets}));
  EXPECT_EQ(result->valueAt(0).str(), "hello");
  EXPECT_EQ(result->valueAt(1).str(), "world");
  EXPECT_EQ(result->valueAt(2).str(), "abc");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
