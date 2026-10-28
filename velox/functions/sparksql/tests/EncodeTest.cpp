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
#include <string_view>

#include <gmock/gmock.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
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

  static std::vector<uint8_t> toBytes(std::string_view input) {
    return std::vector<uint8_t>(input.begin(), input.end());
  }

  void enableLegacyCodingErrorAction() {
    queryCtx_->testingOverrideConfigUnsafe({
        {SparkQueryConfig::qualify(SparkQueryConfig::kLegacyCodingErrorAction),
         "true"},
    });
  }

  void enableLegacyCharsets(bool replaceUnmappable) {
    queryCtx_->testingOverrideConfigUnsafe({
        {SparkQueryConfig::qualify(SparkQueryConfig::kLegacyJavaCharsets),
         "true"},
        {SparkQueryConfig::qualify(SparkQueryConfig::kLegacyCodingErrorAction),
         replaceUnmappable ? "true" : "false"},
    });
  }
};

TEST_F(EncodeTest, verifiedSingleByteCharsets) {
  struct Golden {
    const char* charset;
    const char* input;
    uint8_t expected;
  };
  // JDK 17 encoder goldens for every supported single-byte charset.
  const std::vector<Golden> goldens{
      {"IBM037", "\n", 0x15},
      {"IBM437", "\x1A", 0x1A},
      {"ISO-8859-2", "\xC3\xA9", 0xE9},
      {"ISO-8859-3", "\xC3\xA9", 0xE9},
      {"ISO-8859-4", "\xC3\xA9", 0xE9},
      {"ISO-8859-5", "\xD1\x8F", 0xEF},
      {"ISO-8859-6", "\xD8\xA7", 0xC7},
      {"ISO-8859-7", "\xCE\xB1", 0xE1},
      {"ISO-8859-8", "\xD7\x90", 0xE0},
      {"ISO-8859-9", "\xC3\xA9", 0xE9},
      {"x-iso-8859-11", "\xE0\xB8\x81", 0xA1},
      {"ISO-8859-13", "\xC3\xA9", 0xE9},
      {"ISO-8859-15", "\xC3\xA9", 0xE9},
      {"ISO-8859-16", "\xE2\x82\xAC", 0xA4},
      {"KOI8-R", "\xD1\x8F", 0xD1},
      {"KOI8-U", "\xD1\x8F", 0xD1},
      {"windows-1250", "\xE2\x82\xAC", 0x80},
      {"windows-1251", "\xD1\x8F", 0xFF},
      {"windows-1252", "\xC3\xA9", 0xE9},
      {"windows-1253", "\xCE\xB1", 0xE1},
      {"windows-1254", "\xC3\xA9", 0xE9},
      {"windows-1255", "\xD7\x90", 0xE0},
      {"windows-1256", "\xD8\xA7", 0xC7},
      {"windows-1257", "\xC3\xA9", 0xE9},
      {"windows-1258", "\xC3\xA9", 0xE9},
  };
  for (const bool replaceUnmappable : {false, true}) {
    enableLegacyCharsets(replaceUnmappable);
    for (const auto& [charset, input, expected] : goldens) {
      SCOPED_TRACE(charset);
      EXPECT_THAT(encodeBytes(input, charset), ElementsAre(expected));
    }
    // Spark lowercases U+212A KELVIN SIGN to the ASCII letter k in names.
    EXPECT_THAT(
        encodeBytes("\xD1\x8F", "\xE2\x84\xAAOI8-R"), ElementsAre(0xD1));
    EXPECT_THAT(encodeBytes("\xCE\xB1", "gree\xE2\x84\xAA"), ElementsAre(0xE1));
    EXPECT_THAT(
        encodeBytes(
            "A",
            "UnicodeBigUnmar\xE2\x84\xAA"
            "ed"),
        ElementsAre(0x00, 0x41));
    EXPECT_THAT(
        encodeBytes(
            "A",
            "unicodelittleunmar\xE2\x84\xAA"
            "ed"),
        ElementsAre(0x41, 0x00));
  }
  enableLegacyCharsets(false);
  EXPECT_THAT(encodeBytes("\xC3\xA9", "cp1252"), ElementsAre(0xE9));
  EXPECT_EQ(encode("A", "UTF8"), "A");
  EXPECT_EQ(encode("A", "default"), "A");
  VELOX_ASSERT_USER_THROW(
      encode("\xC2\x81", "windows-1250"), "cannot be encoded");
  VELOX_ASSERT_USER_THROW(
      encode("\xC2\x81", "windows-1252"), "cannot be encoded");
  VELOX_ASSERT_USER_THROW(
      encode("\xE2\x80\x8D", "windows-1252"), "cannot be encoded");
  enableLegacyCharsets(true);
  EXPECT_EQ(encode("\xC2\x81", "windows-1250"), "?");
  EXPECT_EQ(encode("\xC2\x81", "windows-1252"), "?");
  EXPECT_THAT(encodeBytes("\xFF", "IBM037"), ElementsAre(0x3F));
  EXPECT_EQ(encode("\xC0\x80", "windows-1252"), "??");
}

TEST_F(EncodeTest, rejectUnverifiedLegacyCharsets) {
  for (const bool replaceUnmappable : {false, true}) {
    enableLegacyCharsets(replaceUnmappable);
    for (const auto* charset :
         {"Shift_JIS",
          "windows-31j",
          "Big5",
          "GBK",
          "EUC-JP",
          "EUC-KR",
          "GB2312",
          "ISO-2022-JP",
          "ISO-2022-CN",
          "x-JISAutoDetect",
          "x-IBM930",
          "HZ-GB-2312",
          "UTF-32BE-BOM",
          "x-mswin-936",
          "\xE2\x84",
          "\xE2\x84\xAA",
          "\xE2\x84\xAAOI8-R ",
          "\xE2\x84\xAAOI8-R\xE2\x84",
          "U\xC5\xBF-ASCII"}) {
      SCOPED_TRACE(charset);
      VELOX_ASSERT_USER_THROW(encode("A", charset), "unsupported charset");
      VELOX_ASSERT_USER_THROW(encode("", charset), "unsupported charset");
      EXPECT_EQ(encode(std::nullopt, charset), std::nullopt);
      EXPECT_EQ(
          evaluateOnce<std::string>(
              "try(encode(c0, c1))",
              std::optional<std::string>("A"),
              std::optional<std::string>(charset)),
          std::nullopt);
    }
  }
}

TEST_F(EncodeTest, dynamicLegacyCharsets) {
  for (const bool replaceUnmappable : {false, true}) {
    enableLegacyCharsets(replaceUnmappable);
    auto inputs = makeFlatVector<std::string>({
        "A",
        "\xC2\x81",
        "A",
        "\n",
        "A",
        "A",
        "A",
        "A",
        "A",
        "\xD1\x8F",
        "\xD1\x8F",
        "A",
        "\xD1\x8F",
        "A",
    });
    auto charsets = makeFlatVector<std::string>({
        "cp1250",
        "cp1250",
        "no-such-charset",
        "IBM037",
        "UTF8",
        "cp1252",
        "CP1250",
        std::string("windows-1252") + '\0' + "x",
        std::string(1'024, 'x'),
        "\xE2\x84\xAAOI8-R",
        "\xE2\x84\xAAOI8-R",
        std::string("\xE2\x84\xAAOI8-R") + '\0',
        "koi8-r",
        "UnicodeBigUnmar\xE2\x84\xAA"
        "ed",
    });
    auto expected = makeNullableFlatVector<std::string>(
        {
            "A",
            replaceUnmappable ? std::optional<std::string>("?") : std::nullopt,
            std::nullopt,
            "\x15",
            "A",
            "A",
            "A",
            std::nullopt,
            std::nullopt,
            "\xD1",
            "\xD1",
            std::nullopt,
            "\xD1",
            std::string("\0A", 2),
        },
        VARBINARY());
    auto result =
        evaluate("try(encode(c0, c1))", makeRowVector({inputs, charsets}));
    velox::test::assertEqualVectors(expected, result);
  }
}

TEST_F(EncodeTest, zeroCopyOutputLifetime) {
  const std::string ascii(128, 'A');
  const std::string unicode = std::string(128, 'A') + "\xF0\x9F\x98\x80";
  for (const auto* charset : {"UTF-8", "US-ASCII", "ISO-8859-1"}) {
    auto input = makeFlatVector<std::string>({ascii});
    auto result = evaluate<SimpleVector<StringView>>(
        fmt::format("encode(c0, '{}')", charset), makeRowVector({input}));
    EXPECT_EQ(result->valueAt(0).data(), input->valueAt(0).data());
    input.reset();
    EXPECT_EQ(result->valueAt(0).str(), ascii);
  }
  auto input = makeFlatVector<std::string>({unicode});
  auto result = evaluate<SimpleVector<StringView>>(
      "encode(c0, 'UTF-8')", makeRowVector({input}));
  EXPECT_EQ(result->valueAt(0).data(), input->valueAt(0).data());
  input.reset();
  EXPECT_EQ(result->valueAt(0).str(), unicode);
}

TEST_F(EncodeTest, zeroCopyWrappedInputLifetime) {
  enableLegacyCharsets(false);
  const std::string ascii(128, 'A');
  for (const auto* charset :
       {"UTF-8", "US-ASCII", "ISO-8859-1", "windows-1252"}) {
    for (const bool lazy : {false, true}) {
      auto flat = makeFlatVector<std::string>({ascii});
      const auto* data = flat->valueAt(0).data();
      VectorPtr input = wrapInDictionary(makeIndices({0}), flat);
      flat.reset();
      if (lazy) {
        input = wrapInLazyDictionary(input);
      }
      auto result = evaluate<SimpleVector<StringView>>(
          fmt::format("encode(c0, '{}')", charset), makeRowVector({input}));
      EXPECT_EQ(result->valueAt(0).data(), data);
      input.reset();
      EXPECT_EQ(result->valueAt(0).str(), ascii);
    }
  }
}

TEST_F(EncodeTest, utf8) {
  EXPECT_EQ(encode("hello", "UTF-8"), "hello");
  EXPECT_EQ(encode("Spark SQL", "UTF-8"), "Spark SQL");
  EXPECT_EQ(encode("", "UTF-8"), "");
  EXPECT_EQ(encode("abc", "utf-8"), "abc");
}

TEST_F(EncodeTest, usAscii) {
  EXPECT_EQ(encode("hello", "US-ASCII"), "hello");
  EXPECT_EQ(encode("", "US-ASCII"), "");
  EXPECT_EQ(encode("abc", "us-ascii"), "abc");
  EXPECT_THAT(encodeBytes("\x7F", "US-ASCII"), ElementsAre(0x7F));
  VELOX_ASSERT_USER_THROW(
      encode("\xC2\x80", "US-ASCII"),
      "encode: input contains a character that cannot be encoded using 'US-ASCII'");
}

TEST_F(EncodeTest, usAsciiReplacement) {
  enableLegacyCodingErrorAction();
  EXPECT_EQ(encode("\xC3\xA9", "US-ASCII"), "?");
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
  EXPECT_EQ(encode("abc", "iso-8859-1"), "abc");
}

TEST_F(EncodeTest, iso8859Passthrough) {
  EXPECT_THAT(encodeBytes("\xC3\xA9", "ISO-8859-1"), ElementsAre(0xE9));
  EXPECT_THAT(encodeBytes("\xC3\xBF", "ISO-8859-1"), ElementsAre(0xFF));
}

TEST_F(EncodeTest, iso8859Replacement) {
  enableLegacyCodingErrorAction();
  EXPECT_EQ(encode("\xC4\x80", "ISO-8859-1"), "?");
}

TEST_F(EncodeTest, utf16be) {
  EXPECT_THAT(encodeBytes("A", "UTF-16BE"), ElementsAre(0x00, 0x41));
  EXPECT_THAT(encodeBytes("A", "utf-16be"), ElementsAre(0x00, 0x41));
  EXPECT_THAT(encodeBytes("", "UTF-16BE"), IsEmpty());
}

TEST_F(EncodeTest, utf16le) {
  EXPECT_THAT(encodeBytes("A", "UTF-16LE"), ElementsAre(0x41, 0x00));
  EXPECT_THAT(encodeBytes("A", "utf-16le"), ElementsAre(0x41, 0x00));
  EXPECT_THAT(encodeBytes("", "UTF-16LE"), IsEmpty());
}

TEST_F(EncodeTest, utf16) {
  EXPECT_THAT(encodeBytes("A", "UTF-16"), ElementsAre(0xFE, 0xFF, 0x00, 0x41));
  EXPECT_THAT(encodeBytes("A", "utf-16"), ElementsAre(0xFE, 0xFF, 0x00, 0x41));
  EXPECT_THAT(encodeBytes("", "UTF-16"), IsEmpty());
}

TEST_F(EncodeTest, utf32) {
  EXPECT_THAT(encodeBytes("A", "UTF-32"), ElementsAre(0x00, 0x00, 0x00, 0x41));
  EXPECT_THAT(encodeBytes("A", "utf-32"), ElementsAre(0x00, 0x00, 0x00, 0x41));
  EXPECT_THAT(encodeBytes("", "UTF-32"), IsEmpty());
  for (const size_t length : {12, 13, 31, 32, 63, 64, 65, 127, 128, 255, 256}) {
    SCOPED_TRACE(length);
    std::string input(length - 9, 'A');
    input += "\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80";
    std::string expected;
    for (size_t i = 0; i < length - 9; ++i) {
      expected.append("\x00\x00\x00\x41", 4);
    }
    expected.append("\x00\x00\x00\xE9\x00\x00\x20\xAC\x00\x01\xF6\x00", 12);
    EXPECT_EQ(encode(input, "UTF-32"), expected);
  }
}

TEST_F(EncodeTest, wideAsciiBoundaries) {
  struct Golden {
    const char* charset;
    std::string pattern;
    std::string bom;
  };
  const std::vector<Golden> goldens{
      {"UTF-16BE", std::string("\x00\x00\x00\x41\x00\x7F", 6), ""},
      {"UTF-16LE", std::string("\x00\x00\x41\x00\x7F\x00", 6), ""},
      {"UTF-16", std::string("\x00\x00\x00\x41\x00\x7F", 6), "\xFE\xFF"},
      {"UTF-32",
       std::string("\x00\x00\x00\x00\x00\x00\x00\x41\x00\x00\x00\x7F", 12),
       ""},
  };
  for (const auto& [charset, pattern, bom] : goldens) {
    SCOPED_TRACE(charset);
    for (const size_t length :
         {0, 1, 7, 8, 11, 12, 13, 31, 32, 63, 64, 65, 127, 128, 255, 256}) {
      SCOPED_TRACE(length);
      std::string input;
      std::string expected;
      while (input.size() < length) {
        input.append("\x00\x41\x7F", 3);
        expected += pattern;
      }
      input.resize(length);
      expected.resize(length * pattern.size() / 3);
      if (length != 0) {
        expected = bom + expected;
      }
      EXPECT_EQ(encode(input, charset), expected);
      EXPECT_EQ(
          evaluateOnce<std::string>(
              fmt::format("encode(c0, '{}')", charset),
              std::optional<std::string>(input)),
          expected);
    }
  }
}

TEST_F(EncodeTest, supplementaryCodepoint) {
  const std::string emoji = "\xF0\x9F\x98\x80";
  EXPECT_THAT(
      encodeBytes(emoji, "UTF-16BE"), ElementsAre(0xD8, 0x3D, 0xDE, 0x00));
  EXPECT_THAT(
      encodeBytes(emoji, "UTF-16LE"), ElementsAre(0x3D, 0xD8, 0x00, 0xDE));
  EXPECT_THAT(
      encodeBytes(emoji, "UTF-16"),
      ElementsAre(0xFE, 0xFF, 0xD8, 0x3D, 0xDE, 0x00));
  EXPECT_THAT(
      encodeBytes(emoji, "UTF-32"), ElementsAre(0x00, 0x01, 0xF6, 0x00));
}

TEST_F(EncodeTest, multibyte) {
  EXPECT_THAT(encodeBytes("\xC3\xA9", "UTF-16BE"), ElementsAre(0x00, 0xE9));
}

TEST_F(EncodeTest, nullInputs) {
  EXPECT_EQ(encode(std::nullopt, "UTF-8"), std::nullopt);
  EXPECT_EQ(encode("hello", std::nullopt), std::nullopt);
  EXPECT_EQ(encode(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(encode(std::nullopt, "INVALID-CHARSET"), std::nullopt);
}

TEST_F(EncodeTest, unsupportedCharset) {
  VELOX_ASSERT_USER_THROW(
      encode("hello", "INVALID-CHARSET"),
      "encode: unsupported charset 'INVALID-CHARSET'");
  EXPECT_EQ(
      evaluateOnce<std::string>(
          "try(encode(c0, 'INVALID-CHARSET'))",
          std::optional<std::string>("hello")),
      std::nullopt);
}

TEST_F(EncodeTest, charsetScope) {
  const std::vector<std::string> names{
      "UTF8",
      "ASCII",
      "US_ASCII",
      "LATIN1",
      "ISO8859_1",
      "UTF16",
      "Unicode",
      "UTF16BE",
      "UTF16LE",
      "UnicodeLittle",
      "UTF-32BE",
      "UTF-32LE",
      "windows-1252",
      "windows-31j",
      "Shift_JIS",
      "Big5",
      "GBK",
      "IBM037",
      "CESU-8",
      "ISO-2022-JP",
      "ISO-2022-CN",
      "x-JISAutoDetect",
      "default",
      "UTF-32BE-BOM",
      "x-mswin-936",
      "ISO-8859-16",
      "\xE2\x84\xAAOI8-R",
      "UnicodeBigUnmar\xE2\x84\xAA"
      "ed",
      "",
      " UTF-8",
      "UTF-8 ",
      std::string("UTF-8\0GBK", 9)};
  for (const bool replaceUnmappable : {false, true}) {
    SCOPED_TRACE(replaceUnmappable);
    queryCtx_->testingOverrideConfigUnsafe({
        {SparkQueryConfig::qualify(SparkQueryConfig::kLegacyCodingErrorAction),
         replaceUnmappable ? "true" : "false"},
    });
    for (const auto& name : names) {
      SCOPED_TRACE(name);
      VELOX_ASSERT_USER_THROW(
          encode("hello", name),
          fmt::format("encode: unsupported charset '{}'", name));
      VELOX_ASSERT_USER_THROW(
          encode("", name),
          fmt::format("encode: unsupported charset '{}'", name));
    }
    const auto inputs = makeFlatVector<std::string>(
        names.size(), [](auto /*row*/) { return "hello"; });
    const auto charsets = makeFlatVector<std::string>(names);
    const auto result = evaluate<SimpleVector<StringView>>(
        "try(encode(c0, c1))", makeRowVector({inputs, charsets}));
    for (vector_size_t row = 0; row < result->size(); ++row) {
      EXPECT_TRUE(result->isNullAt(row)) << names[row];
    }
    EXPECT_EQ(
        evaluateOnce<std::string>(
            "try(encode(c0, 'windows-1252'))",
            std::optional<std::string>("hello")),
        std::nullopt);
  }
}

TEST_F(EncodeTest, codingErrorAction) {
  for (const auto* charset : {"US-ASCII", "ISO-8859-1"}) {
    SCOPED_TRACE(charset);
    VELOX_ASSERT_USER_THROW(
        encode("\xC4\x80", charset),
        fmt::format(
            "encode: input contains a character that cannot be encoded using '{}'",
            charset));
    EXPECT_EQ(
        evaluateOnce<std::string>(
            fmt::format("try(encode(c0, '{}'))", charset),
            std::optional<std::string>("\xC4\x80")),
        std::nullopt);
  }
  enableLegacyCodingErrorAction();
  for (const auto* charset : {"US-ASCII", "ISO-8859-1"}) {
    SCOPED_TRACE(charset);
    EXPECT_EQ(encode("\xC4\x80", charset), "?");
    EXPECT_EQ(encode("\xF0\x9F\x98\x80", charset), "?");
    EXPECT_EQ(encode("\xFF", charset), "?");
  }
}

TEST_F(EncodeTest, malformedUtf8) {
  EXPECT_THAT(encodeBytes("\xC3", "UTF-8"), ElementsAre(0xEF, 0xBF, 0xBD));
  EXPECT_THAT(
      encodeBytes(std::string("\xC3\x00", 2), "UTF-8"),
      ElementsAre(0xEF, 0xBF, 0xBD, 0x00));
  EXPECT_THAT(encodeBytes("\xFF", "UTF-8"), ElementsAre(0xEF, 0xBF, 0xBD));
  EXPECT_THAT(
      encodeBytes(std::string("\xC0\x80", 2), "UTF-8"),
      ElementsAre(0xEF, 0xBF, 0xBD, 0xEF, 0xBF, 0xBD));
  EXPECT_THAT(
      encodeBytes(std::string("\xF4\x90\x80\x80", 4), "UTF-8"),
      ElementsAre(
          0xEF,
          0xBF,
          0xBD,
          0xEF,
          0xBF,
          0xBD,
          0xEF,
          0xBF,
          0xBD,
          0xEF,
          0xBF,
          0xBD));
  EXPECT_THAT(
      encodeBytes(std::string("\xED\xA0\x80", 3), "UTF-8"),
      ElementsAre(0xEF, 0xBF, 0xBD));
  EXPECT_THAT(encodeBytes("\xFF", "UTF-16BE"), ElementsAre(0xFF, 0xFD));
  EXPECT_THAT(encodeBytes("\xFF", "UTF-16LE"), ElementsAre(0xFD, 0xFF));
  EXPECT_THAT(
      encodeBytes("\xFF", "UTF-16"), ElementsAre(0xFE, 0xFF, 0xFF, 0xFD));
  EXPECT_THAT(
      encodeBytes("\xFF", "UTF-32"), ElementsAre(0x00, 0x00, 0xFF, 0xFD));
}

TEST_F(EncodeTest, realReplacementCharInInput) {
  EXPECT_THAT(
      encodeBytes(std::string("\xEF\xBF\xBD", 3), "UTF-8"),
      ElementsAre(0xEF, 0xBF, 0xBD));
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
  const std::vector<std::string> names{
      "UTF-8",
      "US-ASCII",
      "ISO-8859-1",
      "UTF-16BE",
      "UTF-16LE",
      "UTF-16",
      "UTF-32"};
  const std::vector<std::vector<uint8_t>> expected{
      {0x41, 0x00},
      {0x41, 0x00},
      {0x41, 0x00},
      {0x00, 0x41, 0x00, 0x00},
      {0x41, 0x00, 0x00, 0x00},
      {0xFE, 0xFF, 0x00, 0x41, 0x00, 0x00},
      {0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x00}};
  const auto inputs = makeFlatVector<std::string>(
      names.size(), [](auto /*row*/) { return std::string("A\0", 2); });
  const auto charsets = makeFlatVector<std::string>(names);
  const auto result = evaluate<SimpleVector<StringView>>(
      "encode(c0, c1)", makeRowVector({inputs, charsets}));
  for (vector_size_t row = 0; row < result->size(); ++row) {
    SCOPED_TRACE(names[row]);
    ASSERT_FALSE(result->isNullAt(row));
    EXPECT_EQ(toBytes(std::string_view(result->valueAt(row))), expected[row]);
    EXPECT_EQ(encodeBytes(std::string("A\0", 2), names[row]), expected[row]);
    const auto constantResult = evaluateOnce<std::string>(
        fmt::format("encode(c0, '{}')", names[row]),
        std::optional<std::string>(std::string("A\0", 2)));
    ASSERT_TRUE(constantResult.has_value());
    EXPECT_EQ(toBytes(*constantResult), expected[row]);
  }
}

TEST_F(EncodeTest, unsupportedCharsetNameLengths) {
  VELOX_ASSERT_USER_THROW(
      encode("hello", "ABCDEFGHIJKLMNOPQRSTUVWX"),
      "encode: unsupported charset 'ABCDEFGHIJKLMNOPQRSTUVWX'");
  VELOX_ASSERT_USER_THROW(
      encode("hello", "ABCDEFGHIJKLMNOPQRSTUVWXY"),
      "encode: unsupported charset 'ABCDEFGHIJKLMNOPQRSTUVWXY'");
}

TEST_F(EncodeTest, batch) {
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
