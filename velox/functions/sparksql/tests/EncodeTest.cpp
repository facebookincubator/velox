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
#include <tuple>

#include <gmock/gmock.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
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
};

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

TEST_F(EncodeTest, iso8859) {
  EXPECT_EQ(encode("hello", "ISO-8859-1"), "hello");
  EXPECT_EQ(encode("", "ISO-8859-1"), "");
  EXPECT_EQ(encode("abc", "iso-8859-1"), "abc");
  EXPECT_THAT(encodeBytes("\xC3\xA9", "ISO-8859-1"), ElementsAre(0xE9));
  EXPECT_THAT(encodeBytes("\xC3\xBF", "ISO-8859-1"), ElementsAre(0xFF));
}

TEST_F(EncodeTest, utf16AndUtf32) {
  const std::vector<std::tuple<std::string, std::string, std::vector<uint8_t>>>
      testCases{
          {"UTF-16BE", "utf-16be", {0x00, 0x41}},
          {"UTF-16LE", "utf-16le", {0x41, 0x00}},
          {"UTF-16", "utf-16", {0xFE, 0xFF, 0x00, 0x41}},
          {"UTF-32", "utf-32", {0x00, 0x00, 0x00, 0x41}},
  };
  for (const auto& [charset, lowercaseCharset, expected] : testCases) {
    SCOPED_TRACE(charset);
    EXPECT_THAT(encodeBytes("A", charset), ElementsAreArray(expected));
    EXPECT_THAT(encodeBytes("A", lowercaseCharset), ElementsAreArray(expected));
    EXPECT_THAT(encodeBytes("", charset), IsEmpty());
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
  EXPECT_EQ(
      encode(
          "a\xC3\xA9"
          "b",
          "US-ASCII"),
      "a?b");
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

} // namespace
} // namespace facebook::velox::functions::sparksql::test
