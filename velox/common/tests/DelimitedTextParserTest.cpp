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
#include <gtest/gtest.h>

#include "velox/common/text/DelimitedTextParser.h"

namespace facebook::velox::text {
namespace {

using detail::DelimitedTextScanAction;
using detail::DelimitedTextScanner;

TEST(DelimitedTextParserTest, quotedFields) {
  DelimitedTextBuffers buffers;
  DelimitedTextParser::splitLine(
      R"(one,"two,three","four\"five",)",
      DelimitedTextOptions{.delimiter = ',', .escape = '\\', .quote = '"'},
      0,
      buffers);

  ASSERT_EQ(buffers.fields.size(), 4);
  EXPECT_EQ(buffers.fields[0], "one");
  EXPECT_EQ(buffers.fields[1], "two,three");
  EXPECT_EQ(buffers.fields[2], R"(four"five)");
  EXPECT_EQ(buffers.fields[3], "");
}

TEST(DelimitedTextParserTest, unquotedFieldsRemainZeroCopy) {
  DelimitedTextBuffers buffers;
  const auto initialDecodedCapacity = buffers.decodedFields.capacity();
  const std::string firstField(initialDecodedCapacity + 1, 'a');
  const std::string line{firstField + ",two"};
  DelimitedTextParser::splitLine(
      line,
      DelimitedTextOptions{.delimiter = ',', .escape = '\\', .quote = '"'},
      0,
      buffers);

  EXPECT_TRUE(buffers.decodedFields.empty());
  EXPECT_EQ(buffers.decodedFields.capacity(), initialDecodedCapacity);
  ASSERT_EQ(buffers.fields.size(), 2);
  EXPECT_EQ(buffers.fields[0].data(), line.data());
  EXPECT_EQ(buffers.fields[1].data(), line.data() + firstField.size() + 1);
}

TEST(DelimitedTextParserTest, escapedDelimiters) {
  DelimitedTextBuffers buffers;
  DelimitedTextParser::splitLine(
      R"(one\,two,three\q,"four\,five")",
      DelimitedTextOptions{.delimiter = ',', .escape = '\\', .quote = '"'},
      0,
      buffers);

  EXPECT_THAT(
      buffers.fields,
      testing::ElementsAre("one,two", "threeq", R"(four\,five)"));
}

TEST(DelimitedTextParserTest, malformedFieldPreservesEarlierDecodedViews) {
  DelimitedTextBuffers buffers;
  DelimitedTextParser::splitLine(
      R"(one\,two,"three"four,five)",
      DelimitedTextOptions{.delimiter = ',', .escape = '\\', .quote = '"'},
      0,
      buffers);

  EXPECT_THAT(
      buffers.fields,
      testing::ElementsAre("one,two", R"("three"four)", "five"));
}

TEST(DelimitedTextParserTest, scanner) {
  DelimitedTextScanner scanner(
      DelimitedTextOptions{.delimiter = ',', .escape = '\\', .quote = '"'},
      /*decodeEscapedNewlines=*/true);

  EXPECT_EQ(
      scanner.consume('a', /*isDelimiter=*/false).action,
      DelimitedTextScanAction::kAppend);
  EXPECT_EQ(
      scanner.consume('\\', /*isDelimiter=*/false).action,
      DelimitedTextScanAction::kSkip);
  EXPECT_TRUE(scanner.isEscaping());
  const auto escapedDelimiter = scanner.consume(',', /*isDelimiter=*/true);
  EXPECT_EQ(escapedDelimiter.action, DelimitedTextScanAction::kAppend);
  EXPECT_EQ(escapedDelimiter.value, ',');
  EXPECT_EQ(
      scanner.consume(',', /*isDelimiter=*/true).action,
      DelimitedTextScanAction::kDelimiter);

  DelimitedTextScanner sharedDelimiterAndEscape(
      DelimitedTextOptions{.delimiter = '\\', .escape = '\\', .quote = '\0'},
      /*decodeEscapedNewlines=*/true);
  EXPECT_EQ(
      sharedDelimiterAndEscape.consume('\\').action,
      DelimitedTextScanAction::kDelimiter);
  EXPECT_FALSE(sharedDelimiterAndEscape.isEscaping());

  scanner.reset();
  EXPECT_EQ(
      scanner.consume('\\', /*isDelimiter=*/false).action,
      DelimitedTextScanAction::kSkip);
  const auto escapedNewline = scanner.consume('n', /*isDelimiter=*/false);
  EXPECT_EQ(escapedNewline.action, DelimitedTextScanAction::kAppend);
  EXPECT_EQ(escapedNewline.value, '\n');

  EXPECT_EQ(
      scanner.consume('\\', /*isDelimiter=*/false).action,
      DelimitedTextScanAction::kSkip);
  const auto finalResult = scanner.finish();
  EXPECT_EQ(finalResult.action, DelimitedTextScanAction::kAppend);
  EXPECT_EQ(finalResult.value, '\\');
}

TEST(DelimitedTextParserTest, unquotedRawScanner) {
  DelimitedTextScanner scanner(
      DelimitedTextOptions{.delimiter = ',', .escape = '\\', .quote = '\0'},
      /*decodeEscapedNewlines=*/true);

  EXPECT_EQ(
      scanner.consumeUnquotedRaw('a', /*isDelimiter=*/false),
      DelimitedTextScanAction::kAppend);
  EXPECT_EQ(
      scanner.consumeUnquotedRaw('\\', /*isDelimiter=*/false),
      DelimitedTextScanAction::kSkip);
  EXPECT_TRUE(scanner.isEscaping());
  EXPECT_EQ(
      scanner.consumeUnquotedRaw(',', /*isDelimiter=*/true),
      DelimitedTextScanAction::kAppend);
  EXPECT_FALSE(scanner.isEscaping());
  EXPECT_EQ(
      scanner.consumeUnquotedRaw(',', /*isDelimiter=*/true),
      DelimitedTextScanAction::kDelimiter);

  DelimitedTextScanner sharedDelimiterAndEscape(
      DelimitedTextOptions{.delimiter = '\\', .escape = '\\', .quote = '\0'},
      /*decodeEscapedNewlines=*/true);
  EXPECT_EQ(
      sharedDelimiterAndEscape.consumeUnquotedRaw('\\', /*isDelimiter=*/true),
      DelimitedTextScanAction::kDelimiter);
}

TEST(DelimitedTextParserTest, quotedRawRun) {
  DelimitedTextScanner scanner(
      DelimitedTextOptions{.delimiter = ',', .escape = '\\', .quote = '"'},
      /*decodeEscapedNewlines=*/false);
  EXPECT_EQ(scanner.consume('"').action, DelimitedTextScanAction::kSkip);
  EXPECT_EQ(scanner.quotedRawRunLength(R"(plain text\"tail)"), 10);
  EXPECT_EQ(scanner.consume('\\').action, DelimitedTextScanAction::kSkip);
  EXPECT_EQ(scanner.quotedRawRunLength("escaped"), 0);
  EXPECT_EQ(scanner.consume('"').action, DelimitedTextScanAction::kAppend);
  EXPECT_EQ(scanner.quotedRawRunLength("tail\""), 4);
}

TEST(DelimitedTextParserTest, quoteStateMachine) {
  DelimitedTextBuffers buffers;
  DelimitedTextParser::splitLine(
      R"("a""b",c)",
      DelimitedTextOptions{.delimiter = ',', .escape = '\\', .quote = '"'},
      0,
      buffers);

  EXPECT_THAT(buffers.fields, testing::ElementsAre(R"("a""b")", "c"));
}

TEST(DelimitedTextParserTest, unescapeField) {
  std::string field = R"(one\nline\,two\\)";
  DelimitedTextParser::unescapeField(
      field, std::optional<char>{'\\'}, /*decodeEscapedNewlines=*/true);
  EXPECT_EQ(field, "one\nline,two\\");
}

TEST(DelimitedTextParserTest, configurationBoundaries) {
  DelimitedTextBuffers buffers;
  const auto quotedOptions =
      DelimitedTextOptions{.delimiter = ',', .escape = '\\', .quote = '"'};
  DelimitedTextParser::splitLine("one,two,three", quotedOptions, 2, buffers);
  EXPECT_THAT(buffers.fields, testing::ElementsAre("one", "two"));

  DelimitedTextParser::splitLine("one,", quotedOptions, 2, buffers);
  EXPECT_THAT(buffers.fields, testing::ElementsAre("one", ""));

  const auto literalOptions = DelimitedTextOptions{
      .delimiter = ',', .escape = std::nullopt, .quote = '\0'};
  DelimitedTextParser::splitLine(
      R"("one,two",three\four)", literalOptions, 0, buffers);
  EXPECT_THAT(
      buffers.fields,
      testing::ElementsAre(R"("one)", R"(two")", R"(three\four)"));

  std::string field = R"(one\nline)";
  DelimitedTextParser::unescapeField(
      field, std::nullopt, /*decodeEscapedNewlines=*/true);
  EXPECT_EQ(field, R"(one\nline)");
}

TEST(DelimitedTextParserTest, nullField) {
  EXPECT_TRUE(DelimitedTextParser::isNullField("", ""));
  EXPECT_TRUE(DelimitedTextParser::isNullField("\\N", "\\N"));
  EXPECT_FALSE(DelimitedTextParser::isNullField("null", "\\N"));
}

} // namespace
} // namespace facebook::velox::text
