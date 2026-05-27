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
#include <functional>
#include <optional>

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/lib/Re2Functions.h"
#include "velox/functions/sparksql/RegexFunctions.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql {

using namespace facebook::velox::test;
namespace {

class RegexFunctionsTest : public test::SparkFunctionBaseTest {
 public:
  void SetUp() override {
    SparkFunctionBaseTest::SetUp();
    // For parsing literal integers as INTEGER, not BIGINT,
    // required by regexp_replace because its position argument
    // is INTEGER.
    options_.parseIntegerAsBigint = false;
  }

  std::optional<bool> rlike(
      std::optional<std::string> str,
      std::string pattern) {
    return evaluateOnce<bool>(fmt::format("rlike(c0, '{}')", pattern), str);
  }

  std::optional<std::string> regexp_extract(
      std::optional<std::string> str,
      std::string pattern) {
    return evaluateOnce<std::string>(
        fmt::format("regexp_extract(c0, '{}')", pattern), str);
  }

  std::string testRegexpReplace(
      const std::optional<std::string>& input,
      const std::string& pattern,
      const std::string& replace,
      std::optional<int32_t> position = std::nullopt) {
    auto result = [&] {
      if (!position) {
        return evaluateOnce<std::string>(
            fmt::format("regexp_replace(c0, '{}', '{}')", pattern, replace),
            input);
      } else {
        return evaluateOnce<std::string>(
            fmt::format(
                "regexp_replace(c0, '{}', '{}', {})",
                pattern,
                replace,
                position.value()),
            input);
      }
    }();
    return result.has_value() ? result.value() : "";
  }

  std::shared_ptr<facebook::velox::SimpleVector<facebook::velox::StringView>>
  testingRegexpReplaceRows(
      const std::vector<std::string>& input,
      const std::vector<std::string>& pattern,
      const std::vector<std::string>& replace,
      const std::optional<std::vector<int32_t>>& position = std::nullopt,
      int repeatCount = 1) {
    EXPECT_GT(repeatCount, 0);

    // Repeat the inputs to allow for testing very large dataframes.
    std::vector<std::string> repeatedInput = repeatVector(input, repeatCount);
    std::vector<std::string> repeatedPattern =
        repeatVector(pattern, repeatCount);
    std::vector<std::string> repeatedReplace =
        repeatVector(replace, repeatCount);
    std::vector<int32_t> repeatedPosition = position.has_value()
        ? repeatVector(*position, repeatCount)
        : std::vector<int32_t>();

    auto inputStringVector = makeFlatVector<std::string>(repeatedInput);
    auto patternStringVector = makeFlatVector<std::string>(repeatedPattern);
    auto replaceStringVector = makeFlatVector<std::string>(repeatedReplace);
    auto positionIntVector = makeFlatVector<int32_t>(repeatedPosition);

    std::shared_ptr<SimpleVector<StringView>> result;
    if (position) {
      result = evaluate<SimpleVector<StringView>>(
          "regexp_replace(c0, c1, c2, c3)",
          makeRowVector(
              {inputStringVector,
               patternStringVector,
               replaceStringVector,
               positionIntVector}));
    } else {
      result = evaluate<SimpleVector<StringView>>(
          "regexp_replace(c0, c1, c2)",
          makeRowVector(
              {inputStringVector, patternStringVector, replaceStringVector}));
    }
    return result;
  }

  std::shared_ptr<facebook::velox::SimpleVector<facebook::velox::StringView>>
  testingRegexpReplaceConstantPattern(
      const std::vector<std::string>& input,
      const std::string& pattern,
      const std::vector<std::string>& replace,
      const std::optional<std::vector<int32_t>>& position = std::nullopt) {
    auto inputStringVector = makeFlatVector<std::string>(input);
    auto replaceStringVector = makeFlatVector<std::string>(replace);

    std::shared_ptr<SimpleVector<StringView>> result;
    if (position) {
      auto positionIntVector = makeFlatVector<int32_t>(*position);

      result = evaluate<SimpleVector<StringView>>(
          fmt::format("regexp_replace(c0, '{}', c1, c2)", pattern),
          makeRowVector(
              {inputStringVector, replaceStringVector, positionIntVector}));
    } else {
      result = evaluate<SimpleVector<StringView>>(
          fmt::format("regexp_replace(c0, '{}', c1)", pattern),
          makeRowVector({inputStringVector, replaceStringVector}));
    }
    return result;
  }

  template <typename T>
  std::vector<T> repeatVector(const std::vector<T>& vec, int repeatCount) {
    std::vector<T> result(vec.size() * repeatCount);
    for (int i = 0; i < repeatCount; ++i) {
      std::copy(vec.begin(), vec.end(), result.begin() + i * vec.size());
    }
    return result;
  }

  std::shared_ptr<facebook::velox::FlatVector<facebook::velox::StringView>>
  convertOutput(const std::vector<std::string>& output, size_t repeatCount) {
    std::vector<std::optional<facebook::velox::StringView>> repeatedOutput(
        output.size() * repeatCount);

    using StringView = facebook::velox::StringView;

    for (size_t i = 0; i < repeatCount; ++i) {
      for (size_t j = 0; j < output.size(); ++j) {
        repeatedOutput[i * output.size() + j] = !output[j].empty()
            ? std::optional<StringView>(output[j])
            : std::nullopt;
      }
    }

    return makeNullableFlatVector(repeatedOutput);
  }
};

// A list of known incompatibilities with java.util.regex. Most result in an
// error being thrown; some unsupported character class features result in
// different results.
TEST_F(RegexFunctionsTest, javaRegexIncompatibilities) {
  // Unsupported character classes.
  EXPECT_THROW(rlike(" ", "\\h"), VeloxUserError);
  EXPECT_THROW(rlike(" ", "\\H"), VeloxUserError);
  EXPECT_THROW(rlike(" ", "\\V"), VeloxUserError);
  EXPECT_THROW(rlike(" ", "\\uffff"), VeloxUserError);
  EXPECT_THROW(rlike(" ", "\\e"), VeloxUserError);
  EXPECT_THROW(rlike(" ", "\\c1"), VeloxUserError);
  EXPECT_THROW(rlike(" ", "\\G"), VeloxUserError);
  EXPECT_THROW(rlike(" ", "\\Z"), VeloxUserError);
  EXPECT_THROW(rlike(" ", "\\R"), VeloxUserError);
  // Backreferences not supported.
  EXPECT_THROW(rlike("00", R"((\d)\1)"), VeloxUserError);
  // Possessive quantifiers not supported.
  EXPECT_THROW(rlike(" ", " ?+"), VeloxUserError);
  EXPECT_THROW(rlike(" ", " *+"), VeloxUserError);
  EXPECT_THROW(rlike(" ", " ++"), VeloxUserError);
  EXPECT_THROW(rlike(" ", " {1}+"), VeloxUserError);
  // Possessive quantifiers not supported.
  EXPECT_THROW(rlike(" ", " ?+"), VeloxUserError);
  EXPECT_THROW(rlike(" ", " *+"), VeloxUserError);
  EXPECT_THROW(rlike(" ", " ++"), VeloxUserError);
  EXPECT_THROW(rlike(" ", " {1}+"), VeloxUserError);
  // Lookahead.
  EXPECT_THROW(rlike(" ", "(?= )"), VeloxUserError);
  EXPECT_THROW(rlike(" ", "(?! )"), VeloxUserError);
  EXPECT_THROW(rlike(" ", "(?<= )"), VeloxUserError);
  EXPECT_THROW(rlike(" ", "(?<! )"), VeloxUserError);
  EXPECT_THROW(rlike(" ", "(?<! )"), VeloxUserError);
}

TEST_F(RegexFunctionsTest, allowSimpleConstantRegex) {
  // rlike returns std::optional<bool>; EXPECT_TRUE would check for non-null,
  // not check the result.
  EXPECT_EQ(rlike("a", "a*"), true);
  EXPECT_EQ(rlike("b", "a*"), true);
  EXPECT_EQ(rlike("b", "a+"), false);
  EXPECT_EQ(rlike("a", "^[ab]*$"), true);
  EXPECT_EQ(rlike(std::nullopt, "a*"), std::nullopt);
}

TEST_F(RegexFunctionsTest, blockUnsupportedEdgeCases) {
  // Non-constant pattern.
  EXPECT_THROW(
      evaluateOnce<bool>("rlike('a', c0)", std::optional<std::string>("a*")),
      VeloxUserError);
}

TEST_F(RegexFunctionsTest, regexMatchRegistration) {
  EXPECT_THROW(
      evaluateOnce<std::string>(
          "regexp_extract('a', c0)", std::optional<std::string>("a*")),
      VeloxUserError);
  EXPECT_EQ(regexp_extract("abc", "a."), "ab");
}

TEST_F(RegexFunctionsTest, regexpReplaceRegistration) {
  std::string output = "teeheebc";
  auto result = testRegexpReplace("abc", "a", "teehee");
  EXPECT_EQ(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceEmptyString) {
  std::string output = "";
  auto result = testRegexpReplace("", "empty string", "nothing");
  EXPECT_EQ(result, output);
  output = "abc";
  result = testRegexpReplace("", "", "abc");
  EXPECT_EQ(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceSimple) {
  std::string output = "HeLLo WorLd";
  auto result = testRegexpReplace("Hello World", "l", "L");
  EXPECT_EQ(result, output);
}
TEST_F(RegexFunctionsTest, badUTF8) {
  std::string badUTF = "\xF0\x82\x82\xAC";
  std::string badHalf = "\xF0\x82";
  std::string overlongA = "\xC1\x81"; // 'A' should not be encoded like this.
  std::string invalidCodePoint =
      "\xED\xA0\x80"; // Start of the surrogate pair range in UTF-8
  std::string incompleteSequence = "\xC3"; // Missing the continuation byte.
  std::string illegalContinuation =
      "\x80"; // This is a continuation byte without a start.
  std::string unexpectedContinuation =
      "\xE0\x80\x80\x80"; // Too many continuation bytes.

  VELOX_ASSERT_THROW(
      testingRegexpReplaceRows({badUTF}, {badHalf}, {"Bad"}), "invalid UTF-8");
  VELOX_ASSERT_THROW(
      testingRegexpReplaceRows({overlongA}, {badHalf}, {"Bad"}),
      "invalid UTF-8");
  VELOX_ASSERT_THROW(
      testingRegexpReplaceRows({invalidCodePoint}, {badHalf}, {"Bad"}),
      "invalid UTF-8");
  VELOX_ASSERT_THROW(
      testingRegexpReplaceRows({incompleteSequence}, {badHalf}, {"Bad"}),
      "invalid UTF-8");
  VELOX_ASSERT_THROW(
      testingRegexpReplaceRows({illegalContinuation}, {badHalf}, {"Bad"}),
      "invalid UTF-8");
  VELOX_ASSERT_THROW(
      testingRegexpReplaceRows({unexpectedContinuation}, {badHalf}, {"Bad"}),
      "invalid UTF-8");
}

TEST_F(RegexFunctionsTest, regexpReplaceSimplePosition) {
  std::string output = "Hello WorLd";
  auto result = testRegexpReplace("Hello World", "l", "L", {6});
  EXPECT_EQ(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceNonAscii) {
  std::string output = "♫ Resume is updated!♫ ";
  auto result = testRegexpReplace("♫ Resume is updated¡♫ ", "¡", "!");
  EXPECT_EQ(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceNonAsciiPositionNoChange) {
  std::string output = "♫ Resume is updated¡♫ Some padding";
  auto result =
      testRegexpReplace("♫ Resume is updated¡♫ Some padding", "¡", "!", {21});
  EXPECT_EQ(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceNonAsciiPositionWithChange) {
  std::string output = "♫ Resume is updated!♫ Some padding";
  auto result =
      testRegexpReplace("♫ Resume is updated¡♫ Some padding", "¡", "!", {20});
  EXPECT_EQ(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceMatchSparkSqlTestSimple) {
  std::vector<int32_t> positions = {1, 1, 1};
  const std::vector<std::string> outputVector = {"300", "400", "400-400"};
  auto result = testingRegexpReplaceRows(
      {"100-200", "100-200", "100-200"},
      {"(\\d+)-(\\d+)", "(\\d+)-(\\d+)", "(\\d+)"},
      {"300", "400", "400"},
      positions);
  auto output = convertOutput({"300", "400", "400-400"}, 1);
  assertEqualVectors(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceWithEmptyString) {
  std::string output = "bc";
  auto result = testRegexpReplace("abc", "a", "");
  EXPECT_EQ(result, output);
}

// Verifies the handling of backslash escapes in regexp_replace's replacement
// argument. Three families are covered in a single table:
//   1. Two-byte literal '\\X' (e.g. '\\n', '\\b', '\\f', '\\r', '\\t') is
//      preserved verbatim, so a matched control byte is rewritten to a
//      literal backslash plus letter. Inputs include a multibyte UTF-8
//      character to confirm surrounding bytes do not affect the result.
//   2. Replacement of the form '\\X' where X is neither a digit nor '\\'
//      drops the leading backslash and produces literal X, matching
//      java.util.regex semantics (e.g. '\\{' -> '{', '\\$' -> '$').
//
// Patterns use the explicit hex form '\\x08' for backspace because RE2
// interprets '\\b' in patterns as a word boundary anchor; the other letter
// escapes ('\\f', '\\r', '\\t') match the corresponding control bytes
// directly.
TEST_F(RegexFunctionsTest, regexpReplaceBackslashEscapes) {
  auto result = testingRegexpReplaceRows(
      {"A\nB",
       "A\nB\nC",
       "\xE5\x9B\xBD\nA",
       "A\bB",
       "A\fB",
       "A\rB",
       "A\tB",
       "[{}]",
       "a$b"},
      {"\\n", "\\n", "\\n", "\\x08", "\\f", "\\r", "\\t", "\\[\\{", "\\$"},
      {"\\\\n",
       "\\\\n",
       "\\\\n",
       "\\\\b",
       "\\\\f",
       "\\\\r",
       "\\\\t",
       "\\{",
       "#"});
  auto expected = convertOutput(
      {"A\\nB",
       "A\\nB\\nC",
       "\xE5\x9B\xBD\\nA",
       "A\\bB",
       "A\\fB",
       "A\\rB",
       "A\\tB",
       "{}]",
       "a#b"},
      1);
  assertEqualVectors(result, expected);
}

TEST_F(RegexFunctionsTest, regexpReplacePosition) {
  std::string output1 = "abc";
  std::string output2 = "bc";
  std::string output3 = "aaaaa";
  auto result1 = testRegexpReplace("abca", "a", "", {2});
  auto result2 = testRegexpReplace("abca", "a", "", {1});
  auto result3 = testRegexpReplace("abca", "bc", "aaa", {1});
  EXPECT_EQ(result1, output1);
  EXPECT_EQ(result2, output2);
  EXPECT_EQ(result3, output3);
}

TEST_F(RegexFunctionsTest, regexpReplaceNegativePosition) {
  VELOX_ASSERT_THROW(
      testRegexpReplace("abc", "a", "", {-1}),
      "regexp_replace requires a position >= 1");
}

TEST_F(RegexFunctionsTest, regexpReplaceZeroPosition) {
  VELOX_ASSERT_THROW(
      testRegexpReplace("abc", "a", "", {0}),
      "regexp_replace requires a position >= 1");
}

TEST_F(RegexFunctionsTest, regexpReplacePositionTooLarge) {
  std::string output = "abc";
  auto result1 = testRegexpReplace("abc", "a", "", {1000});
  auto result2 = testRegexpReplace("abc", "a", "", {4});
  EXPECT_EQ(result1, output);
  EXPECT_EQ(result2, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceSpecialCharacters) {
  std::string output = "abca";
  auto result = testRegexpReplace("a.b.c.a", "\\.", "");
  EXPECT_EQ(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceNoReplacement) {
  std::string output = "abcde";
  auto result = testRegexpReplace("abcde", "f", "z");
  EXPECT_EQ(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceMultipleMatches) {
  std::string output = "bb";
  auto result = testRegexpReplace("aa", "a", "b");
  EXPECT_EQ(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceWord) {
  std::string output = "I like cake";
  auto result = testRegexpReplace("I like pie", "pie", "cake");
}

TEST_F(RegexFunctionsTest, regexpReplaceEscapedCharacters) {
  std::string output = "abcde";
  auto result = testRegexpReplace("abc\\de", "\\\\", "");
  EXPECT_EQ(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplacePatternBeforePosition) {
  std::string output = "abcdef";
  auto result = testRegexpReplace("abcdef", "d", "z", {5});
  EXPECT_EQ(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceConstantPattern) {
  std::vector<int32_t> positions = {1, 2};
  const std::vector<std::string> outputVector = {
      "the sky was blue", "coding isn't fun"};
  auto result = testingRegexpReplaceConstantPattern(
      {"the sky is blue", "coding is fun"}, "is", {"was", "isn't"}, positions);
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceDataframe) {
  // Basic Replacement
  std::vector<int32_t> positions = {1, 2};
  const std::vector<std::string> outputVector = {"hi world", "coding was fun"};
  auto result = testingRegexpReplaceRows(
      {"hello world", "coding is fun"},
      {"hello", " is"},
      {"hi", " was"},
      positions);
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}
TEST_F(RegexFunctionsTest, regexpReplaceDataframeMultiple) {
  // Multiple matches
  std::vector<int32_t> positions = {1, 1};
  const std::vector<std::string> outputVector = {
      "fruit fruit fruit", "fruit fruit fruit"};
  auto result = testingRegexpReplaceRows(
      {"apple apple apple", "banana banana banana"},
      {"apple", "banana"},
      {"fruit", "fruit"},
      positions);
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}
TEST_F(RegexFunctionsTest, regexpReplaceDataframeSpecial) {
  // Special characters
  std::vector<int32_t> positions = {1, 1};
  const std::vector<std::string> outputVector = {"a-b-c", "coding"};
  auto result = testingRegexpReplaceRows(
      {"a.b.c", "[coding]"}, {R"(\.)", R"(\[|\])"}, {"-", ""}, positions);
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}
TEST_F(RegexFunctionsTest, regexpReplaceDataframeSizes) {
  // Replacement with different sizes
  std::vector<int32_t> positions = {1, 1};
  const std::vector<std::string> outputVector = {"fantastic day", "short"};
  auto result = testingRegexpReplaceRows(
      {"good day", "shorter"},
      {"good", "shorter"},
      {"fantastic", "short"},
      positions);
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}
TEST_F(RegexFunctionsTest, regexpReplaceDataframeNoMatches) {
  // No matches
  std::vector<int32_t> positions = {1, 1};
  const std::vector<std::string> outputVector = {"apple", "banana"};
  auto result = testingRegexpReplaceRows(
      {"apple", "banana"}, {"orange", "grape"}, {"fruit", "fruit"}, positions);
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}
TEST_F(RegexFunctionsTest, regexpReplaceDataframeOffsetPosition) {
  // Offset position
  std::vector<int32_t> positions = {9, 6};
  const std::vector<std::string> outputVector = {
      "apple pie fruit", "grape fruit grape"};
  auto result = testingRegexpReplaceRows(
      {"apple pie apple", "grape banana grape"},
      {"apple", "banana"},
      {"fruit", "fruit"},
      positions);
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}
TEST_F(RegexFunctionsTest, regexpReplaceEmptyStringsAndPatterns) {
  // Empty strings and patterns
  std::vector<int32_t> positions = {1, 1};
  const std::vector<std::string> outputVector = {"prefix ", "he world"};
  auto result = testingRegexpReplaceRows(
      {"", "hello"}, {"", "llo"}, {"prefix ", " world"}, positions);
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}
TEST_F(RegexFunctionsTest, regexpReplaceDataframeCharacterTypes) {
  // Multiple character types
  std::vector<int32_t> positions = {1, 1};
  const std::vector<std::string> outputVector = {"XXXABC", "YYY"};
  auto result = testingRegexpReplaceRows(
      {"123ABC", "!@#"}, {R"(\d)", R"(\W)"}, {"X", "Y"}, positions);
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}
TEST_F(RegexFunctionsTest, regexpReplaceDataframeBadPosition) {
  // Larger offsets than string size
  std::vector<int32_t> positions = {10, 15};
  const std::vector<std::string> outputVector = {"apple", "banana"};
  auto result = testingRegexpReplaceRows(
      {"apple", "banana"}, {"apple", "banana"}, {"fruit", "fruit"}, positions);
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceDataframeLastCharacter) {
  std::vector<int32_t> positions = {5, 6};
  const std::vector<std::string> outputVector = {"apple", "banana"};
  auto result = testingRegexpReplaceRows(
      {"applez", "bananaz"}, {"z", "z"}, {"", ""}, positions);
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}

// Test to match
// https://github.com/apache/spark/blob/master/sql/core/src/test/scala/org/apache/spark/sql/StringFunctionsSuite.scala#L180-L184
// This test is the crux of why regexp_replace needed to support non-constant
// parameters. Used position {0,0} out of convenience, ideally we create another
// function that does not pass a position parameter.
TEST_F(RegexFunctionsTest, regexpReplaceMatchSparkSqlTest) {
  std::vector<int32_t> positions = {1, 1, 1};
  const std::vector<std::string> outputVector = {"300", "400", "400-400"};
  auto result = testingRegexpReplaceRows(
      {"100-200", "100-200", "100-200"},
      {"(\\d+)-(\\d+)", "(\\d+)-(\\d+)", "(\\d+)"},
      {"300", "400", "400"},
      positions);
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}
TEST_F(RegexFunctionsTest, regexpReplaceRowsNoPosition) {
  const std::vector<std::string> outputVector = {"300", "400", "400-400"};
  auto result = testingRegexpReplaceRows(
      {"100-200", "100-200", "100-200"},
      {"(\\d+)-(\\d+)", "(\\d+)-(\\d+)", "(\\d+)"},
      {"300", "400", "400"});
  auto output = convertOutput(outputVector, 1);
  assertEqualVectors(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceMassiveVectors) {
  std::vector<int32_t> positions = {1, 1, 1};
  const std::vector<std::string> outputVector = {"300", "400", "400-400"};
  auto result = testingRegexpReplaceRows(
      {"100-200", "100-200", "100-200"},
      {"(\\d+)-(\\d+)", "(\\d+)-(\\d+)", "(\\d+)"},
      {"300", "400", "400"},
      positions,
      100000);
  auto output = convertOutput(outputVector, 100000);
  assertEqualVectors(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplaceCacheLimitTest) {
  std::vector<std::string> patterns;
  std::vector<std::string> strings;
  std::vector<std::string> replaces;
  std::vector<std::string> expectedOutputs;
  const core::QueryConfig config({});

  for (int i = 0; i <= config.exprMaxCompiledRegexes(); ++i) {
    patterns.push_back("\\d" + std::to_string(i) + "-\\d" + std::to_string(i));
    strings.push_back("1" + std::to_string(i) + "-2" + std::to_string(i));
    replaces.push_back("X" + std::to_string(i) + "-Y" + std::to_string(i));
    expectedOutputs.push_back(
        "X" + std::to_string(i) + "-Y" + std::to_string(i));
  }

  VELOX_ASSERT_THROW(
      testingRegexpReplaceRows(strings, patterns, replaces),
      "Max number of regex reached");
}

TEST_F(RegexFunctionsTest, regexpReplaceCacheMissLimit) {
  std::vector<std::string> patterns;
  std::vector<std::string> strings;
  std::vector<std::string> replaces;
  std::vector<std::string> expectedOutputs;
  std::vector<int32_t> positions;
  const core::QueryConfig config({});

  for (int i = 0; i <= config.exprMaxCompiledRegexes() - 1; ++i) {
    patterns.push_back("\\d" + std::to_string(i) + "-\\d" + std::to_string(i));
    strings.push_back("1" + std::to_string(i) + "-2" + std::to_string(i));
    replaces.push_back("X" + std::to_string(i) + "-Y" + std::to_string(i));
    expectedOutputs.push_back(
        "X" + std::to_string(i) + "-Y" + std::to_string(i));
    positions.push_back(1);
  }

  auto result =
      testingRegexpReplaceRows(strings, patterns, replaces, positions, 3);
  auto output = convertOutput(expectedOutputs, 3);
  assertEqualVectors(result, output);
}

TEST_F(RegexFunctionsTest, regexpReplacePreprocess) {
  EXPECT_EQ(
      testRegexpReplace("bdztlszhxz_44", "(.*)(_)([0-9]+$)", "$1$2"),
      "bdztlszhxz_");
  EXPECT_EQ(
      testRegexpReplace("1a 2b 14m", "(\\d+)([ab]) ", "3c$2 "), "3ca 3cb 14m");
  EXPECT_EQ(
      testRegexpReplace("1a 2b 14m", "(\\d+)([ab])", "3c$2"), "3ca 3cb 14m");
  EXPECT_EQ(testRegexpReplace("abc", "(?P<alpha>\\w)", "1${alpha}"), "1a1b1c");
  EXPECT_EQ(
      testRegexpReplace("1a1b1c", "(?<digit>\\d)(?<alpha>\\w)", "${alpha}\\$"),
      "a$b$c$");
  EXPECT_EQ(
      testRegexpReplace(
          "1a2b3c", "(?<digit>\\d)(?<alpha>\\w)", "${alpha}${digit}"),
      "a1b2c3");
  EXPECT_EQ(testRegexpReplace("123", "(\\d)", "\\$"), "$$$");
  EXPECT_EQ(
      testRegexpReplace("123", "(?<digit>(?<nest>\\d))", ".${digit}"),
      ".1.2.3");
  EXPECT_EQ(
      testRegexpReplace("123", "(?<digit>(?<nest>\\d))", ".${nest}"), ".1.2.3");
  EXPECT_EQ(testRegexpReplace("[{}]", "\\[\\{", "\\{"), "{}]");
  EXPECT_EQ(testRegexpReplace("[{}]", "\\}\\]", "\\}"), "[{}");
}

TEST_F(RegexFunctionsTest, regexpInstrBasic) {
  auto regexpInstr = [&](std::optional<std::string> str, std::string pattern) {
    return evaluateOnce<int32_t>(
        fmt::format("regexp_instr(c0, '{}')", pattern), str);
  };

  // Basic match at start.
  EXPECT_EQ(regexpInstr("hello world", "world"), 7);
  // Match at position 1.
  EXPECT_EQ(regexpInstr("hello", "h"), 1);
  // No match.
  EXPECT_EQ(regexpInstr("hello", "xyz"), 0);
  // Regex pattern.
  EXPECT_EQ(regexpInstr("abc123def", "[0-9]+"), 4);
  // Empty string input.
  EXPECT_EQ(regexpInstr("", "a"), 0);
  // Match with empty pattern (matches at start).
  EXPECT_EQ(regexpInstr("hello", ""), 1);
  // Null input.
  EXPECT_EQ(regexpInstr(std::nullopt, "a"), std::nullopt);
  // Verify FIRST match is returned when multiple matches exist.
  EXPECT_EQ(regexpInstr("abcabc", "abc"), 1);
  EXPECT_EQ(regexpInstr("xxabcxxabc", "abc"), 3);
  // Pattern with capturing groups — still returns position of whole match.
  EXPECT_EQ(regexpInstr("foo123bar", "(\\d+)"), 4);
  EXPECT_EQ(regexpInstr("hello world", "(wor)(ld)"), 7);
}

TEST_F(RegexFunctionsTest, regexpInstrUnicode) {
  auto regexpInstr = [&](std::optional<std::string> str, std::string pattern) {
    return evaluateOnce<int32_t>(
        fmt::format("regexp_instr(c0, '{}')", pattern), str);
  };

  // Unicode: each character is multi-byte but position should be
  // character-based.
  // \xC3\xA9 = é, \xC3\xA8 = è, \xC3\xA0 = à (each 2 bytes, 1 char).
  EXPECT_EQ(regexpInstr("\xC3\xA9\xC3\xA8\xC3\xA0", "\xC3\xA0"), 3);
  // 4-byte UTF-8 characters (emoji): 🌍🌎🌏 each is 4 bytes.
  EXPECT_EQ(
      regexpInstr(
          "\xF0\x9F\x8C\x8D\xF0\x9F\x8C\x8E\xF0\x9F\x8C\x8F",
          "\xF0\x9F\x8C\x8F"),
      3);
  // Mixed ASCII and multi-byte: "aéb" — match 'b' at char pos 3.
  EXPECT_EQ(
      regexpInstr(
          "a\xC3\xA9"
          "b",
          "b"),
      3);
  // 3-byte UTF-8 characters: Chinese chars.
  EXPECT_EQ(
      regexpInstr(
          "\xE4\xB8\xAD\xE6\x96\x87\xE6\xB5\x8B\xE8\xAF\x95", "\xE6\xB5\x8B"),
      3);
  // Mixed multi-byte with regex metacharacters.
  EXPECT_EQ(regexpInstr("\xC3\xA9\xC3\xA8.test", "\\."), 3);
  // Single character match at the very start (ASCII fast path).
  EXPECT_EQ(regexpInstr("ascii", "a"), 1);
}

TEST_F(RegexFunctionsTest, regexpInstrDynamicPattern) {
  // Test with non-constant (dynamic) pattern using two column inputs.
  auto strings =
      makeFlatVector<StringView>({"hello world", "abc123", "test", "foo bar"});
  auto patterns = makeFlatVector<StringView>({"world", "[0-9]+", "xyz", "bar"});
  auto result =
      evaluate("regexp_instr(c0, c1)", makeRowVector({strings, patterns}));
  auto expected = makeFlatVector<int32_t>({7, 4, 0, 5});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(RegexFunctionsTest, regexpInstrWithIdx) {
  // Spark's regexp_instr accepts idx but silently ignores it — always returns
  // position of entire match regardless of idx value.
  auto regexpInstrIdx =
      [&](std::optional<std::string> str, std::string pattern, int32_t idx) {
        return evaluateOnce<int32_t>(
            fmt::format("regexp_instr(c0, '{}', {})", pattern, idx), str);
      };

  // idx=0 — entire match position.
  EXPECT_EQ(regexpInstrIdx("hello world", "world", 0), 7);
  EXPECT_EQ(regexpInstrIdx("abc123", "[0-9]+", 0), 4);
  // idx>0 — Spark ignores idx, still returns position of entire match.
  EXPECT_EQ(regexpInstrIdx("foo123bar", "(\\d+)", 1), 4);
  EXPECT_EQ(regexpInstrIdx("hello world", "(wor)(ld)", 2), 7);
}

TEST_F(RegexFunctionsTest, regexpInstrNonConstantIdx) {
  // Non-constant idx column — Spark ignores idx values, always returns
  // position of entire match.
  auto strings = makeFlatVector<StringView>({"hello", "world", "abc123"});
  auto patterns = makeFlatVector<StringView>({"h", "or", "[0-9]+"});
  auto indices = makeFlatVector<int32_t>({0, 1, 2});
  auto result = evaluate(
      "regexp_instr(c0, c1, c2)", makeRowVector({strings, patterns, indices}));
  auto expected = makeFlatVector<int32_t>({1, 2, 4});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(RegexFunctionsTest, regexpInstrInvalidPattern) {
  auto regexpInstr = [&](std::optional<std::string> str, std::string pattern) {
    return evaluateOnce<int32_t>(
        fmt::format("regexp_instr(c0, '{}')", pattern), str);
  };

  // Invalid regex pattern should throw (constant pattern path).
  VELOX_ASSERT_THROW(
      regexpInstr("hello", "[invalid"), "Invalid regular expression");

  // Invalid regex pattern with dynamic (non-constant) pattern column.
  auto strings = makeFlatVector<StringView>({"hello"});
  auto patterns = makeFlatVector<StringView>({"[invalid"});
  VELOX_ASSERT_THROW(
      evaluate("regexp_instr(c0, c1)", makeRowVector({strings, patterns})),
      "invalid regular expression");
}

TEST_F(RegexFunctionsTest, regexpInstrMultipleMatches) {
  auto regexpInstr = [&](std::optional<std::string> str, std::string pattern) {
    return evaluateOnce<int32_t>(
        fmt::format("regexp_instr(c0, '{}')", pattern), str);
  };

  // Always returns position of FIRST match.
  EXPECT_EQ(regexpInstr("aaa", "a"), 1);
  EXPECT_EQ(regexpInstr("baa", "a"), 2);
  EXPECT_EQ(regexpInstr("abab", "ab"), 1);
  EXPECT_EQ(regexpInstr("xyzabcxyzabc", "abc"), 4);
  // Overlapping potential matches — first one wins.
  EXPECT_EQ(regexpInstr("aaaa", "aa"), 1);
}

} // namespace
} // namespace facebook::velox::functions::sparksql
