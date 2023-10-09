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

#include "velox/functions/lib/Re2Functions.h"
#include "velox/functions/sparksql/RegexFunctions.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql {

using namespace facebook::velox::test;
namespace {

class RegexFunctionsTest : public test::SparkFunctionBaseTest {
 public:
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

  void testRegexReplaceSimpleOnce(
      const std::vector<std::optional<std::string>>& input,
      std::string pattern,
      std::string replace,
      std::string output,
      std::optional<int> position =
          std::nullopt /*adding the optional position parameter*/) {
    auto valueAt = [&input](vector_size_t row) {
      return input[row] ? StringView(*input[row]) : StringView();
    };
    auto nullAt = [&input](vector_size_t row) {
      return !input[row].has_value();
    };
    auto result = [&] {
      auto inputString =
          makeFlatVector<StringView>(input.size(), valueAt, nullAt);
      auto rowVector = makeRowVector({inputString});

      if (!position) {
        return evaluateOnce<std::string>(
            fmt::format("regex_replace(c0, '{}', '{}')", pattern, replace),
            rowVector);
      } else {
        return evaluateOnce<std::string>(
            fmt::format(
                "regex_replace(c0, '{}', '{}', {})",
                pattern,
                replace,
                *position),
            rowVector);
      }
    }();

    EXPECT_EQ(result, output);
  }

  void testingRegexReplaceRowsSimple(
      const std::vector<std::optional<std::string>>& input,
      const std::vector<std::optional<std::string>>& pattern,
      const std::vector<std::optional<std::string>>& replace,
      const std::vector<std::optional<std::string>>& output,
      const std::optional<std::vector<int64_t>>& positionOpt = std::nullopt,
      int repeatCount = 1) {
    if (repeatCount < 1) {
      return;
    }

    // Repeat the inputs
    std::vector<std::optional<std::string>> repeatedInput(
        input.size() * repeatCount);
    std::vector<std::optional<std::string>> repeatedPattern(
        pattern.size() * repeatCount);
    std::vector<std::optional<std::string>> repeatedReplace(
        replace.size() * repeatCount);
    std::vector<std::optional<StringView>> repeatedOutput(
        output.size() * repeatCount); // Modified to StringView
    std::optional<std::vector<int64_t>> repeatedPosition;

    if (positionOpt.has_value()) {
      repeatedPosition.emplace(positionOpt->size() * repeatCount);
    }

    for (int i = 0; i < repeatCount; ++i) {
      std::copy(
          input.begin(), input.end(), repeatedInput.begin() + i * input.size());
      std::copy(
          pattern.begin(),
          pattern.end(),
          repeatedPattern.begin() + i * pattern.size());
      std::copy(
          replace.begin(),
          replace.end(),
          repeatedReplace.begin() + i * replace.size());

      // Convert to StringView while copying for repeatedOutput
      for (size_t j = 0; j < output.size(); ++j) {
        repeatedOutput[i * output.size() + j] =
            output[j] ? StringView(*output[j]) : StringView();
      }

      if (positionOpt.has_value()) {
        std::copy(
            positionOpt->begin(),
            positionOpt->end(),
            repeatedPosition->begin() + i * positionOpt->size());
      }
    }

    auto valueAtInput = [&repeatedInput](vector_size_t row) -> StringView {
      return repeatedInput[row] ? StringView(*repeatedInput[row])
                                : StringView();
    };

    auto valueAtPattern = [&repeatedPattern](vector_size_t row) -> StringView {
      return repeatedPattern[row] ? StringView(*repeatedPattern[row])
                                  : StringView();
    };

    auto valueAtReplace = [&repeatedReplace](vector_size_t row) -> StringView {
      return repeatedReplace[row] ? StringView(*repeatedReplace[row])
                                  : StringView();
    };

    auto nullAtInput = [&repeatedInput](vector_size_t row) -> bool {
      return !repeatedInput[row].has_value();
    };

    auto nullAtPattern = [&repeatedPattern](vector_size_t row) -> bool {
      return !repeatedPattern[row].has_value();
    };

    auto nullAtReplace = [&repeatedReplace](vector_size_t row) -> bool {
      return !repeatedReplace[row].has_value();
    };

    auto inputString = makeFlatVector<StringView>(
        repeatedInput.size(), valueAtInput, nullAtInput);
    auto patternString = makeFlatVector<StringView>(
        repeatedPattern.size(), valueAtPattern, nullAtPattern);
    auto replaceString = makeFlatVector<StringView>(
        repeatedReplace.size(), valueAtReplace, nullAtReplace);

    std::shared_ptr<SimpleVector<StringView>>
        result; // Modified return type to SimpleVector<StringView>
    if (positionOpt) {
      auto position = *repeatedPosition;

      auto valueAtPosition = [&position](vector_size_t row) -> int64_t {
        return position[row];
      };

      auto nullAtPosition = [&position](vector_size_t row) -> bool {
        return false;
      };

      auto positionInt = makeFlatVector<int64_t>(
          position.size(), valueAtPosition, nullAtPosition);

      result = evaluate<SimpleVector<StringView>>(
          "regex_replace(c0, c1, c2, c3)",
          makeRowVector(
              {inputString, patternString, replaceString, positionInt}));
    } else {
      result = evaluate<SimpleVector<StringView>>(
          "regex_replace(c0, c1, c2)",
          makeRowVector({inputString, patternString, replaceString}));
    }

    auto expectedResult = makeNullableFlatVector<StringView>(repeatedOutput);

    assertEqualVectors(expectedResult, result);
  }

  void testingRegexReplaceSimpleConstantPattern(
      const std::vector<std::optional<std::string>>& input,
      std::string pattern,
      const std::vector<std::optional<std::string>>& replace,
      const std::vector<std::optional<std::string>>& output,
      const std::optional<std::vector<int64_t>>& positionOpt = std::nullopt) {
    auto valueAtInput = [&input](vector_size_t row) -> StringView {
      return input[row] ? StringView(*input[row]) : StringView();
    };

    auto valueAtReplace = [&replace](vector_size_t row) -> StringView {
      return replace[row] ? StringView(*replace[row]) : StringView();
    };

    auto nullAtInput = [&input](vector_size_t row) -> bool {
      return !input[row].has_value();
    };

    auto nullAtReplace = [&replace](vector_size_t row) -> bool {
      return !replace[row].has_value();
    };

    auto inputString =
        makeFlatVector<StringView>(input.size(), valueAtInput, nullAtInput);
    auto replaceString = makeFlatVector<StringView>(
        replace.size(), valueAtReplace, nullAtReplace);

    std::shared_ptr<SimpleVector<StringView>> result;
    if (positionOpt) {
      auto position = *positionOpt;

      auto valueAtPosition = [&position](vector_size_t row) -> int64_t {
        return position[row];
      };

      auto nullAtPosition = [&position](vector_size_t row) -> bool {
        return false;
      };

      auto positionInt = makeFlatVector<int64_t>(
          position.size(), valueAtPosition, nullAtPosition);

      result = evaluate<SimpleVector<StringView>>(
          fmt::format("regex_replace(c0, '{}', c1, c2)", pattern),
          makeRowVector({inputString, replaceString, positionInt}));
    } else {
      result = evaluate<SimpleVector<StringView>>(
          fmt::format("regex_replace(c0, '{}', c1)", pattern),
          makeRowVector({inputString, replaceString}));
    }

    std::vector<std::optional<StringView>> convertedOutput(output.size());

    for (size_t i = 0; i < output.size(); ++i) {
      if (output[i].has_value()) {
        convertedOutput[i] = StringView(*output[i]);
      }
    }

    auto expectedResult = makeNullableFlatVector<StringView>(convertedOutput);

    assertEqualVectors(expectedResult, result);
  };
};

// A list of known incompatibilities with java.util.regex. Most result in an
// error being thrown; some unsupported character class features result in
// different results.
TEST_F(RegexFunctionsTest, JavaRegexIncompatibilities) {
  // Character class union is not supported; parsed as [a\[b]\].
  EXPECT_THROW(rlike("[]", R"([a[b]])"), VeloxUserError);
  // Character class intersection not supported; parsed as [a&\[b]\].
  EXPECT_THROW(rlike("&]", R"([a&&[b]])"), VeloxUserError);
  // Character class difference not supported; parsed as [\w&\[\^b]\].
  EXPECT_THROW(rlike("^]", R"([\w&&[^b]])"), VeloxUserError);
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

TEST_F(RegexFunctionsTest, AllowSimpleConstantRegex) {
  // rlike returns std::optional<bool>; EXPECT_TRUE would check for non-null,
  // not check the result.
  EXPECT_EQ(rlike("a", "a*"), true);
  EXPECT_EQ(rlike("b", "a*"), true);
  EXPECT_EQ(rlike("b", "a+"), false);
  EXPECT_EQ(rlike("a", "^[ab]*$"), true);
  EXPECT_EQ(rlike(std::nullopt, "a*"), std::nullopt);
}

TEST_F(RegexFunctionsTest, BlockUnsupportedEdgeCases) {
  // Non-constant pattern.
  EXPECT_THROW(
      evaluateOnce<bool>("rlike('a', c0)", std::optional<std::string>("a*")),
      VeloxUserError);
  // Unsupported set union syntax.
  EXPECT_THROW(rlike("", "[a[b]]"), VeloxUserError);
}

TEST_F(RegexFunctionsTest, RegexMatchRegistration) {
  EXPECT_THROW(
      evaluateOnce<std::string>(
          "regexp_extract('a', c0)", std::optional<std::string>("a*")),
      VeloxUserError);
  EXPECT_EQ(regexp_extract("abc", "a."), "ab");
  EXPECT_THROW(regexp_extract("[]", "[a[b]]"), VeloxUserError);
}

TEST_F(RegexFunctionsTest, RegexReplaceRegistration) {
  testRegexReplaceSimpleOnce({"abc"}, "a", "teehee", "teeheebc");
}

TEST_F(RegexFunctionsTest, RegexReplaceEmptyString) {
  testRegexReplaceSimpleOnce({""}, "empty string", "nothing", "");
}

TEST_F(RegexFunctionsTest, RegexReplaceSimple) {
  testRegexReplaceSimpleOnce({"Hello World"}, "l", "L", "HeLLo WorLd");
}

TEST_F(RegexFunctionsTest, RegexReplaceSimplePosition) {
  testRegexReplaceSimpleOnce({"Hello World"}, "l", "L", "Hello WorLd", {6});
}

TEST_F(RegexFunctionsTest, RegexReplaceMatchSparkSqlTestSimple) {
  std::vector<int64_t> positions = {1, 1, 1};
  testingRegexReplaceRowsSimple(
      {"100-200", "100-200", "100-200"},
      {"(\\d+)-(\\d+)", "(\\d+)-(\\d+)", "(\\d+)"},
      {"300", "400", "400"},
      {"300", "400", "400-400"},
      positions);
}

TEST_F(RegexFunctionsTest, RegexReplaceWithEmptyString) {
  testRegexReplaceSimpleOnce({"abc"}, "a", "", "bc");
}

TEST_F(RegexFunctionsTest, RegexBadJavaPattern) {
  EXPECT_THROW(
      testRegexReplaceSimpleOnce({"[]"}, "[a[b]]", "", ""), VeloxUserError);
  EXPECT_THROW(
      testRegexReplaceSimpleOnce({"[]"}, "[a&&[b]]", "", ""), VeloxUserError);
  EXPECT_THROW(
      testRegexReplaceSimpleOnce({"[]"}, "[a&&[^b]]", "", ""), VeloxUserError);
}

TEST_F(RegexFunctionsTest, RegexReplacePosition) {
  testRegexReplaceSimpleOnce({"abca"}, "a", "", "abc", {2});
  testRegexReplaceSimpleOnce({"abca"}, "a", "", "bc", {1});
  testRegexReplaceSimpleOnce({"abca"}, "bc", "aaa", "aaaaa", {1});
}

TEST_F(RegexFunctionsTest, RegexReplaceNegativePosition) {
  EXPECT_THROW(
      testRegexReplaceSimpleOnce({"abc"}, "a", "", "abc", {-1}),
      VeloxRuntimeError);
}

TEST_F(RegexFunctionsTest, RegexReplaceZeroPosition) {
  EXPECT_THROW(
      testRegexReplaceSimpleOnce({"abc"}, "a", "", "abc", {0}),
      VeloxRuntimeError);
}

TEST_F(RegexFunctionsTest, RegexReplacePositionTooLarge) {
  testRegexReplaceSimpleOnce({"abc"}, "a", "", "abc", {1000});
  testRegexReplaceSimpleOnce({"abc"}, "a", "", "abc", {4});
}

TEST_F(RegexFunctionsTest, RegexReplaceSpecialCharacters) {
  testRegexReplaceSimpleOnce({"a.b.c.a"}, "\\.", "", "abca");
}

TEST_F(RegexFunctionsTest, RegexReplaceNoReplacement) {
  testRegexReplaceSimpleOnce({"abcde"}, "f", "z", "abcde");
}

TEST_F(RegexFunctionsTest, RegexReplaceMultipleMatches) {
  testRegexReplaceSimpleOnce({"aa"}, "a", "b", "bb");
}

TEST_F(RegexFunctionsTest, RegexReplaceWord) {
  testRegexReplaceSimpleOnce({"I like pie"}, "pie", "cake", "I like cake");
}

TEST_F(RegexFunctionsTest, RegexReplaceEscapedCharacters) {
  testRegexReplaceSimpleOnce({"abc\\de"}, "\\\\", "", "abcde");
}

TEST_F(RegexFunctionsTest, RegexReplacePatternBeforePosition) {
  testRegexReplaceSimpleOnce({"abcdef"}, "d", "z", "abcdef", {5});
}

TEST_F(RegexFunctionsTest, RegexReplaceConstantPattern) {
  std::vector<int64_t> positions = {1, 2};
  testingRegexReplaceSimpleConstantPattern(
      {"the sky is blue", "coding is fun"},
      "is",
      {"was", "isn't"},
      {"the sky was blue", "coding isn't fun"},
      positions);
}

TEST_F(RegexFunctionsTest, RegexReplaceDataframe) {
  // Basic Replacement
  std::vector<int64_t> positions = {1, 2};
  testingRegexReplaceRowsSimple(
      {"hello world", "coding is fun"},
      {"hello", " is"},
      {"hi", " was"},
      {"hi world", "coding was fun"},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeMultiple) {
  // Multiple matches
  std::vector<int64_t> positions = {1, 1};
  testingRegexReplaceRowsSimple(
      {"apple apple apple", "banana banana banana"},
      {"apple", "banana"},
      {"fruit", "fruit"},
      {"fruit fruit fruit", "fruit fruit fruit"},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeSpecial) {
  // Special characters
  std::vector<int64_t> positions = {1, 1};
  testingRegexReplaceRowsSimple(
      {"a.b.c", "[coding]"},
      {R"(\.)", R"(\[|\])"},
      {"-", ""},
      {"a-b-c", "coding"},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeSizes) {
  // Replacement with different sizes
  std::vector<int64_t> positions = {1, 1};
  testingRegexReplaceRowsSimple(
      {"good day", "shorter"},
      {"good", "shorter"},
      {"fantastic", "short"},
      {"fantastic day", "short"},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeNoMatches) {
  // No matches
  std::vector<int64_t> positions = {1, 1};
  testingRegexReplaceRowsSimple(
      {"apple", "banana"},
      {"orange", "grape"},
      {"fruit", "fruit"},
      {"apple", "banana"},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeOffsetPosition) {
  // Offset position
  std::vector<int64_t> positions = {9, 6};
  testingRegexReplaceRowsSimple(
      {"apple pie apple", "grape banana grape"},
      {"apple", "banana"},
      {"fruit", "fruit"},
      {"apple pie fruit", "grape fruit grape"},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceEmptyStringsAndPatterns) {
  // Empty strings and patterns
  std::vector<int64_t> positions = {1, 1};
  testingRegexReplaceRowsSimple(
      {"", "hello"},
      {"", "llo"},
      {"prefix ", " world"},
      {"prefix ", "he world"},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeCharacterTypes) {
  // Multiple character types
  std::vector<int64_t> positions = {1, 1};
  testingRegexReplaceRowsSimple(
      {"123ABC", "!@#"},
      {R"(\d)", R"(\W)"},
      {"X", "Y"},
      {"XXXABC", "YYY"},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeBadPosition) {
  // Larger offsets than string size
  std::vector<int64_t> positions = {10, 15};
  testingRegexReplaceRowsSimple(
      {"apple", "banana"},
      {"apple", "banana"},
      {"fruit", "fruit"},
      {"apple", "banana"},
      positions);
}

TEST_F(RegexFunctionsTest, RegexReplaceDataframeLastCharacter) {
  std::vector<int64_t> positions = {5, 6};
  testingRegexReplaceRowsSimple(
      {"applez", "bananaz"},
      {"z", "z"},
      {"", ""},
      {"apple", "banana"},
      positions);
}

// Test to match
// https://github.com/apache/spark/blob/master/sql/core/src/test/scala/org/apache/spark/sql/StringFunctionsSuite.scala#L180-L184
// This test is the crux of why regex_replace needed to support non-constant
// parameters. Used position {0,0} out of convenience, ideally we create another
// function that does not pass a position parameter.
TEST_F(RegexFunctionsTest, RegexReplaceMatchSparkSqlTest) {
  std::vector<int64_t> positions = {1, 1, 1};
  testingRegexReplaceRowsSimple(
      {"100-200", "100-200", "100-200"},
      {"(\\d+)-(\\d+)", "(\\d+)-(\\d+)", "(\\d+)"},
      {"300", "400", "400"},
      {"300", "400", "400-400"},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceRowsNoPosition) {
  testingRegexReplaceRowsSimple(
      {"100-200", "100-200", "100-200"},
      {"(\\d+)-(\\d+)", "(\\d+)-(\\d+)", "(\\d+)"},
      {"300", "400", "400"},
      {"300", "400", "400-400"});
}

TEST_F(RegexFunctionsTest, RegexReplaceMassiveVectors) {
  std::vector<int64_t> positions = {1, 1, 1};
  testingRegexReplaceRowsSimple(
      {"100-200", "100-200", "100-200"},
      {"(\\d+)-(\\d+)", "(\\d+)-(\\d+)", "(\\d+)"},
      {"300", "400", "400"},
      {"300", "400", "400-400"},
      positions,
      100000);
}

TEST_F(RegexFunctionsTest, RegexReplaceCacheLimitTest) {
  std::vector<std::optional<std::string>> patterns;
  std::vector<std::optional<std::string>> strings;
  std::vector<std::optional<std::string>> replaces;
  std::vector<std::optional<std::string>> expectedOutputs;

  for (int i = 0; i <= kMaxCompiledRegexes; ++i) {
    patterns.push_back("\\d" + std::to_string(i) + "-\\d" + std::to_string(i));
    strings.push_back("1" + std::to_string(i) + "-2" + std::to_string(i));
    replaces.push_back("X" + std::to_string(i) + "-Y" + std::to_string(i));
    expectedOutputs.push_back(
        "X" + std::to_string(i) + "-Y" + std::to_string(i));
  }

  EXPECT_THROW(
      testingRegexReplaceRowsSimple(
          strings, patterns, replaces, expectedOutputs),
      VeloxUserError);
}

TEST_F(RegexFunctionsTest, RegexReplaceCacheMissLimit) {
  std::vector<std::optional<std::string>> patterns;
  std::vector<std::optional<std::string>> strings;
  std::vector<std::optional<std::string>> replaces;
  std::vector<std::optional<std::string>> expectedOutputs;
  std::vector<int64_t> positions;

  for (int i = 0; i <= kMaxCompiledRegexes - 1; ++i) {
    patterns.push_back("\\d" + std::to_string(i) + "-\\d" + std::to_string(i));
    strings.push_back("1" + std::to_string(i) + "-2" + std::to_string(i));
    replaces.push_back("X" + std::to_string(i) + "-Y" + std::to_string(i));
    expectedOutputs.push_back(
        "X" + std::to_string(i) + "-Y" + std::to_string(i));
    positions.push_back(1);
  }

  testingRegexReplaceRowsSimple(
      strings, patterns, replaces, expectedOutputs, positions, 50000);
}

} // namespace
} // namespace facebook::velox::functions::sparksql
