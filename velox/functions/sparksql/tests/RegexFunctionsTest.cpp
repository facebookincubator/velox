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

  void testingRegexReplaceAllConstant(
      const std::vector<std::optional<std::string>>& input,
      std::string pattern,
      std::string replace,
      const std::vector<std::optional<std::vector<std::string>>>& output,
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
        return evaluate<ArrayVector>(
            fmt::format("regex_replace(c0, '{}', '{}')", pattern, replace),
            rowVector);
      } else {
        return evaluate<ArrayVector>(
            fmt::format(
                "regex_replace(c0, '{}', '{}', {})",
                pattern,
                replace,
                *position),
            rowVector);
      }
    }();

    // Creating vectors for output string vectors
    auto sizeAtOutput = [&output](vector_size_t row) {
      return output[row] ? output[row]->size() : 0;
    };
    auto valueAtOutput = [&output](vector_size_t row, vector_size_t idx) {
      return output[row] ? StringView(output[row]->at(idx)) : StringView("");
    };
    auto nullAtOutput = [&output](vector_size_t row) {
      return !output[row].has_value();
    };

    auto expectedResult = makeArrayVector<StringView>(
        output.size(), sizeAtOutput, valueAtOutput, nullAtOutput);

    assertEqualVectors(expectedResult, result);
  }

  void testingRegexReplaceConstantPattern(
      const std::vector<std::optional<std::string>>& input,
      std::string pattern,
      const std::vector<std::optional<std::string>>& replace,
      const std::vector<std::optional<std::vector<std::string>>>& output,
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

    std::shared_ptr<ArrayVector> result;
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

      result = evaluate<ArrayVector>(
          fmt::format("regex_replace(c0, '{}', c1, c2)", pattern),
          makeRowVector({inputString, replaceString, positionInt}));
    } else {
      result = evaluate<ArrayVector>(
          fmt::format("regex_replace(c0, '{}', c1)", pattern),
          makeRowVector({inputString, replaceString}));
    }

    // Creating vectors for output string vectors
    auto sizeAtOutput = [&output](vector_size_t row) {
      return output[row] ? output[row]->size() : 0;
    };

    auto valueAtOutput = [&output](vector_size_t row, vector_size_t idx) {
      return output[row] ? StringView(output[row]->at(idx)) : StringView("");
    };

    auto nullAtOutput = [&output](vector_size_t row) {
      return !output[row].has_value();
    };

    auto expectedResult = makeArrayVector<StringView>(
        output.size(), sizeAtOutput, valueAtOutput, nullAtOutput);

    assertEqualVectors(expectedResult, result);
  };

  void testingRegexReplaceRows(
      const std::vector<std::optional<std::string>>& input,
      const std::vector<std::optional<std::string>>& pattern,
      const std::vector<std::optional<std::string>>& replace,
      const std::vector<std::optional<std::vector<std::string>>>& output,
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
    std::vector<std::optional<std::vector<std::string>>> repeatedOutput(
        output.size() * repeatCount);
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
      std::copy(
          output.begin(),
          output.end(),
          repeatedOutput.begin() + i * output.size());
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

    std::shared_ptr<ArrayVector> result;
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

      result = evaluate<ArrayVector>(
          "regex_replace(c0, c1, c2, c3)",
          makeRowVector(
              {inputString, patternString, replaceString, positionInt}));
    } else {
      result = evaluate<ArrayVector>(
          "regex_replace(c0, c1, c2)",
          makeRowVector({inputString, patternString, replaceString}));
    }

    // Creating vectors for output string vectors
    auto sizeAtOutput = [&repeatedOutput](vector_size_t row) {
      return repeatedOutput[row] ? repeatedOutput[row]->size() : 0;
    };

    auto valueAtOutput = [&repeatedOutput](
                             vector_size_t row, vector_size_t idx) {
      return repeatedOutput[row] ? StringView(repeatedOutput[row]->at(idx))
                                 : StringView("");
    };

    auto nullAtOutput = [&repeatedOutput](vector_size_t row) {
      return !repeatedOutput[row].has_value();
    };

    auto expectedResult = makeArrayVector<StringView>(
        repeatedOutput.size(), sizeAtOutput, valueAtOutput, nullAtOutput);

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
  testingRegexReplaceAllConstant({"abc"}, "a", "teehee", {{{"teeheebc"}}});
}

TEST_F(RegexFunctionsTest, RegexReplaceEmptyString) {
  testingRegexReplaceAllConstant({"abc"}, "a", "", {{{"bc"}}});
}

TEST_F(RegexFunctionsTest, RegexBadJavaPattern) {
  EXPECT_THROW(
      testingRegexReplaceAllConstant({"[]"}, "[a[b]]", "", {{{""}}}),
      VeloxUserError);
  EXPECT_THROW(
      testingRegexReplaceAllConstant({"[]"}, "[a&&[b]]", "", {{{""}}}),
      VeloxUserError);
  EXPECT_THROW(
      testingRegexReplaceAllConstant({"[]"}, "[a&&[^b]]", "", {{{""}}}),
      VeloxUserError);
}

TEST_F(RegexFunctionsTest, RegexReplacePosition) {
  testingRegexReplaceAllConstant({"abca"}, "a", "", {{{"abc"}}}, {2});
  testingRegexReplaceAllConstant({"abca"}, "a", "", {{{"bc"}}}, {1});
  testingRegexReplaceAllConstant({"abca"}, "bc", "aaa", {{{"aaaaa"}}}, {1});
}

TEST_F(RegexFunctionsTest, RegexReplaceNegativePosition) {
  EXPECT_THROW(
      testingRegexReplaceAllConstant({"abc"}, "a", "", {{{"abc"}}}, {-1}),
      VeloxRuntimeError);
}

TEST_F(RegexFunctionsTest, RegexReplaceZeroPosition) {
  EXPECT_THROW(
      testingRegexReplaceAllConstant({"abc"}, "a", "", {{{"abc"}}}, {0}),
      VeloxRuntimeError);
}

TEST_F(RegexFunctionsTest, RegexReplacePositionTooLarge) {
  testingRegexReplaceAllConstant({"abc"}, "a", "", {{{"abc"}}}, {1000});
  testingRegexReplaceAllConstant({"abc"}, "a", "", {{{"abc"}}}, {4});
}

TEST_F(RegexFunctionsTest, RegexReplaceSpecialCharacters) {
  testingRegexReplaceAllConstant({"a.b.c.a"}, "\\.", "", {{{"abca"}}});
}

TEST_F(RegexFunctionsTest, RegexReplaceNoReplacement) {
  testingRegexReplaceAllConstant({"abcde"}, "f", "z", {{{"abcde"}}});
}

TEST_F(RegexFunctionsTest, RegexReplaceMultipleMatches) {
  testingRegexReplaceAllConstant({"aa"}, "a", "b", {{{"bb"}}});
}

TEST_F(RegexFunctionsTest, RegexReplaceWord) {
  testingRegexReplaceAllConstant(
      {"I like pie"}, "pie", "cake", {{{"I like cake"}}});
}

TEST_F(RegexFunctionsTest, RegexReplaceEscapedCharacters) {
  testingRegexReplaceAllConstant({"abc\\de"}, "\\\\", "", {{{"abcde"}}});
}

TEST_F(RegexFunctionsTest, RegexReplacePatternBeforePosition) {
  testingRegexReplaceAllConstant({"abcdef"}, "d", "z", {{{"abcdef"}}}, 5);
}

TEST_F(RegexFunctionsTest, RegexReplaceConstantPattern) {
  std::vector<int64_t> positions = {1, 2};
  testingRegexReplaceConstantPattern(
      {"the sky is blue", "coding is fun"},
      "is",
      {"was", "isn't"},
      {{std::vector<std::string>{"the sky was blue"}},
       {std::vector<std::string>{"coding isn't fun"}}},
      positions);
}

TEST_F(RegexFunctionsTest, RegexReplaceDataframe) {
  // Basic Replacement
  std::vector<int64_t> positions = {1, 2};
  testingRegexReplaceRows(
      {"hello world", "coding is fun"},
      {"hello", " is"},
      {"hi", " was"},
      {{std::vector<std::string>{"hi world"}},
       {std::vector<std::string>{"coding was fun"}}},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeMultiple) {
  // Multiple matches
  std::vector<int64_t> positions = {1, 1};
  testingRegexReplaceRows(
      {"apple apple apple", "banana banana banana"},
      {"apple", "banana"},
      {"fruit", "fruit"},
      {{std::vector<std::string>{"fruit fruit fruit"}},
       {std::vector<std::string>{"fruit fruit fruit"}}},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeSpecial) {
  // Special characters
  std::vector<int64_t> positions = {1, 1};
  testingRegexReplaceRows(
      {"a.b.c", "[coding]"},
      {R"(\.)", R"(\[|\])"},
      {"-", ""},
      {{std::vector<std::string>{"a-b-c"}},
       {std::vector<std::string>{"coding"}}},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeSizes) {
  // Replacement with different sizes
  std::vector<int64_t> positions = {1, 1};
  testingRegexReplaceRows(
      {"good day", "shorter"},
      {"good", "shorter"},
      {"fantastic", "short"},
      {{std::vector<std::string>{"fantastic day"}},
       {std::vector<std::string>{"short"}}},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeNoMatches) {
  // No matches
  std::vector<int64_t> positions = {1, 1};
  testingRegexReplaceRows(
      {"apple", "banana"},
      {"orange", "grape"},
      {"fruit", "fruit"},
      {{std::vector<std::string>{"apple"}},
       {std::vector<std::string>{"banana"}}},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeOffsetPosition) {
  // Offset position
  std::vector<int64_t> positions = {9, 6};
  testingRegexReplaceRows(
      {"apple pie apple", "grape banana grape"},
      {"apple", "banana"},
      {"fruit", "fruit"},
      {{std::vector<std::string>{"apple pie fruit"}},
       {std::vector<std::string>{"grape fruit grape"}}},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceEmptyStringsAndPatterns) {
  // Empty strings and patterns
  std::vector<int64_t> positions = {1, 1};
  testingRegexReplaceRows(
      {"", "hello"},
      {"", "llo"},
      {"prefix ", " world"},
      {{std::vector<std::string>{"prefix "}},
       {std::vector<std::string>{"he world"}}},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeCharacterTypes) {
  // Multiple character types
  std::vector<int64_t> positions = {1, 1};
  testingRegexReplaceRows(
      {"123ABC", "!@#"},
      {R"(\d)", R"(\W)"},
      {"X", "Y"},
      {{std::vector<std::string>{"XXXABC"}}, {std::vector<std::string>{"YYY"}}},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceDataframeBadPosition) {
  // Larger offsets than string size
  std::vector<int64_t> positions = {10, 15};
  testingRegexReplaceRows(
      {"apple", "banana"},
      {"apple", "banana"},
      {"fruit", "fruit"},
      {{std::vector<std::string>{"apple"}},
       {std::vector<std::string>{"banana"}}},
      positions);
}

TEST_F(RegexFunctionsTest, RegexReplaceDataframeLastCharacter) {
  std::vector<int64_t> positions = {5, 6};
  testingRegexReplaceRows(
      {"applez", "bananaz"},
      {"z", "z"},
      {"", ""},
      {{std::vector<std::string>{"apple"}},
       {std::vector<std::string>{"banana"}}},
      positions);
}

// Test to match
// https://github.com/apache/spark/blob/master/sql/core/src/test/scala/org/apache/spark/sql/StringFunctionsSuite.scala#L180-L184
// This test is the crux of why regex_replace needed to support non-constant
// parameters. Used position {0,0} out of convenience, ideally we create another
// function that does not pass a position parameter.
TEST_F(RegexFunctionsTest, RegexReplaceMatchSparkSqlTest) {
  std::vector<int64_t> positions = {1, 1, 1};
  testingRegexReplaceRows(
      {"100-200", "100-200", "100-200"},
      {"(\\d+)-(\\d+)", "(\\d+)-(\\d+)", "(\\d+)"},
      {"300", "400", "400"},
      {{std::vector<std::string>{"300"}},
       {std::vector<std::string>{"400"}},
       {std::vector<std::string>{"400-400"}}},
      positions);
}
TEST_F(RegexFunctionsTest, RegexReplaceRowsNoPosition) {
  testingRegexReplaceRows(
      {"100-200", "100-200", "100-200"},
      {"(\\d+)-(\\d+)", "(\\d+)-(\\d+)", "(\\d+)"},
      {"300", "400", "400"},
      {{std::vector<std::string>{"300"}},
       {std::vector<std::string>{"400"}},
       {std::vector<std::string>{"400-400"}}});
}

TEST_F(RegexFunctionsTest, RegexReplaceMassiveVectors) {
  std::vector<int64_t> positions = {1, 1, 1};
  testingRegexReplaceRows(
      {"100-200", "100-200", "100-200"},
      {"(\\d+)-(\\d+)", "(\\d+)-(\\d+)", "(\\d+)"},
      {"300", "400", "400"},
      {{std::vector<std::string>{"300"}},
       {std::vector<std::string>{"400"}},
       {std::vector<std::string>{"400-400"}}},
      positions,
      100000);
}

} // namespace
} // namespace facebook::velox::functions::sparksql
