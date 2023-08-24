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

  void regex_replace_constant(
    const std::vector<std::optional<std::string>>& input,
    std::string pattern,
    std::string replace,
    const std::vector<std::optional<std::vector<std::string>>>& output,
    std::optional<int> position = std::nullopt  /*adding the optional position parameter*/ ){
      auto valueAt = [&input](vector_size_t row) {
          return input[row] ? StringView(*input[row]) : StringView();
      };
      auto nullAt = [&input](vector_size_t row) { return !input[row].has_value(); };
      auto result  = [&] {
        auto inputString =
          makeFlatVector<StringView>(input.size(), valueAt, nullAt);
        auto rowVector = makeRowVector({inputString});

        if (!position) {
          return evaluate<ArrayVector>(fmt::format("regex_replace(c0, '{}', '{}')", pattern, replace), rowVector);
        } else {
          return evaluate<ArrayVector>(fmt::format("regex_replace(c0, '{}', '{}', {})", pattern, replace, *position), rowVector);
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
  void regex_replace_rows_position(
    const std::vector<std::optional<std::string>>& input,
    const std::vector<std::optional<std::string>>& pattern,
    const std::vector<std::optional<std::string>>& replace,
    const std::vector<std::optional<std::vector<std::string>>>& output,
    const std::vector<int64_t>& position  /*adding the optional position parameter*/ ){

      auto valueAtInput = [&input](vector_size_t row) -> StringView {
          return input[row] ? StringView(*input[row]) : StringView();
      };

      auto valueAtPattern = [&pattern](vector_size_t row) -> StringView {
          return pattern[row] ? StringView(*pattern[row]) : StringView();
      };

      auto valueAtReplace = [&replace](vector_size_t row) -> StringView {
          return replace[row] ? StringView(*replace[row]) : StringView();
      };

      // auto valueAtPosition = [&position](vector_size_t row) -> std::optional<int> {
      //     return position[row];
      // };

      auto nullAtInput = [&input](vector_size_t row) -> bool {
          return !input[row].has_value();
      };

      auto nullAtPattern = [&pattern](vector_size_t row) -> bool {
          return !pattern[row].has_value();
      };

      auto nullAtReplace = [&replace](vector_size_t row) -> bool {
          return !replace[row].has_value();
      };

      // auto nullAtPosition = [&position](vector_size_t row) -> bool {
      //     return !position[row].has_value();
      // };


      auto inputString = makeFlatVector<StringView>(input.size(), valueAtInput, nullAtInput);
      auto patternString = makeFlatVector<StringView>(pattern.size(), valueAtPattern, nullAtPattern);
      auto replaceString = makeFlatVector<StringView>(replace.size(), valueAtReplace, nullAtReplace);
      auto positionInt = makeFlatVector<int64_t>(position);
      auto result = evaluate<ArrayVector>("regex_replace(c0, c1, c2, c3)", makeRowVector({inputString,
                                                                        patternString, replaceString,
                                                                        positionInt}));

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
  regex_replace_constant({"abc"},"a", "teehee", {{{"teeheebc"}}});
}

TEST_F(RegexFunctionsTest, RegexReplaceTwoParams){
  regex_replace_constant({"abc"}, "a", "", {{{"bc"}}});
}

TEST_F(RegexFunctionsTest, RegexReplacePosition){
  regex_replace_constant({"abca"}, "a", "", {{{"abc"}}}, {2});
}

TEST_F(RegexFunctionsTest, RegexReplaceNegativePosition){
  EXPECT_THROW(regex_replace_constant({"abc"}, "a", "", {{{"abc"}}}, {-1}), VeloxRuntimeError);
}

TEST_F(RegexFunctionsTest, RegexReplaceDataframe){
  regex_replace_rows_position({"abca", "this is the second"},
                               {"a", "is"},
                               {"", "is not"},
                               {{{"abc", "this is not the second"}}},
                               {2, 0});
}

} // namespace
} // namespace facebook::velox::functions::sparksql
