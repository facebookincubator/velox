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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/type/Type.h"

using namespace facebook::velox::test;
namespace facebook::velox::functions::sparksql::test {
namespace {

class SplitTest : public SparkFunctionBaseTest {
 protected:
  void testSplit(
      const std::vector<std::string>& input,
      std::string delim,
      std::optional<int32_t> limit,
      size_t numRows,
      const std::vector<std::vector<std::string>>& expected) {
    auto strings = makeFlatVector(input);
    auto delims = makeFlatVector<StringView>(
        numRows, [&](vector_size_t row) { return StringView{delim}; });
    if (limit.has_value()) {
      auto limits = makeFlatVector<int32_t>(
          numRows, [&](vector_size_t row) { return limit.value(); });
      auto input = makeRowVector({strings, delims, limits});
      auto result = evaluate("split(c0, c1, c2)", input);
      auto expectedResult = makeArrayVector(expected);
      assertEqualVectors(expectedResult, result);
    } else {
      auto input = makeRowVector({strings, delims});
      auto result = evaluate("split(c0, c1)", input);
      auto expectedResult = makeArrayVector(expected);
      assertEqualVectors(expectedResult, result);
    }
  };
};

TEST_F(SplitTest, basic) {
  auto limit = -1;
  auto delim = ",";
  auto numRows = 3;
  auto input = std::vector<std::string>{
      {"I,he,she,they"}, // Simple
      {"one,,,four,"}, // Empty strings
      {""}, // The whole string is empty
  };
  auto expected = std::vector<std::vector<std::string>>({
      {"I", "he", "she", "they"},
      {"one", "", "", "four", ""},
      {""},
  });

  // No limit provided.
  testSplit(input, delim, std::nullopt, numRows, expected);

  // Limit <= 0.
  testSplit(input, delim, limit, numRows, expected);

  // High limit, the limit greater than the input string size.
  limit = 10;
  testSplit(input, delim, limit, numRows, expected);

  // Small limit, the limit is smaller than or equals to input string size.
  limit = 3;
  expected = {
      {"I", "he", "she,they"},
      {"one", "", ",four,"},
      {""},
  };
  testSplit(input, delim, limit, numRows, expected);

  // limit = 1, the resulting array only has one entry to contain all input.
  limit = 1;
  expected = {
      {"I,he,she,they"},
      {"one,,,four,"},
      {""},
  };
  testSplit(input, delim, limit, numRows, expected);

  // Non-ascii delimiter.
  delim = "లేదా";
  input = {
      {"синяя сливаలేదా赤いトマトలేదా黃苹果లేదాbrown pear"},
      {"зелёное небоలేదాలేదాలేదా緑の空లేదా"},
      {""},
  };
  expected = {
      {"синяя слива", "赤いトマト", "黃苹果", "brown pear"},
      {"зелёное небо", "", "", "緑の空", ""},
      {""},
  };
  testSplit(input, delim, std::nullopt, numRows, expected);
  limit = -1;
  testSplit(input, delim, limit, numRows, expected);
  limit = 10;
  testSplit(input, delim, limit, numRows, expected);
  limit = 3;
  expected = {
      {"синяя слива", "赤いトマト", "黃苹果లేదాbrown pear"},
      {"зелёное небо", "", "లేదా緑の空లేదా"},
      {""},
  };
  testSplit(input, delim, limit, numRows, expected);
  limit = 1;
  expected = {
      {"синяя сливаలేదా赤いトマトలేదా黃苹果లేదాbrown pear"},
      {"зелёное небоలేదాలేదాలేదా緑の空లేదా"},
      {""},
  };
  testSplit(input, delim, limit, numRows, expected);

  // Cover case that delimiter not exists in input string.
  delim = "A";
  testSplit(input, delim, std::nullopt, numRows, expected);
}

TEST_F(SplitTest, emptyDelimiter) {
  auto limit = -1;
  auto delim = "";
  auto numRows = 3;
  auto input = std::vector<std::string>{
      {"I,he,she,they"}, // Simple
      {"one,,,four,"}, // Empty strings
      {""}, // The whole string is empty
  };
  auto expected = std::vector<std::vector<std::string>>({
      {"I", ",", "h", "e", ",", "s", "h", "e", ",", "t", "h", "e", "y"},
      {"o", "n", "e", ",", ",", ",", "f", "o", "u", "r", ","},
      {""},
  });

  // No limit provided.
  testSplit(input, delim, std::nullopt, numRows, expected);

  // Limit <= 0.
  testSplit(input, delim, limit, numRows, expected);

  // High limit, the limit greater than the input string size.
  limit = 20;
  testSplit(input, delim, limit, numRows, expected);

  // Small limit, the limit is smaller than or equals to input string size.
  limit = 3;
  expected = {
      {"I", ",", "h"},
      {"o", "n", "e"},
      {""},
  };
  testSplit(input, delim, limit, numRows, expected);

  // limit = 1.
  limit = 1;
  expected = {
      {"I"},
      {"o"},
      {""},
  };
  testSplit(input, delim, limit, numRows, expected);

  // Non-ascii, empty delimiter.
  input = std::vector<std::string>{
      {"синяя赤いトマト緑の"},
      {"Hello世界🙂"},
      {""},
  };
  expected = {
      {"с", "и", "н", "я", "я", "赤", "い", "ト", "マ", "ト", "緑", "の"},
      {"H", "e", "l", "l", "o", "世", "界", "🙂"},
      {""},
  };
  testSplit(input, delim, std::nullopt, numRows, expected);

  expected = {
      {"с", "и"},
      {"H", "e"},
      {""},
  };
  testSplit(input, delim, 2, numRows, expected);
}

TEST_F(SplitTest, regexDelimiter) {
  auto delim = "\\s*[a-z]+\\s*";
  auto input = std::vector<std::string>{
      "1a 2b 14m",
      "1a 2b 14",
      "",
      "a123b",
  };
  auto numRows = 4;
  auto limit = -1;
  auto expected = std::vector<std::vector<std::string>>({
      {"1", "2", "14", ""},
      {"1", "2", "14"},
      {""},
      {"", "123", ""},
  });

  // No limit.
  testSplit(input, delim, std::nullopt, numRows, expected);
  // Limit < 0.
  testSplit(input, delim, limit, numRows, expected);
  // High limit.
  limit = 10;
  testSplit(input, delim, limit, numRows, expected);
  // Small limit.
  limit = 3;
  expected = {
      {"1", "2", "14m"},
      {"1", "2", "14"},
      {""},
      {"", "123", ""},
  };
  testSplit(input, delim, limit, numRows, expected);
  limit = 1;
  expected = {
      {"1a 2b 14m"},
      {"1a 2b 14"},
      {""},
      {"a123b"},
  };
  testSplit(input, delim, limit, numRows, expected);

  delim = "A|";
  numRows = 3;
  input = std::vector<std::string>{
      {"синяя赤いトマト緑の"},
      {"Hello世界🙂"},
      {""},
  };
  expected = std::vector<std::vector<std::string>>({
      {"с", "и", "н", "я", "я", "赤", "い", "ト", "マ", "ト", "緑", "の", ""},
      {"H", "e", "l", "l", "o", "世", "界", "🙂", ""},
      {""},
  });

  testSplit(input, delim, std::nullopt, numRows, expected);

  limit = 2;
  expected = {
      {"с", "иняя赤いトマト緑の"},
      {"H", "ello世界🙂"},
      {""},
  };
  testSplit(input, delim, limit, numRows, expected);
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
