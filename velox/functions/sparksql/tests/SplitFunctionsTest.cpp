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

#include <optional>
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {

using namespace facebook::velox::test;
namespace {

class SplitTest : public SparkFunctionBaseTest {
 protected:
  void testSplit(
      const std::vector<std::optional<std::string>>& input,
      std::optional<std::string> pattern,
      const std::vector<std::optional<std::vector<std::string>>>& output,
      std::optional<int32_t> limit = std::nullopt);

  void testSplitEncodings(
      const std::vector<VectorPtr>& inputs,
      const std::vector<std::optional<std::vector<std::string>>>& output);

  ArrayVectorPtr toArrayVector(
      const std::vector<std::optional<std::vector<std::string>>>& vector);
};

ArrayVectorPtr SplitTest::toArrayVector(
    const std::vector<std::optional<std::vector<std::string>>>& vector) {
  // Creating vectors for output string vectors
  auto sizeAt = [&vector](vector_size_t row) {
    return vector[row] ? vector[row]->size() : 0;
  };
  auto valueAt = [&vector](vector_size_t row, vector_size_t idx) {
    return vector[row] ? StringView(vector[row]->at(idx)) : StringView("");
  };
  auto nullAt = [&vector](vector_size_t row) {
    return !vector[row].has_value();
  };
  return makeArrayVector<StringView>(vector.size(), sizeAt, valueAt, nullAt);
}

void SplitTest::testSplit(
    const std::vector<std::optional<std::string>>& input,
    std::optional<std::string> pattern,
    const std::vector<std::optional<std::vector<std::string>>>& output,
    std::optional<int32_t> limit) {
  auto valueAt = [&input](vector_size_t row) {
    return input[row] ? StringView(*input[row]) : StringView();
  };

  // Creating vectors for input strings
  auto nullAt = [&input](vector_size_t row) { return !input[row].has_value(); };

  auto result = [&] {
    auto inputString =
        makeFlatVector<StringView>(input.size(), valueAt, nullAt);
    auto rowVector = makeRowVector({inputString});

    // Evaluating the function for each input and seed
    std::string patternString = pattern.has_value()
        ? std::string(", '") + pattern.value() + "'"
        : ", ''";
    const std::string limitString = limit.has_value()
        ? ", '" + std::to_string(limit.value()) + "'::INTEGER"
        : "";
    std::string expressionString =
        std::string("split(c0") + patternString + limitString + ")";
    return evaluate<ArrayVector>(expressionString, rowVector);
  }();

  const auto expectedResult = toArrayVector(output);

  // Checking the results
  assertEqualVectors(expectedResult, result);
}

void SplitTest::testSplitEncodings(
    const std::vector<VectorPtr>& inputs,
    const std::vector<std::optional<std::vector<std::string>>>& output) {
  const auto expected = toArrayVector(output);
  std::vector<core::TypedExprPtr> inputExprs = {
      std::make_shared<core::FieldAccessTypedExpr>(inputs[0]->type(), "c0"),
      std::make_shared<core::FieldAccessTypedExpr>(inputs[1]->type(), "c1")};
  if (inputs.size() > 2) {
    inputExprs.emplace_back(
        std::make_shared<core::FieldAccessTypedExpr>(inputs[2]->type(), "c2"));
  }
  const auto expr = std::make_shared<const core::CallTypedExpr>(
      expected->type(), std::move(inputExprs), "split");
  testEncodings(expr, inputs, expected);
}

TEST_F(SplitTest, reallocationAndCornerCases) {
  testSplit(
      {"boo:and:foo", "abcfd", "abcfd:", "", ":ab::cfd::::"},
      ":",
      {{{"boo", "and", "foo"}},
       {{"abcfd"}},
       {{"abcfd", ""}},
       {{""}},
       {{"", "ab", "", "cfd", "", "", "", ""}}});
}

TEST_F(SplitTest, nulls) {
  testSplit(
      {std::nullopt, "abcfd", "abcfd:", std::nullopt, ":ab::cfd::::"},
      ":",
      {{std::nullopt},
       {{"abcfd"}},
       {{"abcfd", ""}},
       {{std::nullopt}},
       {{"", "ab", "", "cfd", "", "", "", ""}}});
}

TEST_F(SplitTest, defaultArguments) {
  testSplit(
      {"boo:and:foo", "abcfd"}, ":", {{{"boo", "and", "foo"}}, {{"abcfd"}}});
}

TEST_F(SplitTest, longStrings) {
  testSplit(
      {"abcdefghijklkmnopqrstuvwxyz"},
      ",",
      {{{"abcdefghijklkmnopqrstuvwxyz"}}});
}

TEST_F(SplitTest, zeroLengthPattern) {
  testSplit(
      {"abcdefg", "abc:+%/n?(^)", ""},
      std::nullopt,
      {{{"a", "b", "c", "d", "e", "f", "g"}},
       {{"a", "b", "c", ":", "+", "%", "/", "n", "?", "(", "^", ")"}},
       {{""}}});
  testSplit(
      {"abcdefg", "ab:c+%/n?(^)", ""},
      std::nullopt,
      {{{"a", "b", "c"}}, {{"a", "b", ":"}}, {{""}}},
      3);
  testSplit(
      {"abcdefg", "abc:+%/n?(^)", ""},
      std::nullopt,
      {{{"a", "b", "c", "d", "e", "f", "g"}},
       {{"a", "b", "c", ":", "+", "%", "/", "n", "?", "(", "^", ")"}},
       {{""}}},
      20);
}

TEST_F(SplitTest, encodings) {
  auto strings = makeFlatVector<StringView>(
      {"abcdef",
       "oneAtwoBthreeC",
       "aa2bb3cc",
       "hello",
       "aacbbcddc",
       "morning",
       "",
       "",
       "today",
       "tomorrow"});
  auto patterns = makeFlatVector<StringView>(
      {"",
       "[ABC]",
       "[1-9]+",
       "e.*o",
       "c",
       "(mo)|ni",
       ":",
       "",
       ".",
       "tomorrow"});
  auto limits = makeFlatVector<int32_t>({0, -1, -1, 0, -1, -2, -1, -4, -1, -1});
  std::vector<std::optional<std::vector<std::string>>> expected = {
      {{"a", "b", "c", "d", "e", "f"}},
      {{"one", "two", "three", ""}},
      {{"aa", "bb", "cc"}},
      {{"h", ""}},
      {{"aa", "bb", "dd", ""}},
      {{"", "r", "ng"}},
      {{""}},
      {{""}},
      {{
          "",
          "",
          "",
          "",
          "",
          "",
      }},
      {{"", ""}}};
  testSplitEncodings({strings, patterns, limits}, expected);
  testSplitEncodings({strings, patterns}, expected);

  limits = makeFlatVector<int32_t>({3, 3, 2, 1, 5, 2, 1, 1, 2, 2});
  expected = {
      {{"a", "b", "c"}},
      {{"one", "two", "threeC"}},
      {{"aa", "bb3cc"}},
      {{"hello"}},
      {{"aa", "bb", "dd", ""}},
      {{"", "rning"}},
      {{""}},
      {{""}},
      {{"", "oday"}},
      {{"", ""}}};
  testSplitEncodings({strings, patterns, limits}, expected);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
