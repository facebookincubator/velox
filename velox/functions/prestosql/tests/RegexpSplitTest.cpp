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
#include <exception>
#include <optional>

#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox {
namespace {

class RegexSplitTest : public functions::test::FunctionBaseTest {
 protected:
  auto regexp_split(
    const std::optional<StringView>& string,
    const std::string& pattern) {
    return evaluate(
      fmt::format("regexp_split(c0, '{}')", pattern),
      makeRowVector({makeNullableFlatVector<StringView>({string})}));
  }
};

TEST_F(RegexSplitTest, RegexpSplit) {
  test::assertEqualVectors(
      makeNullableArrayVector<std::string>({{"h","w","f","r"}}),
      regexp_split("h w f r", " "));
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"ab"}}), regexp_split("ab", "c"));
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"ab", "d"}}),
      regexp_split("abcd", "c"));
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"", "ab", "d", ""}}),
      regexp_split("cabcdc", "c"));
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"", "", "", ""}}),
      regexp_split("ccc", "c"));

  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"a", "b", "c", "d"}}),
      regexp_split("a.b:c;d", "[\\.:;]"));
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"a", "b:c;d"}}),
      regexp_split("a.b:c;d", "\\."));
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"a.b", "c;d"}}),
      regexp_split("a.b:c;d", ":"));

  // Character classes
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"", "", "", ""}}),
      regexp_split("abc", "\\w"));
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"a", "", "b", "", "c", ""}}),
      regexp_split("a12b34c5", "\\d"));
  test::assertEqualVectors(
      makeNullableArrayVector<std::string>({{"h","w","f","r"}}),
      regexp_split("h1w23f45r", "\\d+"));

  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{""}}), regexp_split("", "\\d"));
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({std::nullopt}),
      regexp_split(std::nullopt, "abc"));

  // Capture groups
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"", "", "14m"}}),
      regexp_split("1a 2b 14m", "(\\d+)([ab]) "));
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"", " ", " 14m"}}),
      regexp_split("1a 2b 14m", "(\\d+)([ab])"));

  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"", "", "", ""}}),
      regexp_split("abc", "(?P<alpha>\\w)"));
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"", "", "", ""}}),
      regexp_split("1a2b3c", "(?<digit>\\d)(?<alpha>\\w)"));
  test::assertEqualVectors(
      makeNullableArrayVector<StringView>({{"", "", "", ""}}),
      regexp_split("123", "(?<digit>(?<nest>\\d))"));

  EXPECT_THROW(regexp_split("123", "(?<d"), VeloxUserError);
  EXPECT_THROW(regexp_split("123", R"((?''digit''\d))"), VeloxUserError);
  EXPECT_THROW(regexp_split("123", "(?P<>\\d)"), VeloxUserError);
}

} // namespace
} // namespace facebook::velox
