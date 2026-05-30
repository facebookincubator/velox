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
//
// Cases ported from pcre4j's `MatcherReplacementTests.java`.  Same
// provenance notes as PatternPortedTest.cpp.
//

#include "velox/external/regex_compat/tests/BackendTestBase.h"
#include "velox/external/regex_compat/tests/JavaMatcherAdapter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::regex_compat::test {
namespace {

template <typename R>
using ReplacementPortedTest = BackendTest<R>;
TYPED_TEST_SUITE(ReplacementPortedTest, AllBackends);

// replaceAll: literal replacement, no group refs.
TYPED_TEST(ReplacementPortedTest, replaceAllLiteral) {
  TypeParam re("o");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string s = "foo bar";
  int n = TypeParam::GlobalReplace(&s, re, "0");
  EXPECT_EQ(2, n);
  EXPECT_EQ("f00 bar", s);
}

// replaceAll: numbered group refs ($1).
TYPED_TEST(ReplacementPortedTest, replaceAllNumberedGroup) {
  TypeParam re("(\\d+)");
  std::string s = "abc 42 xyz 7";
  int n = TypeParam::GlobalReplace(&s, re, "<$1>");
  EXPECT_EQ(2, n);
  EXPECT_EQ("abc <42> xyz <7>", s);
}

// replaceAll: numbered group refs $0 (whole match).
TYPED_TEST(ReplacementPortedTest, replaceAllZeroGroupRef) {
  TypeParam re("\\d+");
  std::string s = "a 1 b 2";
  int n = TypeParam::GlobalReplace(&s, re, "[$0]");
  EXPECT_EQ(2, n);
  EXPECT_EQ("a [1] b [2]", s);
}

// replaceAll: named group ${name}.
TYPED_TEST(ReplacementPortedTest, replaceAllNamedGroup) {
  TypeParam re("(?<digit>\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string s = "a 1 b 22";
  int n = TypeParam::GlobalReplace(&s, re, "[${digit}]");
  EXPECT_EQ(2, n);
  EXPECT_EQ("a [1] b [22]", s);
}

// replaceAll: dollar-sign literally via backslash escape.
TYPED_TEST(ReplacementPortedTest, replaceAllEscapedDollar) {
  TypeParam re("x");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string s = "x x";
  int n = TypeParam::GlobalReplace(&s, re, "\\$");
  EXPECT_EQ(2, n);
  EXPECT_EQ("$ $", s);
}

// replaceAll: backslash literally via double-backslash.
TYPED_TEST(ReplacementPortedTest, replaceAllEscapedBackslash) {
  TypeParam re("x");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string s = "x";
  // In C++ string literal, "\\\\" is the two-char string `\\` which Java sees
  // as escaped backslash → single literal '\'.
  int n = TypeParam::GlobalReplace(&s, re, "\\\\");
  EXPECT_EQ(1, n);
  EXPECT_EQ("\\", s);
}

// replaceAll: zero-match (pattern doesn't match) leaves input unchanged.
TYPED_TEST(ReplacementPortedTest, replaceAllNoMatchKeepsInput) {
  TypeParam re("z+");
  std::string s = "hello";
  int n = TypeParam::GlobalReplace(&s, re, "X");
  EXPECT_EQ(0, n);
  EXPECT_EQ("hello", s);
}

// replaceAll across multiple groups in replacement.
TYPED_TEST(ReplacementPortedTest, replaceAllMultiGroupCombination) {
  TypeParam re("(\\w+) (\\w+)");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string s = "hello world";
  int n = TypeParam::GlobalReplace(&s, re, "$2 $1");
  EXPECT_EQ(1, n);
  EXPECT_EQ("world hello", s);
}

// replaceFirst: only the first match is replaced (via Adapter).
TYPED_TEST(ReplacementPortedTest, replaceFirstOnlyFirst) {
  TypeParam re("\\d+");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "a 1 b 2 c 3");
  std::string out = m.replaceFirst("X");
  EXPECT_EQ("a X b 2 c 3", out);
}

// replaceFirst with group reference.
TYPED_TEST(ReplacementPortedTest, replaceFirstWithGroupRef) {
  TypeParam re("(\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "a 1 b 22 c 333");
  std::string out = m.replaceFirst("[$1]");
  EXPECT_EQ("a [1] b 22 c 333", out);
}

// Empty pattern replacement (pcre4j edge case).
TYPED_TEST(ReplacementPortedTest, replaceAllEmptyReplacement) {
  TypeParam re("\\d+");
  std::string s = "a 1 b 22";
  int n = TypeParam::GlobalReplace(&s, re, "");
  EXPECT_EQ(2, n);
  EXPECT_EQ("a  b ", s);
}

} // namespace
} // namespace facebook::velox::regex_compat::test
