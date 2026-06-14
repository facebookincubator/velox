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

// ============== Newly ported from pcre4j MatcherReplacementTests ==============

// pcre4j quoteReplacement(...)
TYPED_TEST(ReplacementPortedTest, quoteReplacementBasic) {
  EXPECT_EQ("hello", JavaMatcherAdapter<TypeParam>::quoteReplacement("hello"));
}
TYPED_TEST(ReplacementPortedTest, quoteReplacementBackslash) {
  EXPECT_EQ("hello\\\\world",
            JavaMatcherAdapter<TypeParam>::quoteReplacement("hello\\world"));
}
TYPED_TEST(ReplacementPortedTest, quoteReplacementDollar) {
  EXPECT_EQ("price: \\$100",
            JavaMatcherAdapter<TypeParam>::quoteReplacement("price: $100"));
}
TYPED_TEST(ReplacementPortedTest, quoteReplacementBoth) {
  EXPECT_EQ("\\$100 \\\\ \\$200",
            JavaMatcherAdapter<TypeParam>::quoteReplacement("$100 \\ $200"));
}
TYPED_TEST(ReplacementPortedTest, quoteReplacementEmpty) {
  EXPECT_EQ("", JavaMatcherAdapter<TypeParam>::quoteReplacement(""));
}

// pcre4j replaceAllBasic
TYPED_TEST(ReplacementPortedTest, replaceAllBasic) {
  TypeParam re("world");
  std::string s = "hello world";
  TypeParam::GlobalReplace(&s, re, "universe");
  EXPECT_EQ("hello universe", s);
}

// pcre4j replaceAllMultiple
TYPED_TEST(ReplacementPortedTest, replaceAllMultiple) {
  TypeParam re("o");
  std::string s = "hello world";
  int n = TypeParam::GlobalReplace(&s, re, "0");
  EXPECT_EQ(2, n);
  EXPECT_EQ("hell0 w0rld", s);
}

// pcre4j replaceAllWithGroupReference (covered by replaceAllNumberedGroup
// already, but we mirror pcre4j name)
TYPED_TEST(ReplacementPortedTest, replaceAllWithGroupReference) {
  TypeParam re("(\\d+)");
  std::string s = "value: 42";
  TypeParam::GlobalReplace(&s, re, "<$1>");
  EXPECT_EQ("value: <42>", s);
}

// pcre4j replaceAllWithNamedGroupReference
TYPED_TEST(ReplacementPortedTest, replaceAllWithNamedGroupReferenceBasic) {
  TypeParam re("(?<digit>\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string s = "value: 42";
  TypeParam::GlobalReplace(&s, re, "<${digit}>");
  EXPECT_EQ("value: <42>", s);
}

// pcre4j replaceAllUnicode
TYPED_TEST(ReplacementPortedTest, replaceAllUnicode) {
  TypeParam re("\xf0\x9f\x8c\x90"); // U+1F310 globe
  std::string s = "hi \xf0\x9f\x8c\x90 there";
  TypeParam::GlobalReplace(&s, re, "\xf0\x9f\x8c\x8d"); // U+1F30D earth
  EXPECT_EQ("hi \xf0\x9f\x8c\x8d there", s);
}

// pcre4j replaceFirstBasic
TYPED_TEST(ReplacementPortedTest, replaceFirstBasic) {
  TypeParam re("o");
  JavaMatcherAdapter<TypeParam> m(&re, "foo bar");
  EXPECT_EQ("f0o bar", m.replaceFirst("0"));
}

// pcre4j replaceFirstWithGroupReference
TYPED_TEST(ReplacementPortedTest, replaceFirstWithGroupReferenceMulti) {
  TypeParam re("(\\d+)");
  JavaMatcherAdapter<TypeParam> m(&re, "a 1 b 22 c");
  EXPECT_EQ("a <1> b 22 c", m.replaceFirst("<$1>"));
}

// pcre4j replaceFirstNoMatch
TYPED_TEST(ReplacementPortedTest, replaceFirstNoMatch) {
  TypeParam re("xyz");
  JavaMatcherAdapter<TypeParam> m(&re, "hello world");
  EXPECT_EQ("hello world", m.replaceFirst("ZZZ"));
}

// pcre4j replaceAllWithFullMatchReference  ($0)
TYPED_TEST(ReplacementPortedTest, replaceAllWithFullMatchReference) {
  TypeParam re("\\w+");
  std::string s = "hello world";
  TypeParam::GlobalReplace(&s, re, "[$0]");
  EXPECT_EQ("[hello] [world]", s);
}

// pcre4j replaceAllWithNamedGroupReferenceYearMonth
TYPED_TEST(ReplacementPortedTest, replaceAllWithNamedGroupReferenceYearMonth) {
  TypeParam re("(?<year>\\d{4})-(?<month>\\d{2})");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string s = "date: 2024-01, also 2025-12";
  TypeParam::GlobalReplace(&s, re, "${month}/${year}");
  EXPECT_EQ("date: 01/2024, also 12/2025", s);
}

// pcre4j replaceFirstWithFullMatchReference
TYPED_TEST(ReplacementPortedTest, replaceFirstWithFullMatchReference) {
  TypeParam re("\\w+");
  JavaMatcherAdapter<TypeParam> m(&re, "hello world");
  EXPECT_EQ("[hello] world", m.replaceFirst("[$0]"));
}

// pcre4j appendReplacementStringBuilder (basic appendReplacement + appendTail walk)
TYPED_TEST(ReplacementPortedTest, appendReplacementBasic) {
  TypeParam re("(\\w+)");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "one two three");
  std::string sb;
  while (m.find()) {
    m.appendReplacement(sb, "[$1]");
  }
  m.appendTail(sb);
  EXPECT_EQ("[one] [two] [three]", sb);
}

// pcre4j appendReplacementWithNamedGroup
TYPED_TEST(ReplacementPortedTest, appendReplacementWithNamedGroup) {
  TypeParam re("(?<word>\\w+)");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "one two three");
  std::string sb;
  while (m.find()) {
    m.appendReplacement(sb, "${word}!");
  }
  m.appendTail(sb);
  EXPECT_EQ("one! two! three!", sb);
}

// pcre4j appendReplacementEscapedCharacters: replacement "\\$\\\\" → literal "$\"
TYPED_TEST(ReplacementPortedTest, appendReplacementEscapedCharacters) {
  TypeParam re("\\d+");
  JavaMatcherAdapter<TypeParam> m(&re, "test123value");
  std::string sb;
  while (m.find()) {
    // C++ literal "\\$\\\\" = 4 chars: \ $ \ \   → in Java replacement
    // syntax: \\$ -> literal '$',  \\\\ -> literal '\'.  Net replacement: "$\".
    m.appendReplacement(sb, "\\$\\\\");
  }
  m.appendTail(sb);
  EXPECT_EQ("test$\\value", sb);
}

// pcre4j appendReplacementLiteralText
TYPED_TEST(ReplacementPortedTest, appendReplacementLiteralText) {
  TypeParam re("world");
  JavaMatcherAdapter<TypeParam> m(&re, "hello world!");
  std::string sb;
  while (m.find()) {
    m.appendReplacement(sb, "universe");
  }
  m.appendTail(sb);
  EXPECT_EQ("hello universe!", sb);
}

// pcre4j appendTailOnly: no matches, just appendTail → echoes input.
TYPED_TEST(ReplacementPortedTest, appendTailOnly) {
  TypeParam re("xyz");
  JavaMatcherAdapter<TypeParam> m(&re, "hello world");
  std::string sb;
  // No find() call → no match → appendTail copies entire input.
  m.appendTail(sb);
  EXPECT_EQ("hello world", sb);
}

// pcre4j appendReplacementNoMatch: appendReplacement without a successful
// find() throws IllegalStateException in Java; we throw std::logic_error.
TYPED_TEST(ReplacementPortedTest, appendReplacementWithoutMatchThrows) {
  TypeParam re("\\d+");
  JavaMatcherAdapter<TypeParam> m(&re, "hello world");
  std::string sb;
  EXPECT_THROW(m.appendReplacement(sb, "test"), std::logic_error);
}

// pcre4j appendReplacementMultipleGroups: "$3$2$1" reverses 3 chars.
TYPED_TEST(ReplacementPortedTest, appendReplacementMultipleGroups) {
  TypeParam re("(\\w)(\\w)(\\w)");
  JavaMatcherAdapter<TypeParam> m(&re, "abc def ghi");
  std::string sb;
  while (m.find()) {
    m.appendReplacement(sb, "$3$2$1");
  }
  m.appendTail(sb);
  EXPECT_EQ("cba fed ihg", sb);
}

// pcre4j appendReplacementGroupZero
TYPED_TEST(ReplacementPortedTest, appendReplacementGroupZero) {
  TypeParam re("\\w+");
  JavaMatcherAdapter<TypeParam> m(&re, "hello world");
  std::string sb;
  while (m.find()) {
    m.appendReplacement(sb, "[$0]");
  }
  m.appendTail(sb);
  EXPECT_EQ("[hello] [world]", sb);
}

// pcre4j appendReplacementUnicode: 4-byte UTF-8 needle / 4-byte UTF-8 repl.
TYPED_TEST(ReplacementPortedTest, appendReplacementUnicode) {
  TypeParam re("\xf0\x9f\x8c\x90"); // U+1F310 globe
  JavaMatcherAdapter<TypeParam> m(&re, "hi \xf0\x9f\x8c\x90 there");
  std::string sb;
  while (m.find()) {
    m.appendReplacement(sb, "\xf0\x9f\x8c\x8d"); // U+1F30D earth
  }
  m.appendTail(sb);
  EXPECT_EQ("hi \xf0\x9f\x8c\x8d there", sb);
}

// pcre4j appendReplacementWithEscapedDollarSign: replacement "\\$5" →
// literal "$5" (not group 5).
TYPED_TEST(ReplacementPortedTest, appendReplacementWithEscapedDollarSign) {
  TypeParam re("\\d+");
  JavaMatcherAdapter<TypeParam> m(&re, "price: 100");
  std::string sb;
  while (m.find()) {
    m.appendReplacement(sb, "\\$5");
  }
  m.appendTail(sb);
  EXPECT_EQ("price: $5", sb);
}

// pcre4j appendReplacementBackslashEscapesNextChar: \\X → X literal
TYPED_TEST(ReplacementPortedTest, appendReplacementBackslashEscapesNextChar) {
  TypeParam re("x");
  JavaMatcherAdapter<TypeParam> m(&re, "x");
  std::string sb;
  ASSERT_TRUE(m.find());
  m.appendReplacement(sb, "\\$\\\\\\?");
  m.appendTail(sb);
  // Java: \\$ → '$', \\\\ → '\', \\? → '?'.  Net: "$\?"
  EXPECT_EQ("$\\?", sb);
}

} // namespace
} // namespace facebook::velox::regex_compat::test
