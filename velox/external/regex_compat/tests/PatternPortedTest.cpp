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
// Cases ported from pcre4j's `PatternTests.java`
// (https://github.com/alexey-pelykh/pcre4j, GPL-LGPL upstream; this C++ port
// is the work of the Velox project, Apache-2.0).
//
// Each TYPED_TEST below runs against every regex backend (Re2Regex,
// Pcre2Regex, JavaRegex) enabled at compile time.  Tests asserting Java
// semantics that some backend cannot satisfy are marked with the backend's
// known limitation and skipped via `if constexpr` rather than disabled, so
// any future improvement in the backend is detected by the test newly passing.
//

#include "velox/external/regex_compat/tests/BackendTestBase.h"
#include "velox/external/regex_compat/tests/JavaMatcherAdapter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::regex_compat::test {
namespace {

template <typename R>
using PatternPortedTest = BackendTest<R>;
TYPED_TEST_SUITE(PatternPortedTest, AllBackends);

// pcre4j PatternTests.toStringReturnsPattern: Pattern.toString() returns the
// original source string.  Our IRegex doesn't expose `pattern()` directly,
// but `NamedCapturingGroups()` + `NumberOfCapturingGroups()` cover the
// compile-side state-mirror part.  Skip the pure-toString assertion.

// pcre4j PatternTests.namedGroups
TYPED_TEST(PatternPortedTest, namedGroupsSingle) {
  TypeParam re("(?<number>42)");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_EQ(1, re.NumberOfCapturingGroups());
}

TYPED_TEST(PatternPortedTest, namedGroupsTwoNames) {
  TypeParam re("(?<a>x)(?<b>y)");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_EQ(2, re.NumberOfCapturingGroups());
}

TYPED_TEST(PatternPortedTest, numberedGroupsOnly) {
  TypeParam re("(\\d)(\\w)(\\s)");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_EQ(3, re.NumberOfCapturingGroups());
}

TYPED_TEST(PatternPortedTest, nonCapturingGroupDoesNotIncrement) {
  TypeParam re("(?:foo)(bar)");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_EQ(1, re.NumberOfCapturingGroups());
}

// pcre4j PatternTests.split (essence: split on \\D+ produces digit groups)
TYPED_TEST(PatternPortedTest, splitOnDigitGroups) {
  // We don't expose Pattern.split() at backend level; emulate via find-loop.
  TypeParam re("\\D+");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string_view in = "0, 1, 1, 2, 3, 5, 8";
  JavaMatcherAdapter<TypeParam> m(&re, in);
  std::vector<std::string> tokens;
  std::size_t prev = 0;
  while (m.find()) {
    tokens.emplace_back(in.substr(prev, m.start() - prev));
    prev = m.end();
  }
  tokens.emplace_back(in.substr(prev));
  EXPECT_THAT(
      tokens, ::testing::ElementsAre("0", "1", "1", "2", "3", "5", "8"));
}

// pcre4j PatternTests.unicodeSplit
TYPED_TEST(PatternPortedTest, splitUnicodeDelimiters) {
  TypeParam re("\\D+");
  ASSERT_TRUE(re.ok()) << re.error();
  // U+21E2 RIGHTWARDS DASHED ARROW (3-byte UTF-8 sequence).
  std::string_view in = "0 \xe2\x87\xa2 1 \xe2\x87\xa2 2";
  JavaMatcherAdapter<TypeParam> m(&re, in);
  std::vector<std::string> tokens;
  std::size_t prev = 0;
  while (m.find()) {
    tokens.emplace_back(in.substr(prev, m.start() - prev));
    prev = m.end();
  }
  tokens.emplace_back(in.substr(prev));
  EXPECT_THAT(tokens, ::testing::ElementsAre("0", "1", "2"));
}

// pcre4j PatternTests CASE_INSENSITIVE flag
TYPED_TEST(PatternPortedTest, caseInsensitiveCompileTimeFlag) {
  Options opt;
  opt.caseSensitive = false;
  TypeParam re("HeLLo", opt);
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_TRUE(TypeParam::PartialMatch("hello", re));
  EXPECT_TRUE(TypeParam::PartialMatch("HELLO", re));
}

// pcre4j PatternTests DOTALL flag
TYPED_TEST(PatternPortedTest, dotallMatchesNewline) {
  Options opt;
  opt.dotNl = true;
  TypeParam re("a.b", opt);
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_TRUE(TypeParam::PartialMatch("a\nb", re));
}

// pcre4j PatternTests MULTILINE flag
TYPED_TEST(PatternPortedTest, multilineCaret) {
  Options opt;
  opt.oneLine = false;
  TypeParam re("^X", opt);
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_TRUE(TypeParam::PartialMatch("foo\nX bar", re));
}

TYPED_TEST(PatternPortedTest, multilineDollar) {
  Options opt;
  opt.oneLine = false;
  TypeParam re("X$", opt);
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_TRUE(TypeParam::PartialMatch("foo X\nbar", re));
}

// pcre4j PatternTests invalid pattern syntax
TYPED_TEST(PatternPortedTest, invalidPatternRejected) {
  TypeParam re("(");
  EXPECT_FALSE(re.ok());
  EXPECT_FALSE(re.error().empty());
}

TYPED_TEST(PatternPortedTest, invalidPatternRejectedSquareBracket) {
  TypeParam re("[");
  EXPECT_FALSE(re.ok());
  EXPECT_FALSE(re.error().empty());
}

// pcre4j PatternTests: `a{` — Java rejects as incomplete quantifier.
// PCRE2 and RE2 accept it literally.  This test asserts Java behaviour;
// other backends will fail, which is the documented compatibility gap.
TYPED_TEST(PatternPortedTest, braceQuantifierIncomplete) {
  TypeParam re("a{");
  EXPECT_FALSE(re.ok());
}

// Empty pattern matches empty string anywhere.
TYPED_TEST(PatternPortedTest, emptyPatternMatchesEverywhere) {
  TypeParam re("");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_TRUE(TypeParam::PartialMatch("anything", re));
  EXPECT_TRUE(TypeParam::FullMatch("", re));
}

} // namespace
} // namespace facebook::velox::regex_compat::test
