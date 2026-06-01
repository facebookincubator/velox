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
#include "velox/external/regex_compat/Pcre2Regex.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::regex_compat {
namespace {

TEST(Pcre2RegexTest, compileOk) {
  Pcre2Regex re("\\d+");
  EXPECT_TRUE(re.ok());
  EXPECT_EQ(0, re.NumberOfCapturingGroups());
  EXPECT_EQ("", re.error());
}

TEST(Pcre2RegexTest, compileError) {
  Pcre2Regex re("(unclosed");
  EXPECT_FALSE(re.ok());
  EXPECT_FALSE(re.error().empty());
}

TEST(Pcre2RegexTest, surrogateBlockCompilesInRawByteMode) {
  Pcre2Regex re("\\p{InHIGH_SURROGATES}");
  EXPECT_TRUE(re.ok()) << re.error();
}

TEST(Pcre2RegexTest, javaNamedGroupAccepted) {
  // PCRE2 natively understands (?<name>...) — no translation needed.
  Pcre2Regex re("(?<num>\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_EQ(1, re.NumberOfCapturingGroups());
  const auto& names = re.NamedCapturingGroups();
  ASSERT_NE(names.end(), names.find("num"));
  EXPECT_EQ(1, names.at("num"));
}

TEST(Pcre2RegexTest, matchUnanchored) {
  Pcre2Regex re("(\\d+)");
  std::string_view sub[2];
  std::string_view in = "abc 42 xyz";
  EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kUnanchored, sub, 2));
  EXPECT_EQ("42", sub[0]);
  EXPECT_EQ("42", sub[1]);
}

TEST(Pcre2RegexTest, matchAnchorBoth) {
  Pcre2Regex re("[a-z]+");
  std::string_view sub[1];
  std::string_view in = "abc";
  EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kAnchorBoth, sub, 1));
}

TEST(Pcre2RegexTest, matchAnchorBothRejectsTrailing) {
  Pcre2Regex re("[a-z]+");
  std::string_view sub[1];
  std::string_view in = "abc1";
  EXPECT_FALSE(re.Match(in, 0, in.size(), Anchor::kAnchorBoth, sub, 1));
}

TEST(Pcre2RegexTest, fullPartialMatch) {
  Pcre2Regex re("[a-z]+");
  EXPECT_TRUE(Pcre2Regex::FullMatch("abc", re));
  EXPECT_FALSE(Pcre2Regex::FullMatch("abc1", re));
  EXPECT_TRUE(Pcre2Regex::PartialMatch("abc1", re));
}

TEST(Pcre2RegexTest, globalReplaceWithNumberedGroup) {
  // PCRE2 with SUBSTITUTE_EXTENDED natively understands $1.
  Pcre2Regex re("(\\d+)");
  std::string s = "a1b22c333";
  int n = Pcre2Regex::GlobalReplace(&s, re, "[$1]");
  EXPECT_EQ(3, n);
  EXPECT_EQ("a[1]b[22]c[333]", s);
}

TEST(Pcre2RegexTest, globalReplaceWithNamedGroup) {
  // PCRE2 natively understands ${name}.
  Pcre2Regex re("(?<n>\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string s = "a1b22c";
  int n = Pcre2Regex::GlobalReplace(&s, re, "[${n}]");
  EXPECT_EQ(2, n);
  EXPECT_EQ("a[1]b[22]c", s);
}

TEST(Pcre2RegexTest, caseInsensitiveOption) {
  Options opt;
  opt.caseSensitive = false;
  Pcre2Regex re("hello", opt);
  EXPECT_TRUE(Pcre2Regex::PartialMatch("HELLO world", re));
}

TEST(Pcre2RegexTest, defaultWordClassIsAscii) {
  Pcre2Regex re("(?<!あ)\\w");
  ASSERT_TRUE(re.ok()) << re.error();

  std::string_view sub[1];
  std::string_view in = "###あぃいa###";
  ASSERT_TRUE(re.Match(in, 0, in.size(), Anchor::kUnanchored, sub, 1));
  EXPECT_EQ("a", sub[0]);
}

TEST(Pcre2RegexTest, matchCanCrossLoneSurrogateUtf8) {
  Pcre2Regex re("(.)([^a])xyz");
  ASSERT_TRUE(re.ok()) << re.error();

  const std::string in =
      std::string("\xED\xA0\x80", 3) + "\xF0\x90\x80\x80" + "xyz";
  std::string_view sub[3];
  ASSERT_TRUE(re.Match(in, 0, in.size(), Anchor::kUnanchored, sub, 3));
  EXPECT_EQ(in, sub[0]);
  EXPECT_EQ(std::string("\xED\xA0\x80", 3), sub[1]);
  EXPECT_EQ("\xF0\x90\x80\x80", sub[2]);
}

TEST(Pcre2RegexTest, rawSurrogateQuantifierAppliesToWholeCodeUnit) {
  Pcre2Regex re(std::string("\xED\xA0\x80", 3) + "?\xF0\x90\x80\x82");
  ASSERT_TRUE(re.ok()) << re.error();

  const std::string in = "\xF0\x90\x80\x82";
  std::string_view sub[1];
  ASSERT_TRUE(re.Match(in, 0, in.size(), Anchor::kUnanchored, sub, 1));
  EXPECT_EQ(in, sub[0]);
}

TEST(Pcre2RegexTest, surrogateHexRangeClassMatchesLoneSurrogateOnly) {
  Pcre2Regex re("[\\x{d800}-\\x{dbff}\\x{dc00}-\\x{dfff}]");
  ASSERT_TRUE(re.ok()) << re.error();

  const std::string lone = std::string("xxx") + "\xED\xB2\xA9" + "\xED\xA0\xBD" + "yyy";
  std::string_view sub[1];
  ASSERT_TRUE(re.Match(lone, 0, lone.size(), Anchor::kUnanchored, sub, 1));
  EXPECT_EQ(std::string("\xED\xB2\xA9", 3), sub[0]);

  const std::string validPair = "\xF0\x9F\x92\xA9";
  EXPECT_FALSE(re.Match(validPair, 0, validPair.size(), Anchor::kUnanchored, sub, 1));
}

TEST(Pcre2RegexTest, rawUtf8SurrogateRangeClassMatchesWholeCodeUnit) {
  Pcre2Regex re(std::string("[") + "\xED\xA0\x80" + "-" + "\xED\xBF\xBF" + "]");
  ASSERT_TRUE(re.ok()) << re.error();

  const std::string lone = std::string("\xED\xBF\xBF", 3) + "\xED\xA0\x80";
  std::string_view sub[1];
  ASSERT_TRUE(re.Match(lone, 0, lone.size(), Anchor::kUnanchored, sub, 1));
  EXPECT_EQ(std::string("\xED\xBF\xBF", 3), sub[0]);

  const std::string validPair = "\xF0\x90\x8F\xBF";
  EXPECT_FALSE(re.Match(validPair, 0, validPair.size(), Anchor::kUnanchored, sub, 1));
}

TEST(Pcre2RegexTest, rawUtf8SurrogateLiteralInClassMatchesWholeCodeUnit) {
  Pcre2Regex re(std::string("[a") + "\xED\xA0\x80" + "]");
  ASSERT_TRUE(re.ok()) << re.error();

  std::string_view sub[1];
  const std::string lone = std::string("\xED\xA0\x80", 3);
  ASSERT_TRUE(re.Match(lone, 0, lone.size(), Anchor::kUnanchored, sub, 1));
  EXPECT_EQ(lone, sub[0]);

  EXPECT_FALSE(re.Match(lone.substr(0, 1), 0, 1, Anchor::kUnanchored, sub, 1));

  const std::string ascii = "a";
  ASSERT_TRUE(re.Match(ascii, 0, ascii.size(), Anchor::kUnanchored, sub, 1));
  EXPECT_EQ(ascii, sub[0]);
}

TEST(Pcre2RegexTest, lookaheadSupported) {
  // PCRE2 supports lookahead (unlike RE2).  This is the headline reason for
  // adding PCRE2 as an alternative backend.
  Pcre2Regex re("(?=foo)bar");
  // (?=foo)bar matches "bar" only when preceded immediately by "foo".  But
  // since (?=foo) doesn't consume "foo", the match position is at "foo" and
  // the engine tries to match "bar" there — which fails.  This pattern is
  // semantically equivalent to: match "foo" that's followed by "bar".  Use
  // a more illustrative example:
  Pcre2Regex re2("\\d+(?=px)");
  ASSERT_TRUE(re2.ok()) << re2.error();
  std::string_view sub[1];
  std::string_view in = "size 42px wide";
  EXPECT_TRUE(re2.Match(in, 0, in.size(), Anchor::kUnanchored, sub, 1));
  EXPECT_EQ("42", sub[0]);
}

TEST(Pcre2RegexTest, backrefSupported) {
  // PCRE2 supports backreferences (unlike RE2).
  Pcre2Regex re("(\\w+) \\1");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_TRUE(Pcre2Regex::PartialMatch("hello hello", re));
  EXPECT_FALSE(Pcre2Regex::PartialMatch("hello world", re));
}

} // namespace
} // namespace facebook::velox::regex_compat
