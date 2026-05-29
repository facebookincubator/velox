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
#include "velox/external/regex_compat/Re2Regex.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::regex_compat {
namespace {

TEST(Re2RegexTest, compileOk) {
  Re2Regex re("\\d+");
  EXPECT_TRUE(re.ok());
  EXPECT_EQ(0, re.NumberOfCapturingGroups());
  EXPECT_EQ("", re.error());
}

TEST(Re2RegexTest, compileError) {
  Re2Regex re("(unclosed");
  EXPECT_FALSE(re.ok());
  EXPECT_FALSE(re.error().empty());
}

TEST(Re2RegexTest, javaNamedGroupAccepted) {
  // Java syntax (?<name>...) should be translated to RE2 (?P<name>...) by
  // prepareRegexpReplacePattern before reaching re2::RE2.
  Re2Regex re("(?<num>\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_EQ(1, re.NumberOfCapturingGroups());
  const auto& names = re.NamedCapturingGroups();
  ASSERT_NE(names.end(), names.find("num"));
  EXPECT_EQ(1, names.at("num"));
}

TEST(Re2RegexTest, matchUnanchored) {
  Re2Regex re("(\\d+)");
  std::string_view sub[2];
  std::string_view in = "abc 42 xyz";
  EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kUnanchored, sub, 2));
  EXPECT_EQ("42", sub[0]);
  EXPECT_EQ("42", sub[1]);
}

TEST(Re2RegexTest, matchAnchorBoth) {
  Re2Regex re("[a-z]+");
  std::string_view sub[1];
  std::string_view in = "abc";
  EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kAnchorBoth, sub, 1));
}

TEST(Re2RegexTest, matchAnchorBothRejectsTrailing) {
  Re2Regex re("[a-z]+");
  std::string_view sub[1];
  std::string_view in = "abc1";
  EXPECT_FALSE(re.Match(in, 0, in.size(), Anchor::kAnchorBoth, sub, 1));
}

TEST(Re2RegexTest, fullPartialMatch) {
  Re2Regex re("[a-z]+");
  EXPECT_TRUE(Re2Regex::FullMatch("abc", re));
  EXPECT_FALSE(Re2Regex::FullMatch("abc1", re));
  EXPECT_TRUE(Re2Regex::PartialMatch("abc1", re));
}

TEST(Re2RegexTest, globalReplaceWithNumberedGroup) {
  // Java $1 should be translated to RE2 \1 by prepareRegexpReplaceReplacement.
  Re2Regex re("(\\d+)");
  std::string s = "a1b22c333";
  int n = Re2Regex::GlobalReplace(&s, re, "[$1]");
  EXPECT_EQ(3, n);
  EXPECT_EQ("a[1]b[22]c[333]", s);
}

TEST(Re2RegexTest, globalReplaceWithNamedGroup) {
  // Java ${name} should be translated to RE2 \N by prepareRegexpReplaceReplacement.
  Re2Regex re("(?<n>\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string s = "a1b22c";
  int n = Re2Regex::GlobalReplace(&s, re, "[${n}]");
  EXPECT_EQ(2, n);
  EXPECT_EQ("a[1]b[22]c", s);
}

TEST(Re2RegexTest, caseInsensitiveOption) {
  Options opt;
  opt.caseSensitive = false;
  Re2Regex re("hello", opt);
  EXPECT_TRUE(Re2Regex::PartialMatch("HELLO world", re));
}

TEST(Re2RegexTest, lookaroundUnsupportedByRe2) {
  // RE2 doesn't support lookahead; prepareRegexpReplacePattern doesn't help here
  // (it only rewrites named groups).  We expect ok()==false with an error.
  // RE2's exact error message is "invalid perl operator: (?=" — we assert on a
  // substring so this stays robust across re2 versions.
  Re2Regex re("(?=foo)bar");
  EXPECT_FALSE(re.ok());
  EXPECT_THAT(re.error(), ::testing::HasSubstr("(?="));
}

} // namespace
} // namespace facebook::velox::regex_compat
