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
#if VELOX_REGEX_COMPAT_HAS_JAVA

#include "velox/external/regex_compat/JavaRegex.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::regex_compat {
namespace {

TEST(JavaRegexTest, compileOk) {
  JavaRegex re("\\d+");
  EXPECT_TRUE(re.ok());
  EXPECT_EQ(0, re.NumberOfCapturingGroups());
}

TEST(JavaRegexTest, compileError) {
  JavaRegex re("(unclosed");
  EXPECT_FALSE(re.ok());
  EXPECT_FALSE(re.error().empty());
}

TEST(JavaRegexTest, namedGroup) {
  JavaRegex re("(?<num>\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_EQ(1, re.NumberOfCapturingGroups());
  // Pattern.namedGroups() is JDK 20+; treat as best-effort.
  if (!re.NamedCapturingGroups().empty()) {
    EXPECT_EQ(1, re.NamedCapturingGroups().at("num"));
  }
}

TEST(JavaRegexTest, matchUnanchored) {
  JavaRegex re("(\\d+)");
  std::string_view sub[2];
  std::string_view in = "abc 42 xyz";
  EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kUnanchored, sub, 2));
  EXPECT_EQ("42", sub[0]);
  EXPECT_EQ("42", sub[1]);
}

TEST(JavaRegexTest, matchAnchorBoth) {
  JavaRegex re("[a-z]+");
  std::string_view sub[1];
  std::string_view in = "abc";
  EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kAnchorBoth, sub, 1));
}

TEST(JavaRegexTest, matchAnchorBothRejectsTrailing) {
  JavaRegex re("[a-z]+");
  std::string_view sub[1];
  std::string_view in = "abc1";
  EXPECT_FALSE(re.Match(in, 0, in.size(), Anchor::kAnchorBoth, sub, 1));
}

TEST(JavaRegexTest, fullPartialMatch) {
  JavaRegex re("[a-z]+");
  EXPECT_TRUE(JavaRegex::FullMatch("abc", re));
  EXPECT_FALSE(JavaRegex::FullMatch("abc1", re));
  EXPECT_TRUE(JavaRegex::PartialMatch("abc1", re));
}

TEST(JavaRegexTest, globalReplaceWithNumberedGroup) {
  JavaRegex re("(\\d+)");
  std::string s = "a1b22c333";
  int n = JavaRegex::GlobalReplace(&s, re, "[$1]");
  EXPECT_EQ(3, n);
  EXPECT_EQ("a[1]b[22]c[333]", s);
}

TEST(JavaRegexTest, globalReplaceWithNamedGroup) {
  JavaRegex re("(?<n>\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string s = "a1b22c";
  int n = JavaRegex::GlobalReplace(&s, re, "[${n}]");
  EXPECT_EQ(2, n);
  EXPECT_EQ("a[1]b[22]c", s);
}

TEST(JavaRegexTest, caseInsensitiveOption) {
  Options opt;
  opt.caseSensitive = false;
  JavaRegex re("hello", opt);
  EXPECT_TRUE(JavaRegex::PartialMatch("HELLO world", re));
}

TEST(JavaRegexTest, lookaheadSupported) {
  // Java natively supports lookahead.
  JavaRegex re("\\d+(?=px)");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string_view sub[1];
  std::string_view in = "size 42px wide";
  EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kUnanchored, sub, 1));
  EXPECT_EQ("42", sub[0]);
}

TEST(JavaRegexTest, backrefSupported) {
  // Java natively supports backreferences.
  JavaRegex re("(\\w+) \\1");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_TRUE(JavaRegex::PartialMatch("hello hello", re));
  EXPECT_FALSE(JavaRegex::PartialMatch("hello world", re));
}

TEST(JavaRegexTest, javaSpecificPropertyInLC) {
  // Java's \p{InGreek} (Unicode block "Greek"). This is one of the
  // Java-specific property tokens that PCRE2 cannot understand natively —
  // serves as a sentinel for the future Java->PCRE2 translator scope.
  JavaRegex re("\\p{InGreek}+");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_TRUE(JavaRegex::PartialMatch("hello \xce\xb1\xce\xb2\xce\xb3 world", re));
}

} // namespace
} // namespace facebook::velox::regex_compat

#endif // VELOX_REGEX_COMPAT_HAS_JAVA
