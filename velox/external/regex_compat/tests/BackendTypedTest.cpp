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
// Typed test suite that exercises the regex-compat API common to all three
// backends (Re2Regex / Pcre2Regex / JavaRegex).  Each TYPED_TEST below is
// compiled and executed once per backend type, so one source line generates
// `len(AllBackends)` assertions of identical behaviour.
//

#include "velox/external/regex_compat/tests/BackendTestBase.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::regex_compat::test {
namespace {

TYPED_TEST_SUITE(BackendTest, AllBackends);

TYPED_TEST(BackendTest, compileOk) {
  TypeParam re("\\d+");
  EXPECT_TRUE(re.ok());
  EXPECT_EQ(0, re.NumberOfCapturingGroups());
}

TYPED_TEST(BackendTest, compileError) {
  TypeParam re("(unclosed");
  EXPECT_FALSE(re.ok());
  EXPECT_FALSE(re.error().empty());
}

TYPED_TEST(BackendTest, javaNamedGroup) {
  // Java syntax (?<name>...) — every backend must accept it.
  TypeParam re("(?<num>\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_EQ(1, re.NumberOfCapturingGroups());
}

TYPED_TEST(BackendTest, matchUnanchored) {
  TypeParam re("(\\d+)");
  std::string_view sub[2];
  std::string_view in = "abc 42 xyz";
  EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kUnanchored, sub, 2));
  EXPECT_EQ("42", sub[0]);
  EXPECT_EQ("42", sub[1]);
}

TYPED_TEST(BackendTest, matchAnchorBoth) {
  TypeParam re("[a-z]+");
  std::string_view sub[1];
  std::string_view in = "abc";
  EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kAnchorBoth, sub, 1));
}

TYPED_TEST(BackendTest, matchAnchorBothRejectsTrailing) {
  TypeParam re("[a-z]+");
  std::string_view sub[1];
  std::string_view in = "abc1";
  EXPECT_FALSE(re.Match(in, 0, in.size(), Anchor::kAnchorBoth, sub, 1));
}

TYPED_TEST(BackendTest, fullPartialMatch) {
  TypeParam re("[a-z]+");
  EXPECT_TRUE(TypeParam::FullMatch("abc", re));
  EXPECT_FALSE(TypeParam::FullMatch("abc1", re));
  EXPECT_TRUE(TypeParam::PartialMatch("abc1", re));
}

TYPED_TEST(BackendTest, globalReplaceNumbered) {
  TypeParam re("(\\d+)");
  std::string s = "a1b22c333";
  int n = TypeParam::GlobalReplace(&s, re, "[$1]");
  EXPECT_EQ(3, n);
  EXPECT_EQ("a[1]b[22]c[333]", s);
}

TYPED_TEST(BackendTest, globalReplaceNamed) {
  TypeParam re("(?<n>\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  std::string s = "a1b22c";
  int n = TypeParam::GlobalReplace(&s, re, "[${n}]");
  EXPECT_EQ(2, n);
  EXPECT_EQ("a[1]b[22]c", s);
}

TYPED_TEST(BackendTest, caseInsensitive) {
  Options opt;
  opt.caseSensitive = false;
  TypeParam re("hello", opt);
  EXPECT_TRUE(TypeParam::PartialMatch("HELLO world", re));
}

TYPED_TEST(BackendTest, dotAllOption) {
  // Dot matches newline only when dotNl is on.
  {
    TypeParam re(".+");
    std::string_view sub[1];
    std::string_view in = "ab\ncd";
    EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kUnanchored, sub, 1));
    EXPECT_EQ("ab", sub[0]); // stopped at \n
  }
  {
    Options opt;
    opt.dotNl = true;
    TypeParam re(".+", opt);
    std::string_view sub[1];
    std::string_view in = "ab\ncd";
    EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kUnanchored, sub, 1));
    EXPECT_EQ("ab\ncd", sub[0]); // dot now matched \n
  }
}

TYPED_TEST(BackendTest, multilineAnchors) {
  Options opt;
  opt.oneLine = false; // MULTILINE
  TypeParam re("^bar", opt);
  std::string_view sub[1];
  std::string_view in = "foo\nbar";
  EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kUnanchored, sub, 1));
  EXPECT_EQ("bar", sub[0]);
}

TYPED_TEST(BackendTest, emptyGroupMatch) {
  // Group that didn't participate in the match — must yield an empty
  // string_view (data == nullptr per contract).
  TypeParam re("(a)|(b)");
  std::string_view sub[3];
  std::string_view in = "a";
  EXPECT_TRUE(re.Match(in, 0, in.size(), Anchor::kAnchorBoth, sub, 3));
  EXPECT_EQ("a", sub[0]);
  EXPECT_EQ("a", sub[1]);
  EXPECT_EQ(nullptr, sub[2].data()); // group 2 did not match
}

} // namespace
} // namespace facebook::velox::regex_compat::test
