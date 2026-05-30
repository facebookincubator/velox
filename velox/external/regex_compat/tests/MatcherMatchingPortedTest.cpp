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
// Cases ported from pcre4j's `MatcherMatchingTests.java`.  Same provenance
// notes as PatternPortedTest.cpp.
//

#include "velox/external/regex_compat/tests/BackendTestBase.h"
#include "velox/external/regex_compat/tests/JavaMatcherAdapter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::regex_compat::test {
namespace {

template <typename R>
using MatchingPortedTest = BackendTest<R>;
TYPED_TEST_SUITE(MatchingPortedTest, AllBackends);

// Matcher.find() walks all matches.
TYPED_TEST(MatchingPortedTest, findWalksAllMatches) {
  TypeParam re("\\d+");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "a 1 b 22 c 333");
  std::vector<std::string> found;
  while (m.find()) {
    found.emplace_back(m.group(0).value());
  }
  EXPECT_THAT(found, ::testing::ElementsAre("1", "22", "333"));
}

TYPED_TEST(MatchingPortedTest, findNoMatch) {
  TypeParam re("xyz");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "abc def");
  EXPECT_FALSE(m.find());
}

TYPED_TEST(MatchingPortedTest, findWithStartIndex) {
  TypeParam re("\\d+");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "1 2 3 4");
  ASSERT_TRUE(m.find(2));
  EXPECT_EQ("2", m.group(0).value());
}

// Matcher.matches() — full-input anchored.
TYPED_TEST(MatchingPortedTest, matchesFullInput) {
  TypeParam re("\\d+");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "42");
  EXPECT_TRUE(m.matches());
}

TYPED_TEST(MatchingPortedTest, matchesRejectsPartial) {
  TypeParam re("\\d+");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "42x");
  EXPECT_FALSE(m.matches());
}

// Matcher.lookingAt() — anchor at start, may end early.
TYPED_TEST(MatchingPortedTest, lookingAtPrefixOnly) {
  TypeParam re("\\d+");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "42x");
  EXPECT_TRUE(m.lookingAt());
  EXPECT_EQ("42", m.group(0).value());
}

TYPED_TEST(MatchingPortedTest, lookingAtRejectsLateMatch) {
  TypeParam re("\\d+");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "x42");
  EXPECT_FALSE(m.lookingAt());
}

// Matcher.group(int) and Matcher.start/end accessors.
TYPED_TEST(MatchingPortedTest, groupAccessor) {
  TypeParam re("(\\d+)-(\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "foo 10-200 bar");
  ASSERT_TRUE(m.find());
  EXPECT_EQ("10-200", m.group(0).value());
  EXPECT_EQ("10", m.group(1).value());
  EXPECT_EQ("200", m.group(2).value());
  EXPECT_EQ(4, m.start());
  EXPECT_EQ(10, m.end());
  EXPECT_EQ(4, m.start(1));
  EXPECT_EQ(6, m.end(1));
  EXPECT_EQ(7, m.start(2));
  EXPECT_EQ(10, m.end(2));
}

TYPED_TEST(MatchingPortedTest, groupCountAccessor) {
  TypeParam re("(a)(b)(c)(d)");
  JavaMatcherAdapter<TypeParam> m(&re, "abcd");
  EXPECT_EQ(4, m.groupCount());
}

// Matcher.group(String) — named groups.  JavaRegex relies on JDK 20+
// Pattern.namedGroups() which our build host has, but other JDKs may not;
// we keep this test conservative and skip if name table is empty.
TYPED_TEST(MatchingPortedTest, groupAccessorByName) {
  TypeParam re("(?<lo>\\d+)-(?<hi>\\d+)");
  ASSERT_TRUE(re.ok()) << re.error();
  if (re.NamedCapturingGroups().empty()) {
    GTEST_SKIP() << "Backend doesn't expose named group table";
  }
  JavaMatcherAdapter<TypeParam> m(&re, "10-200");
  ASSERT_TRUE(m.find());
  EXPECT_EQ("10", m.group("lo").value());
  EXPECT_EQ("200", m.group("hi").value());
}

// Matcher.reset() — restart from beginning.
TYPED_TEST(MatchingPortedTest, resetRestartsCursor) {
  TypeParam re("\\d");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "a1b2c3");
  EXPECT_TRUE(m.find());
  EXPECT_EQ("1", m.group(0).value());
  EXPECT_TRUE(m.find());
  EXPECT_EQ("2", m.group(0).value());
  m.reset();
  EXPECT_TRUE(m.find());
  EXPECT_EQ("1", m.group(0).value());
}

// Matcher.reset(input) — re-bind to new input.
TYPED_TEST(MatchingPortedTest, resetWithNewInput) {
  TypeParam re("\\d");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "abc");
  EXPECT_FALSE(m.find());
  m.reset("9 8 7");
  EXPECT_TRUE(m.find());
  EXPECT_EQ("9", m.group(0).value());
}

// Empty group sentinel.
TYPED_TEST(MatchingPortedTest, groupDidNotParticipate) {
  TypeParam re("(a)|(b)");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "a");
  ASSERT_TRUE(m.find());
  EXPECT_EQ("a", m.group(1).value());
  EXPECT_EQ(std::nullopt, m.group(2));
  EXPECT_EQ(-1, m.start(2));
  EXPECT_EQ(-1, m.end(2));
}

} // namespace
} // namespace facebook::velox::regex_compat::test
