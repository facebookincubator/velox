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

// pcre4j MatcherMatchingTests.captureGroups — group(0) + start/end/start("name") symmetry
TYPED_TEST(MatchingPortedTest, captureGroupsByNameAndIndex) {
  TypeParam re("(?<four>4)(.*)(?<two>2)");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "4test2");
  ASSERT_TRUE(m.find());
  EXPECT_EQ("4test2", m.group(0).value());
  EXPECT_EQ("4", m.group(1).value());
  EXPECT_EQ("test", m.group(2).value());
  EXPECT_EQ("2", m.group(3).value());
  EXPECT_EQ(3, m.groupCount());
  if (!re.NamedCapturingGroups().empty()) {
    EXPECT_EQ("4", m.group("four").value());
    EXPECT_EQ("2", m.group("two").value());
  }
}

// pcre4j MatcherMatchingTests.matchesTrueInRegion / matchesFalseRegion
TYPED_TEST(MatchingPortedTest, matchesWithinRegion) {
  TypeParam re("42");
  JavaMatcherAdapter<TypeParam> m(&re, "[42]");
  EXPECT_TRUE(m.region(1, 3).matches());  // region "42" — full match
  JavaMatcherAdapter<TypeParam> m2(&re, "[42!]");
  EXPECT_FALSE(m2.region(1, 4).matches()); // region "42!" — not full
}

// pcre4j MatcherMatchingTests.lookingAtTrueInRegion / lookingAtFalseRegion
TYPED_TEST(MatchingPortedTest, lookingAtWithinRegion) {
  TypeParam re("42");
  JavaMatcherAdapter<TypeParam> m(&re, "[42]");
  EXPECT_TRUE(m.region(1, 3).lookingAt());
  JavaMatcherAdapter<TypeParam> m2(&re, "[!42]");
  EXPECT_FALSE(m2.region(1, 4).lookingAt());  // region "!42" — '!' first, doesn't match start
}

// pcre4j MatcherMatchingTests.findTrueInRegion / findFalseInRegion
TYPED_TEST(MatchingPortedTest, findWithinRegion) {
  TypeParam re("42");
  JavaMatcherAdapter<TypeParam> m(&re, "[42]");
  EXPECT_TRUE(m.region(1, 3).find());
  EXPECT_EQ("42", m.group(0).value());
  TypeParam re2("42!");
  JavaMatcherAdapter<TypeParam> m2(&re2, "[42]");
  EXPECT_FALSE(m2.region(1, 3).find());
}

// pcre4j MatcherMatchingTests.findFalseAtOffset
TYPED_TEST(MatchingPortedTest, findFalseAtOffset) {
  TypeParam re("42");
  JavaMatcherAdapter<TypeParam> m(&re, "!!test");
  EXPECT_FALSE(m.find(2));
}

// pcre4j MatcherMatchingTests.findMultipleWithinRegion
TYPED_TEST(MatchingPortedTest, findMultipleWithinRegion) {
  TypeParam re("42");
  JavaMatcherAdapter<TypeParam> m(&re, "42!42!42!42");
  m.region(2, 8); // region content: "!42!42!"
  std::vector<int> matchStarts;
  while (m.find()) {
    matchStarts.push_back(m.start());
  }
  // Should match "42" at offsets 3 and 6 (within the region [2,8)).
  EXPECT_THAT(matchStarts, ::testing::ElementsAre(3, 6));
}

// pcre4j MatcherMatchingTests.findMultipleOutsideRegion
TYPED_TEST(MatchingPortedTest, findMultipleOutsideRegion) {
  TypeParam re("42");
  JavaMatcherAdapter<TypeParam> m(&re, "42!__!__!42");
  m.region(2, 8); // region content: "!__!__!" — no "42" inside
  EXPECT_FALSE(m.find());
}

// pcre4j MatcherMatchingTests.emptyGroup — `!*` matches empty at position 0
TYPED_TEST(MatchingPortedTest, emptyGroup) {
  TypeParam re("!*");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "42");
  ASSERT_TRUE(m.find());
  EXPECT_EQ("", m.group(0).value());
  EXPECT_EQ(0, m.start());
  EXPECT_EQ(0, m.end());
  EXPECT_EQ(0, m.groupCount());
}

// pcre4j MatcherMatchingTests.unmatchedGroups — alternation where only one branch participates
TYPED_TEST(MatchingPortedTest, unmatchedGroupsInAlternation) {
  TypeParam re("42((?<exclamation>!)|(?<question>\\?))");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "42!");
  ASSERT_TRUE(m.find());
  EXPECT_EQ("42!", m.group(0).value());
  EXPECT_EQ("!", m.group(1).value());     // outer group matches '!'
  EXPECT_EQ("!", m.group(2).value());     // exclamation = '!'
  EXPECT_EQ(std::nullopt, m.group(3));    // question did NOT match
  EXPECT_EQ(3, m.groupCount());
  if (!re.NamedCapturingGroups().empty()) {
    EXPECT_EQ("!", m.group("exclamation").value());
    EXPECT_EQ(std::nullopt, m.group("question"));
  }
}

// pcre4j MatcherMatchingTests.positiveLookaround — lookahead/lookbehind both ways.
// Asserts Java semantics: pattern compiles and matches "42" in "(42)".
// Backends without lookaround (RE2) will fail this test; that's a recorded
// compatibility-rate data point, not a bug.
TYPED_TEST(MatchingPortedTest, positiveLookaround) {
  TypeParam re("(?<=(?<lWrapper>\\W))?(\\d+)(?=(?<rWrapper>\\W))?");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "(42)");
  ASSERT_TRUE(m.find());
  EXPECT_EQ("42", m.group(0).value());
}

// pcre4j MatcherMatchingTests.positiveUnmatchedLookaround —
// lookbehind not satisfied at the start; lookahead not satisfied at end.
TYPED_TEST(MatchingPortedTest, positiveUnmatchedLookaround) {
  TypeParam re("(?<=(?<lWrapper>\\W))?(\\d+)(?=(?<rWrapper>\\W))?");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "42]");
  ASSERT_TRUE(m.find());
  EXPECT_EQ("42", m.group(0).value());
}

// pcre4j MatcherMatchingTests.emptyStringMatches — pattern "^$" on empty input matches.
TYPED_TEST(MatchingPortedTest, emptyStringMatches) {
  TypeParam re("^$");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "");
  EXPECT_TRUE(m.matches());
}

// pcre4j MatcherMatchingTests.emptyStringFind — pattern "^$" on empty input finds once.
TYPED_TEST(MatchingPortedTest, emptyStringFind) {
  TypeParam re("^$");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "");
  ASSERT_TRUE(m.find());
  EXPECT_EQ(0, m.start());
  EXPECT_EQ(0, m.end());
  EXPECT_EQ("", m.group(0).value());
  EXPECT_EQ(0, m.groupCount());
}

// pcre4j MatcherMatchingTests.findAtEndOfString — find($, len(input)) finds zero-width
// match at end.
TYPED_TEST(MatchingPortedTest, findAtEndOfString) {
  TypeParam re("$");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "abc");
  EXPECT_TRUE(m.find(3));
}

// pcre4j MatcherMatchingTests.findExhaustedInRegion — multiple matches in region,
// then no more.
TYPED_TEST(MatchingPortedTest, findExhaustedInRegion) {
  TypeParam re("a");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "aaa");
  m.region(0, 2);  // region "aa"
  EXPECT_TRUE(m.find());   // first 'a'
  EXPECT_TRUE(m.find());   // second 'a'
  EXPECT_FALSE(m.find());  // no more in region
}

// pcre4j MatcherMatchingTests.findWithZeroWidthMatchExhaustsRegion —
// Java spec: $ matches at region end (zero-width), then no more matches.
TYPED_TEST(MatchingPortedTest, findWithZeroWidthMatchExhaustsRegion) {
  TypeParam re("$");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "ab");
  m.region(0, 1);
  ASSERT_TRUE(m.find());
  EXPECT_EQ(1, m.start());
  EXPECT_EQ(1, m.end());
  EXPECT_FALSE(m.find());
}

} // namespace
} // namespace facebook::velox::regex_compat::test
