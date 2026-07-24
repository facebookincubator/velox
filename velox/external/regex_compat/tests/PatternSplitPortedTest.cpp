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
// Cases ported from pcre4j's `PatternSplitTests.java`.
//
// Java's `Pattern.split` is implemented here as a free helper that drives
// the backend's find() loop through `JavaMatcherAdapter`, so engine
// differences in find()/match propagate naturally to split() output.
//
// Skipped:
//   * splitWithDelimiters* — `String[] splitWithDelimiters(...)` is Java 21+
//     and not in our embedded JDK 17 surface.
//

#include "velox/external/regex_compat/tests/BackendTestBase.h"
#include "velox/external/regex_compat/tests/JavaMatcherAdapter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace facebook::velox::regex_compat::test {
namespace {

// Java-canonical split: find()-loop walk, trailing-empty trim when limit==0,
// at-most-`limit` parts when limit>0, no trim when limit<0.
template <typename R>
std::vector<std::string>
javaSplit(R& re, std::string_view input, int limit = 0) {
  JavaMatcherAdapter<R> m(&re, input);
  std::vector<std::string> parts;
  int matches = 0;
  std::size_t index = 0;
  const bool matchLimited = limit > 0;
  while (m.find()) {
    if (matchLimited && matches == limit - 1) {
      break;
    }
    const std::size_t s = static_cast<std::size_t>(m.start());
    const std::size_t e = static_cast<std::size_t>(m.end());
    // Java skips zero-width matches that don't advance past the current
    // segment start.
    if (s == index && s == e) {
      continue;
    }
    parts.emplace_back(input.substr(index, s - index));
    index = e;
    ++matches;
  }
  if (matches == 0) {
    return {std::string(input)};
  }
  parts.emplace_back(input.substr(index));
  if (limit == 0) {
    while (!parts.empty() && parts.back().empty()) {
      parts.pop_back();
    }
  }
  return parts;
}

template <typename R>
using SplitPortedTest = BackendTest<R>;
TYPED_TEST_SUITE(SplitPortedTest, AllBackends);

// --- limit=0 trailing empty strings removal ---

TYPED_TEST(SplitPortedTest, splitTrailingEmptyStringsRemovedWithDefaultLimit) {
  TypeParam re(",");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_THAT(
      javaSplit(re, "a,b,c,,,"), ::testing::ElementsAre("a", "b", "c"));
}

TYPED_TEST(SplitPortedTest, splitTrailingEmptyStringsRemovedWithZeroLimit) {
  TypeParam re(",");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_THAT(
      javaSplit(re, "a,b,c,,,", 0), ::testing::ElementsAre("a", "b", "c"));
}

TYPED_TEST(SplitPortedTest, splitAllEmptyWithZeroLimit) {
  TypeParam re(",");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_TRUE(javaSplit(re, ",,,", 0).empty());
}

// --- Positive limit ---

TYPED_TEST(SplitPortedTest, splitPositiveLimitOne) {
  TypeParam re(",");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_THAT(javaSplit(re, "a,b,c", 1), ::testing::ElementsAre("a,b,c"));
}

TYPED_TEST(SplitPortedTest, splitPositiveLimitExceedsMatches) {
  TypeParam re(",");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_THAT(
      javaSplit(re, "a,b,c", 10), ::testing::ElementsAre("a", "b", "c"));
}

// --- Empty input and no-match ---

TYPED_TEST(SplitPortedTest, splitEmptyInput) {
  TypeParam re(",");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_THAT(javaSplit(re, ""), ::testing::ElementsAre(""));
}

TYPED_TEST(SplitPortedTest, splitNoMatch) {
  TypeParam re(",");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_THAT(javaSplit(re, "abc"), ::testing::ElementsAre("abc"));
}

// --- Regex-based delimiter edge cases ---

TYPED_TEST(SplitPortedTest, splitMultiCharDelimiter) {
  TypeParam re("\\s*,\\s*");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_THAT(
      javaSplit(re, "a , b , c"), ::testing::ElementsAre("a", "b", "c"));
}

TYPED_TEST(SplitPortedTest, splitDelimiterAtStartAndEnd) {
  TypeParam re(",");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_THAT(
      javaSplit(re, ",a,b,c,"), ::testing::ElementsAre("", "a", "b", "c"));
}

TYPED_TEST(SplitPortedTest, splitConsecutiveDelimiters) {
  TypeParam re(",");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_THAT(
      javaSplit(re, "a,,b,,c"),
      ::testing::ElementsAre("a", "", "b", "", "c"));
}

TYPED_TEST(SplitPortedTest, splitSingleCharInput) {
  TypeParam re(",");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_TRUE(javaSplit(re, ",").empty());
}

// --- splitAsStream edge cases (Java's splitAsStream is just a stream view
// over the same split logic; we reuse javaSplit here). ---

TYPED_TEST(SplitPortedTest, splitAsStreamTrailingEmpties) {
  TypeParam re(",");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_THAT(
      javaSplit(re, "a,b,c,,,"), ::testing::ElementsAre("a", "b", "c"));
}

TYPED_TEST(SplitPortedTest, splitAsStreamEmptyInput) {
  TypeParam re(",");
  ASSERT_TRUE(re.ok()) << re.error();
  EXPECT_THAT(javaSplit(re, ""), ::testing::ElementsAre(""));
}

} // namespace
} // namespace facebook::velox::regex_compat::test
