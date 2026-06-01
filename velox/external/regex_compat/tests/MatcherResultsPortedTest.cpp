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
// Cases ported from pcre4j's `MatcherResultsTests.java`.  Java's
// `Matcher.results()` returns a Stream<MatchResult>; we model it as a
// find()-loop that snapshots (start, end, group) per match.  Cases that
// depend purely on Java's stream API (sum reductions, etc.) are skipped.
//

#include "velox/external/regex_compat/tests/BackendTestBase.h"
#include "velox/external/regex_compat/tests/JavaMatcherAdapter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

namespace facebook::velox::regex_compat::test {
namespace {

template <typename R>
using ResultsPortedTest = BackendTest<R>;
TYPED_TEST_SUITE(ResultsPortedTest, AllBackends);

// Snapshot tuple (start, end, group(0)) for each match found.
template <typename R>
std::vector<std::tuple<int, int, std::string>> snapshotAll(
    JavaMatcherAdapter<R>& m) {
  std::vector<std::tuple<int, int, std::string>> out;
  while (m.find()) {
    out.emplace_back(m.start(), m.end(), std::string(m.group(0).value()));
  }
  return out;
}

TYPED_TEST(ResultsPortedTest, resultsBasic) {
  TypeParam re("\\d+");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "a1b22c333d");
  auto r = snapshotAll(m);
  ASSERT_EQ(3u, r.size());
  EXPECT_EQ(std::make_tuple(1, 2, std::string("1")), r[0]);
  EXPECT_EQ(std::make_tuple(3, 5, std::string("22")), r[1]);
  EXPECT_EQ(std::make_tuple(6, 9, std::string("333")), r[2]);
}

TYPED_TEST(ResultsPortedTest, resultsNoMatches) {
  TypeParam re("xyz");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "hello world");
  EXPECT_TRUE(snapshotAll(m).empty());
}

TYPED_TEST(ResultsPortedTest, resultsSingleMatch) {
  TypeParam re("world");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "hello world!");
  auto r = snapshotAll(m);
  ASSERT_EQ(1u, r.size());
  EXPECT_EQ(std::make_tuple(6, 11, std::string("world")), r[0]);
}

TYPED_TEST(ResultsPortedTest, resultsWithGroups) {
  TypeParam re("(\\w)(\\d)");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "a1 b2 c3");
  std::vector<std::tuple<std::string, std::string, std::string>> r;
  while (m.find()) {
    r.emplace_back(
        std::string(m.group(0).value()),
        std::string(m.group(1).value()),
        std::string(m.group(2).value()));
  }
  ASSERT_EQ(3u, r.size());
  EXPECT_EQ(std::make_tuple("a1", "a", "1"), r[0]);
  EXPECT_EQ(std::make_tuple("b2", "b", "2"), r[1]);
  EXPECT_EQ(std::make_tuple("c3", "c", "3"), r[2]);
}

// Snapshots are independent: collecting first must not perturb later reads.
TYPED_TEST(ResultsPortedTest, resultsImmutableSnapshots) {
  TypeParam re("\\w+");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "one two three");
  auto r = snapshotAll(m);
  ASSERT_EQ(3u, r.size());
  EXPECT_EQ(std::make_tuple(0, 3, std::string("one")), r[0]);
  EXPECT_EQ(std::make_tuple(4, 7, std::string("two")), r[1]);
  EXPECT_EQ(std::make_tuple(8, 13, std::string("three")), r[2]);
}

// Zero-width matches via positive lookahead — RE2 lacks lookaround.
TYPED_TEST(ResultsPortedTest, resultsZeroWidthMatches) {
  TypeParam re("(?=\\d)");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "a1b2c3");
  auto r = snapshotAll(m);
  ASSERT_EQ(3u, r.size());
  for (auto& [s, e, g] : r) {
    EXPECT_EQ(s, e);
    EXPECT_EQ("", g);
  }
  EXPECT_EQ(1, std::get<0>(r[0]));
  EXPECT_EQ(3, std::get<0>(r[1]));
  EXPECT_EQ(5, std::get<0>(r[2]));
}

TYPED_TEST(ResultsPortedTest, resultsEmptyString) {
  TypeParam re(".*");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "");
  auto r = snapshotAll(m);
  ASSERT_EQ(1u, r.size());
  EXPECT_EQ(std::make_tuple(0, 0, std::string("")), r[0]);
}

// \p{L}+ over Cyrillic "мир" and CJK "世界" — Unicode property class.
TYPED_TEST(ResultsPortedTest, resultsUnicode) {
  TypeParam re("\\p{L}+");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(
      &re, "hello \xD0\xBC\xD0\xB8\xD1\x80 \xE4\xB8\x96\xE7\x95\x8C");
  std::vector<std::string> groups;
  while (m.find()) {
    groups.emplace_back(m.group(0).value());
  }
  EXPECT_THAT(
      groups,
      ::testing::ElementsAre(
          "hello", "\xD0\xBC\xD0\xB8\xD1\x80", "\xE4\xB8\x96\xE7\x95\x8C"));
}

// After find() once, continuing iteration yields the remainder only.
TYPED_TEST(ResultsPortedTest, resultsDoesNotReset) {
  TypeParam re("\\w+");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "one two three");
  ASSERT_TRUE(m.find());
  EXPECT_EQ("one", m.group(0).value());
  auto rest = snapshotAll(m);
  ASSERT_EQ(2u, rest.size());
  EXPECT_EQ("two", std::get<2>(rest[0]));
  EXPECT_EQ("three", std::get<2>(rest[1]));
}

// After reset() we re-iterate from the beginning.
TYPED_TEST(ResultsPortedTest, resultsAfterReset) {
  TypeParam re("\\w+");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "one two three");
  ASSERT_TRUE(m.find());
  m.reset();
  auto r = snapshotAll(m);
  ASSERT_EQ(3u, r.size());
  EXPECT_EQ("one", std::get<2>(r[0]));
  EXPECT_EQ("two", std::get<2>(r[1]));
  EXPECT_EQ("three", std::get<2>(r[2]));
}

TYPED_TEST(ResultsPortedTest, resultsWithNamedGroups) {
  TypeParam re("(?<word>\\w+)");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "hello world");
  std::vector<std::pair<std::string, std::string>> r;
  while (m.find()) {
    r.emplace_back(
        std::string(m.group(0).value()), std::string(m.group(1).value()));
  }
  ASSERT_EQ(2u, r.size());
  EXPECT_EQ(std::make_pair(std::string("hello"), std::string("hello")), r[0]);
  EXPECT_EQ(std::make_pair(std::string("world"), std::string("world")), r[1]);
}

TYPED_TEST(ResultsPortedTest, resultsCount) {
  TypeParam re("a");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "abracadabra");
  EXPECT_EQ(5u, snapshotAll(m).size());
}

} // namespace
} // namespace facebook::velox::regex_compat::test
