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
// Cases ported from pcre4j's `MatcherMatchResultTests.java`.
//
// Most cases there exercise Java-specific `MatchResult` snapshot semantics
// (immutability of the snapshot when the matcher advances, IllegalState/
// IndexOutOfBounds/IllegalArgument exception contracts, namedGroups() map
// equality, hasMatch() flag).  Those are Java API-contract tests, not
// regex-engine behavior, so they are skipped here — they would produce
// identical pass/fail across all three backends and add no engine-compat
// signal.
//
// We port only the two cases that exercise engine behavior the existing
// MatcherMatchingPortedTest doesn't already cover:
//   * matchResultByGroupNumber       — 3 consecutive whitespace-separated
//                                      capturing groups, sweep over all
//                                      group indices.
//   * matchResultNamedGroupAccessors — 3 named groups in a date pattern.
//

#include "velox/external/regex_compat/tests/BackendTestBase.h"
#include "velox/external/regex_compat/tests/JavaMatcherAdapter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::regex_compat::test {
namespace {

template <typename R>
using MatchResultPortedTest = BackendTest<R>;
TYPED_TEST_SUITE(MatchResultPortedTest, AllBackends);

TYPED_TEST(MatchResultPortedTest, matchResultByGroupNumber) {
  TypeParam re("(\\w+)\\s+(\\w+)\\s+(\\w+)");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "one two three");
  ASSERT_TRUE(m.find());
  EXPECT_EQ(3, m.groupCount());
  EXPECT_EQ("one two three", m.group(0).value());
  EXPECT_EQ("one", m.group(1).value());
  EXPECT_EQ("two", m.group(2).value());
  EXPECT_EQ("three", m.group(3).value());
  EXPECT_EQ(0, m.start(0));
  EXPECT_EQ(13, m.end(0));
  EXPECT_EQ(0, m.start(1));
  EXPECT_EQ(3, m.end(1));
  EXPECT_EQ(4, m.start(2));
  EXPECT_EQ(7, m.end(2));
  EXPECT_EQ(8, m.start(3));
  EXPECT_EQ(13, m.end(3));
}

TYPED_TEST(MatchResultPortedTest, matchResultNamedGroupAccessors) {
  TypeParam re("(?<year>\\d{4})-(?<month>\\d{2})-(?<day>\\d{2})");
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "date: 2024-01-15");
  ASSERT_TRUE(m.find());
  EXPECT_EQ("2024", m.group("year").value());
  EXPECT_EQ("01", m.group("month").value());
  EXPECT_EQ("15", m.group("day").value());
  EXPECT_EQ(6, m.start("year"));
  EXPECT_EQ(10, m.end("year"));
  EXPECT_EQ(11, m.start("month"));
  EXPECT_EQ(13, m.end("month"));
  EXPECT_EQ(14, m.start("day"));
  EXPECT_EQ(16, m.end("day"));
}

} // namespace
} // namespace facebook::velox::regex_compat::test
