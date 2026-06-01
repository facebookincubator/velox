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
// Ported from org.pcre4j.regex.translate.RangeSetTest (Java).
//
#include "velox/functions/lib/java_pcre2_translator/RangeSet.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace facebook::velox::functions::java_pcre2_translator::test {

TEST(RangeSet, emptySet) {
  EXPECT_TRUE(RangeSet::empty().isEmpty());
  EXPECT_FALSE(RangeSet::empty().contains('a'));
}

TEST(RangeSet, singleCodePoint) {
  auto s = RangeSet::single('a');
  EXPECT_FALSE(s.isEmpty());
  EXPECT_TRUE(s.contains('a'));
  EXPECT_FALSE(s.contains('b'));
}

TEST(RangeSet, range) {
  auto az = RangeSet::range('a', 'z');
  EXPECT_TRUE(az.contains('a'));
  EXPECT_TRUE(az.contains('m'));
  EXPECT_TRUE(az.contains('z'));
  EXPECT_FALSE(az.contains('A'));
  EXPECT_FALSE(az.contains('{'));
}

TEST(RangeSet, unionDisjoint) {
  auto u = RangeSet::range('a', 'z').unionWith(RangeSet::range('A', 'Z'));
  EXPECT_TRUE(u.contains('a'));
  EXPECT_TRUE(u.contains('A'));
  EXPECT_FALSE(u.contains('1'));
}

TEST(RangeSet, unionOverlapping) {
  auto u = RangeSet::range('a', 'c').unionWith(RangeSet::range('b', 'd'));
  EXPECT_TRUE(u.contains('a'));
  EXPECT_TRUE(u.contains('b'));
  EXPECT_TRUE(u.contains('d'));
  EXPECT_FALSE(u.contains('e'));
  EXPECT_EQ(1, u.rangeCount());
}

TEST(RangeSet, intersectOverlap) {
  auto i = RangeSet::range('a', 'c').intersect(RangeSet::range('b', 'd'));
  EXPECT_FALSE(i.contains('a'));
  EXPECT_TRUE(i.contains('b'));
  EXPECT_TRUE(i.contains('c'));
  EXPECT_FALSE(i.contains('d'));
}

TEST(RangeSet, intersectDisjoint) {
  auto i = RangeSet::range('a', 'c').intersect(RangeSet::range('d', 'f'));
  EXPECT_TRUE(i.isEmpty());
}

TEST(RangeSet, complementEmpty) {
  auto c = RangeSet::empty().complement();
  EXPECT_EQ(RangeSet::all(), c.unionWith(RangeSet::empty()));
  EXPECT_TRUE(c.contains(0));
  EXPECT_TRUE(c.contains(0x10FFFF));
}

TEST(RangeSet, complementRange) {
  auto notAz = RangeSet::range('a', 'z').complement();
  EXPECT_FALSE(notAz.contains('a'));
  EXPECT_FALSE(notAz.contains('z'));
  EXPECT_TRUE(notAz.contains('A'));
  EXPECT_TRUE(notAz.contains('0'));
  EXPECT_TRUE(notAz.contains(0x10FFFF));
}

TEST(RangeSet, subtract) {
  auto diff = RangeSet::range('a', 'f').subtract(RangeSet::range('c', 'f'));
  EXPECT_TRUE(diff.contains('a'));
  EXPECT_TRUE(diff.contains('b'));
  EXPECT_FALSE(diff.contains('c'));
  EXPECT_FALSE(diff.contains('f'));
}

TEST(RangeSet, toPcre2ClassBodySinglePrintable) {
  EXPECT_EQ("a", RangeSet::single('a').toPcre2ClassBody());
}

TEST(RangeSet, toPcre2ClassBodySingleNonPrintable) {
  EXPECT_EQ("\\x{9}", RangeSet::single('\t').toPcre2ClassBody());
}

TEST(RangeSet, toPcre2ClassBodyRange) {
  EXPECT_EQ("a-z", RangeSet::range('a', 'z').toPcre2ClassBody());
}

TEST(RangeSet, toPcre2ClassBodyEscapesSpecialChars) {
  EXPECT_EQ("\\-", RangeSet::single('-').toPcre2ClassBody());
  EXPECT_EQ("\\]", RangeSet::single(']').toPcre2ClassBody());
  EXPECT_EQ("\\^", RangeSet::single('^').toPcre2ClassBody());
}

TEST(RangeSet, toPcre2ClassBodyMultipleRanges) {
  auto u = RangeSet::range('a', 'z').unionWith(RangeSet::range('A', 'Z'));
  const auto body = u.toPcre2ClassBody();
  EXPECT_TRUE(
      body.find("A-Z") != std::string::npos ||
      body.find("a-z") != std::string::npos);
}

TEST(RangeSet, singleRejectsNegative) {
  EXPECT_THROW(RangeSet::single(-1), std::invalid_argument);
}

TEST(RangeSet, singleRejectsAboveMax) {
  EXPECT_THROW(RangeSet::single(0x110000), std::invalid_argument);
}

TEST(RangeSet, singleAcceptsBoundaries) {
  EXPECT_EQ(1, RangeSet::single(0).rangeCount());
  EXPECT_EQ(1, RangeSet::single(0x10FFFF).rangeCount());
}

TEST(RangeSet, rangeRejectsNegativeLo) {
  EXPECT_THROW(RangeSet::range(-1, 5), std::invalid_argument);
}

TEST(RangeSet, rangeRejectsHiAboveMax) {
  EXPECT_THROW(RangeSet::range(0, 0x110000), std::invalid_argument);
}

TEST(RangeSet, rangeRejectsInverted) {
  EXPECT_THROW(RangeSet::range(5, 4), std::invalid_argument);
}

TEST(RangeSet, unionMergesAdjacentRanges) {
  auto merged = RangeSet::range('a', 'c').unionWith(RangeSet::range('d', 'f'));
  EXPECT_EQ(1, merged.rangeCount())
      << "adjacent ranges must be merged; got: " << merged.toPcre2ClassBody();
  EXPECT_EQ("a-f", merged.toPcre2ClassBody());
}

TEST(RangeSet, unionMergesOverlappingRanges) {
  auto merged = RangeSet::range('a', 'e').unionWith(RangeSet::range('c', 'g'));
  EXPECT_EQ(1, merged.rangeCount());
  EXPECT_EQ("a-g", merged.toPcre2ClassBody());
}

} // namespace facebook::velox::functions::java_pcre2_translator::test
