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

#include <gtest/gtest.h>

#include "velox/dwio/common/RowRanges.h"

namespace facebook::velox::dwio::common {
namespace {

class RowRangesTest : public testing::Test {};

TEST(RowRangesTest, rowRange) {
  RowRange r(5, 10);
  EXPECT_EQ(r.from_, 5);
  EXPECT_EQ(r.to_, 10);
  EXPECT_EQ(r.count(), 6); // 10 - 5 + 1
  EXPECT_EQ(r.toString(), "[5, 10]");
  auto ru1 = RowRange::TryUnion(RowRange(1, 5), RowRange(6, 8));
  EXPECT_TRUE(ru1.has_value());
  EXPECT_EQ(ru1->toString(), "[1, 8]");

  auto ru2 = RowRange::TryUnion(RowRange(1, 5), RowRange(7, 8));
  EXPECT_FALSE(ru2.has_value());

  auto ri = RowRange::intersection(RowRange(2, 6), RowRange(4, 10));
  EXPECT_TRUE(ri.has_value());
  EXPECT_EQ(ri->toString(), "[4, 6]");

  auto ri2 = RowRange::intersection(RowRange(1, 3), RowRange(4, 5));
  EXPECT_FALSE(ri2.has_value());
}

TEST(RowRangesTest, createSingle) {
  RowRanges rr = RowRanges::createSingle(10);
  EXPECT_EQ(rr.rowCount(), 10);
  EXPECT_EQ(rr.toString(), "[[0, 9]]");

  RowRanges rr0 = RowRanges::createSingle(0);
  EXPECT_EQ(rr0.rowCount(), 0);
  EXPECT_EQ(rr0.toString(), "[]");
}

TEST(RowRangesTest, unionWith) {
  RowRanges a(RowRange(113, 241));
  RowRanges b(RowRange(221, 340));
  EXPECT_EQ(RowRanges::unionWith(a, b).toString(), "[[113, 340]]");

  RowRanges c(RowRange(113, 230));
  RowRanges d(RowRange(231, 340));
  EXPECT_EQ(RowRanges::unionWith(c, d).toString(), "[[113, 340]]");

  RowRanges e(RowRange(113, 230));
  RowRanges f(RowRange(232, 340));
  EXPECT_EQ(RowRanges::unionWith(e, f).toString(), "[[113, 230], [232, 340]]");

  RowRanges x;
  x.add(RowRange(5, 8));
  x.add(RowRange(10, 20));
  x.add(RowRange(25, 30));
  EXPECT_EQ(x.toString(), "[[5, 8], [10, 20], [25, 30]]");

  RowRanges y;
  y.add(RowRange(9, 18));

  EXPECT_EQ(RowRanges::unionWith(x, y).toString(), "[[5, 20], [25, 30]]");
}

TEST(RowRangesTest, intersection) {
  RowRanges a(RowRange(113, 241));
  RowRanges b(RowRange(221, 340));
  EXPECT_EQ(RowRanges::intersection(a, b).toString(), "[[221, 241]]");

  RowRanges c(RowRange(113, 230));
  RowRanges d(RowRange(231, 340));
  EXPECT_EQ(RowRanges::intersection(c, d).toString(), "[]");

  RowRanges x;
  x.add(RowRange(0, 100));
  x.add(RowRange(200, 300));
  RowRanges y;
  y.add(RowRange(50, 250));
  EXPECT_EQ(
      RowRanges::intersection(x, y).toString(), "[[50, 100], [200, 250]]");
}

TEST(RowRangesTest, overlapAndCount) {
  RowRanges r;
  r.add(RowRange(0, 4));
  r.add(RowRange(10, 12));
  EXPECT_TRUE(r.isOverlapping(2, 3));
  EXPECT_TRUE(r.isOverlapping(3, 10));
  EXPECT_FALSE(r.isOverlapping(5, 9));

  EXPECT_EQ(r.rowCount(), 8);
}
TEST(RowRangesTest, addAdjacentChains) {
  RowRanges r;
  r.add(RowRange(0, 0));
  EXPECT_EQ(r.toString(), "[[0, 0]]");
  r.add(RowRange(1, 1));
  EXPECT_EQ(r.toString(), "[[0, 1]]");
  r.add(RowRange(2, 2));
  EXPECT_EQ(r.toString(), "[[0, 2]]");
  EXPECT_EQ(r.rowCount(), 3);
}

TEST(RowRangesTest, addNonAdjacent) {
  RowRanges r;
  r.add(RowRange(5, 10));
  r.add(RowRange(12, 15));
  EXPECT_EQ(r.toString(), "[[5, 10], [12, 15]]");
  EXPECT_EQ(r.rowCount(), (10 - 5 + 1) + (15 - 12 + 1));
}

TEST(RowRangesTest, unionChainedAdjacent) {
  RowRanges a(RowRange(0, 3));
  RowRanges b(RowRange(4, 7));
  RowRanges c(RowRange(8, 10));
  auto u = RowRanges::unionWith(RowRanges::unionWith(a, b), c);
  EXPECT_EQ(u.toString(), "[[0, 10]]");
}

TEST(RowRangesTest, unionInterleaved) {
  RowRanges a;
  a.add(RowRange(0, 2));
  a.add(RowRange(10, 12));
  RowRanges b;
  b.add(RowRange(1, 11));
  auto u = RowRanges::unionWith(a, b);
  EXPECT_EQ(u.toString(), "[[0, 12]]");
}

TEST(RowRangesTest, intersectionTouchingEdge) {
  RowRanges a(RowRange(100, 200));
  RowRanges b(RowRange(200, 300));
  auto inter = RowRanges::intersection(a, b);
  EXPECT_EQ(inter.toString(), "[[200, 200]]");
  EXPECT_EQ(inter.rowCount(), 1);
}

TEST(RowRangesTest, intersectionSinglePoint) {
  RowRanges a(RowRange(50, 50));
  RowRanges b(RowRange(0, 100));
  auto inter = RowRanges::intersection(a, b);
  EXPECT_EQ(inter.toString(), "[[50, 50]]");
  EXPECT_EQ(inter.rowCount(), 1);
}

TEST(RowRangesTest, intersectionContainment) {
  RowRanges a(RowRange(0, 100));
  RowRanges b(RowRange(20, 30));
  EXPECT_EQ(RowRanges::intersection(a, b).toString(), "[[20, 30]]");
  RowRanges c(RowRange(101, 110));
  EXPECT_EQ(RowRanges::intersection(a, c).toString(), "[]");
}

TEST(RowRangesTest, overlapExactAndGap) {
  RowRanges r;
  r.add(RowRange(100, 110));
  EXPECT_TRUE(r.isOverlapping(100, 110));
  EXPECT_TRUE(r.isOverlapping(102, 108));
  EXPECT_FALSE(r.isOverlapping(90, 99));
  EXPECT_FALSE(r.isOverlapping(111, 120));
}

TEST(RowRangesTest, overlapAcrossSegments) {
  RowRanges r;
  r.add(RowRange(0, 10));
  r.add(RowRange(20, 30));
  EXPECT_FALSE(r.isOverlapping(11, 19));
  EXPECT_TRUE(r.isOverlapping(5, 15));
  EXPECT_TRUE(r.isOverlapping(15, 25));
}

TEST(RowRangeTest, differenceNoOverlap) {
  auto v = RowRange::difference({0, 5}, {10, 15});
  EXPECT_EQ(v.size(), 1);
  EXPECT_EQ(v[0].from_, 0);
  EXPECT_EQ(v[0].to_, 5);
}

TEST(RowRangeTest, differenceFullOverlap) {
  auto v = RowRange::difference({0, 5}, {0, 5});
  EXPECT_TRUE(v.empty());
}

TEST(RowRangeTest, differenceOverlapStart) {
  auto v = RowRange::difference({0, 5}, {0, 2});
  EXPECT_EQ(v.size(), 1);
  EXPECT_EQ(v[0].from_, 3);
  EXPECT_EQ(v[0].to_, 5);
}

TEST(RowRangeTest, differenceOverlapEnd) {
  auto v = RowRange::difference({0, 5}, {3, 5});
  EXPECT_EQ(v.size(), 1);
  EXPECT_EQ(v[0].from_, 0);
  EXPECT_EQ(v[0].to_, 2);
}

TEST(RowRangeTest, differenceOverlapMiddle) {
  auto v = RowRange::difference({0, 10}, {3, 7});
  EXPECT_EQ(v.size(), 2);
  EXPECT_EQ(v[0].from_, 0);
  EXPECT_EQ(v[0].to_, 2);
  EXPECT_EQ(v[1].from_, 8);
  EXPECT_EQ(v[1].to_, 10);
}

TEST(RowRangesTest, intersectOneOverlap) {
  RowRanges rr;
  rr = RowRanges::createSingle(6); // [0,5]
  rr = RowRanges::unionWith(rr, RowRanges({10, 15}));
  auto i = rr.intersectOne({3, 12});
  EXPECT_TRUE(i.has_value());
  EXPECT_EQ(i->from_, 3);
  EXPECT_EQ(i->to_, 5);
}

TEST(RowRangesTest, intersectOneNoOverlap) {
  RowRanges rr;
  rr = RowRanges::createSingle(6); // [0,5]
  rr = RowRanges::unionWith(rr, RowRanges({10, 15}));
  auto i = rr.intersectOne({6, 9});
  EXPECT_FALSE(i.has_value());
}

TEST(RowRangesTest, noOverlapBefore) {
  RowRanges rs;
  rs.add(RowRange(20, 30));

  auto [chunk, overlap] = RowRanges::firstSplitByIntersection({10, 19}, rs);
  EXPECT_EQ(chunk.from_, 10);
  EXPECT_EQ(chunk.to_, 19);
  EXPECT_FALSE(overlap);
}

TEST(RowRangesTest, noOverlapAfter) {
  RowRanges rs;
  rs.add(RowRange(5, 10));

  auto [chunk, overlap] = RowRanges::firstSplitByIntersection({15, 20}, rs);
  EXPECT_EQ(chunk.from_, 15);
  EXPECT_EQ(chunk.to_, 20);
  EXPECT_FALSE(overlap);
}

TEST(RowRangesTest, overlapInMiddle) {
  RowRanges rs;
  rs.add(RowRange(15, 18));

  auto [chunk, overlap] = RowRanges::firstSplitByIntersection({12, 20}, rs);
  EXPECT_EQ(chunk.from_, 12);
  EXPECT_EQ(chunk.to_, 14);
  EXPECT_FALSE(overlap);
}

TEST(RowRangesTest, overlapStartInsideValidRange) {
  RowRanges rs;
  rs.add(RowRange(10, 20));

  auto [chunk, overlap] = RowRanges::firstSplitByIntersection({15, 25}, rs);
  EXPECT_EQ(chunk.from_, 15);
  EXPECT_EQ(chunk.to_, 20);
  EXPECT_TRUE(overlap);
}

TEST(RowRangesTest, overlapExact) {
  RowRanges rs;
  rs.add(RowRange(10, 20));

  auto [chunk, overlap] = RowRanges::firstSplitByIntersection({10, 20}, rs);
  EXPECT_EQ(chunk.from_, 10);
  EXPECT_EQ(chunk.to_, 20);
  EXPECT_TRUE(overlap);
}

TEST(RowRangesTest, overlapPartialEnd) {
  RowRanges rs;
  rs.add(RowRange(15, 18));

  auto [chunk, overlap] = RowRanges::firstSplitByIntersection({15, 25}, rs);
  EXPECT_EQ(chunk.from_, 15);
  EXPECT_EQ(chunk.to_, 18);
  EXPECT_TRUE(overlap);
}

TEST(RowRangesTest, emptyRanges) {
  RowRanges rs;

  auto [chunk, overlap] = RowRanges::firstSplitByIntersection({5, 10}, rs);
  EXPECT_EQ(chunk.from_, 5);
  EXPECT_EQ(chunk.to_, 10);
  EXPECT_FALSE(overlap);
}

TEST(RowRangesTest, multipleValidRanges) {
  RowRanges rs;
  rs.add(RowRange(8, 9));
  rs.add(RowRange(12, 14));
  rs.add(RowRange(20, 25));

  auto [chunk, overlap] = RowRanges::firstSplitByIntersection({10, 22}, rs);
  EXPECT_EQ(chunk.from_, 10);
  EXPECT_EQ(chunk.to_, 11);
  EXPECT_FALSE(overlap);
}

TEST(RowRangesTest, startInsideGap) {
  RowRanges rs;
  rs.add(RowRange(5, 9));
  rs.add(RowRange(15, 20));

  auto [chunk, overlap] = RowRanges::firstSplitByIntersection({12, 18}, rs);
  EXPECT_EQ(chunk.from_, 12);
  EXPECT_EQ(chunk.to_, 14);
  EXPECT_FALSE(overlap);
}

TEST(RowRangesTest, exactBoundaryNonOverlap) {
  RowRanges rs;
  rs.add(RowRange(10, 15));

  auto [chunk, overlap] = RowRanges::firstSplitByIntersection({16, 20}, rs);
  EXPECT_EQ(chunk.from_, 16);
  EXPECT_EQ(chunk.to_, 20);
  EXPECT_FALSE(overlap);
}

TEST(RowRangesTest, exactBoundaryOverlap) {
  RowRanges rs;
  rs.add(RowRange(10, 15));

  auto [chunk, overlap] = RowRanges::firstSplitByIntersection({15, 18}, rs);
  EXPECT_EQ(chunk.from_, 15);
  EXPECT_EQ(chunk.to_, 15);
  EXPECT_TRUE(overlap);
}

TEST(RowRangesTest, onePointOverlap) {
  RowRanges rs;
  rs.add(RowRange(8, 8));

  auto [chunk, overlap] = RowRanges::firstSplitByIntersection({8, 10}, rs);
  EXPECT_EQ(chunk.from_, 8);
  EXPECT_EQ(chunk.to_, 8);
  EXPECT_TRUE(overlap);
}

TEST(RowRangesTest, unionBothEmpty) {
  RowRanges a;
  RowRanges b;
  auto u = RowRanges::unionWith(a, b);
  EXPECT_EQ(u.toString(), "[]");
}

TEST(RowRangesTest, unionOneEmpty) {
  RowRanges a(RowRange(5, 10));
  RowRanges b;
  auto u1 = RowRanges::unionWith(a, b);
  EXPECT_EQ(u1.toString(), "[[5, 10]]");
  auto u2 = RowRanges::unionWith(b, a);
  EXPECT_EQ(u2.toString(), "[[5, 10]]");
}

TEST(RowRangesTest, unionDisjointNonAdjacent) {
  RowRanges a(RowRange(0, 2));
  RowRanges b(RowRange(5, 7));
  auto u = RowRanges::unionWith(a, b);
  EXPECT_EQ(u.toString(), "[[0, 2], [5, 7]]");
}

TEST(RowRangesTest, unionAdjacentRanges) {
  RowRanges a(RowRange(0, 4));
  RowRanges b(RowRange(5, 9));
  auto u = RowRanges::unionWith(a, b);
  EXPECT_EQ(u.toString(), "[[0, 9]]");
}

TEST(RowRangesTest, unionOverlappingRanges) {
  RowRanges a(RowRange(0, 5));
  RowRanges b(RowRange(3, 10));
  auto u = RowRanges::unionWith(a, b);
  EXPECT_EQ(u.toString(), "[[0, 10]]");
}

TEST(RowRangesTest, unionMultipleRanges) {
  RowRanges a;
  a.add(RowRange(1, 2));
  a.add(RowRange(5, 6));
  RowRanges b;
  b.add(RowRange(2, 5));
  b.add(RowRange(8, 9));
  auto u = RowRanges::unionWith(a, b);
  EXPECT_EQ(u.toString(), "[[1, 6], [8, 9]]");
}

// Tests for Complement using toString() comparison
TEST(RowRangesTest, complementEmptySrcFullRange) {
  RowRanges src;
  auto c = RowRanges::complement(src, 5);
  EXPECT_EQ(c.toString(), "[[0, 4]]");
}

TEST(RowRangesTest, complementFullSrcEmpty) {
  RowRanges src(RowRange(0, 2));
  auto c = RowRanges::complement(src, 3);
  EXPECT_EQ(c.toString(), "[]");
}

TEST(RowRangesTest, complementPrefix) {
  RowRanges src(RowRange(0, 0));
  auto c = RowRanges::complement(src, 5);
  EXPECT_EQ(c.toString(), "[[1, 4]]");
}

TEST(RowRangesTest, complementSuffix) {
  RowRanges src(RowRange(3, 4));
  auto c = RowRanges::complement(src, 5);
  EXPECT_EQ(c.toString(), "[[0, 2]]");
}

TEST(RowRangesTest, complementMiddle) {
  RowRanges src(RowRange(2, 3));
  auto c = RowRanges::complement(src, 6);
  EXPECT_EQ(c.toString(), "[[0, 1], [4, 5]]");
}

TEST(RowRangesTest, complementMultipleRanges) {
  RowRanges src;
  src.add(RowRange(1, 2));
  src.add(RowRange(5, 6));
  auto c = RowRanges::complement(src, 10);
  EXPECT_EQ(c.toString(), "[[0, 0], [3, 4], [7, 9]]");
}

TEST(RowRangesTest, complementNoSpace) {
  RowRanges src;
  auto c = RowRanges::complement(src, 0);
  EXPECT_EQ(c.toString(), "[]");
}

TEST(RowRangesTest, unionWithBothEmpty) {
  RowRanges a;
  RowRanges b;
  a.unionWith(b);
  EXPECT_EQ(a.toString(), "[]");
}

TEST(RowRangesTest, unionWithOneEmpty) {
  RowRanges a(RowRange(5, 10));
  RowRanges b;
  a.unionWith(b);
  EXPECT_EQ(a.toString(), "[[5, 10]]");

  RowRanges c;
  RowRanges d(RowRange(2, 4));
  c.unionWith(d);
  EXPECT_EQ(c.toString(), "[[2, 4]]");
}

TEST(RowRangesTest, unionWithDisjointNonAdjacent) {
  RowRanges a(RowRange(0, 2));
  RowRanges b(RowRange(5, 7));
  a.unionWith(b);
  EXPECT_EQ(a.toString(), "[[0, 2], [5, 7]]");
}

TEST(RowRangesTest, unionWithAdjacentRanges) {
  RowRanges a(RowRange(0, 4));
  RowRanges b(RowRange(5, 9));
  a.unionWith(b);
  EXPECT_EQ(a.toString(), "[[0, 9]]");
}

TEST(RowRangesTest, unionWithOverlappingRanges) {
  RowRanges a(RowRange(0, 5));
  RowRanges b(RowRange(3, 10));
  a.unionWith(b);
  EXPECT_EQ(a.toString(), "[[0, 10]]");
}

TEST(RowRangesTest, unionWithMultipleRanges) {
  RowRanges a;
  a.add(RowRange(1, 2));
  a.add(RowRange(5, 6));
  RowRanges b;
  b.add(RowRange(2, 5));
  b.add(RowRange(8, 9));
  a.unionWith(b);
  EXPECT_EQ(a.toString(), "[[1, 6], [8, 9]]");
}

// Tests for intersectWith()
TEST(RowRangesTest, intersectWithBothEmpty) {
  RowRanges a;
  RowRanges b;
  a.intersectWith(b);
  EXPECT_EQ(a.toString(), "[]");
}

TEST(RowRangesTest, intersectWithNoOverlap) {
  RowRanges a(RowRange(0, 2));
  RowRanges b(RowRange(5, 7));
  a.intersectWith(b);
  EXPECT_EQ(a.toString(), "[]");
}

TEST(RowRangesTest, intersectWithPartialOverlap) {
  RowRanges a(RowRange(0, 5));
  RowRanges b(RowRange(3, 10));
  a.intersectWith(b);
  EXPECT_EQ(a.toString(), "[[3, 5]]");
}

TEST(RowRangesTest, intersectWithCompleteOverlap) {
  RowRanges a(RowRange(2, 8));
  RowRanges b(RowRange(2, 8));
  a.intersectWith(b);
  EXPECT_EQ(a.toString(), "[[2, 8]]");
}

TEST(RowRangesTest, intersectWithMultipleIntervals) {
  RowRanges a;
  a.add(RowRange(0, 3));
  a.add(RowRange(5, 9));
  RowRanges b;
  b.add(RowRange(2, 6));
  b.add(RowRange(8, 10));
  a.intersectWith(b);
  EXPECT_EQ(a.toString(), "[[2, 3], [5, 6], [8, 9]]");
}

} // namespace
} // namespace facebook::velox::dwio::common
