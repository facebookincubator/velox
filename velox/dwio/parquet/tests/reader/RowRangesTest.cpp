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

#include "velox/dwio/parquet/reader/RowRanges.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::parquet;

class RowRangesTest : public ParquetTestBase {};

TEST(RowRangesTest, rowRange) {
  RowRange r(5, 10);
  EXPECT_EQ(r.from_, 5);
  EXPECT_EQ(r.to_, 10);
  EXPECT_EQ(r.count(), 6); // 10 - 5 + 1
  EXPECT_EQ(r.toString(), "[5, 10]");
  auto ru1 = RowRange::TryUnion(RowRange(1, 5), RowRange(6, 8));
  ASSERT_TRUE(ru1.has_value());
  EXPECT_EQ(ru1->toString(), "[1, 8]");

  auto ru2 = RowRange::TryUnion(RowRange(1, 5), RowRange(7, 8));
  EXPECT_FALSE(ru2.has_value());

  auto ri = RowRange::intersection(RowRange(2, 6), RowRange(4, 10));
  ASSERT_TRUE(ri.has_value());
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

TEST(RowRangesTest, union) {
  RowRanges a(RowRange(113, 241));
  RowRanges b(RowRange(221, 340));
  EXPECT_EQ(RowRanges::Union(a, b).toString(), "[[113, 340]]");

  RowRanges c(RowRange(113, 230));
  RowRanges d(RowRange(231, 340));
  EXPECT_EQ(RowRanges::Union(c, d).toString(), "[[113, 340]]");

  RowRanges e(RowRange(113, 230));
  RowRanges f(RowRange(232, 340));
  EXPECT_EQ(RowRanges::Union(e, f).toString(), "[[113, 230], [232, 340]]");

  RowRanges x;
  x.add(RowRange(5, 8));
  x.add(RowRange(10, 20));
  x.add(RowRange(25, 30));
  EXPECT_EQ(x.toString(), "[[5, 8], [10, 20], [25, 30]]");

  RowRanges y;
  y.add(RowRange(9, 18));

  EXPECT_EQ(RowRanges::Union(x, y).toString(), "[[5, 20], [25, 30]]");
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
  auto u = RowRanges::Union(RowRanges::Union(a, b), c);
  EXPECT_EQ(u.toString(), "[[0, 10]]");
}

TEST(RowRangesTest, unionInterleaved) {
  RowRanges a;
  a.add(RowRange(0, 2));
  a.add(RowRange(10, 12));
  RowRanges b;
  b.add(RowRange(1, 11));
  auto u = RowRanges::Union(a, b);
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
