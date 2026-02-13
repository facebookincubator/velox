/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/exec/SpatialIndex.h"
#include <gtest/gtest.h>
#include <algorithm>

using namespace ::testing;
using namespace facebook::velox::exec;

namespace facebook::velox::exec::test {

class SpatialIndexTest : public virtual testing::Test {
 protected:
  void makeIndex(
      std::vector<Envelope> envelopes,
      uint32_t branchSize = SpatialIndex::kDefaultRTreeBranchSize) {
    branchSize_ = branchSize;
    Envelope bounds = Envelope::of(envelopes);
    index_ = SpatialIndex(std::move(bounds), std::move(envelopes), branchSize);
  }

  Envelope indexBounds() const {
    return index_.bounds();
  }

  void assertQuery(
      double minX,
      double minY,
      double maxX,
      double maxY,
      std::vector<int32_t> expected) const {
    std::vector<int32_t> actual =
        index_.query(Envelope::from(minX, minY, maxX, maxY));
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(actual, expected);
  }

  SpatialIndex index_;
  uint32_t branchSize_ = SpatialIndex::kDefaultRTreeBranchSize;
};

TEST_F(SpatialIndexTest, testEnvelope) {
  Envelope empty = Envelope::empty();
  ASSERT_TRUE(empty.isEmpty());
  ASSERT_FALSE(Envelope::intersects(empty, empty));

  Envelope point =
      Envelope{.minX = 0, .minY = 0, .maxX = 0, .maxY = 0, .rowIndex = -1};
  ASSERT_FALSE(point.isEmpty());
  ASSERT_FALSE(Envelope::intersects(empty, point));
  ASSERT_TRUE(Envelope::intersects(point, point));
}

TEST_F(SpatialIndexTest, testNaNHandling) {
  float nan = std::numeric_limits<float>::quiet_NaN();

  Envelope envWithNaN{
      .minX = nan, .minY = 0, .maxX = 1, .maxY = 1, .rowIndex = 0};
  ASSERT_TRUE(envWithNaN.isEmpty());

  Envelope envWithNaN2{
      .minX = 0, .minY = 0, .maxX = nan, .maxY = 1, .rowIndex = 0};
  ASSERT_TRUE(envWithNaN2.isEmpty());

  Envelope envWithNaN3{
      .minX = 0, .minY = nan, .maxX = 1, .maxY = 1, .rowIndex = 0};
  ASSERT_TRUE(envWithNaN3.isEmpty());

  Envelope envWithNaN4{
      .minX = 0, .minY = 0, .maxX = 1, .maxY = nan, .rowIndex = 0};
  ASSERT_TRUE(envWithNaN4.isEmpty());

  Envelope envWithNaN5{
      .minX = nan, .minY = nan, .maxX = nan, .maxY = nan, .rowIndex = 0};
  ASSERT_TRUE(envWithNaN5.isEmpty());
}

TEST_F(SpatialIndexTest, testEmptyIndex) {
  makeIndex(std::vector<Envelope>{});
  Envelope bounds = indexBounds();
  ASSERT_EQ(bounds.minX, std::numeric_limits<double>::infinity());
  ASSERT_EQ(bounds.minY, std::numeric_limits<double>::infinity());
  ASSERT_EQ(bounds.maxX, -std::numeric_limits<double>::infinity());
  ASSERT_EQ(bounds.maxY, -std::numeric_limits<double>::infinity());
  ASSERT_EQ(bounds.rowIndex, -1);

  assertQuery(0, 0, 1, 1, {});
}

TEST_F(SpatialIndexTest, testSingleEnvelope) {
  makeIndex(
      std::vector<Envelope>{Envelope{
          .minX = 1, .minY = 11, .maxX = 2, .maxY = 12, .rowIndex = 0}});

  Envelope bounds = indexBounds();
  ASSERT_EQ(bounds.minX, 1);
  ASSERT_EQ(bounds.minY, 11);
  ASSERT_EQ(bounds.maxX, 2);
  ASSERT_EQ(bounds.maxY, 12);

  assertQuery(1.5, 11.5, 1.5, 11.5, {0});
  assertQuery(0.5, 10.5, 1.5, 11.5, {0});
  assertQuery(0, 10, 0.5, 10.5, {});
  assertQuery(3, 13, 4, 14, {});
}

TEST_F(SpatialIndexTest, testPointProbe) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{.minX = 1, .minY = 0, .maxX = 1, .maxY = 0, .rowIndex = 6},
          Envelope{.minX = 0, .minY = 0, .maxX = 0, .maxY = 0, .rowIndex = 5},
          Envelope{.minX = 0, .minY = 0, .maxX = 1, .maxY = 1, .rowIndex = 4},
          Envelope{.minX = -1, .minY = -1, .maxX = 0, .maxY = 0, .rowIndex = 3},
          Envelope{.minX = -1, .minY = -1, .maxX = 1, .maxY = 1, .rowIndex = 2},
          Envelope{
              .minX = 0.5, .minY = 0.5, .maxX = 1, .maxY = 1, .rowIndex = 1},
      });
  Envelope bounds = indexBounds();
  ASSERT_EQ(bounds.minX, -1);
  ASSERT_EQ(bounds.minY, -1);
  ASSERT_EQ(bounds.maxX, 1);
  ASSERT_EQ(bounds.maxY, 1);
  ASSERT_EQ(bounds.rowIndex, -1);

  assertQuery(0, 0, 0, 0, {2, 3, 4, 5});
  assertQuery(0, 1, 0, 1, {2, 4});
}

TEST_F(SpatialIndexTest, testFloatImprecision) {
  // Since the index casts doubles to floats then nudges the result,
  // we should make sure that the index gives the right results on
  // cases where the double doesn't have an exact float representation.
  float float1 = 1.0f;
  float float1Down =
      std::nextafterf(float1, -std::numeric_limits<float>::infinity());
  float float2 = 2.0f;
  float float2Up =
      std::nextafterf(float2, std::numeric_limits<float>::infinity());

  double baseMax = static_cast<double>(float2);
  double baseMaxUp =
      std::nextafter<double>(baseMax, std::numeric_limits<double>::infinity());
  double baseMaxDown =
      std::nextafter<double>(baseMax, -std::numeric_limits<double>::infinity());
  double baseMin = static_cast<double>(float1);
  double baseMinUp =
      std::nextafter<double>(baseMin, std::numeric_limits<double>::infinity());
  double baseMinDown =
      std::nextafter<double>(baseMin, -std::numeric_limits<double>::infinity());

  makeIndex(
      std::vector<Envelope>{
          Envelope::from(baseMin, baseMin, baseMax, baseMax, 1),
          Envelope::from(baseMinUp, baseMinUp, baseMaxUp, baseMaxUp, 2),
          Envelope::from(baseMinDown, baseMinDown, baseMaxDown, baseMaxDown, 3),
      });

  Envelope bounds = indexBounds();
  ASSERT_EQ(bounds.minX, float1Down);
  ASSERT_EQ(bounds.minY, float1Down);
  ASSERT_EQ(bounds.maxX, float2Up);
  ASSERT_EQ(bounds.maxY, float2Up);

  assertQuery(2.1, 2.1, 2.1, 2.1, {});
  assertQuery(baseMin, baseMin, baseMin, baseMin, {1, 2, 3});
  assertQuery(baseMinDown, baseMinDown, baseMinDown, baseMinDown, {1, 2, 3});
  assertQuery(baseMinUp, baseMinUp, baseMinUp, baseMinUp, {1, 2, 3});
  assertQuery(baseMax, baseMax, baseMax, baseMax, {1, 2, 3});
  assertQuery(baseMaxDown, baseMaxDown, baseMaxDown, baseMaxDown, {1, 2, 3});
  assertQuery(baseMaxUp, baseMaxUp, baseMaxUp, baseMaxUp, {1, 2, 3});
}

TEST_F(SpatialIndexTest, testFloatImprecisionSubnormal) {
  // Check that our bumping rules work for subnormal floats as well.
  float subnormalFloatDown =
      std::nextafterf(0.0, -std::numeric_limits<float>::infinity());
  float subnormalFloatUp =
      std::nextafterf(0.0, std::numeric_limits<float>::infinity());

  double subnormalDoubleDown =
      std::nextafter<double>(0.0, -std::numeric_limits<double>::infinity());
  double subnormalDoubleUp =
      std::nextafter<double>(0.0, std::numeric_limits<double>::infinity());

  makeIndex(
      std::vector<Envelope>{
          Envelope::from(0.0, 0.0, 0.0, 0.0, 1),
          Envelope::from(
              subnormalDoubleDown,
              subnormalDoubleDown,
              subnormalDoubleDown,
              subnormalDoubleDown,
              2),
          Envelope::from(
              subnormalDoubleUp,
              subnormalDoubleUp,
              subnormalDoubleUp,
              subnormalDoubleUp,
              3),
          Envelope::from(
              subnormalDoubleDown,
              subnormalDoubleDown,
              subnormalDoubleUp,
              subnormalDoubleUp,
              4),
      });

  Envelope bounds = indexBounds();
  ASSERT_EQ(bounds.minX, subnormalFloatDown);
  ASSERT_EQ(bounds.minY, subnormalFloatDown);
  ASSERT_EQ(bounds.maxX, subnormalFloatUp);
  ASSERT_EQ(bounds.maxY, subnormalFloatUp);

  assertQuery(0.1, 0.1, 0.1, 0.1, {});
  assertQuery(0.0, 0.0, 0.0, 0.0, {1, 2, 3, 4});
  assertQuery(
      subnormalDoubleDown,
      subnormalDoubleDown,
      subnormalDoubleDown,
      subnormalDoubleDown,
      {1, 2, 3, 4});
  assertQuery(
      subnormalDoubleUp,
      subnormalDoubleUp,
      subnormalDoubleUp,
      subnormalDoubleUp,
      {1, 2, 3, 4});
  assertQuery(
      subnormalDoubleDown,
      subnormalDoubleDown,
      subnormalDoubleUp,
      subnormalDoubleUp,
      {1, 2, 3, 4});
}

TEST_F(SpatialIndexTest, testNegativeCoordinates) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{
              .minX = -5, .minY = -5, .maxX = -1, .maxY = -1, .rowIndex = 0},
          Envelope{
              .minX = -10, .minY = -10, .maxX = -6, .maxY = -6, .rowIndex = 1},
          Envelope{
              .minX = -3, .minY = -8, .maxX = 2, .maxY = -4, .rowIndex = 2}});

  Envelope bounds = indexBounds();
  ASSERT_EQ(bounds.minX, -10);
  ASSERT_EQ(bounds.minY, -10);
  ASSERT_EQ(bounds.maxX, 2);
  ASSERT_EQ(bounds.maxY, -1);

  assertQuery(-7, -7, -7, -7, {1});
  assertQuery(-2, -5, -2, -5, {0, 2});
  assertQuery(0, 0, 1, 1, {});
}

TEST_F(SpatialIndexTest, testOverlappingEnvelopes) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{.minX = 0, .minY = 0, .maxX = 10, .maxY = 10, .rowIndex = 0},
          Envelope{.minX = 5, .minY = 5, .maxX = 15, .maxY = 15, .rowIndex = 1},
          Envelope{.minX = 2, .minY = 2, .maxX = 8, .maxY = 8, .rowIndex = 2},
          Envelope{
              .minX = 7, .minY = 7, .maxX = 12, .maxY = 12, .rowIndex = 3}});

  assertQuery(6, 6, 6, 6, {0, 1, 2});
  assertQuery(8, 8, 8, 8, {0, 1, 2, 3});
  assertQuery(9, 9, 9, 9, {0, 1, 3});
  assertQuery(3, 3, 3, 3, {0, 2});
  assertQuery(13, 13, 13, 13, {1});
}

TEST_F(SpatialIndexTest, testNonOverlappingEnvelopes) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{.minX = 0, .minY = 0, .maxX = 1, .maxY = 1, .rowIndex = 0},
          Envelope{.minX = 2, .minY = 2, .maxX = 3, .maxY = 3, .rowIndex = 1},
          Envelope{.minX = 4, .minY = 4, .maxX = 5, .maxY = 5, .rowIndex = 2},
          Envelope{.minX = 6, .minY = 6, .maxX = 7, .maxY = 7, .rowIndex = 3}});

  assertQuery(0.5, 0.5, 0.5, 0.5, {0});
  assertQuery(2.5, 2.5, 2.5, 2.5, {1});
  assertQuery(4.5, 4.5, 4.5, 4.5, {2});
  assertQuery(6.5, 6.5, 6.5, 6.5, {3});
  assertQuery(1.5, 1.5, 1.5, 1.5, {});
}

TEST_F(SpatialIndexTest, testLargeQueryEnvelope) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{.minX = 1, .minY = 1, .maxX = 2, .maxY = 2, .rowIndex = 0},
          Envelope{.minX = 3, .minY = 3, .maxX = 4, .maxY = 4, .rowIndex = 1},
          Envelope{.minX = 5, .minY = 5, .maxX = 6, .maxY = 6, .rowIndex = 2}});

  assertQuery(0, 0, 10, 10, {0, 1, 2});
  assertQuery(-100, -100, 100, 100, {0, 1, 2});
}

TEST_F(SpatialIndexTest, testSmallQueryEnvelope) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{
              .minX = 0, .minY = 0, .maxX = 100, .maxY = 100, .rowIndex = 0},
          Envelope{
              .minX = 50,
              .minY = 50,
              .maxX = 150,
              .maxY = 150,
              .rowIndex = 1}});

  assertQuery(25, 25, 26, 26, {0});
  assertQuery(75, 75, 76, 76, {0, 1});
  assertQuery(125, 125, 126, 126, {1});
  assertQuery(0.1, 0.1, 0.2, 0.2, {0});
}

TEST_F(SpatialIndexTest, testEdgeTouching) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{.minX = 0, .minY = 0, .maxX = 5, .maxY = 5, .rowIndex = 0},
          Envelope{.minX = 5, .minY = 0, .maxX = 10, .maxY = 5, .rowIndex = 1},
          Envelope{.minX = 0, .minY = 5, .maxX = 5, .maxY = 10, .rowIndex = 2},
          Envelope{
              .minX = 5, .minY = 5, .maxX = 10, .maxY = 10, .rowIndex = 3}});

  assertQuery(5, 5, 5, 5, {0, 1, 2, 3});
  assertQuery(5, 2, 5, 2, {0, 1});
  assertQuery(2, 5, 2, 5, {0, 2});
}

TEST_F(SpatialIndexTest, testCornerTouching) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{.minX = 0, .minY = 0, .maxX = 5, .maxY = 5, .rowIndex = 0},
          Envelope{
              .minX = 5, .minY = 5, .maxX = 10, .maxY = 10, .rowIndex = 1}});

  assertQuery(5, 5, 5, 5, {0, 1});
  assertQuery(4.9, 4.9, 5.1, 5.1, {0, 1});
}

TEST_F(SpatialIndexTest, testInfiniteValues) {
  float inf = std::numeric_limits<float>::infinity();
  float negInf = -std::numeric_limits<float>::infinity();

  makeIndex(
      std::vector<Envelope>{
          Envelope{.minX = 0, .minY = 0, .maxX = 1, .maxY = 1, .rowIndex = 0}});

  assertQuery(inf, inf, inf, inf, {});
  assertQuery(negInf, negInf, negInf, negInf, {});
  assertQuery(negInf, negInf, inf, inf, {0});
}

TEST_F(SpatialIndexTest, testLargeDataset) {
  std::vector<Envelope> envelopes;
  envelopes.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    envelopes.push_back(
        Envelope{
            .minX = static_cast<float>(i),
            .minY = static_cast<float>(i),
            .maxX = static_cast<float>(i + 1),
            .maxY = static_cast<float>(i + 1),
            .rowIndex = i});
  }
  makeIndex(std::move(envelopes));

  assertQuery(500.5, 500.5, 500.5, 500.5, {500});
  assertQuery(100.5, 100.5, 104.5, 104.5, {100, 101, 102, 103, 104});
  assertQuery(-1, -1, -1, -1, {});
  assertQuery(1001, 1001, 1001, 1001, {});

  Envelope bounds = indexBounds();
  ASSERT_EQ(bounds.minX, 0);
  ASSERT_EQ(bounds.minY, 0);
  ASSERT_EQ(bounds.maxX, 1000);
  ASSERT_EQ(bounds.maxY, 1000);
}

TEST_F(SpatialIndexTest, testVeryLargeCoordinates) {
  float largeVal = 1e20f;
  makeIndex(
      std::vector<Envelope>{
          Envelope{
              .minX = -largeVal,
              .minY = -largeVal,
              .maxX = -largeVal + 1,
              .maxY = -largeVal + 1,
              .rowIndex = 0},
          Envelope{
              .minX = largeVal - 1,
              .minY = largeVal - 1,
              .maxX = largeVal,
              .maxY = largeVal,
              .rowIndex = 1}});

  assertQuery(
      -largeVal + 0.5, -largeVal + 0.5, -largeVal + 0.5, -largeVal + 0.5, {0});
  assertQuery(
      largeVal - 0.5, largeVal - 0.5, largeVal - 0.5, largeVal - 0.5, {1});
  assertQuery(0, 0, 0, 0, {});
}

TEST_F(SpatialIndexTest, testQueryOutsideBounds) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{.minX = 0, .minY = 0, .maxX = 10, .maxY = 10, .rowIndex = 0},
          Envelope{
              .minX = 5, .minY = 5, .maxX = 15, .maxY = 15, .rowIndex = 1}});

  assertQuery(-10, -10, -5, -5, {});
  assertQuery(20, 20, 25, 25, {});
  assertQuery(-10, 5, -5, 10, {});
  assertQuery(5, 20, 10, 25, {});
}

TEST_F(SpatialIndexTest, testPartialOverlap) {
  makeIndex(
      std::vector<Envelope>{Envelope{
          .minX = 0, .minY = 0, .maxX = 10, .maxY = 10, .rowIndex = 0}});

  assertQuery(-5, -5, 5, 5, {0});
  assertQuery(5, -5, 15, 5, {0});
  assertQuery(-5, 5, 5, 15, {0});
  assertQuery(5, 5, 15, 15, {0});
}

TEST_F(SpatialIndexTest, testMixedSizeEnvelopes) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{
              .minX = 0, .minY = 0, .maxX = 0.1, .maxY = 0.1, .rowIndex = 0},
          Envelope{
              .minX = 1, .minY = 1, .maxX = 100, .maxY = 100, .rowIndex = 1},
          Envelope{
              .minX = 50, .minY = 50, .maxX = 51, .maxY = 51, .rowIndex = 2}});

  assertQuery(0.05, 0.05, 0.05, 0.05, {0});
  assertQuery(50, 50, 100, 100, {1, 2});
  assertQuery(25, 25, 25, 25, {1});
}

TEST_F(SpatialIndexTest, testZeroAreaEnvelopes) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{.minX = 0, .minY = 0, .maxX = 0, .maxY = 0, .rowIndex = 0},
          Envelope{.minX = 1, .minY = 1, .maxX = 1, .maxY = 1, .rowIndex = 1},
          Envelope{.minX = 2, .minY = 2, .maxX = 2, .maxY = 2, .rowIndex = 2}});

  assertQuery(0, 0, 0, 0, {0});
  assertQuery(1, 1, 1, 1, {1});
  assertQuery(2, 2, 2, 2, {2});
  assertQuery(0.5, 0.5, 0.5, 0.5, {});
  assertQuery(0, 0, 2, 2, {0, 1, 2});
}

TEST_F(SpatialIndexTest, testIdenticalEnvelopes) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{.minX = 5, .minY = 5, .maxX = 10, .maxY = 10, .rowIndex = 0},
          Envelope{.minX = 5, .minY = 5, .maxX = 10, .maxY = 10, .rowIndex = 1},
          Envelope{
              .minX = 5, .minY = 5, .maxX = 10, .maxY = 10, .rowIndex = 2}});

  assertQuery(7, 7, 7, 7, {0, 1, 2});
  assertQuery(5, 5, 10, 10, {0, 1, 2});
  assertQuery(4, 4, 4, 4, {});
}

TEST_F(SpatialIndexTest, testDifferentBranchSizes) {
  std::vector<uint32_t> branchSizes = {2, 3, 4, 8, 16, 32, 64, 128, 256};

  for (uint32_t branchSize : branchSizes) {
    std::vector<Envelope> envelopes;
    envelopes.reserve(100);
    for (int i = 0; i < 100; ++i) {
      envelopes.push_back(
          Envelope{
              .minX = static_cast<float>(i),
              .minY = static_cast<float>(i),
              .maxX = static_cast<float>(i + 1),
              .maxY = static_cast<float>(i + 1),
              .rowIndex = i});
    }
    makeIndex(std::move(envelopes), branchSize);

    assertQuery(50.5, 50.5, 50.5, 50.5, {50});
    assertQuery(10.5, 10.5, 14.5, 14.5, {10, 11, 12, 13, 14});
    assertQuery(-1, -1, -1, -1, {});
    assertQuery(101, 101, 101, 101, {});

    Envelope bounds = indexBounds();
    ASSERT_EQ(bounds.minX, 0);
    ASSERT_EQ(bounds.minY, 0);
    ASSERT_EQ(bounds.maxX, 100);
    ASSERT_EQ(bounds.maxY, 100);
  }
}

TEST_F(SpatialIndexTest, testSmallBranchSize) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{.minX = 0, .minY = 0, .maxX = 10, .maxY = 10, .rowIndex = 0},
          Envelope{.minX = 5, .minY = 5, .maxX = 15, .maxY = 15, .rowIndex = 1},
          Envelope{.minX = 2, .minY = 2, .maxX = 8, .maxY = 8, .rowIndex = 2},
          Envelope{
              .minX = 7, .minY = 7, .maxX = 12, .maxY = 12, .rowIndex = 3}},
      2);

  assertQuery(6, 6, 6, 6, {0, 1, 2});
  assertQuery(8, 8, 8, 8, {0, 1, 2, 3});
  assertQuery(9, 9, 9, 9, {0, 1, 3});
  assertQuery(3, 3, 3, 3, {0, 2});
  assertQuery(13, 13, 13, 13, {1});
}

TEST_F(SpatialIndexTest, testLargeBranchSize) {
  makeIndex(
      std::vector<Envelope>{
          Envelope{.minX = 0, .minY = 0, .maxX = 10, .maxY = 10, .rowIndex = 0},
          Envelope{.minX = 5, .minY = 5, .maxX = 15, .maxY = 15, .rowIndex = 1},
          Envelope{.minX = 2, .minY = 2, .maxX = 8, .maxY = 8, .rowIndex = 2},
          Envelope{
              .minX = 7, .minY = 7, .maxX = 12, .maxY = 12, .rowIndex = 3}},
      512);

  assertQuery(6, 6, 6, 6, {0, 1, 2});
  assertQuery(8, 8, 8, 8, {0, 1, 2, 3});
  assertQuery(9, 9, 9, 9, {0, 1, 3});
  assertQuery(3, 3, 3, 3, {0, 2});
  assertQuery(13, 13, 13, 13, {1});
}

} // namespace facebook::velox::exec::test
