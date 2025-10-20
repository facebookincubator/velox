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

#include "velox/exec/SpatialIndex.h"
#include <gtest/gtest.h>
#include <algorithm>

using namespace ::testing;
using namespace facebook::velox::exec;

namespace facebook::velox::exec::test {

class SpatialIndexTest : public virtual testing::Test {
 protected:
  SpatialIndex index_;

  void makeIndex(std::vector<Envelope> envelopes) {
    index_ = SpatialIndex(std::move(envelopes));
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

TEST_F(SpatialIndexTest, testPointProbe) {
  makeIndex(std::vector<Envelope>{
      Envelope{.minX = 1, .minY = 0, .maxX = 1, .maxY = 0, .rowIndex = 6},
      Envelope{.minX = 0, .minY = 0, .maxX = 0, .maxY = 0, .rowIndex = 5},
      Envelope{.minX = 0, .minY = 0, .maxX = 1, .maxY = 1, .rowIndex = 4},
      Envelope{.minX = -1, .minY = -1, .maxX = 0, .maxY = 0, .rowIndex = 3},
      Envelope{.minX = -1, .minY = -1, .maxX = 1, .maxY = 1, .rowIndex = 2},
      Envelope{.minX = 0.5, .minY = 0.5, .maxX = 1, .maxY = 1, .rowIndex = 1},
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

  makeIndex(std::vector<Envelope>{
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

  makeIndex(std::vector<Envelope>{
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

} // namespace facebook::velox::exec::test
