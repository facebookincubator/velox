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

#include "velox/exec/HilbertIndex.h"
#include <gtest/gtest.h>
#include <limits>

using namespace ::testing;
using namespace facebook::velox::exec;

namespace facebook::velox::exec::test {

class HilbertIndexTest : public virtual testing::Test {};

TEST_F(HilbertIndexTest, testOrder) {
  HilbertIndex hilbert(0, 0, 4, 4);

  uint32_t h0 = hilbert.indexOf(0.0, 0.0);
  uint32_t h1 = hilbert.indexOf(1.0, 1.0);
  uint32_t h2 = hilbert.indexOf(1.0, 3.0);
  uint32_t h3 = hilbert.indexOf(3.0, 3.0);
  uint32_t h4 = hilbert.indexOf(3.0, 1.0);

  ASSERT_LT(h0, h1);
  ASSERT_LT(h1, h2);
  ASSERT_LT(h2, h3);
  ASSERT_LT(h3, h4);
}

TEST_F(HilbertIndexTest, testOutOfBounds) {
  HilbertIndex hilbert(0, 0, 1, 1);

  ASSERT_EQ(hilbert.indexOf(2.0, 2.0), std::numeric_limits<uint32_t>::max());
}

TEST_F(HilbertIndexTest, testDegenerateRectangle) {
  HilbertIndex hilbert(0, 0, 0, 0);

  ASSERT_EQ(hilbert.indexOf(0.0, 0.0), 0);
  ASSERT_EQ(hilbert.indexOf(2.0, 2.0), std::numeric_limits<uint32_t>::max());
}

TEST_F(HilbertIndexTest, testDegenerateHorizontalRectangle) {
  HilbertIndex hilbert(0, 0, 4, 0);

  ASSERT_EQ(hilbert.indexOf(0.0, 0.0), 0);
  ASSERT_LT(hilbert.indexOf(1.0, 0.0), hilbert.indexOf(2.0, 0.0));
  ASSERT_EQ(hilbert.indexOf(0.0, 2.0), std::numeric_limits<uint32_t>::max());
  ASSERT_EQ(hilbert.indexOf(2.0, 2.0), std::numeric_limits<uint32_t>::max());
}

TEST_F(HilbertIndexTest, testDegenerateVerticalRectangle) {
  HilbertIndex hilbert(0, 0, 0, 4);

  ASSERT_EQ(hilbert.indexOf(0.0, 0.0), 0);
  ASSERT_LT(hilbert.indexOf(0.0, 1.0), hilbert.indexOf(0.0, 2.0));
  ASSERT_EQ(hilbert.indexOf(2.0, 0.0), std::numeric_limits<uint32_t>::max());
  ASSERT_EQ(hilbert.indexOf(2.0, 2.0), std::numeric_limits<uint32_t>::max());
}

TEST_F(HilbertIndexTest, testNegativeCoordinates) {
  HilbertIndex hilbert(-10, -10, 10, 10);

  uint32_t h0 = hilbert.indexOf(-5.0, -5.0);
  uint32_t h1 = hilbert.indexOf(0.0, 0.0);
  uint32_t h2 = hilbert.indexOf(5.0, 5.0);

  ASSERT_LT(h0, h1);
  ASSERT_LT(h1, h2);

  ASSERT_EQ(
      hilbert.indexOf(-15.0, -15.0), std::numeric_limits<uint32_t>::max());
  ASSERT_EQ(hilbert.indexOf(15.0, 15.0), std::numeric_limits<uint32_t>::max());
}

TEST_F(HilbertIndexTest, testFloatingPointPrecision) {
  HilbertIndex hilbert(0, 0, 1, 1);

  uint32_t h1 = hilbert.indexOf(0.1, 0.1);
  uint32_t h2 = hilbert.indexOf(0.2, 0.2);
  uint32_t h3 = hilbert.indexOf(0.9, 0.9);

  ASSERT_LT(h1, h2);
  ASSERT_LT(h2, h3);
}

TEST_F(HilbertIndexTest, testBoundaryPoints) {
  HilbertIndex hilbert(0, 0, 10, 10);

  uint32_t h0 = hilbert.indexOf(0.0, 0.0);
  uint32_t h1 = hilbert.indexOf(10.0, 10.0);
  uint32_t h2 = hilbert.indexOf(0.0, 10.0);
  // Bottom-right corner is at the end of the range, so may be MAX

  ASSERT_NE(h0, std::numeric_limits<uint32_t>::max());
  ASSERT_NE(h1, std::numeric_limits<uint32_t>::max());
  ASSERT_NE(h2, std::numeric_limits<uint32_t>::max());
}

TEST_F(HilbertIndexTest, testLargeCoordinates) {
  HilbertIndex hilbert(0, 0, 1000000, 1000000);

  uint32_t h1 = hilbert.indexOf(100000, 100000);
  uint32_t h2 = hilbert.indexOf(500000, 500000);
  uint32_t h3 = hilbert.indexOf(900000, 900000);

  ASSERT_LT(h1, h2);
  ASSERT_LT(h2, h3);
}

TEST_F(HilbertIndexTest, testDensityInSmallRegion) {
  HilbertIndex hilbert(0, 0, 100, 100);

  std::vector<uint32_t> indices;
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      indices.push_back(hilbert.indexOf(i * 10.0f + 5.0f, j * 10.0f + 5.0f));
    }
  }

  std::set<uint32_t> uniqueIndices(indices.begin(), indices.end());
  ASSERT_EQ(indices.size(), 100);
  ASSERT_GT(uniqueIndices.size(), 90);
}

TEST_F(HilbertIndexTest, testSpatialLocality) {
  HilbertIndex hilbert(0, 0, 100, 100);

  uint32_t h1 = hilbert.indexOf(50.0, 50.0);
  uint32_t h2 = hilbert.indexOf(50.1, 50.1);
  uint32_t h3 = hilbert.indexOf(50.2, 50.2);
  uint32_t h4 = hilbert.indexOf(90.0, 90.0);

  uint32_t diff12 = std::abs(static_cast<int32_t>(h1 - h2));
  uint32_t diff23 = std::abs(static_cast<int32_t>(h2 - h3));
  uint32_t diff14 = std::abs(static_cast<int32_t>(h1 - h4));

  ASSERT_LT(diff12, diff14);
  ASSERT_LT(diff23, diff14);
}

TEST_F(HilbertIndexTest, testIdenticalPoints) {
  HilbertIndex hilbert(0, 0, 10, 10);

  uint32_t h1 = hilbert.indexOf(5.0, 5.0);
  uint32_t h2 = hilbert.indexOf(5.0, 5.0);

  ASSERT_EQ(h1, h2);
}

TEST_F(HilbertIndexTest, testExtremelySmallBounds) {
  HilbertIndex hilbert(0, 0, 0.001, 0.001);

  uint32_t h1 = hilbert.indexOf(0.0, 0.0);
  uint32_t h2 = hilbert.indexOf(0.0005, 0.0005);

  ASSERT_NE(h1, std::numeric_limits<uint32_t>::max());
  ASSERT_NE(h2, std::numeric_limits<uint32_t>::max());
}

} // namespace facebook::velox::exec::test
