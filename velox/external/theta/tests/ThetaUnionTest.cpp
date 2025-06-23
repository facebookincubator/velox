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

// Adapted from Apache DataSketches

#include "velox/external/theta/ThetaUnion.h"
#include "TestUtils.h"

#include <gtest/gtest.h>
#include <stdexcept>

namespace facebook::velox::common::theta {

TEST(ThetaUnion, empty) {
  updateThetaSketch sketch1 = updateThetaSketch::builder().build();
  ThetaUnion u = ThetaUnion::builder().build();
  compactThetaSketch sketch2 = u.getResult();
  EXPECT_TRUE(sketch2.getNumRetained() == 0);
  EXPECT_TRUE(sketch2.isEmpty());
  EXPECT_FALSE(sketch2.isEstimationMode());

  u.update(sketch1);
  sketch2 = u.getResult();
  EXPECT_TRUE(sketch2.getNumRetained() == 0);
  EXPECT_TRUE(sketch2.isEmpty());
  EXPECT_FALSE(sketch2.isEstimationMode());
}

TEST(ThetaUnion, nonEmptyNoRetainedKeys) {
  updateThetaSketch update_sketch =
      updateThetaSketch::builder().setP(0.001f).build();
  update_sketch.update(1);
  ThetaUnion u = ThetaUnion::builder().build();
  u.update(update_sketch);
  compactThetaSketch sketch = u.getResult();
  EXPECT_TRUE(sketch.getNumRetained() == 0);
  EXPECT_FALSE(sketch.isEmpty());
  EXPECT_TRUE(sketch.isEstimationMode());
  EXPECT_TRUE(sketch.getTheta() == Approx(0.001).margin(1e-10));
}

TEST(ThetaUnion, exactModeHalfOverlap) {
  auto sketch1 = updateThetaSketch::builder().build();
  int value = 0;
  for (int i = 0; i < 1000; i++)
    sketch1.update(value++);

  auto sketch2 = updateThetaSketch::builder().build();
  value = 500;
  for (int i = 0; i < 1000; i++)
    sketch2.update(value++);

  auto u = ThetaUnion::builder().build();
  u.update(sketch1);
  u.update(sketch2);
  auto sketch3 = u.getResult();
  EXPECT_FALSE(sketch3.isEmpty());
  EXPECT_FALSE(sketch3.isEstimationMode());
  EXPECT_TRUE(sketch3.getEstimate() == 1500.0);

  u.reset();
  sketch3 = u.getResult();
  EXPECT_TRUE(sketch3.getNumRetained() == 0);
  EXPECT_TRUE(sketch3.isEmpty());
  EXPECT_FALSE(sketch3.isEstimationMode());
}

TEST(ThetaUnion, exactModeHalfOverlapWrappedCompact) {
  auto sketch1 = updateThetaSketch::builder().build();
  int value = 0;
  for (int i = 0; i < 1000; i++)
    sketch1.update(value++);
  auto bytes1 = sketch1.compact().serialize();

  auto sketch2 = updateThetaSketch::builder().build();
  value = 500;
  for (int i = 0; i < 1000; i++)
    sketch2.update(value++);
  auto bytes2 = sketch2.compact().serialize();

  auto u = ThetaUnion::builder().build();
  u.update(wrappedCompactThetaSketch::wrap(bytes1.data(), bytes1.size()));
  u.update(wrappedCompactThetaSketch::wrap(bytes2.data(), bytes2.size()));
  compactThetaSketch sketch3 = u.getResult();
  EXPECT_FALSE(sketch3.isEmpty());
  EXPECT_FALSE(sketch3.isEstimationMode());
  EXPECT_TRUE(sketch3.getEstimate() == 1500.0);
}

TEST(ThetaUnion, estimationModeHalfOverlap) {
  auto sketch1 = updateThetaSketch::builder().build();
  int value = 0;
  for (int i = 0; i < 10000; i++)
    sketch1.update(value++);

  auto sketch2 = updateThetaSketch::builder().build();
  value = 5000;
  for (int i = 0; i < 10000; i++)
    sketch2.update(value++);

  auto u = ThetaUnion::builder().build();
  u.update(sketch1);
  u.update(sketch2);
  auto sketch3 = u.getResult();
  EXPECT_FALSE(sketch3.isEmpty());
  EXPECT_TRUE(sketch3.isEstimationMode());
  EXPECT_TRUE(sketch3.getEstimate() == Approx(15000).margin(15000 * 0.01));

  u.reset();
  sketch3 = u.getResult();
  EXPECT_TRUE(sketch3.getNumRetained() == 0);
  EXPECT_TRUE(sketch3.isEmpty());
  EXPECT_FALSE(sketch3.isEstimationMode());
}

TEST(ThetaUnion, seedMismatch) {
  updateThetaSketch sketch = updateThetaSketch::builder().build();
  sketch.update(1); // non-empty should not be ignored
  ThetaUnion u = ThetaUnion::builder().setSeed(123).build();
  EXPECT_THROW(u.update(sketch), VeloxRuntimeError);
}

TEST(ThetaUnion, largerK) {
  auto update_sketch1 = updateThetaSketch::builder().set_lg_k(14).build();
  for (int i = 0; i < 16384; ++i)
    update_sketch1.update(i);

  auto update_sketch2 = updateThetaSketch::builder().set_lg_k(14).build();
  for (int i = 0; i < 26384; ++i)
    update_sketch2.update(i);

  auto update_sketch3 = updateThetaSketch::builder().set_lg_k(14).build();
  for (int i = 0; i < 86384; ++i)
    update_sketch3.update(i);

  auto union1 = ThetaUnion::builder().set_lg_k(16).build();
  union1.update(update_sketch2);
  union1.update(update_sketch1);
  union1.update(update_sketch3);
  auto result1 = union1.getResult();
  EXPECT_TRUE(result1.getEstimate() == update_sketch3.getEstimate());

  auto union2 = ThetaUnion::builder().set_lg_k(16).build();
  union2.update(update_sketch1);
  union2.update(update_sketch3);
  union2.update(update_sketch2);
  auto result2 = union2.getResult();
  EXPECT_TRUE(result2.getEstimate() == update_sketch3.getEstimate());
}

} // namespace facebook::velox::common::theta
