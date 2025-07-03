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

#include "velox/external/theta/ThetaSketch.h"
#include "TestUtils.h"

#include <gtest/gtest.h>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace facebook::velox::common::theta {

const std::string inputPath = "test_sketch_files/";

TEST(ThetaSketch, Empty) {
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  EXPECT_TRUE(update_sketch.isEmpty());
  EXPECT_FALSE(update_sketch.isEstimationMode());
  EXPECT_TRUE(update_sketch.getTheta() == 1.0);
  EXPECT_TRUE(update_sketch.getEstimate() == 0.0);
  EXPECT_TRUE(update_sketch.getLowerBound(1) == 0.0);
  EXPECT_TRUE(update_sketch.getUpperBound(1) == 0.0);
  EXPECT_TRUE(update_sketch.isOrdered());

  compactThetaSketch compact_sketch = update_sketch.compact();
  EXPECT_TRUE(compact_sketch.isEmpty());
  EXPECT_FALSE(compact_sketch.isEstimationMode());
  EXPECT_TRUE(compact_sketch.getTheta() == 1.0);
  EXPECT_TRUE(compact_sketch.getEstimate() == 0.0);
  EXPECT_TRUE(compact_sketch.getLowerBound(1) == 0.0);
  EXPECT_TRUE(compact_sketch.getUpperBound(1) == 0.0);
  EXPECT_TRUE(compact_sketch.isOrdered());

  // empty is forced to be ordered
  EXPECT_TRUE(update_sketch.compact(false).isOrdered());
}

TEST(ThetaSketch, NonEmptyNoRetainedKeys) {
  updateThetaSketch update_sketch =
      updateThetaSketch::builder().setP(0.001f).build();
  update_sketch.update(1);
  EXPECT_TRUE(update_sketch.getNumRetained() == 0);
  EXPECT_FALSE(update_sketch.isEmpty());
  EXPECT_TRUE(update_sketch.isEstimationMode());
  EXPECT_TRUE(update_sketch.getEstimate() == 0.0);
  EXPECT_TRUE(update_sketch.getLowerBound(1) == 0.0);
  EXPECT_TRUE(update_sketch.getUpperBound(1) > 0);

  compactThetaSketch compact_sketch = update_sketch.compact();
  EXPECT_TRUE(compact_sketch.getNumRetained() == 0);
  EXPECT_FALSE(compact_sketch.isEmpty());
  EXPECT_TRUE(compact_sketch.isEstimationMode());
  EXPECT_TRUE(compact_sketch.getEstimate() == 0.0);
  EXPECT_TRUE(compact_sketch.getLowerBound(1) == 0.0);
  EXPECT_TRUE(compact_sketch.getUpperBound(1) > 0);

  update_sketch.reset();
  EXPECT_TRUE(update_sketch.isEmpty());
  EXPECT_FALSE(update_sketch.isEstimationMode());
  EXPECT_TRUE(update_sketch.getTheta() == 1.0);
  EXPECT_TRUE(update_sketch.getEstimate() == 0.0);
  EXPECT_TRUE(update_sketch.getLowerBound(1) == 0.0);
  EXPECT_TRUE(update_sketch.getUpperBound(1) == 0.0);
}

TEST(ThetaSketch, SingleItem) {
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  update_sketch.update(1);
  EXPECT_FALSE(update_sketch.isEmpty());
  EXPECT_FALSE(update_sketch.isEstimationMode());
  EXPECT_TRUE(update_sketch.getTheta() == 1.0);
  EXPECT_TRUE(update_sketch.getEstimate() == 1.0);
  EXPECT_TRUE(update_sketch.getLowerBound(1) == 1.0);
  EXPECT_TRUE(update_sketch.getUpperBound(1) == 1.0);
  EXPECT_TRUE(update_sketch.isOrdered()); // one item is ordered

  compactThetaSketch compact_sketch = update_sketch.compact();
  EXPECT_FALSE(compact_sketch.isEmpty());
  EXPECT_FALSE(compact_sketch.isEstimationMode());
  EXPECT_TRUE(compact_sketch.getTheta() == 1.0);
  EXPECT_TRUE(compact_sketch.getEstimate() == 1.0);
  EXPECT_TRUE(compact_sketch.getLowerBound(1) == 1.0);
  EXPECT_TRUE(compact_sketch.getUpperBound(1) == 1.0);
  EXPECT_TRUE(compact_sketch.isOrdered());

  // single item is forced to be ordered
  EXPECT_TRUE(update_sketch.compact(false).isOrdered());
}

TEST(ThetaSketch, ResizeExact) {
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  for (int i = 0; i < 2000; i++)
    update_sketch.update(i);
  EXPECT_FALSE(update_sketch.isEmpty());
  EXPECT_FALSE(update_sketch.isEstimationMode());
  EXPECT_TRUE(update_sketch.getTheta() == 1.0);
  EXPECT_TRUE(update_sketch.getEstimate() == 2000.0);
  EXPECT_TRUE(update_sketch.getLowerBound(1) == 2000.0);
  EXPECT_TRUE(update_sketch.getUpperBound(1) == 2000.0);
  EXPECT_FALSE(update_sketch.isOrdered());

  compactThetaSketch compact_sketch = update_sketch.compact();
  EXPECT_FALSE(compact_sketch.isEmpty());
  EXPECT_FALSE(compact_sketch.isEstimationMode());
  EXPECT_TRUE(compact_sketch.getTheta() == 1.0);
  EXPECT_TRUE(compact_sketch.getEstimate() == 2000.0);
  EXPECT_TRUE(compact_sketch.getLowerBound(1) == 2000.0);
  EXPECT_TRUE(compact_sketch.getUpperBound(1) == 2000.0);
  EXPECT_TRUE(compact_sketch.isOrdered());

  update_sketch.reset();
  EXPECT_TRUE(update_sketch.isEmpty());
  EXPECT_FALSE(update_sketch.isEstimationMode());
  EXPECT_TRUE(update_sketch.getTheta() == 1.0);
  EXPECT_TRUE(update_sketch.getEstimate() == 0.0);
  EXPECT_TRUE(update_sketch.getLowerBound(1) == 0.0);
  EXPECT_TRUE(update_sketch.getUpperBound(1) == 0.0);
  EXPECT_TRUE(update_sketch.isOrdered());
}

TEST(ThetaSketch, estimation) {
  updateThetaSketch update_sketch =
      updateThetaSketch::builder()
          .setResizeFactor(updateThetaSketch::resizeFactor::X1)
          .build();
  const int n = 8000;
  for (int i = 0; i < n; i++)
    update_sketch.update(i);
  // std::cerr << update_sketch.to_string();
  EXPECT_FALSE(update_sketch.isEmpty());
  EXPECT_TRUE(update_sketch.isEstimationMode());
  EXPECT_TRUE(update_sketch.getTheta() < 1.0);
  EXPECT_TRUE(
      update_sketch.getEstimate() == Approx((double)n).margin(n * 0.01));
  EXPECT_TRUE(update_sketch.getLowerBound(1) < n);
  EXPECT_TRUE(update_sketch.getUpperBound(1) > n);

  const uint32_t k = 1 << ThetaConstants::DEFAULT_LG_K;
  EXPECT_TRUE(update_sketch.getNumRetained() >= k);
  update_sketch.trim();
  EXPECT_TRUE(update_sketch.getNumRetained() == k);

  compactThetaSketch compact_sketch = update_sketch.compact();
  EXPECT_FALSE(compact_sketch.isEmpty());
  EXPECT_TRUE(compact_sketch.isOrdered());
  EXPECT_TRUE(compact_sketch.isEstimationMode());
  EXPECT_TRUE(compact_sketch.getTheta() < 1.0);
  EXPECT_TRUE(
      compact_sketch.getEstimate() == Approx((double)n).margin(n * 0.01));
  EXPECT_TRUE(compact_sketch.getLowerBound(1) < n);
  EXPECT_TRUE(compact_sketch.getUpperBound(1) > n);
}

TEST(ThetaSketch, DeserializeCompactV1EmptyFromJava) {
  std::ifstream is;
  is.exceptions(std::ios::failbit | std::ios::badbit);
  is.open(inputPath + "theta_compact_empty_from_java_v1.sk", std::ios::binary);
  auto sketch = compactThetaSketch::deserialize(is);
  EXPECT_TRUE(sketch.isEmpty());
  EXPECT_FALSE(sketch.isEstimationMode());
  EXPECT_TRUE(sketch.getNumRetained() == 0);
  EXPECT_TRUE(sketch.getTheta() == 1.0);
  EXPECT_TRUE(sketch.getEstimate() == 0.0);
  EXPECT_TRUE(sketch.getLowerBound(1) == 0.0);
  EXPECT_TRUE(sketch.getUpperBound(1) == 0.0);
}

TEST(ThetaSketch, DeserializeCompactV2EmptyFromJava) {
  std::ifstream is;
  is.exceptions(std::ios::failbit | std::ios::badbit);
  is.open(inputPath + "theta_compact_empty_from_java_v2.sk", std::ios::binary);
  auto sketch = compactThetaSketch::deserialize(is);
  EXPECT_TRUE(sketch.isEmpty());
  EXPECT_FALSE(sketch.isEstimationMode());
  EXPECT_TRUE(sketch.getNumRetained() == 0);
  EXPECT_TRUE(sketch.getTheta() == 1.0);
  EXPECT_TRUE(sketch.getEstimate() == 0.0);
  EXPECT_TRUE(sketch.getLowerBound(1) == 0.0);
  EXPECT_TRUE(sketch.getUpperBound(1) == 0.0);
}

TEST(ThetaSketch, DeserializeCompactV1EstimationFromJava) {
  std::ifstream is;
  is.exceptions(std::ios::failbit | std::ios::badbit);
  is.open(
      inputPath + "theta_compact_estimation_from_java_v1.sk", std::ios::binary);
  auto sketch = compactThetaSketch::deserialize(is);
  EXPECT_FALSE(sketch.isEmpty());
  EXPECT_TRUE(sketch.isEstimationMode());
  EXPECT_TRUE(sketch.isOrdered());
  EXPECT_TRUE(sketch.getNumRetained() == 4342);
  EXPECT_TRUE(sketch.getTheta() == Approx(0.531700444213199).margin(1e-10));
  EXPECT_TRUE(sketch.getEstimate() == Approx(8166.25234614053).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(2) == Approx(7996.956955317471).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(2) == Approx(8339.090301078124).margin(1e-10));

  // the same construction process in Java must have produced exactly the same
  // sketch
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  const int n = 8192;
  for (int i = 0; i < n; i++)
    update_sketch.update(i);
  EXPECT_TRUE(sketch.getNumRetained() == update_sketch.getNumRetained());
  EXPECT_TRUE(
      sketch.getTheta() == Approx(update_sketch.getTheta()).margin(1e-10));
  EXPECT_TRUE(
      sketch.getEstimate() ==
      Approx(update_sketch.getEstimate()).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(1) ==
      Approx(update_sketch.getLowerBound(1)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(1) ==
      Approx(update_sketch.getUpperBound(1)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(2) ==
      Approx(update_sketch.getLowerBound(2)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(2) ==
      Approx(update_sketch.getUpperBound(2)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(3) ==
      Approx(update_sketch.getLowerBound(3)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(3) ==
      Approx(update_sketch.getUpperBound(3)).margin(1e-10));
  compactThetaSketch compact_sketch = update_sketch.compact();
  // the sketches are ordered, so the iteration sequence must match exactly
  auto iter = sketch.begin();
  for (const auto& key : compact_sketch) {
    EXPECT_TRUE(*iter == key);
    ++iter;
  }
}

TEST(ThetaSketch, DeserializeCompactV2EstimationFromJava) {
  std::ifstream is;
  is.exceptions(std::ios::failbit | std::ios::badbit);
  is.open(
      inputPath + "theta_compact_estimation_from_java_v2.sk", std::ios::binary);
  auto sketch = compactThetaSketch::deserialize(is);
  EXPECT_FALSE(sketch.isEmpty());
  EXPECT_TRUE(sketch.isEstimationMode());
  EXPECT_TRUE(sketch.isOrdered());
  EXPECT_TRUE(sketch.getNumRetained() == 4342);
  EXPECT_TRUE(sketch.getTheta() == Approx(0.531700444213199).margin(1e-10));
  EXPECT_TRUE(sketch.getEstimate() == Approx(8166.25234614053).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(2) == Approx(7996.956955317471).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(2) == Approx(8339.090301078124).margin(1e-10));

  // the same construction process in Java must have produced exactly the same
  // sketch
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  const int n = 8192;
  for (int i = 0; i < n; i++)
    update_sketch.update(i);
  EXPECT_TRUE(sketch.getNumRetained() == update_sketch.getNumRetained());
  EXPECT_TRUE(
      sketch.getTheta() == Approx(update_sketch.getTheta()).margin(1e-10));
  EXPECT_TRUE(
      sketch.getEstimate() ==
      Approx(update_sketch.getEstimate()).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(1) ==
      Approx(update_sketch.getLowerBound(1)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(1) ==
      Approx(update_sketch.getUpperBound(1)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(2) ==
      Approx(update_sketch.getLowerBound(2)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(2) ==
      Approx(update_sketch.getUpperBound(2)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(3) ==
      Approx(update_sketch.getLowerBound(3)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(3) ==
      Approx(update_sketch.getUpperBound(3)).margin(1e-10));
  compactThetaSketch compact_sketch = update_sketch.compact();
  // the sketches are ordered, so the iteration sequence must match exactly
  auto iter = sketch.begin();
  for (const auto& key : compact_sketch) {
    EXPECT_TRUE(*iter == key);
    ++iter;
  }
}

TEST(ThetaSketch, DerializeDeserializeStreamAndBytesEquivalence) {
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  const int n = 8192;
  for (int i = 0; i < n; i++)
    update_sketch.update(i);

  std::stringstream s(std::ios::in | std::ios::out | std::ios::binary);
  auto compact_sketch = update_sketch.compact();
  compact_sketch.serialize(s);
  auto bytes = compact_sketch.serialize();
  EXPECT_TRUE(bytes.size() == static_cast<size_t>(s.tellp()));
  EXPECT_TRUE(bytes.size() == compact_sketch.getSerializedSizeBytes());
  for (size_t i = 0; i < bytes.size(); ++i) {
    EXPECT_TRUE(((char*)bytes.data())[i] == (char)s.get());
  }

  s.seekg(0); // rewind
  compactThetaSketch deserialized_sketch1 = compactThetaSketch::deserialize(s);
  compactThetaSketch deserialized_sketch2 =
      compactThetaSketch::deserialize(bytes.data(), bytes.size());
  EXPECT_TRUE(bytes.size() == static_cast<size_t>(s.tellg()));
  EXPECT_TRUE(deserialized_sketch2.isEmpty() == deserialized_sketch1.isEmpty());
  EXPECT_TRUE(
      deserialized_sketch2.isOrdered() == deserialized_sketch1.isOrdered());
  EXPECT_TRUE(
      deserialized_sketch2.getNumRetained() ==
      deserialized_sketch1.getNumRetained());
  EXPECT_TRUE(
      deserialized_sketch2.getTheta() == deserialized_sketch1.getTheta());
  EXPECT_TRUE(
      deserialized_sketch2.getEstimate() == deserialized_sketch1.getEstimate());
  EXPECT_TRUE(
      deserialized_sketch2.getLowerBound(1) ==
      deserialized_sketch1.getLowerBound(1));
  EXPECT_TRUE(
      deserialized_sketch2.getUpperBound(1) ==
      deserialized_sketch1.getUpperBound(1));
  // the sketches are ordered, so the iteration sequence must match exactly
  auto iter = deserialized_sketch1.begin();
  for (auto key : deserialized_sketch2) {
    EXPECT_TRUE(*iter == key);
    ++iter;
  }
}

TEST(ThetaSketch, DeserializeEmptyBufferOverrun) {
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  auto bytes = update_sketch.compact().serialize();
  EXPECT_TRUE(bytes.size() == 8);
  EXPECT_THROW(
      compactThetaSketch::deserialize(bytes.data(), bytes.size() - 1),
      VeloxUserError);
}

TEST(ThetaSketch, DeserializeSingleItemBufferOverrun) {
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  update_sketch.update(1);
  auto bytes = update_sketch.compact().serialize();
  EXPECT_THROW(
      compactThetaSketch::deserialize(bytes.data(), 7), VeloxUserError);
  EXPECT_THROW(
      compactThetaSketch::deserialize(bytes.data(), bytes.size() - 1),
      VeloxUserError);
}

TEST(ThetaSketch, DeserializeExactModeBufferOverrun) {
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  for (int i = 0; i < 1000; ++i)
    update_sketch.update(i);
  auto bytes = update_sketch.compact().serialize();
  EXPECT_THROW(
      compactThetaSketch::deserialize(bytes.data(), 7), VeloxUserError);
  EXPECT_THROW(
      compactThetaSketch::deserialize(bytes.data(), 8), VeloxUserError);
  EXPECT_THROW(
      compactThetaSketch::deserialize(bytes.data(), 16), VeloxUserError);
  EXPECT_THROW(
      compactThetaSketch::deserialize(bytes.data(), bytes.size() - 1),
      VeloxUserError);
}

TEST(ThetaSketch, DeserializeEstimationModeBufferOverrun) {
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  for (int i = 0; i < 10000; ++i)
    update_sketch.update(i);
  auto bytes = update_sketch.compact().serialize();
  EXPECT_THROW(
      compactThetaSketch::deserialize(bytes.data(), 7), VeloxUserError);
  EXPECT_THROW(
      compactThetaSketch::deserialize(bytes.data(), 8), VeloxUserError);
  EXPECT_THROW(
      compactThetaSketch::deserialize(bytes.data(), 16), VeloxUserError);
  EXPECT_THROW(
      compactThetaSketch::deserialize(bytes.data(), 24), VeloxUserError);
  EXPECT_THROW(
      compactThetaSketch::deserialize(bytes.data(), bytes.size() - 1),
      VeloxUserError);
}

TEST(ThetaSketch, ConversionConstructorAndWrappedCompact) {
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  const int n = 8192;
  for (int i = 0; i < n; i++)
    update_sketch.update(i);

  // unordered
  auto unordered_compact1 = update_sketch.compact(false);
  compactThetaSketch unordered_compact2(update_sketch, false);
  auto it = unordered_compact1.begin();
  for (auto entry : unordered_compact2) {
    EXPECT_TRUE(*it == entry);
    ++it;
  }

  // ordered
  auto ordered_compact1 = update_sketch.compact();
  compactThetaSketch ordered_compact2(update_sketch, true);
  it = ordered_compact1.begin();
  for (auto entry : ordered_compact2) {
    EXPECT_TRUE(*it == entry);
    ++it;
  }

  // wrapped compact
  auto bytes = ordered_compact1.serialize();
  auto ordered_compact3 =
      wrappedCompactThetaSketch::wrap(bytes.data(), bytes.size());
  it = ordered_compact1.begin();
  for (auto entry : ordered_compact3) {
    EXPECT_TRUE(*it == entry);
    ++it;
  }
  EXPECT_TRUE(ordered_compact3.getEstimate() == ordered_compact1.getEstimate());
  EXPECT_TRUE(
      ordered_compact3.getLowerBound(1) == ordered_compact1.getLowerBound(1));
  EXPECT_TRUE(
      ordered_compact3.getUpperBound(1) == ordered_compact1.getUpperBound(1));
  EXPECT_TRUE(
      ordered_compact3.isEstimationMode() ==
      ordered_compact1.isEstimationMode());
  EXPECT_TRUE(ordered_compact3.getTheta() == ordered_compact1.getTheta());

  // seed mismatch
  EXPECT_THROW(
      wrappedCompactThetaSketch::wrap(bytes.data(), bytes.size(), 0),
      VeloxUserError);
}

TEST(ThetaSketch, WrapCompactV1EmptyFromJava) {
  std::ifstream is;
  is.exceptions(std::ios::failbit | std::ios::badbit);
  is.open(
      inputPath + "theta_compact_empty_from_java_v1.sk",
      std::ios::binary | std::ios::ate);

  std::vector<uint8_t> buf;
  if (is) {
    auto size = is.tellg();
    buf.reserve(size);
    buf.assign(size, 0);
    is.seekg(0, std::ios_base::beg);
    is.read((char*)(buf.data()), buf.size());
  }

  auto sketch = wrappedCompactThetaSketch::wrap(buf.data(), buf.size());
  EXPECT_TRUE(sketch.isEmpty());
  EXPECT_FALSE(sketch.isEstimationMode());
  EXPECT_TRUE(sketch.getNumRetained() == 0);
  EXPECT_TRUE(sketch.getTheta() == 1.0);
  EXPECT_TRUE(sketch.getEstimate() == 0.0);
  EXPECT_TRUE(sketch.getLowerBound(1) == 0.0);
  EXPECT_TRUE(sketch.getUpperBound(1) == 0.0);
}

TEST(ThetaSketch, WrapCompactV2EmptyFromJava) {
  std::ifstream is;
  is.exceptions(std::ios::failbit | std::ios::badbit);
  is.open(
      inputPath + "theta_compact_empty_from_java_v2.sk",
      std::ios::binary | std::ios::ate);

  std::vector<uint8_t> buf;
  if (is) {
    auto size = is.tellg();
    buf.reserve(size);
    buf.assign(size, 0);
    is.seekg(0, std::ios_base::beg);
    is.read((char*)(buf.data()), buf.size());
  }

  auto sketch = wrappedCompactThetaSketch::wrap(buf.data(), buf.size());
  EXPECT_TRUE(sketch.isEmpty());
  EXPECT_FALSE(sketch.isEstimationMode());
  EXPECT_TRUE(sketch.getNumRetained() == 0);
  EXPECT_TRUE(sketch.getTheta() == 1.0);
  EXPECT_TRUE(sketch.getEstimate() == 0.0);
  EXPECT_TRUE(sketch.getLowerBound(1) == 0.0);
  EXPECT_TRUE(sketch.getUpperBound(1) == 0.0);
}

TEST(ThetaSketch, WrapCompactV1EstimationFromJava) {
  std::ifstream is;
  is.exceptions(std::ios::failbit | std::ios::badbit);
  is.open(
      inputPath + "theta_compact_estimation_from_java_v1.sk",
      std::ios::binary | std::ios::ate);
  std::vector<uint8_t> buf;
  if (is) {
    auto size = is.tellg();
    buf.reserve(size);
    buf.assign(size, 0);
    is.seekg(0, std::ios_base::beg);
    is.read((char*)(buf.data()), buf.size());
  }

  auto sketch = wrappedCompactThetaSketch::wrap(buf.data(), buf.size());
  EXPECT_FALSE(sketch.isEmpty());
  EXPECT_TRUE(sketch.isEstimationMode());
  //  EXPECT_TRUE(sketch.isOrdered());       // v1 may not be ordered
  EXPECT_TRUE(sketch.getNumRetained() == 4342);
  EXPECT_TRUE(sketch.getTheta() == Approx(0.531700444213199).margin(1e-10));
  EXPECT_TRUE(sketch.getEstimate() == Approx(8166.25234614053).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(2) == Approx(7996.956955317471).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(2) == Approx(8339.090301078124).margin(1e-10));

  // the same construction process in Java must have produced exactly the same
  // sketch
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  const int n = 8192;
  for (int i = 0; i < n; i++)
    update_sketch.update(i);
  EXPECT_TRUE(sketch.getNumRetained() == update_sketch.getNumRetained());
  EXPECT_TRUE(
      sketch.getTheta() == Approx(update_sketch.getTheta()).margin(1e-10));
  EXPECT_TRUE(
      sketch.getEstimate() ==
      Approx(update_sketch.getEstimate()).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(1) ==
      Approx(update_sketch.getLowerBound(1)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(1) ==
      Approx(update_sketch.getUpperBound(1)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(2) ==
      Approx(update_sketch.getLowerBound(2)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(2) ==
      Approx(update_sketch.getUpperBound(2)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(3) ==
      Approx(update_sketch.getLowerBound(3)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(3) ==
      Approx(update_sketch.getUpperBound(3)).margin(1e-10));
  compactThetaSketch compact_sketch = update_sketch.compact();
  // the sketches are ordered, so the iteration sequence must match exactly
  auto iter = sketch.begin();
  for (const auto key : compact_sketch) {
    EXPECT_TRUE(*iter == key);
    ++iter;
  }
}

TEST(ThetaSketch, WrapCompactV2EstimationFromJava) {
  std::ifstream is;
  is.exceptions(std::ios::failbit | std::ios::badbit);
  is.open(
      inputPath + "theta_compact_estimation_from_java_v2.sk",
      std::ios::binary | std::ios::ate);
  std::vector<uint8_t> buf;
  if (is) {
    auto size = is.tellg();
    buf.reserve(size);
    buf.assign(size, 0);
    is.seekg(0, std::ios_base::beg);
    is.read((char*)(buf.data()), buf.size());
  }

  auto sketch = wrappedCompactThetaSketch::wrap(buf.data(), buf.size());
  EXPECT_FALSE(sketch.isEmpty());
  EXPECT_TRUE(sketch.isEstimationMode());
  //  EXPECT_TRUE(sketch.isOrdered());       // v1 may not be ordered
  EXPECT_TRUE(sketch.getNumRetained() == 4342);
  EXPECT_TRUE(sketch.getTheta() == Approx(0.531700444213199).margin(1e-10));
  EXPECT_TRUE(sketch.getEstimate() == Approx(8166.25234614053).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(2) == Approx(7996.956955317471).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(2) == Approx(8339.090301078124).margin(1e-10));

  // the same construction process in Java must have produced exactly the same
  // sketch
  updateThetaSketch update_sketch = updateThetaSketch::builder().build();
  const int n = 8192;
  for (int i = 0; i < n; i++)
    update_sketch.update(i);
  EXPECT_TRUE(sketch.getNumRetained() == update_sketch.getNumRetained());
  EXPECT_TRUE(
      sketch.getTheta() == Approx(update_sketch.getTheta()).margin(1e-10));
  EXPECT_TRUE(
      sketch.getEstimate() ==
      Approx(update_sketch.getEstimate()).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(1) ==
      Approx(update_sketch.getLowerBound(1)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(1) ==
      Approx(update_sketch.getUpperBound(1)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(2) ==
      Approx(update_sketch.getLowerBound(2)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(2) ==
      Approx(update_sketch.getUpperBound(2)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getLowerBound(3) ==
      Approx(update_sketch.getLowerBound(3)).margin(1e-10));
  EXPECT_TRUE(
      sketch.getUpperBound(3) ==
      Approx(update_sketch.getUpperBound(3)).margin(1e-10));
  compactThetaSketch compact_sketch = update_sketch.compact();
  // the sketches are ordered, so the iteration sequence must match exactly
  auto iter = sketch.begin();
  for (const auto key : compact_sketch) {
    EXPECT_TRUE(*iter == key);
    ++iter;
  }
}

TEST(ThetaSketch, SerializeDeserializeSmallCompressed) {
  auto update_sketch = updateThetaSketch::builder().build();
  for (int i = 0; i < 10; i++)
    update_sketch.update(i);
  auto compact_sketch = update_sketch.compact();

  auto bytes = compact_sketch.serializeCompressed();
  EXPECT_TRUE(bytes.size() == compact_sketch.getSerializedSizeBytes(true));
  { // deserialize bytes
    auto deserialized_sketch =
        compactThetaSketch::deserialize(bytes.data(), bytes.size());
    EXPECT_TRUE(
        deserialized_sketch.getNumRetained() ==
        compact_sketch.getNumRetained());
    EXPECT_TRUE(deserialized_sketch.getTheta() == compact_sketch.getTheta());
    auto iter = deserialized_sketch.begin();
    for (const auto key : compact_sketch) {
      EXPECT_TRUE(*iter == key);
      ++iter;
    }
  }
  { // wrap bytes
    auto wrapped_sketch =
        wrappedCompactThetaSketch::wrap(bytes.data(), bytes.size());
    EXPECT_TRUE(
        wrapped_sketch.getNumRetained() == compact_sketch.getNumRetained());
    EXPECT_TRUE(wrapped_sketch.getTheta() == compact_sketch.getTheta());
    auto iter = wrapped_sketch.begin();
    for (const auto key : compact_sketch) {
      EXPECT_TRUE(*iter == key);
      ++iter;
    }
  }

  std::stringstream s(std::ios::in | std::ios::out | std::ios::binary);
  compact_sketch.serializeCompressed(s);
  EXPECT_TRUE(
      static_cast<size_t>(s.tellp()) ==
      compact_sketch.getSerializedSizeBytes(true));
  auto deserialized_sketch = compactThetaSketch::deserialize(s);
  EXPECT_TRUE(
      deserialized_sketch.getNumRetained() == compact_sketch.getNumRetained());
  EXPECT_TRUE(deserialized_sketch.getTheta() == compact_sketch.getTheta());
  auto iter = deserialized_sketch.begin();
  for (const auto key : compact_sketch) {
    EXPECT_TRUE(*iter == key);
    ++iter;
  }
}

TEST(ThetaSketch, SerializeDeserializeCompressed) {
  auto update_sketch = updateThetaSketch::builder().build();
  for (int i = 0; i < 10000; i++)
    update_sketch.update(i);
  auto compact_sketch = update_sketch.compact();

  auto bytes = compact_sketch.serializeCompressed();
  EXPECT_TRUE(bytes.size() == compact_sketch.getSerializedSizeBytes(true));
  { // deserialize bytes
    auto deserialized_sketch =
        compactThetaSketch::deserialize(bytes.data(), bytes.size());
    EXPECT_TRUE(
        deserialized_sketch.getNumRetained() ==
        compact_sketch.getNumRetained());
    EXPECT_TRUE(deserialized_sketch.getTheta() == compact_sketch.getTheta());
    auto iter = deserialized_sketch.begin();
    for (const auto key : compact_sketch) {
      EXPECT_TRUE(*iter == key);
      ++iter;
    }
  }
  { // wrap bytes
    auto wrapped_sketch =
        wrappedCompactThetaSketch::wrap(bytes.data(), bytes.size());
    EXPECT_TRUE(
        wrapped_sketch.getNumRetained() == compact_sketch.getNumRetained());
    EXPECT_TRUE(wrapped_sketch.getTheta() == compact_sketch.getTheta());
    auto iter = wrapped_sketch.begin();
    for (const auto key : compact_sketch) {
      EXPECT_TRUE(*iter == key);
      ++iter;
    }
  }

  std::stringstream s(std::ios::in | std::ios::out | std::ios::binary);
  compact_sketch.serializeCompressed(s);
  EXPECT_TRUE(
      static_cast<size_t>(s.tellp()) ==
      compact_sketch.getSerializedSizeBytes(true));
  auto deserialized_sketch = compactThetaSketch::deserialize(s);
  EXPECT_TRUE(
      deserialized_sketch.getNumRetained() == compact_sketch.getNumRetained());
  EXPECT_TRUE(deserialized_sketch.getTheta() == compact_sketch.getTheta());
  auto iter = deserialized_sketch.begin();
  for (const auto key : compact_sketch) {
    EXPECT_TRUE(*iter == key);
    ++iter;
  }
}

// The sketch reaches capacity for the first time at 2 * K * 15/16,
// but at that point it is still in exact mode, so the serialized size is not
// the maximum (theta in not serialized in the exact mode). So we need to catch
// the second time, but some updates will be ignored in the estimation mode, so
// we update more than enough times keeping track of the maximum. Potentially
// the exact number of updates to reach the peak can be figured out given this
// particular sequence, but not assuming that might be even better (say, in case
// we change the load factor or hash function or just out of principle not to
// rely on implementation details too much).
TEST(ThetaSketch, maxSerializedSize) {
  const uint8_t lg_k = 10;
  auto sketch = updateThetaSketch::builder().set_lg_k(lg_k).build();
  int value = 0;

  // this will go over the first peak, which is not the highest
  for (int i = 0; i < (1 << lg_k) * 2; ++i)
    sketch.update(value++);

  // this will to over the second peak keeping track of the max size
  size_t max_size_bytes = 0;
  for (int i = 0; i < (1 << lg_k) * 2; ++i) {
    sketch.update(value++);
    auto bytes = sketch.compact().serialize();
    max_size_bytes = std::max(max_size_bytes, bytes.size());
  }
  EXPECT_TRUE(
      max_size_bytes == compactThetaSketch::getMaxSerializedSizeBytes(lg_k));
}

} // namespace facebook::velox::common::theta
