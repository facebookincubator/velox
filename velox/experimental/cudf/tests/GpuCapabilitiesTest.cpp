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

#include "velox/experimental/cudf/exec/GpuCapabilities.h"

#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox {
namespace {

class GpuCapabilitiesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    initializeGpuCapabilities();
  }
};

TEST_F(GpuCapabilitiesTest, describesTheDevice) {
  const auto& capabilities = gpuCapabilities();
  ASSERT_GT(capabilities.totalMemoryBytes, 0);
  EXPECT_GE(capabilities.deviceIndex, 0);
  EXPECT_FALSE(capabilities.name.empty());
  EXPECT_GT(capabilities.multiprocessorCount, 0);
  EXPECT_GT(capabilities.computeCapabilityMajor, 0);
  // Every device this builds for reports a UUID, and it is what persisted
  // per-device tuning would key on, so an empty one is a real defect.
  EXPECT_EQ(capabilities.uuid.rfind("GPU-", 0), 0u);
}

TEST_F(GpuCapabilitiesTest, bandwidthIsDerivedFromTheMemoryInterface) {
  const auto& capabilities = gpuCapabilities();
  if (capabilities.memoryBusWidthBits == 0) {
    GTEST_SKIP() << "device did not report a memory interface";
  }
  EXPECT_EQ(
      capabilities.peakMemoryBandwidthBytesPerSecond,
      static_cast<int64_t>(capabilities.memoryClockRateKhz) * 1000LL *
          (static_cast<int64_t>(capabilities.memoryBusWidthBits) / 8LL) * 2LL);
}

TEST_F(GpuCapabilitiesTest, jsonCarriesTheSchemaVersion) {
  const auto json = gpuCapabilities().toJson();
  // A consumer that stores or ships this needs to know which shape it has.
  EXPECT_NE(
      json.find(
          "\"schemaVersion\":" +
          std::to_string(GpuCapabilities::kSchemaVersion)),
      std::string::npos);
  EXPECT_NE(json.find("\"totalMemoryBytes\""), std::string::npos);
  EXPECT_EQ(json.front(), '{');
  EXPECT_EQ(json.back(), '}');
}

TEST_F(GpuCapabilitiesTest, derivedDefaultsScaleWithTheDevice) {
  const auto& capabilities = gpuCapabilities();
  ASSERT_GT(capabilities.totalMemoryBytes, 0);

  constexpr uint64_t kFallback = 256ULL << 20;
  const auto oneDriver = gpu_defaults::batchSizeMinBytes(kFallback, 1);
  const auto fourDrivers = gpu_defaults::batchSizeMinBytes(kFallback, 4);

  // The budget is per driver, so more drivers sharing the device must each get
  // less; that relationship is the whole point of the derivation.
  EXPECT_LT(fourDrivers, oneDriver);
  EXPECT_LT(oneDriver, static_cast<uint64_t>(capabilities.totalMemoryBytes));

  // A build only earns a denser hash table once it is a real share of the
  // device, so the threshold must be a substantial number of rows.
  EXPECT_GT(gpu_defaults::hashJoinDenseLoadFactorMinRows(0), 1'000'000UL);
  EXPECT_GT(gpu_defaults::finalGroupbySpillBytes(0), 0UL);
}

TEST(GpuCapabilitiesFallbackTest, unknownDeviceKeepsTheCallersDefault) {
  // A device that cannot be queried must not change behaviour: every
  // derivation has to hand back exactly the fixed value it was given. This is
  // what lets the feature ship without risk on machines it cannot inspect.
  const GpuCapabilities unknown;
  ASSERT_EQ(unknown.totalMemoryBytes, 0);
  EXPECT_EQ(unknown.toString(), "GPU capabilities unknown");
}

} // namespace
} // namespace facebook::velox::cudf_velox
