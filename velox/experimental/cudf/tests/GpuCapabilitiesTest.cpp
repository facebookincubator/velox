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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox {
namespace {

class GpuCapabilitiesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    initializeGpuCapabilities(kCurrentDevice);
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
  EXPECT_THAT(capabilities.uuid, ::testing::StartsWith("GPU-"));
}

TEST_F(GpuCapabilitiesTest, bandwidthIsPlausibleForTheDevice) {
  const auto& capabilities = gpuCapabilities();
  if (capabilities.memoryClockRateKhz == 0) {
    // CUDA 13 removed the memory clock from cudaDeviceProp, and the attribute
    // it was replaced by is not available everywhere either.
    GTEST_SKIP() << "runtime did not report a memory clock";
  }
  ASSERT_GT(capabilities.memoryBusWidthBits, 0);
  // A unit slip in the kHz-to-Hz or bits-to-bytes conversion moves the result
  // by three orders of magnitude, so bound it instead of recomputing the
  // formula from the same two fields. No shipping discrete GPU is outside
  // 10 GB/s .. 20 TB/s.
  EXPECT_GT(capabilities.peakMemoryBandwidthBytesPerSecond, 10'000'000'000LL);
  EXPECT_LT(
      capabilities.peakMemoryBandwidthBytesPerSecond, 20'000'000'000'000LL);
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

TEST_F(GpuCapabilitiesTest, jsonEscapesDeviceStrings) {
  // The name and bus id come from the driver and go out to a coordinator, so
  // the serializer has to quote them rather than paste them in.
  GpuCapabilities capabilities;
  capabilities.totalMemoryBytes = 1;
  capabilities.name = R"(a "quoted" \ name)";
  const auto json = capabilities.toJson();
  EXPECT_NE(json.find(R"(\"quoted\")"), std::string::npos);
}

TEST_F(GpuCapabilitiesTest, derivedDefaultsScaleWithTheDevice) {
  const auto& capabilities = gpuCapabilities();
  ASSERT_GT(capabilities.totalMemoryBytes, 0);

  constexpr int64_t kFallback = 256LL << 20;
  const auto oneDriver =
      gpu_defaults::batchSizeMinBytes(capabilities, kFallback, 1);
  const auto fourDrivers =
      gpu_defaults::batchSizeMinBytes(capabilities, kFallback, 4);

  // The budget is per driver, so more drivers sharing the device must each get
  // no more; on a device small enough that both hit the 32 MiB floor they are
  // equal, which is why this is not a strict inequality.
  EXPECT_LE(fourDrivers, oneDriver);
  EXPECT_GE(fourDrivers, 32LL << 20);
  EXPECT_LT(oneDriver, capabilities.totalMemoryBytes);

  // A build only earns a denser hash table once it is a real share of the
  // device, so the threshold must be a substantial number of rows.
  EXPECT_GT(
      gpu_defaults::hashJoinDenseLoadFactorMinRows(
          capabilities, /*fallback=*/0),
      1'000'000);
  EXPECT_GT(
      gpu_defaults::finalGroupbySpillBytes(capabilities, /*fallback=*/0), 0);
}

TEST(GpuCapabilitiesFallbackTest, unknownDeviceKeepsTheCallersDefault) {
  // A device that cannot be queried must not change behaviour: every
  // derivation has to hand back exactly the fixed value it was given. This is
  // what lets the feature ship without risk on machines it cannot inspect.
  const GpuCapabilities unknown;
  ASSERT_EQ(unknown.totalMemoryBytes, 0);
  EXPECT_EQ(unknown.toString(), "GPU capabilities unknown");

  constexpr int64_t kBatchFallback = 256LL << 20;
  constexpr int64_t kSpillFallback = 1LL << 30;
  constexpr int64_t kRowsFallback = 1'000'000;
  EXPECT_EQ(
      gpu_defaults::batchSizeMinBytes(unknown, kBatchFallback, 1),
      kBatchFallback);
  // Not even the 32 MiB floor applies on an unknown device; the caller's
  // constant has to come back untouched whatever the driver count.
  EXPECT_EQ(
      gpu_defaults::batchSizeMinBytes(unknown, kBatchFallback, 64),
      kBatchFallback);
  EXPECT_EQ(
      gpu_defaults::finalGroupbySpillBytes(unknown, kSpillFallback),
      kSpillFallback);
  EXPECT_EQ(
      gpu_defaults::hashJoinDenseLoadFactorMinRows(unknown, kRowsFallback),
      kRowsFallback);
}

} // namespace
} // namespace facebook::velox::cudf_velox
