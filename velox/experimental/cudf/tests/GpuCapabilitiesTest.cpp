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

TEST(GpuCapabilitiesTest, describesTheDevice) {
  const auto capabilities = readGpuCapabilities(kCurrentDevice);
  ASSERT_GT(capabilities.totalMemoryBytes, 0);
  EXPECT_GE(capabilities.deviceIndex, 0);
  EXPECT_FALSE(capabilities.name.empty());
  EXPECT_GT(capabilities.multiprocessorCount, 0);
  EXPECT_GT(capabilities.computeCapabilityMajor, 0);
  // Every device this builds for reports a UUID, and it is what persisted
  // per-device tuning would key on, so an empty one is a real defect.
  EXPECT_THAT(capabilities.uuid, ::testing::StartsWith("GPU-"));
}

TEST(GpuCapabilitiesTest, bandwidthIsPlausibleForTheDevice) {
  const auto capabilities = readGpuCapabilities(kCurrentDevice);
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

TEST(GpuCapabilitiesTest, unreadableDeviceYieldsAnEmptyDescription) {
  // A device that cannot be queried must not throw and must be distinguishable
  // from one that was read, so a caller can keep its fixed default. An index
  // well past any real device count is the reachable way to force that path.
  const auto capabilities = readGpuCapabilities(1 << 20);
  EXPECT_EQ(capabilities.totalMemoryBytes, 0);
  EXPECT_EQ(capabilities.deviceIndex, -1);
  EXPECT_EQ(capabilities.toString(), "GPU capabilities unknown");
}

TEST(GpuCapabilitiesTest, jsonCarriesTheDescription) {
  const auto json = readGpuCapabilities(kCurrentDevice).toJson();
  EXPECT_NE(json.find("\"totalMemoryBytes\""), std::string::npos);
  EXPECT_NE(json.find("\"computeCapability\""), std::string::npos);
  EXPECT_EQ(json.front(), '{');
  EXPECT_EQ(json.back(), '}');
}

TEST(GpuCapabilitiesTest, jsonEscapesDeviceStrings) {
  // The name and bus id come from the driver, so the serializer has to quote
  // them rather than paste them in.
  GpuCapabilities capabilities;
  capabilities.name = R"(a "quoted" \ name)";
  const auto json = capabilities.toJson();
  EXPECT_NE(json.find(R"(\"quoted\")"), std::string::npos);
}

TEST(GpuCapabilitiesTest, publishesTheDeviceOnce) {
  const auto expected = readGpuCapabilities(kCurrentDevice);
  ASSERT_GT(expected.totalMemoryBytes, 0);

  initializeGpuCapabilities(expected.deviceIndex);
  EXPECT_EQ(gpuCapabilities().deviceIndex, expected.deviceIndex);
  EXPECT_EQ(gpuCapabilities().totalMemoryBytes, expected.totalMemoryBytes);

  // Only the first call reads, so a later one naming an unreadable device
  // cannot replace what was published.
  initializeGpuCapabilities(1 << 20);
  EXPECT_EQ(gpuCapabilities().deviceIndex, expected.deviceIndex);
  EXPECT_EQ(gpuCapabilities().totalMemoryBytes, expected.totalMemoryBytes);
}

} // namespace
} // namespace facebook::velox::cudf_velox
