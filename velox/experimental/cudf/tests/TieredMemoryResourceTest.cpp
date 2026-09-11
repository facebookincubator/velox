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

#include "velox/experimental/cudf/exec/GpuResources.h"

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

using namespace facebook::velox::cudf_velox;

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr std::size_t kCapBytes = 64 * kMiB;

} // namespace

class TieredMemoryResourceTest : public testing::Test {};

TEST_F(TieredMemoryResourceTest, deviceTierBelowCap) {
  TieredMemoryResource mr{kCapBytes};
  auto const stream = rmm::cuda_stream_default;

  void* small = mr.allocate(stream, 16 * kMiB);
  ASSERT_NE(small, nullptr);
  {
    auto const stats = mr.stats();
    EXPECT_EQ(stats.capBytes, kCapBytes);
    EXPECT_EQ(stats.deviceBytes, 16 * kMiB);
    EXPECT_EQ(stats.overflowAllocations, 0);
    EXPECT_EQ(stats.overflowBytes, 0);
  }

  // 16 MiB + 100 MiB exceeds the 64 MiB device cap, so this must land in the
  // managed overflow tier without disturbing the device tier.
  void* large = mr.allocate(stream, 100 * kMiB);
  ASSERT_NE(large, nullptr);
  {
    auto const stats = mr.stats();
    EXPECT_EQ(stats.deviceBytes, 16 * kMiB);
    EXPECT_EQ(stats.overflowAllocations, 1);
    EXPECT_EQ(stats.overflowBytes, 100 * kMiB);
  }

  // The overflow pointer must be usable from the device.
  ASSERT_EQ(cudaMemsetAsync(large, 0, 100 * kMiB, stream.value()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.value()), cudaSuccess);

  mr.deallocate(stream, large, 100 * kMiB);
  mr.deallocate(stream, small, 16 * kMiB);
  ASSERT_EQ(cudaStreamSynchronize(stream.value()), cudaSuccess);

  auto const stats = mr.stats();
  EXPECT_EQ(stats.deviceBytes, 0);
  EXPECT_EQ(stats.overflowBytes, 0);
  EXPECT_EQ(stats.deviceBytesPeak, 16 * kMiB);
  EXPECT_EQ(stats.overflowBytesPeak, 100 * kMiB);
}

TEST_F(TieredMemoryResourceTest, spillsOnceCapIsReached) {
  TieredMemoryResource mr{kCapBytes};
  auto const stream = rmm::cuda_stream_default;

  constexpr std::size_t kBlockBytes = 8 * kMiB;
  constexpr std::size_t kNumBlocks = (128 * kMiB) / kBlockBytes;

  std::vector<void*> blocks;
  blocks.reserve(kNumBlocks);
  for (std::size_t i = 0; i < kNumBlocks; ++i) {
    void* ptr = mr.allocate(stream, kBlockBytes);
    ASSERT_NE(ptr, nullptr);
    blocks.push_back(ptr);

    auto const stats = mr.stats();
    EXPECT_LE(stats.deviceBytes, kCapBytes);
    if ((i + 1) * kBlockBytes <= kCapBytes) {
      EXPECT_EQ(stats.deviceBytes, (i + 1) * kBlockBytes);
      EXPECT_EQ(stats.overflowAllocations, 0);
    } else {
      // Everything past the cap goes to the overflow tier.
      EXPECT_EQ(stats.deviceBytes, kCapBytes);
      EXPECT_EQ(stats.overflowAllocations, (i + 1) - (kCapBytes / kBlockBytes));
    }
  }

  for (auto* ptr : blocks) {
    mr.deallocate(stream, ptr, kBlockBytes);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream.value()), cudaSuccess);

  auto const stats = mr.stats();
  EXPECT_EQ(stats.deviceBytes, 0);
  EXPECT_EQ(stats.overflowBytes, 0);
  EXPECT_EQ(stats.deviceBytesPeak, kCapBytes);
}
