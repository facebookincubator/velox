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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/tests/utils/OperatorTestBase.h"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::cudf_velox {
namespace {

constexpr int64_t kAllocSize = 8 << 20; // 8MB.

class GpuResourcesTest : public OperatorTestBase {
 protected:
  // The memory resource configuration has to be in place before registerCudf()
  // creates the resources, so each test registers cuDF itself.
  void SetUp() override {
    OperatorTestBase::SetUp();
    auto& config = CudfConfig::getInstance();
    savedMemoryResource_ = config.memoryResource;
    savedOutputMemoryResource_ = config.outputMemoryResource;
  }

  void TearDown() override {
    unregisterCudf();
    auto& config = CudfConfig::getInstance();
    config.memoryResource = savedMemoryResource_;
    config.outputMemoryResource = savedOutputMemoryResource_;
    OperatorTestBase::TearDown();
  }

  // Asserts that an allocation from each of 'resources' raises the live byte
  // count, and that freeing them returns it to the value it started at.
  void assertTracksAllocations(
      const std::vector<rmm::device_async_resource_ref>& resources) {
    const auto baseline = cudfAllocatedBytes();
    ASSERT_GE(baseline, 0);

    auto stream = cudfGlobalStreamPool().get_stream();
    {
      std::vector<rmm::device_buffer> buffers;
      buffers.reserve(resources.size());
      for (const auto& resource : resources) {
        buffers.emplace_back(kAllocSize, stream, resource);
      }
      // A resource may round the request up, hence the inequality.
      EXPECT_GE(
          cudfAllocatedBytes(),
          baseline + static_cast<int64_t>(resources.size()) * kAllocSize);
    }
    EXPECT_EQ(cudfAllocatedBytes(), baseline);
  }

  std::string savedMemoryResource_;
  std::string savedOutputMemoryResource_;
};

TEST_F(GpuResourcesTest, allocatedBytesWithoutCudf) {
  {
    SCOPED_TRACE("Before registration");
    ASSERT_FALSE(cudfIsRegistered());
    EXPECT_EQ(cudfAllocatedBytes(), -1);
  }

  registerCudf();
  ASSERT_GE(cudfAllocatedBytes(), 0);
  unregisterCudf();

  {
    SCOPED_TRACE("After unregistration");
    EXPECT_EQ(cudfAllocatedBytes(), -1);
  }
}

TEST_F(GpuResourcesTest, allocatedBytesTracksLiveAllocations) {
  registerCudf();
  ASSERT_TRUE(cudfIsRegistered());

  assertTracksAllocations({cudf::get_current_device_resource_ref()});
}

TEST_F(GpuResourcesTest, allocatedBytesIncludesSeparateOutputResource) {
  auto& config = CudfConfig::getInstance();
  config.memoryResource = "async";
  config.outputMemoryResource = "cuda";
  registerCudf();

  assertTracksAllocations(
      {cudf::get_current_device_resource_ref(), get_output_mr()});
}

using ResourceKind = CudfMemoryResourceKind;
using AdmissionStatus = CudfDeviceMemoryAdmissionStatus;
constexpr uint64_t kAllocationBytes = 4096;
constexpr uint64_t kImpossibleHeadroom = std::numeric_limits<uint64_t>::max();

class GpuMemoryAdmissionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    clearCurrentDeviceMemoryInfo();
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
        cudaSuccess);
  }

  void TearDown() override {
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    // The tracking contains non-owning pool handles. Invalidate them before
    // releasing either resource, including on assertion-failure paths.
    clearCurrentDeviceMemoryInfo();
    output_.reset();
    primary_.reset();
    EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
  }

  rmm::device_async_resource_ref makeResource(
      std::string_view mode,
      ResourceKind kind = ResourceKind::kPrimary) {
    auto& resource = kind == ResourceKind::kPrimary ? primary_ : output_;
    // Pools grow on demand from empty. Arena requires a nonzero capacity;
    // one percent is the smallest supported size in the production factory.
    resource.emplace(createMemoryResource(mode, mode == "arena" ? 1 : 0, kind));
    return resource.value();
  }

  cuda::stream_ref stream() const {
    return cuda::stream_ref{stream_};
  }

 private:
  cudaStream_t stream_{nullptr};
  std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> primary_;
  std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> output_;
};

TEST_F(GpuMemoryAdmissionTest, unregisteredResourceDoesNotInferAdmission) {
  const auto info = currentDeviceMemoryInfo();
  ASSERT_TRUE(info.has_value());
  EXPECT_FALSE(info->hasAdmissionStats);
  const auto admission =
      tryReserveCurrentDeviceMemory(kAllocationBytes, kImpossibleHeadroom);
  EXPECT_EQ(admission.status, AdmissionStatus::kUnavailable);
  EXPECT_TRUE(admission.mayProceed());
}

TEST_F(GpuMemoryAdmissionTest, sharedAsyncPoolSharesCapacityAndReservations) {
  auto resource = makeResource("async");
  sharePrimaryMemoryResourceWithOutput();
  rmm::device_buffer buffer(kAllocationBytes, stream(), resource);
  stream().sync();

  const auto primary = currentDeviceMemoryInfo();
  const auto output = currentDeviceMemoryInfo(ResourceKind::kOutput);
  ASSERT_TRUE(primary.has_value());
  ASSERT_TRUE(output.has_value());
  ASSERT_TRUE(primary->hasAdmissionStats);
  ASSERT_TRUE(output->hasAdmissionStats);
  ASSERT_TRUE(primary->hasPoolStats);
  ASSERT_TRUE(output->hasPoolStats);
  EXPECT_GE(primary->poolUsedBytes, kAllocationBytes);
  EXPECT_EQ(primary->poolReservedBytes, output->poolReservedBytes);
  EXPECT_EQ(primary->poolUsedBytes, output->poolUsedBytes);
  EXPECT_EQ(primary->poolReusableBytes, output->poolReusableBytes);

  const auto admission = tryReserveCurrentDeviceMemory(kAllocationBytes, 0);
  auto release =
      folly::makeGuard([&] { releaseDeviceMemoryReservation(admission); });
  ASSERT_EQ(admission.status, AdmissionStatus::kAdmitted);
  const auto reserved = currentDeviceMemoryInfo(ResourceKind::kOutput);
  ASSERT_TRUE(reserved.has_value());
  EXPECT_EQ(reserved->inProcessReservedBytes, kAllocationBytes);
}

TEST_F(GpuMemoryAdmissionTest, distinctAsyncPoolsReportTheirOwnCapacity) {
  auto primaryResource = makeResource("async");
  rmm::device_buffer primaryBuffer(kAllocationBytes, stream(), primaryResource);
  auto outputResource = makeResource("async", ResourceKind::kOutput);
  stream().sync();
  const auto primaryBefore = currentDeviceMemoryInfo();
  const auto outputBefore = currentDeviceMemoryInfo(ResourceKind::kOutput);
  ASSERT_TRUE(primaryBefore.has_value());
  ASSERT_TRUE(outputBefore.has_value());
  ASSERT_TRUE(primaryBefore->hasPoolStats);
  ASSERT_TRUE(outputBefore->hasPoolStats);

  rmm::device_buffer outputBuffer(
      2 * kAllocationBytes, stream(), outputResource);
  stream().sync();
  const auto primaryAfter = currentDeviceMemoryInfo();
  const auto outputAfter = currentDeviceMemoryInfo(ResourceKind::kOutput);
  ASSERT_TRUE(primaryAfter.has_value());
  ASSERT_TRUE(outputAfter.has_value());
  EXPECT_EQ(primaryAfter->poolUsedBytes, primaryBefore->poolUsedBytes);
  EXPECT_GT(outputAfter->poolUsedBytes, outputBefore->poolUsedBytes);

  const auto admission =
      tryReserveCurrentDeviceMemory(kAllocationBytes, 0, ResourceKind::kOutput);
  auto release =
      folly::makeGuard([&] { releaseDeviceMemoryReservation(admission); });
  ASSERT_EQ(admission.status, AdmissionStatus::kAdmitted);
  const auto reserved = currentDeviceMemoryInfo();
  ASSERT_TRUE(reserved.has_value());
  EXPECT_EQ(reserved->inProcessReservedBytes, kAllocationBytes);
}

TEST_F(GpuMemoryAdmissionTest, cudaOutputDoesNotBorrowPrimaryPool) {
  auto primaryResource = makeResource("async");
  rmm::device_buffer primaryBuffer(kAllocationBytes, stream(), primaryResource);
  makeResource("cuda", ResourceKind::kOutput);
  stream().sync();

  const auto primary = currentDeviceMemoryInfo();
  const auto output = currentDeviceMemoryInfo(ResourceKind::kOutput);
  ASSERT_TRUE(primary.has_value());
  ASSERT_TRUE(output.has_value());
  ASSERT_TRUE(primary->hasPoolStats);
  EXPECT_TRUE(output->hasAdmissionStats);
  EXPECT_FALSE(output->hasPoolStats);
  EXPECT_EQ(output->poolReusableBytes, 0);
  EXPECT_EQ(
      tryReserveCurrentDeviceMemory(
          kAllocationBytes, kImpossibleHeadroom, ResourceKind::kOutput)
          .status,
      AdmissionStatus::kInsufficient);
}

TEST_F(GpuMemoryAdmissionTest, poolOutputDoesNotBorrowPrimaryPool) {
  makeResource("async");
  auto outputResource = makeResource("pool", ResourceKind::kOutput);
  const auto primary = currentDeviceMemoryInfo();
  const auto output = currentDeviceMemoryInfo(ResourceKind::kOutput);
  ASSERT_TRUE(primary.has_value());
  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(primary->hasAdmissionStats);
  EXPECT_FALSE(output->hasAdmissionStats);
  EXPECT_EQ(
      tryReserveCurrentDeviceMemory(
          kAllocationBytes, kImpossibleHeadroom, ResourceKind::kOutput)
          .status,
      AdmissionStatus::kUnavailable);
  rmm::device_buffer buffer(kAllocationBytes, stream(), outputResource);
  stream().sync();
}

TEST_F(GpuMemoryAdmissionTest, poolPrimaryDoesNotDisableAsyncOutputAdmission) {
  makeResource("pool");
  makeResource("async", ResourceKind::kOutput);
  const auto primary = currentDeviceMemoryInfo();
  const auto output = currentDeviceMemoryInfo(ResourceKind::kOutput);
  ASSERT_TRUE(primary.has_value());
  ASSERT_TRUE(output.has_value());
  EXPECT_FALSE(primary->hasAdmissionStats);
  EXPECT_TRUE(output->hasAdmissionStats);
  EXPECT_TRUE(output->hasPoolStats);
  EXPECT_EQ(
      tryReserveCurrentDeviceMemory(
          kAllocationBytes, kImpossibleHeadroom, ResourceKind::kOutput)
          .status,
      AdmissionStatus::kInsufficient);
}

class UnobservableGpuMemoryAdmissionTest
    : public GpuMemoryAdmissionTest,
      public ::testing::WithParamInterface<const char*> {};

TEST_P(UnobservableGpuMemoryAdmissionTest, letsAllocatorDecide) {
  const std::string_view mode{GetParam()};
  // Use the same capability check as RMM before constructing its async
  // managed resource. These modes are unavailable on CUDA 12.x builds.
  if ((mode == "managed_async" || mode == "prefetch_managed_async") &&
      !rmm::detail::runtime_async_managed_alloc::is_supported()) {
    GTEST_SKIP() << "Async managed memory requires CUDA 13.0 or higher and "
                    "concurrent managed access support";
  }
  // A device-only estimate must not reject an allocation without observing
  // allocator capacity: pool/arena retain reusable bytes, and managed memory
  // may allocate beyond device memory. Impossible headroom distinguishes the
  // explicit unavailable fallback from accidental physical-only admission.
  auto resource = makeResource(mode);
  const auto info = currentDeviceMemoryInfo();
  ASSERT_TRUE(info.has_value());
  EXPECT_FALSE(info->hasAdmissionStats);
  const auto admission =
      tryReserveCurrentDeviceMemory(kAllocationBytes, kImpossibleHeadroom);
  EXPECT_EQ(admission.status, AdmissionStatus::kUnavailable);
  EXPECT_TRUE(admission.mayProceed());
  rmm::device_buffer buffer(kAllocationBytes, stream(), resource);
  stream().sync();
}

INSTANTIATE_TEST_SUITE_P(
    AllocatorModes,
    UnobservableGpuMemoryAdmissionTest,
    ::testing::Values(
        "pool",
        "arena",
        "managed",
        "managed_pool",
        "managed_async",
        "prefetch_managed",
        "prefetch_managed_pool",
        "prefetch_managed_async"));

} // namespace
} // namespace facebook::velox::cudf_velox
