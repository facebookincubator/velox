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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/tests/utils/OperatorTestBase.h"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <string>
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

} // namespace
} // namespace facebook::velox::cudf_velox
