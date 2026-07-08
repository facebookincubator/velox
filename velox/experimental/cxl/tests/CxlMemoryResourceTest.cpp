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

#include "velox/experimental/cxl/CxlMemoryResource.h"

#include <numa.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/common/testutil/TestValue.h"

#include <gtest/gtest.h>

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::cxl {
namespace {

class CxlMemoryResourceTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    // CxlMemoryResource::create creates a SHARED arbitrator, whose factory must
    // be registered first.
    memory::SharedArbitrator::registerFactory();
    TestValue::enable();
  }

  static void TearDownTestCase() {
    memory::SharedArbitrator::unregisterFactory();
  }
};

// Real CXL hardware is absent on CI, so the node-is-CXL check is bypassed via a
// TestValue hook to exercise resource creation end to end on a DRAM node.
DEBUG_ONLY_TEST_F(CxlMemoryResourceTest, buildsResource) {
  SCOPED_TESTVALUE_SET(
      "facebook::velox::cxl::checkNodeIsCxl",
      std::function<void(bool*)>([](bool* hasCpus) { *hasCpus = false; }));

  constexpr int64_t kMaxCapacity = 64 << 20; // 64MB
  auto resource = CxlMemoryResource::create(0, kMaxCapacity);
  ASSERT_NE(resource, nullptr);
  EXPECT_EQ(resource->tag(), std::string{CxlMemoryResource::kTag});
  EXPECT_NE(resource->allocator(), nullptr);
  EXPECT_NE(resource->arbitrator(), nullptr);
}

TEST_F(CxlMemoryResourceTest, rejectsNonCxlNode) {
  // The node hosting CPU 0 is guaranteed to have CPUs, so it is DRAM, not a
  // CXL device, and binding a CXL resource there must fail.
  const int32_t cpuNode = numa_node_of_cpu(0);
  VELOX_ASSERT_USER_THROW(
      CxlMemoryResource::create(cpuNode), "requires a CPU-less NUMA node");
}

} // namespace
} // namespace facebook::velox::cxl
