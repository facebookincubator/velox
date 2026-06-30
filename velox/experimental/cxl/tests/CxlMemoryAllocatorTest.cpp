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

#include "velox/experimental/cxl/CxlMemoryAllocator.h"

#include <numa.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"

#include <gtest/gtest.h>

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::cxl {
namespace {

using namespace facebook::velox::memory;

// A node id guaranteed to be out of range on any host: one past the highest
// node libnuma reports.
int32_t outOfRangeNode() {
  return numa_max_node() + 1;
}

// Injection point that treats the bound node as CPU-less, so the allocator can
// be built on a DRAM node on CXL-less hosts.
const char* const kCheckNodeIsCxl = "facebook::velox::cxl::checkNodeIsCxl";

class CxlMemoryAllocatorTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    TestValue::enable();
  }

  // Options sized small enough to keep the test cheap; node 0 always exists on
  // a NUMA-capable host.
  static MemoryAllocator::Options options() {
    MemoryAllocator::Options opts;
    opts.capacity = 64 << 20; // 64MB
    return opts;
  }
};

// Real CXL hardware is absent on CI, so node 0 is bound as if it were CXL.
DEBUG_ONLY_TEST_F(CxlMemoryAllocatorTest, numaNodeAccessor) {
  SCOPED_TESTVALUE_SET(
      kCheckNodeIsCxl,
      std::function<void(bool*)>([](bool* hasCpus) { *hasCpus = false; }));
  CxlMemoryAllocator allocator(options(), 0);
  EXPECT_EQ(allocator.numaNode(), 0);
  EXPECT_EQ(allocator.kind(), MemoryAllocator::Kind::kMmap);
}

DEBUG_ONLY_TEST_F(CxlMemoryAllocatorTest, allocateFreeRoundTrip) {
  SCOPED_TESTVALUE_SET(
      kCheckNodeIsCxl,
      std::function<void(bool*)>([](bool* hasCpus) { *hasCpus = false; }));
  CxlMemoryAllocator allocator(options(), 0);
  ASSERT_EQ(allocator.numAllocated(), 0);

  constexpr MachinePageCount kNumPages = 16;
  Allocation allocation;
  ASSERT_TRUE(allocator.allocateNonContiguous(kNumPages, allocation));
  EXPECT_GE(allocation.numPages(), kNumPages);
  EXPECT_GE(allocator.numAllocated(), kNumPages);

  // Touch every page so the binding actually faults memory in.
  for (uint32_t i = 0; i < allocation.numRuns(); ++i) {
    auto run = allocation.runAt(i);
    auto* bytes = run.data<uint8_t>();
    for (MachinePageCount page = 0; page < run.numPages(); ++page) {
      bytes[page * AllocationTraits::kPageSize] = static_cast<uint8_t>(page);
    }
  }

  EXPECT_TRUE(allocator.checkConsistency());

  allocator.freeNonContiguous(allocation);
  EXPECT_EQ(allocation.numPages(), 0);
  EXPECT_EQ(allocator.numAllocated(), 0);
  EXPECT_TRUE(allocator.checkConsistency());
}

TEST_F(CxlMemoryAllocatorTest, invalidNodeThrows) {
  VELOX_ASSERT_THROW(
      CxlMemoryAllocator(options(), -1), "Invalid CXL NUMA node");
  VELOX_ASSERT_THROW(
      CxlMemoryAllocator(options(), outOfRangeNode()),
      "CXL NUMA node is out of range");
}

TEST_F(CxlMemoryAllocatorTest, rejectsNonCxlNode) {
  // The node hosting CPU 0 is guaranteed to have CPUs, so it is DRAM, not a
  // CXL device, and binding the allocator there must fail.
  const int32_t cpuNode = numa_node_of_cpu(0);
  VELOX_ASSERT_USER_THROW(
      CxlMemoryAllocator(options(), cpuNode), "requires a CPU-less NUMA node");
}

} // namespace
} // namespace facebook::velox::cxl
