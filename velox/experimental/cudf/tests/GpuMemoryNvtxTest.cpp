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

#include "velox/experimental/cudf/exec/GpuMemoryNvtx.h"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

// These tests run without a profiler attached, so NVTX resolves its entry
// points to no-ops and hands back id 0 for every counter it is asked to
// register. What is exercised here is this module's own bookkeeping: that
// registration is idempotent per owner, that a counter set carries the epoch it
// was issued in, that a reset retires those sets, and that none of the entry
// points throws or races. Whether Nsight renders the result can only be checked
// against a real capture.

namespace facebook::velox::cudf_velox {
namespace {

GpuMemoryOwner makeOwner(int32_t operatorId, std::string planNodeId = "3") {
  GpuMemoryOwner owner;
  owner.taskUuid = "task-uuid";
  owner.taskId = "task-id.0.0.0";
  owner.queryId = "query-1";
  owner.planNodeId = std::move(planNodeId);
  owner.planNodeType = "TableScanNode";
  owner.pipelineId = 0;
  owner.driverId = 0;
  owner.operatorId = operatorId;
  owner.operatorType = "TableScan";
  return owner;
}

GpuMemoryPlanLocation makeLocation(std::string planNodeId = "3") {
  GpuMemoryPlanLocation location;
  location.path.push_back(
      GpuMemoryPlanPathEntry{std::move(planNodeId), "TableScanNode"});
  location.order = 0;
  return location;
}

class GpuMemoryNvtxTest : public ::testing::Test {
 protected:
  void SetUp() override {
    resetGpuMemoryNvtxCounters();
  }

  void TearDown() override {
    resetGpuMemoryNvtxCounters();
  }
};

TEST_F(GpuMemoryNvtxTest, RegistrationIssuesACurrentEpoch) {
  const auto counters =
      registerGpuMemoryNvtxOwner(1, 10, makeOwner(1), makeLocation());
  EXPECT_NE(counters.epoch, 0u)
      << "a registered owner must be distinguishable from an unregistered one";
}

TEST_F(GpuMemoryNvtxTest, RegistrationIsIdempotentPerOwner) {
  const auto first =
      registerGpuMemoryNvtxOwner(1, 10, makeOwner(1), makeLocation());
  const auto second =
      registerGpuMemoryNvtxOwner(1, 10, makeOwner(1), makeLocation());

  EXPECT_EQ(first.epoch, second.epoch);
  EXPECT_EQ(first.ownerCounterId, second.ownerCounterId);
  EXPECT_EQ(first.queryCounterId, second.queryCounterId);
  EXPECT_EQ(first.planNodeCounterId, second.planNodeCounterId);
}

TEST_F(GpuMemoryNvtxTest, OwnersOfOneQueryShareTheirAncestorCounters) {
  const auto first =
      registerGpuMemoryNvtxOwner(1, 10, makeOwner(1), makeLocation());
  const auto second =
      registerGpuMemoryNvtxOwner(2, 10, makeOwner(2), makeLocation());

  EXPECT_EQ(first.queryCounterId, second.queryCounterId);
  EXPECT_EQ(first.taskCounterId, second.taskCounterId);
  EXPECT_EQ(first.planNodeCounterId, second.planNodeCounterId)
      << "same plan node key must resolve to one aggregate";
}

TEST_F(GpuMemoryNvtxTest, UnattributedOwnerHasNoAncestorCounters) {
  const auto counters = registerGpuMemoryNvtxUnattributedOwner();

  EXPECT_NE(counters.epoch, 0u);
  EXPECT_EQ(counters.queryCounterId, 0u);
  EXPECT_EQ(counters.taskCounterId, 0u);
  EXPECT_EQ(counters.planNodeCounterId, 0u)
      << "an allocation outside any operator belongs to no query or plan node";
}

TEST_F(GpuMemoryNvtxTest, ResetRetiresPreviouslyIssuedCounters) {
  const auto before =
      registerGpuMemoryNvtxOwner(1, 10, makeOwner(1), makeLocation());
  resetGpuMemoryNvtxCounters();
  const auto after =
      registerGpuMemoryNvtxOwner(1, 10, makeOwner(1), makeLocation());

  EXPECT_NE(before.epoch, after.epoch)
      << "a set cached across a reset must stop sampling";

  // Sampling a retired set is a no-op rather than a write into a counter that
  // no longer has a scope.
  GpuMemoryUpdate update;
  update.ownerId = 1;
  update.globalCurrentBytes = 1024;
  sampleGpuMemoryNvtxCounters(before, update);
  sampleGpuMemoryNvtxCounters(after, update);
}

TEST_F(GpuMemoryNvtxTest, SamplingAnUnregisteredSetIsANoOp) {
  GpuMemoryUpdate update;
  update.ownerId = 7;
  update.globalCurrentBytes = 2048;
  sampleGpuMemoryNvtxCounters(GpuMemoryNvtxCounters{}, update);
}

TEST_F(GpuMemoryNvtxTest, ConcurrentRegistrationAndSamplingAgree) {
  constexpr int kThreads = 8;
  constexpr int kPerThread = 500;

  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([thread, &mismatches] {
      const auto ownerId = static_cast<uint64_t>(thread + 1);
      const auto expected = registerGpuMemoryNvtxOwner(
          ownerId, 10, makeOwner(thread), makeLocation());
      for (int i = 0; i < kPerThread; ++i) {
        // Re-registering must keep returning the same set while other threads
        // register and sample concurrently.
        const auto counters = registerGpuMemoryNvtxOwner(
            ownerId, 10, makeOwner(thread), makeLocation());
        if (counters.ownerCounterId != expected.ownerCounterId ||
            counters.epoch != expected.epoch) {
          mismatches.fetch_add(1);
        }
        GpuMemoryUpdate update;
        update.ownerId = ownerId;
        update.globalCurrentBytes = static_cast<uint64_t>(i);
        update.ownerCurrentBytes = static_cast<uint64_t>(i);
        sampleGpuMemoryNvtxCounters(counters, update);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(mismatches.load(), 0);
}

TEST_F(GpuMemoryNvtxTest, MarkersDoNotThrow) {
  markGpuMemoryDataLoss("test reason");

  GpuMemoryUpdate state;
  state.ownerId = 1;
  state.globalCurrentBytes = 4096;
  markGpuMemoryAllocationFailure(1, 512, state, 128, 4096, "cudaSuccess");
}

} // namespace
} // namespace facebook::velox::cudf_velox
