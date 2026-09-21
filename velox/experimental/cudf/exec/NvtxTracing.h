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

#pragma once

#include <cstdint>
#include <string>

namespace facebook::velox::exec {
class Task;
} // namespace facebook::velox::exec

namespace facebook::velox::cudf_velox {

/// NVTX domain holding one range per stretch of time a Driver occupied an OS
/// thread. Separate from "velox" so it lands on its own row in Nsight: the
/// operator ranges there have a median duration of ~2.4 us and are invisible
/// without deep zoom, whereas these bands are milliseconds wide and show, at a
/// glance, that a thread changed hands.
struct VeloxThreadDomain {
  static constexpr char const* name{"velox.thread"};
};

/// Records that `label` (a Driver) is running on the calling thread right now.
///
/// Closes the band for whatever Driver was last seen on this thread and opens a
/// new one. Called on every wrapped operator entry, so the cost is a
/// thread-local compare in the common case where nothing changed.
///
/// The band is an approximation of thread occupancy, not a measurement of it:
/// it starts at the first cuDF operator call after the swap rather than at
/// enqueue, and ends at the first call of the next Driver rather than when this
/// one blocked. Knowing precisely would need a hook in core Velox's Driver
/// loop, which this deliberately avoids.
void noteDriverOnThread(uint64_t key, const char* label);

/// Registers `task` the first time it is seen and returns the NVTX category id
/// assigned to it, which callers stamp on every range so that a capture can be
/// grouped and filtered by task. Returns 0, NVTX's "no category", for a null
/// task.
///
/// Registration also names the category after the full Presto task id,
/// queryId.stageId.stageExecutionId.id.attemptNumber, and emits one mark
/// carrying the same. Note that Task::shortId() is unusable as the name: it
/// hashes only the query id, so every task of a query shares one value.
///
/// A category rather than a domain: Nsight renders a row per domain per thread,
/// and a Task's Drivers are spread over every thread in the pool, so a domain
/// per Task would cost roughly numTasks * numThreads mostly-empty rows. A
/// category adds no rows, and because it rides on the range itself it stays
/// correct no matter which thread a Driver lands on.
uint32_t nvtxRegisterTask(const exec::Task* task);

/// Builds the label for a Driver's thread-occupancy band, for example
/// "t3.0.0.0 p1d0", where the leading field is the task id with its query id
/// stripped. The task appears here, on a band that is milliseconds wide, rather
/// than on the operator ranges, which are too short to read a longer name off
/// of and already carry the task as their category.
std::string
nvtxDriverLabel(const exec::Task* task, int32_t pipelineId, int32_t driverId);

/// Packs the identity of a physical operator into the 64-bit NVTX payload:
/// pipelineId, driverId, operatorId and splitGroupId. A viewer can then place
/// a range on the right Driver lane without parsing the range name.
constexpr int64_t packOperatorIdentity(
    int32_t pipelineId,
    int32_t driverId,
    int32_t operatorId,
    uint32_t splitGroupId) {
  // splitGroupId is kUngroupedGroupId (0xFFFFFFFF) for ungrouped execution;
  // truncate it to 16 bits, where it reads as 0xFFFF.
  return (static_cast<int64_t>(pipelineId & 0xFFFF) << 48) |
      (static_cast<int64_t>(driverId & 0xFFFF) << 32) |
      (static_cast<int64_t>(operatorId & 0xFFFF) << 16) |
      static_cast<int64_t>(splitGroupId & 0xFFFF);
}

} // namespace facebook::velox::cudf_velox
