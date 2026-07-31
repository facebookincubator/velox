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
#include <string_view>

namespace facebook::velox::cudf_velox {

/// Identifies one concrete operator instance that originated an allocation.
struct GpuMemoryOwner {
  std::string taskUuid;
  std::string taskId;
  std::string queryId;
  std::string planNodeId;
  /// Display only. Deliberately excluded from equality and hashing, because
  /// registerOperator() looks an owner up before it can resolve the type.
  std::string planNodeType;
  int32_t pipelineId{-1};
  int32_t driverId{-1};
  /// Position within the pipeline.
  int32_t operatorId{-1};
  std::string operatorType;

  bool operator==(const GpuMemoryOwner& other) const {
    return taskUuid == other.taskUuid && taskId == other.taskId &&
        queryId == other.queryId && planNodeId == other.planNodeId &&
        pipelineId == other.pipelineId && driverId == other.driverId &&
        operatorId == other.operatorId && operatorType == other.operatorType;
  }
};

/// Returns what identifies the query level, which is not always the query id.
///
/// QueryCtx::create() defaults queryId to empty and only some embeddings fill
/// it in. Keying on it alone would merge unrelated queries into one row. A task
/// belongs to one query, so its uuid is a safe fallback; the cost is one row
/// per task where the id is missing. The id still labels the row.
inline std::string_view gpuMemoryQueryKey(const GpuMemoryOwner& owner) {
  return owner.queryId.empty() ? owner.taskUuid : owner.queryId;
}

/// Refers to stable owner and plan node records in the ledger.
struct GpuMemoryOwnerHandle {
  uint64_t ownerId{0};
  /// The ledger's numeric handle for the plan node aggregate, unrelated to
  /// GpuMemoryOwner::planNodeId.
  uint64_t planNodeKey{0};

  bool operator==(const GpuMemoryOwnerHandle&) const = default;
};

/// Every counter affected by one logical-memory transition, after it.
///
/// Each value comes from an independent atomic, so they agree with one another
/// only in single-threaded execution and once allocation activity quiesces.
struct GpuMemoryUpdate {
  uint64_t ownerId{0};
  uint64_t planNodeKey{0};
  uint64_t globalCurrentBytes{0};
  /// Monotonic for the life of the process, so it is reported through the
  /// snapshot and the allocation-failure marker rather than as a counter, where
  /// it would carry one query's peak into every later query.
  uint64_t globalPeakBytes{0};
  uint64_t queryCurrentBytes{0};
  uint64_t queryPeakBytes{0};
  /// A task executes exactly one plan fragment, so this is the fragment level.
  uint64_t taskCurrentBytes{0};
  uint64_t planNodeCurrentBytes{0};
  uint64_t ownerCurrentBytes{0};
};

} // namespace facebook::velox::cudf_velox
