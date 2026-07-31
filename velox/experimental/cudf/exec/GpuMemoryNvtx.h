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

#include "velox/experimental/cudf/exec/GpuMemoryOwner.h"
#include "velox/experimental/cudf/exec/GpuMemoryPlanPath.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace facebook::velox::cudf_velox {

/// Publishes Velox-cuDF logical GPU memory as NVTX counters.
///
/// Counters are registered in the NVTX default domain rather than the "velox"
/// domain that carries the operator ranges, because Nsight renders a named
/// domain as an extra level above every counter row.
///
/// Their scopes form one tree under a single root scope: query, then plan
/// fragment, then one scope per plan node, then one per pipeline, with each
/// operator instance a counter leaf. Plan nodes are siblings of one another
/// rather than nested by the plan tree; their labels carry the plan order.
///
/// Every entry point is safe to call when no profiler is attached, in which
/// case NVTX resolves its entry points to no-ops, and none of them can throw.

/// The counters one operator instance moves on every transition.
///
/// Held by the caller so that sampling needs neither a lookup nor a lock. Zero
/// is NVTX's "no counter" value, so a default-constructed set samples nothing.
struct GpuMemoryNvtxCounters {
  /// Registration cycle these ids belong to. A reset retires every id, and a
  /// set cached before it must not be sampled afterwards.
  uint64_t epoch{0};
  uint64_t globalCounterId{0};
  uint64_t ownerCounterId{0};
  uint64_t queryCounterId{0};
  uint64_t queryPeakCounterId{0};
  /// A task executes one plan fragment, so this is the fragment level.
  uint64_t taskCounterId{0};
  uint64_t planNodeCounterId{0};
};

/// Registers the scope tree and counters for one owner, and returns the
/// counters its transitions move. Idempotent per owner and per shared level.
///
/// An empty 'planLocation' places the operator under the fragment's "unmapped"
/// scope, so an unresolved plan node id still reports its bytes.
GpuMemoryNvtxCounters registerGpuMemoryNvtxOwner(
    uint64_t ownerId,
    uint64_t planNodeKey,
    const GpuMemoryOwner& owner,
    const GpuMemoryPlanLocation& planLocation) noexcept;

/// For allocations made outside any operator. Called lazily, so a process that
/// never makes one shows no always-zero row.
GpuMemoryNvtxCounters registerGpuMemoryNvtxUnattributedOwner() noexcept;

/// Takes no lock. Does nothing for a counter set that is empty or predates a
/// reset.
void sampleGpuMemoryNvtxCounters(
    const GpuMemoryNvtxCounters& counters,
    const GpuMemoryUpdate& update) noexcept;

/// Emits a factual allocation-failure marker with the logical state at failure.
///
/// 'cudaStatus' reports whether the device byte counts could be collected on
/// this cold path.
void markGpuMemoryAllocationFailure(
    uint64_t ownerId,
    std::size_t requestedBytes,
    const GpuMemoryUpdate& state,
    std::size_t cudaFreeBytes,
    std::size_t cudaTotalBytes,
    std::string_view cudaStatus) noexcept;

/// Emits a diagnostic marker when ledger accounting loses an event.
void markGpuMemoryDataLoss(std::string_view reason) noexcept;

/// Zeroes every counter and clears the registered hierarchy.
void resetGpuMemoryNvtxCounters() noexcept;

} // namespace facebook::velox::cudf_velox
