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
#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::exec {
class Operator;
}

namespace facebook::velox::cudf_velox {

struct GpuMemoryOwnerSnapshot {
  GpuMemoryOwnerHandle handle;
  /// Identity captured at registration.
  GpuMemoryOwner owner;
  uint64_t currentBytes{0};
  uint64_t peakBytes{0};
  /// Allocated successfully over the ledger's lifetime.
  uint64_t totalBytes{0};
};

/// A process-wide view of the accounting.
///
/// Exists to make the accounting testable; no production code reads it. It
/// carries only what separates correct accounting from incorrect: what is live,
/// what the peak was, what was ever allocated, whether the pointer map drained,
/// and whether anything was lost. Anything more would be state maintained on
/// the allocation path for no reader.
///
/// Levels are updated independently and without a shared lock, so a snapshot
/// taken mid-flight can catch them at slightly different instants. It is exact
/// single-threaded and once allocation activity quiesces.
struct GpuMemorySnapshot {
  uint64_t currentBytes{0};
  uint64_t peakBytes{0};
  uint64_t totalBytes{0};
  /// Counted from the pointer map rather than tracked separately.
  uint64_t liveAllocations{0};
  /// Accounting events that could not be represented.
  uint64_t dataLossEvents{0};
  /// Ordered by descending live bytes.
  std::vector<GpuMemoryOwnerSnapshot> owners;
};

/// Attributes every tracked GPU allocation to the operator instance that made
/// it, and maintains live, peak and total byte counters per operator instance,
/// plan node, task, query and process.
///
/// The allocation and deallocation paths take no shared lock. Pointer ownership
/// lives in a concurrent hash map, every counter is an atomic, and each owner
/// record caches the ancestor counters and NVTX counter ids it needs, so a
/// transition performs no lookup outside that map. A registration mutex guards
/// the creation of records, which then keep stable addresses for their
/// lifetime.
///
/// That holds only while callers reuse the handle they were issued. Resolving
/// an owner from an operator takes the registration mutex, so a caller that
/// resolves per allocation rather than caching reintroduces the serialization.
///
/// A free is always charged to the allocation-time owner, even when it happens
/// on another thread or under another active operator, which is what makes an
/// owner breakdown at a later process peak meaningful.
class GpuMemoryLedger {
 public:
  GpuMemoryLedger();
  ~GpuMemoryLedger();

  GpuMemoryLedger(const GpuMemoryLedger&) = delete;
  GpuMemoryLedger& operator=(const GpuMemoryLedger&) = delete;

  /// Registers an owner and returns its stable handle.
  ///
  /// 'planLocation' places the owner in the NVTX counter hierarchy. An
  /// unresolved location reports it as unmapped. Registering an already known
  /// owner returns the existing handle without creating a second record.
  GpuMemoryOwnerHandle registerOwner(
      const GpuMemoryOwner& owner,
      const GpuMemoryPlanLocation& planLocation = {});

  /// Resolves the operator's display type and plan path only when the owner is
  /// new, since that walks the plan tree.
  GpuMemoryOwnerHandle registerOperator(exec::Operator* op);

  /// Returns the resulting counter values, or nothing for a null address. A
  /// duplicate live address is reported as data loss.
  std::optional<GpuMemoryUpdate> recordAllocation(
      void* address,
      std::size_t bytes,
      GpuMemoryOwnerHandle handle) noexcept;

  /// Debits the allocation-time owner, using the size recorded at allocation
  /// rather than 'bytes'. A disagreement between the two, like an unknown
  /// address, is reported as data loss.
  std::optional<GpuMemoryUpdate> recordDeallocation(
      void* address,
      std::size_t bytes) noexcept;

  /// Counters as they stand, for the allocation-failure marker.
  GpuMemoryUpdate currentState(GpuMemoryOwnerHandle handle) const noexcept;

  /// See GpuMemorySnapshot for what this guarantees under concurrency.
  [[nodiscard]] GpuMemorySnapshot snapshot() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace facebook::velox::cudf_velox
