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
#include <functional>
#include <limits>
#include <memory>
#include <string>

namespace facebook::velox::core {
class QueryCtx;
}

namespace facebook::velox::exec {
class Task;
}

namespace facebook::velox::memory {

class MemoryAllocator;
class MemoryArbitrator;
class MemoryPool;
class MemoryReclaimer;

/// Describes an externally-provided memory resource (e.g. a GPU or tiered
/// memory backend) registered with the memory subsystem and referenced by
/// 'tag' when building per-query memory pools. The constructor enforces
/// non-empty tag and non-null allocator, arbitrator, and reclaimerFactory;
/// once constructed, the resource is immutable.
class CustomMemoryResource {
 public:
  using ReclaimerFactory = std::function<std::unique_ptr<MemoryReclaimer>()>;
  using QueryReclaimerFactory = std::function<
      std::unique_ptr<MemoryReclaimer>(core::QueryCtx*, MemoryPool*)>;
  using TaskReclaimerFactory = std::function<std::unique_ptr<MemoryReclaimer>(
      const std::shared_ptr<exec::Task>&,
      int64_t /*priority*/,
      const std::string& /*resourceTag*/)>;

  /// Optional execution-context-aware overrides. Missing factories retain the
  /// legacy factory behavior at that level. A supplied factory's result is
  /// authoritative: nullptr means no reclaimer, never select a fallback.
  /// Factories must not capture a particular query or task in this process-wide
  /// resource. Each invocation receives the current execution context;
  /// factories must support concurrent calls for different queries/tasks.
  struct ExecutionReclaimerFactories {
    QueryReclaimerFactory query{};
    TaskReclaimerFactory task{};
  };

  CustomMemoryResource(
      std::string tag,
      std::shared_ptr<MemoryAllocator> allocator,
      std::shared_ptr<MemoryArbitrator> arbitrator,
      ReclaimerFactory reclaimerFactory,
      int64_t maxCapacity = std::numeric_limits<int64_t>::max());

  /// Opt-in execution factories; the original constructor is unchanged.
  CustomMemoryResource(
      std::string tag,
      std::shared_ptr<MemoryAllocator> allocator,
      std::shared_ptr<MemoryArbitrator> arbitrator,
      ReclaimerFactory reclaimerFactory,
      int64_t maxCapacity,
      ExecutionReclaimerFactories executionReclaimerFactories);

  /// Unique identifier for this resource.
  const std::string& tag() const {
    return tag_;
  }

  /// Capacity of the per-query root pool created from this resource.
  int64_t maxCapacity() const {
    return maxCapacity_;
  }

  /// Allocator backing pools tagged with this resource.
  MemoryAllocator* allocator() const {
    return allocator_.get();
  }

  /// Arbitrator routing capacity decisions for pools tagged with this
  /// resource.
  MemoryArbitrator* arbitrator() const {
    return arbitrator_.get();
  }

  /// Returns a fresh reclaimer for a new pool by invoking the factory
  /// supplied at construction. Used for legacy roots/tasks and node pools.
  /// A nullptr result means that the pool has no reclaimer.
  std::unique_ptr<MemoryReclaimer> newReclaimer() const;

  bool hasQueryReclaimerFactory() const {
    return static_cast<bool>(executionReclaimerFactories_.query);
  }

  /// Calls the explicit query factory; requires hasQueryReclaimerFactory().
  /// Inputs are borrowed during creation. Install the returned reclaimer on
  /// the root during query setup, before creating Tasks or allocating memory.
  std::unique_ptr<MemoryReclaimer> newQueryReclaimer(
      core::QueryCtx* queryCtx,
      MemoryPool* pool) const;

  /// Uses the explicit task factory when supplied, otherwise the legacy
  /// factory. Does not reinterpret a nullptr result from either factory.
  std::unique_ptr<MemoryReclaimer> newTaskReclaimer(
      const std::shared_ptr<exec::Task>& task,
      int64_t priority) const;

 private:
  const std::string tag_;
  const int64_t maxCapacity_;
  const std::shared_ptr<MemoryAllocator> allocator_;
  const std::shared_ptr<MemoryArbitrator> arbitrator_;
  const ReclaimerFactory reclaimerFactory_;
  const ExecutionReclaimerFactories executionReclaimerFactories_;
};

} // namespace facebook::velox::memory
