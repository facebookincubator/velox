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
#include <string_view>
#include <variant>

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

/// Pool creation currently happens before the pool exists, so 'pool' can be
/// nullptr.
struct PoolReclaimerContext {
  MemoryPool* pool;
};

struct QueryReclaimerContext {
  core::QueryCtx* queryCtx;
  MemoryPool* pool;
};

struct TaskReclaimerContext {
  std::shared_ptr<exec::Task> task;
  int64_t priority;
  /// Borrowed for the duration of the factory call.
  std::string_view resourceTag;
};

using ReclaimerContext = std::
    variant<PoolReclaimerContext, QueryReclaimerContext, TaskReclaimerContext>;

/// Describes an externally-provided memory resource (e.g. a GPU or tiered
/// memory backend) registered with the memory subsystem and referenced by
/// 'tag' when building custom memory pool hierarchies. Roots need not belong to
/// queries. Construction enforces
/// non-empty tag and non-null allocator, arbitrator, and reclaimerFactory;
/// once constructed, the resource is immutable.
class CustomMemoryResource {
 public:
  using ReclaimerFactory =
      std::function<std::unique_ptr<MemoryReclaimer>(const ReclaimerContext&)>;

  /// The factory receives the current pool, query, or task context. A nullptr
  /// result means no reclaimer; there is no implicit fallback. Factories on
  /// resources shared across queries must support concurrent calls and should
  /// not capture a particular query or task.
  CustomMemoryResource(
      std::string tag,
      std::shared_ptr<MemoryAllocator> allocator,
      std::shared_ptr<MemoryArbitrator> arbitrator,
      ReclaimerFactory reclaimerFactory,
      int64_t maxCapacity = std::numeric_limits<int64_t>::max());

  /// Unique identifier for this resource.
  const std::string& tag() const {
    return tag_;
  }

  /// Maximum capacity of a root created through the resource-based overload.
  /// The arbitrator determines its currently granted capacity.
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

  /// Returns a fresh reclaimer by invoking the factory with 'context'.
  /// Resource roots and node pools use the default pool context. Query setup
  /// explicitly supplies a query context and installs the result before
  /// creating Tasks or allocating memory. Task creation supplies a task
  /// context. A nullptr result means that the pool has no reclaimer.
  std::unique_ptr<MemoryReclaimer> newReclaimer(
      const ReclaimerContext& context = PoolReclaimerContext{nullptr}) const;

 private:
  const std::string tag_;
  const int64_t maxCapacity_;
  const std::shared_ptr<MemoryAllocator> allocator_;
  const std::shared_ptr<MemoryArbitrator> arbitrator_;
  const ReclaimerFactory reclaimerFactory_;
};

} // namespace facebook::velox::memory
