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

namespace facebook::velox::memory {

class MemoryAllocator;
class MemoryArbitrator;
class MemoryReclaimer;

/// Describes an externally-provided memory resource (e.g. a GPU or tiered
/// memory backend) registered with MemoryManager and referenced by 'tag' when
/// building per-query memory pools.
struct CustomMemoryResource {
  /// Unique non-empty identifier.
  std::string tag;

  /// Capacity of the per-query root pool created for this resource.
  int64_t maxCapacity{std::numeric_limits<int64_t>::max()};

  /// Allocator backing pools tagged with this resource.
  std::shared_ptr<MemoryAllocator> allocator;

  /// Arbitrator routing capacity decisions for pools tagged with this
  /// resource.
  std::shared_ptr<MemoryArbitrator> arbitrator;

  /// Required. Builds a reclaimer for pools tagged with this resource.
  /// Invoked by the caller (not the framework) when constructing the
  /// per-query root pool. For reclaimers that need a QueryCtx-aware view,
  /// the caller skips this factory and attaches the reclaimer post-build via
  /// MemoryPool::setReclaimer.
  std::function<std::unique_ptr<MemoryReclaimer>()> reclaimerFactory;
};

} // namespace facebook::velox::memory
