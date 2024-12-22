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

#include <vector>
#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/HashStringAllocator.h"

namespace facebook::velox::optimizer {

/// Single threaded cache in front of HashStringAllocator free
/// list. Speeds up allocation and fre of plan candidates. This is
/// about 2x faster than uncached HashStringAllocator. Overall amounts
/// to ~2% of optimization time.
class ArenaCache {
  static constexpr int32_t kMaxSize = 512;
  static constexpr int32_t kGranularity = 16;

 public:
  explicit ArenaCache(velox::HashStringAllocator& allocator)
      : allocator_(allocator), allocated_(kMaxSize / kGranularity) {}

  void* allocate(size_t size) {
    auto sizeClass = velox::bits::roundUp(size, kGranularity) / kGranularity;
    if (sizeClass < kMaxSize / kGranularity) {
      if (!allocated_[sizeClass].empty()) {
        void* result = allocated_[sizeClass].back();
        allocated_[sizeClass].pop_back();
        totalSize_ -= sizeClass;
        return result;
      }
    }
    return allocator_.allocate(sizeClass * kGranularity)->begin();
  }

  void free(void* ptr) {
    auto header = velox::HashStringAllocator::headerOf(ptr);
    int32_t sizeClass = header->size() / kGranularity;
    if (sizeClass < kMaxSize / kGranularity) {
      totalSize_ += sizeClass;
      allocated_[sizeClass].push_back(ptr);
      return;
    }
    allocator_.free(header);
  }

 private:
  velox::HashStringAllocator& allocator_;
  std::vector<std::vector<void*>> allocated_;
  uint64_t totalSize_{0};
};

} // namespace facebook::velox::optimizer
