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

namespace facebook::verax {

class ArenaCache {
  static constexpr int32_t kMaxSize = 512;
  static constexpr int32_t kGranularity = 16;

 public:
  explicit ArenaCache(velox::HashStringAllocator& allocator)
      : allocator_(allocator), allocated_(kMaxSize / kGranularity) {}

  void* allocate(size_t size) {
    auto roundedSize = velox::bits::roundUp(size, kGranularity) / kGranularity;
    if (roundedSize < kMaxSize / kGranularity) {
      if (!allocated_[roundedSize].empty()) {
        void* result = allocated_[roundedSize].back();
        allocated_[roundedSize].pop_back();
        totalSize_ -= roundedSize;
        return result;
      }
    }
    return allocator_.allocate(size)->begin();
  }

  void free(void* ptr) {
    auto header = velox::HashStringAllocator::headerOf(ptr);
    int32_t size = header->size() / kGranularity;
    if (size < kMaxSize / kGranularity) {
      totalSize_ += size;
      allocated_[size].push_back(ptr);
      return;
    }
    allocator_.free(header);
  }

 private:
  velox::HashStringAllocator& allocator_;
  std::vector<std::vector<void*>> allocated_;
  uint64_t totalSize_{0};
};

} // namespace facebook::verax
