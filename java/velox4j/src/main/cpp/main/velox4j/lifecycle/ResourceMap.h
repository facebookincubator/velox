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

#include <velox/common/base/Exceptions.h>
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace facebook::velox4j {

using ResourceHandle = uint32_t;
static_assert(std::numeric_limits<ResourceHandle>::min() == 0);

/// Safely casts a value from type F to type T with bounds checking.
/// E.g., from uint64_t to uint32_t. If the number to cast doesn't fit
/// into uint32_t, an error will be thrown.
template <typename T, typename F>
T safeCast(F from) {
  F min = 0;
  F max = static_cast<F>(std::numeric_limits<T>::max());
  VELOX_CHECK_GE(from, min, "Safe casting a negative number");
  VELOX_CHECK_LE(from, max, "Number overflow");
  return static_cast<T>(from);
}

/// A utility class that maps resource objects to their shared pointers.
/// @tparam TResource class of the object to hold.
template <typename TResource>
class ResourceMap {
 public:
  ResourceMap() : resourceId_(kInitResourceId) {}

  ResourceHandle insert(TResource holder) {
    ResourceHandle result = safeCast<ResourceHandle>(resourceId_++);
    const std::lock_guard<std::mutex> lock(mtx_);
    map_.insert(std::pair<ResourceHandle, TResource>(result, holder));
    return result;
  }

  void erase(ResourceHandle moduleId) {
    const std::lock_guard<std::mutex> lock(mtx_);
    VELOX_CHECK_EQ(
        map_.erase(moduleId),
        1,
        "ResourceHandle not found in resource map when trying to erase: " +
            std::to_string(moduleId));
  }

  TResource lookup(ResourceHandle moduleId) const {
    const std::lock_guard<std::mutex> lock(mtx_);
    auto it = map_.find(moduleId);
    VELOX_CHECK(
        it != map_.end(),
        "ResourceHandle not found in resource map when trying to lookup: " +
            std::to_string(moduleId));
    return it->second;
  }

  void clear() {
    const std::lock_guard<std::mutex> lock(mtx_);
    map_.clear();
  }

  size_t size() const {
    const std::lock_guard<std::mutex> lock(mtx_);
    return map_.size();
  }

  size_t nextId() const {
    return resourceId_;
  }

 private:
  // Initialize the resource id starting value to a number greater than zero
  // to allow for easier debugging of uninitialized java variables.
  static constexpr size_t kInitResourceId = 4;

  std::atomic<size_t> resourceId_{0};

  // map from resource ids returned to Java and resource pointers
  std::unordered_map<ResourceHandle, TResource> map_;
  mutable std::mutex mtx_;
};

} // namespace facebook::velox4j
