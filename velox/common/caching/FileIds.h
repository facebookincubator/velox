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

#include "velox/common/caching/StringIdMap.h"

namespace facebook::velox {

// Returns a process-wide map of file path to id and id to file path.
StringIdMap& fileIds();

class FileSizes {
 public:
  static void setSize(uint64_t id, uint64_t size) {
    std::lock_guard<std::mutex> l(mutex_);
    auto previous = sizeLocked(id);
    if (previous && previous != size) {
      LOG(FATAL) << "Changing size of file " << id << " from " << previous
                 << " to " << size;
    }
    idToSize_[id] = size;
  }
  static uint64_t size(uint64_t id) {
    std::lock_guard<std::mutex> l(mutex_);
    return sizeLocked(id, true);
  }

 private:
  static uint64_t sizeLocked(uint64_t id, bool mustFind = false) {
    auto it = idToSize_.find(id);
    if (it != idToSize_.end()) {
      return it->second;
    }
    if (mustFind) {
      LOG(FATAL) << "No size known for file " << id;
    }
    return 0;
  }

  static std::mutex mutex_;
  static std::unordered_map<uint64_t, uint64_t> idToSize_;
};

} // namespace facebook::velox
