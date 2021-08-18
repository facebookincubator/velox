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

#include <folly/CPortability.h>
#include <atomic>
#include <cstdint>
#include <cstring>

namespace facebook {
namespace velox {
namespace memory {
// Memory tracking methods. All methods are single threaded. Aggregate nodes
// will have their stats updated by a global aggregation thread. This also means
// that aggregate nodes should not allocate memory and should do so by creating
// children nodes.
struct MemoryUsage {
 public:
  int64_t getCurrentBytes() const;
  int64_t getMaxBytes() const;

  void incrementCurrentBytes(int64_t size);
  void setCurrentBytes(int64_t size);

 private:
  // Can contain other stats.
  std::atomic<int64_t> currentBytes_{0};
  std::atomic<int64_t> maxBytes_{0};
};
} // namespace memory
} // namespace velox
} // namespace facebook
