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

#include <atomic>
namespace facebook::velox::exec {

// A simple one way status flag that uses a non atomic flag to avoid
// unnecessary atomic operations.
class OneWayStatusFlag {
 public:
  bool check() const {
    return fastStatus_ || atomicStatus_.load(std::memory_order_acquire);
  }

  void set() {
    if (!fastStatus_) {
      atomicStatus_.store(true, std::memory_order_relaxed);
      fastStatus_ = true;
    }
  }

  // Operator overload to convert OneWayStatusFlag to bool
  operator bool() const {
    return check();
  }

 private:
  bool fastStatus_{false};
  std::atomic<bool> atomicStatus_{false};
};

} // namespace facebook::velox::exec
