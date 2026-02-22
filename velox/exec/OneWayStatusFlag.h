/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

class OneWayStatusFlag {
 public:
  bool check() const noexcept {
    return status_.load(std::memory_order_acquire);
  }

  void set() noexcept {
    if (!status_.load(std::memory_order_relaxed)) {
      status_.store(true, std::memory_order_release);
    }
  }

  explicit operator bool() const noexcept {
    return check();
  }

 private:
  std::atomic_bool status_{false};
};

} // namespace facebook::velox::exec
