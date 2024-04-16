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
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::dwio::common::unit_loader_tools {

class Measure {
 public:
  explicit Measure(const std::function<void(uint64_t)>& blockedOnIoMsCallback)
      : blockedOnIoMsCallback_{blockedOnIoMsCallback},
        startTime_{std::chrono::high_resolution_clock::now()} {}

  Measure(const Measure&) = delete;
  Measure(Measure&&) = delete;
  Measure& operator=(const Measure&) = delete;
  Measure& operator=(Measure&& other) = delete;

  ~Measure() {
    auto timeBlockedOnIo =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - startTime_);
    blockedOnIoMsCallback_(timeBlockedOnIo.count());
  }

 private:
  const std::function<void(uint64_t)>& blockedOnIoMsCallback_;
  const std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;
};

inline std::optional<Measure> measureBlockedOnIo(
    const std::function<void(uint64_t)>& blockedOnIoMsCallback) {
  if (blockedOnIoMsCallback) {
    return std::make_optional<Measure>(blockedOnIoMsCallback);
  }
  return std::nullopt;
}

// This class can create many callbacks that can be distributed to unit loader
// factories. Only when the last created callback is activated, this class will
// emit the original callback.
// If the callbacks created are never activated (because of an exception for
// example), this will still work, since they will emit the signal on the
// destructor.
// The class expects that all callbacks are created before they start emiting
// signals.
class CallbackOnLastSignal {
 public:
  explicit CallbackOnLastSignal(std::function<void()> callback);

  ~CallbackOnLastSignal();

  std::function<void()> getCallback();

 private:
  std::shared_ptr<std::atomic_size_t> callbacksCount_;
  std::function<void()> callback_;
};

} // namespace facebook::velox::dwio::common::unit_loader_tools
