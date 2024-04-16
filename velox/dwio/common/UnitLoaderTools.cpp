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

#include "velox/dwio/common/UnitLoaderTools.h"

#include <atomic>

#include "folly/synchronization/CallOnce.h"

namespace facebook::velox::dwio::common::unit_loader_tools {

namespace {

class EnsureCallOnce {
 public:
  explicit EnsureCallOnce(
      std::function<void()> func,
      std::shared_ptr<std::atomic_size_t> counter = nullptr)
      : flag_{std::make_shared<folly::once_flag>()}, func_{std::move(func)} {
    if (counter) {
      ++*counter;
    }
  }

  EnsureCallOnce(EnsureCallOnce&& other) noexcept {
    func_ = std::move(other.func_);
    flag_ = std::move(other.flag_);
  }

  EnsureCallOnce(const EnsureCallOnce& other)
      : flag_{other.flag_}, func_{other.func_} {}

  EnsureCallOnce& operator=(const EnsureCallOnce&) = delete;
  EnsureCallOnce& operator=(EnsureCallOnce&&) = delete;

  void operator()() {
    folly::call_once(*flag_, func_);
  }

 private:
  std::shared_ptr<folly::once_flag> flag_;
  std::function<void()> func_;
};

class EnsureAlwaysCalls {
 public:
  explicit EnsureAlwaysCalls(std::function<void()> func)
      : func_{std::move(func)} {}

  EnsureAlwaysCalls(EnsureAlwaysCalls&& other) noexcept {
    func_ = std::move(other.func_);
  }

  EnsureAlwaysCalls(const EnsureAlwaysCalls& other) : func_{other.func_} {}

  EnsureAlwaysCalls& operator=(const EnsureAlwaysCalls&) = delete;
  EnsureAlwaysCalls& operator=(EnsureAlwaysCalls&&) = delete;

  void operator()() {
    func_();
  }

  ~EnsureAlwaysCalls() {
    if (func_) {
      func_();
    }
  }

 private:
  std::function<void()> func_;
};

} // namespace

CallbackOnLastSignal::CallbackOnLastSignal(
    std::function<void()> callback)
      : callbacksCount_{std::make_shared<std::atomic_size_t>(0)},
      callback_{ callback ?[callback = EnsureCallOnce(std::move(callback)),
                         callbacksCount = callbacksCount_]() mutable {
        // Only the last one will trigger the call
        if (--*callbacksCount == 0) {
          callback();
        }
      } : std::function<void()>()} {}

CallbackOnLastSignal::~CallbackOnLastSignal() {}

std::function<void()> CallbackOnLastSignal::getCallback() {
  if (!callback_) {
    return std::function<void()>();
  }
  return EnsureAlwaysCalls(EnsureCallOnce(callback_, callbacksCount_));
}

} // namespace facebook::velox::dwio::common::unit_loader_tools
