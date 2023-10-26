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

#include "velox/dwio/common/ExecutorBarrier.h"
#include "folly/ExceptionWrapper.h"
#include "velox/dwio/common/exception/Exception.h"

namespace facebook::velox::dwio::common {

namespace {

class BarrierElement {
 public:
  BarrierElement(
      size_t& count,
      std::mutex& mutex,
      std::condition_variable& cv,
      folly::exception_wrapper& exception)
      : count_{count},
        mutex_{&mutex},
        cv_{cv},
        exception_{exception},
        executed_{false} {
    std::lock_guard lock{*mutex_};
    ++count_;
  }

  BarrierElement(BarrierElement&& other) noexcept
      : count_{other.count_},
        mutex_{other.mutex_},
        cv_{other.cv_},
        exception_{other.exception_},
        executed_{other.executed_} {
    // Move away
    other.mutex_ = nullptr;
  }

  BarrierElement(const BarrierElement& other) = delete;
  BarrierElement& operator=(BarrierElement&& other) = delete;
  BarrierElement& operator=(const BarrierElement& other) = delete;

  ~BarrierElement() {
    // If this object wasn't moved away
    if (mutex_) {
      std::lock_guard lock{*mutex_};
      if (!executed_) {
        if (!exception_.has_exception_ptr()) {
          exception_ =
              facebook::velox::dwio::common::exception::LoggedException(
                  "Function scheduled in executor was never invoked.");
        }
      }
      if (--count_ == 0) {
        cv_.notify_all();
      }
    }
  }

  void executed() {
    executed_ = true;
  }

 private:
  size_t& count_;
  std::mutex* mutex_;
  std::condition_variable& cv_;
  folly::exception_wrapper& exception_;
  bool executed_{false};
};

} // namespace

auto ExecutorBarrier::wrapMethod(folly::Func f) {
  return [f = std::move(f),
          this,
          barrierElement =
              BarrierElement(count_, mutex_, cv_, exception_)]() mutable {
    try {
      barrierElement.executed();
      f();
    } catch (const std::exception& e) {
      std::lock_guard lock{mutex_};
      if (!exception_.has_exception_ptr()) {
        exception_ = folly::exception_wrapper(
            ::folly::exception_wrapper::from_catch_ref_t{},
            std::current_exception(),
            e);
      }
    } catch (...) {
      std::lock_guard lock{mutex_};
      if (!exception_.has_exception_ptr()) {
        exception_ = folly::exception_wrapper(std::current_exception());
      }
    }
  };
}

void ExecutorBarrier::add(folly::Func f) {
  executor_.add(wrapMethod(std::move(f)));
}

void ExecutorBarrier::addWithPriority(folly::Func f, int8_t priority) {
  executor_.addWithPriority(wrapMethod(std::move(f)), priority);
}

} // namespace facebook::velox::dwio::common
