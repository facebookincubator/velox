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

#include <functional>
#include <variant>
#include <vector>

#include "folly/Range.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::dwio::common {

// Use this class if you want to return a result or a set of actions needed
// before achieving that result.
// For example, if IO needs to be performed, instead of blocking and doing the
// IO on the same thread, the reader can return the NEEDS_MORE_IO state and a
// callback, or a set of callbacks, if parallelizable, to perform the IO.
// Some states may not require actions.
// If there's a result, no action is expected.
template <typename StateEnum, StateEnum HasResultState, typename ResultType>
class StepResult {
 public:
  template <
      typename T = ResultType,
      typename std::enable_if_t<!std::is_void_v<T>, int> = 0>
  explicit StepResult(std::conditional_t<true, ResultType, T> result)
      : state_{HasResultState}, resultOrActions_(std::move(result)) {}

  explicit StepResult(StateEnum state = HasResultState)
      : state_{state}, resultOrActions_{std::vector<std::function<void()>>{}} {
    VELOX_CHECK(
        std::is_void_v<ResultType> || state != HasResultState,
        "Result state must have a result for non-void ResultType");
  }

  StepResult(StateEnum state, std::function<void()> action)
      : state_{state}, resultOrActions_{std::move(action)} {
    VELOX_CHECK(
        std::is_void_v<ResultType> || state != HasResultState,
        "Result state must have a result for non-void ResultType");
  }

  StepResult(StateEnum state, std::vector<std::function<void()>> actions)
      : state_{state}, resultOrActions_{std::move(actions)} {
    VELOX_CHECK(
        std::is_void_v<ResultType> || state != HasResultState,
        "Result state must have a result for non-void ResultType");
  }

  StateEnum state() const {
    return state_;
  }

  template <typename T = ResultType>
  typename std::enable_if_t<!std::is_void_v<T>, bool> hasResult() const {
    return resultOrActions_.index() == 2;
  }

  template <typename T = ResultType>
  typename std::enable_if_t<!std::is_void_v<T>, ResultType> result() const {
    VELOX_CHECK(hasResult(), "Result is not set");
    return std::get<ResultType>(resultOrActions_);
  }

  folly::Range<std::function<void()>*> actions() {
    switch (resultOrActions_.index()) {
      case 0: // Action
        return {&std::get<std::function<void()>>(resultOrActions_), 1};
      case 1: // Actions
        return {
            std::get<std::vector<std::function<void()>>>(resultOrActions_)
                .data(),
            std::get<std::vector<std::function<void()>>>(resultOrActions_)
                .size()};
      case 2: // Result (can't happen for void)
        return {};
      default:
        VELOX_FAIL("Unexpected variant index");
    }
  }

  template <
      typename OtherStateEnum,
      OtherStateEnum OtherHasResultState,
      typename OtherResultType>
  void mergeActionsFrom(
      StepResult<OtherStateEnum, OtherHasResultState, OtherResultType> other) {
    switch (other.resultOrActions_.index()) {
      case 0: // Action
        mergeAction(
            std::move(std::get<std::function<void()>>(other.resultOrActions_)));
        break;
      case 1: // Actions
        for (auto& action : std::get<std::vector<std::function<void()>>>(
                 other.resultOrActions_)) {
          mergeAction(std::move(action));
        }
        break;
      case 2: // Result (can't happen for void)
        VELOX_FAIL("Can't merge actions of class that has a result");
    }
  }

  size_t runAllActions() {
    size_t numActions = 0;
    switch (resultOrActions_.index()) {
      case 0: // Action
        std::get<std::function<void()>>(resultOrActions_)();
        numActions = 1;
        break;
      case 1: // Actions
      {
        auto& actions =
            std::get<std::vector<std::function<void()>>>(resultOrActions_);
        numActions = actions.size();
        for (auto& action : actions) {
          action();
        }
        break;
      }
      default:
        break;
    }
    return numActions;
  }

 private:
  // Otherwise mergeActionsFrom can't read other's private members, since
  // they're different classes
  template <
      typename OtherStateEnum,
      OtherStateEnum OtherHasResultState,
      typename OtherResultType>
  friend class StepResult;

  void mergeAction(std::function<void()> action) {
    switch (resultOrActions_.index()) {
      case 0: // Action
        resultOrActions_ = std::vector<std::function<void()>>{
            std::move(std::get<std::function<void()>>(resultOrActions_)),
            std::move(action)};
        break;
      case 1: // Actions
        std::get<std::vector<std::function<void()>>>(resultOrActions_)
            .push_back(std::move(action));
        break;
      case 2: // Result (can't happen for void)
        VELOX_FAIL("Can't merge actions if destination class has a result");
    }
  }

  StateEnum state_;
  std::conditional_t<
      std::is_void_v<ResultType>,
      std::variant<std::function<void()>, std::vector<std::function<void()>>>,
      std::variant<
          std::function<void()>,
          std::vector<std::function<void()>>,
          ResultType>>
      resultOrActions_;
};

} // namespace facebook::velox::dwio::common
