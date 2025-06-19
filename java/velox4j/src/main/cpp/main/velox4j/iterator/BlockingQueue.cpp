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
#include "velox4j/iterator/BlockingQueue.h"

#include <fmt/core.h>
#include <folly/Unit.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <velox/common/base/Exceptions.h>
#include <exception>
#include <utility>

namespace facebook::velox4j {
using namespace facebook::velox;

std::string BlockingQueue::stateToString(State state) {
  switch (state) {
    case OPEN:
      return "OPEN";
    case FINISHED:
      return "FINISHED";
    case CLOSED:
      return "CLOSED";
  }
  VELOX_FAIL("unknown state");
}

BlockingQueue::BlockingQueue() {
  waitExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(1);
}

BlockingQueue::~BlockingQueue() {
  close();
}

std::optional<RowVectorPtr> BlockingQueue::read(ContinueFuture& future) {
  {
    std::unique_lock lock(mutex_);
    ensureNotClosed();
    if (!queue_.empty()) {
      auto rowVector = queue_.front();
      queue_.pop();
      return rowVector;
    }
    // Queue is empty, checks the current state. If it's FINISHED,
    // returns a nullptr to Velox to signal the stream is gracefully ended.
    if (state_ == FINISHED) {
      return nullptr;
    }
  }

  // Blocked. Async wait for a new element.
  auto [readPromise, readFuture] =
      makeVeloxContinuePromiseContract(fmt::format("BlockingQueue::read"));
  // Returns a future that is fulfilled immediately to signal Velox
  // that this stream is still open and is currently waiting for input.
  future = std::move(readFuture);
  {
    std::lock_guard l(mutex_);
    VELOX_CHECK(promises_.empty());
    promises_.emplace_back(std::move(readPromise));
  }

  waitExecutor_->add([this]() -> void {
    std::unique_lock lock(mutex_);
    // Async wait for a new element.
    notEmptyCv_.wait(
        lock, [this]() { return state_ != OPEN || !queue_.empty(); });
    switch (state_) {
      case OPEN: {
        VELOX_CHECK(!queue_.empty());
      } // Falls through.
      case FINISHED: {
        VELOX_CHECK(promises_.size() == 1);
        for (auto& p : promises_) {
          p.setValue();
        }
        promises_.clear();
      } break;
      case CLOSED: {
        try {
          VELOX_FAIL("BlockingQueue was just closed");
        } catch (const std::exception& e) {
          // Velox should guarantee the continue future is only requested once
          // while it's not fulfilled.
          VELOX_CHECK(promises_.size() == 1);
          for (auto& p : promises_) {
            p.setException(e);
            promises_.clear();
            return;
          }
        }
      } break;
    }
  });

  return std::nullopt;
}

void BlockingQueue::put(RowVectorPtr rowVector) {
  {
    std::lock_guard lock(mutex_);
    ensureOpen();
    queue_.push(std::move(rowVector));
  }
  notEmptyCv_.notify_one();
}

void BlockingQueue::noMoreInput() {
  {
    std::lock_guard lock(mutex_);
    ensureOpen();
    state_ = FINISHED;
  }
  notEmptyCv_.notify_one();
}

bool BlockingQueue::empty() const {
  {
    std::lock_guard lock(mutex_);
    return queue_.empty();
  }
}

void BlockingQueue::ensureOpen() const {
  VELOX_CHECK(
      state_ == OPEN,
      "Queue is not open. Current state is {}",
      stateToString(state_));
}

void BlockingQueue::ensureNotClosed() const {
  VELOX_CHECK(state_ != CLOSED, "Queue was closed.");
}

void BlockingQueue::close() {
  {
    std::lock_guard lock(mutex_);
    ensureNotClosed();
    state_ = CLOSED;
  }
  notEmptyCv_.notify_one();
  waitExecutor_->join();
}
} // namespace facebook::velox4j
