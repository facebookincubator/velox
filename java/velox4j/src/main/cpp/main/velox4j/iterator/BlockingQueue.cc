#include "BlockingQueue.h"

namespace velox4j {
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
    condVar_.wait(lock, [this]() { return state_ != OPEN || !queue_.empty(); });
    switch (state_) {
      case OPEN: {
        VELOX_CHECK(!queue_.empty());
        // Fall through.
      }
      case FINISHED: {
        VELOX_CHECK(promises_.size() == 1);
        for (auto& p : promises_) {
          p.setValue();
        }
        promises_.clear();
        break;
      }
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
        break;
      }
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
  condVar_.notify_one();
}

void BlockingQueue::noMoreInput() {
  {
    std::lock_guard lock(mutex_);
    ensureOpen();
    state_ = FINISHED;
  }
  condVar_.notify_one();
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
  condVar_.notify_one();
  waitExecutor_->join();
}
} // namespace velox4j
