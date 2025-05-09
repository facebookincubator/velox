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

#include <folly/executors/IOThreadPoolExecutor.h>
#include "velox4j/connector/ExternalStream.h"

namespace velox4j {

class BlockingQueue : public ExternalStream {
 public:
  enum State { OPEN = 0, FINISHED = 1, CLOSED = 2 };

  static std::string stateToString(State state);

  BlockingQueue();

  // Delete copy/move CTORs.
  BlockingQueue(BlockingQueue&&) = delete;
  BlockingQueue(const BlockingQueue&) = delete;
  BlockingQueue& operator=(const BlockingQueue&) = delete;
  BlockingQueue& operator=(BlockingQueue&&) = delete;

  ~BlockingQueue() override;

  std::optional<facebook::velox::RowVectorPtr> read(
      facebook::velox::ContinueFuture& future) override;

  void put(facebook::velox::RowVectorPtr rowVector);

  void noMoreInput();

  bool empty() const;

 private:
  void ensureOpen() const;

  void ensureNotClosed() const;

  void close();

  mutable std::mutex mutex_;
  mutable std::condition_variable condVar_;
  std::queue<facebook::velox::RowVectorPtr> queue_;
  std::unique_ptr<folly::IOThreadPoolExecutor> waitExecutor_;
  std::vector<facebook::velox::ContinuePromise> promises_{};
  State state_{OPEN};
};
} // namespace velox4j
