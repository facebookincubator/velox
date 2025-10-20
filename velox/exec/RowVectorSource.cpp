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
#include "velox/exec/RowVectorSource.h"
#include <boost/circular_buffer.hpp>

namespace facebook::velox::exec {
namespace {
class ScopedPromiseNotification {
 public:
  explicit ScopedPromiseNotification(size_t initSize) {
    promises_.reserve(initSize);
  }

  ~ScopedPromiseNotification() {
    for (auto& promise : promises_) {
      promise.setValue();
    }
  }

  void add(std::vector<ContinuePromise>&& promises) {
    promises_.reserve(promises_.size() + promises.size());
    for (auto& promise : promises) {
      promises_.emplace_back(std::move(promise));
    }
    promises.clear();
  }

  void add(ContinuePromise&& promise) {
    promises_.emplace_back(std::move(promise));
  }

 private:
  std::vector<ContinuePromise> promises_;
};

class RowVectorSourceQueue {
 public:
  explicit RowVectorSourceQueue(int queueSize) : data_(queueSize) {}

  BlockingReason next(
      RowVectorPtr& data,
      ContinueFuture* future,
      ScopedPromiseNotification& notification) {
    data.reset();
    if (data_.empty()) {
      if (atEnd_) {
        return BlockingReason::kNotBlocked;
      }
      consumerPromises_.emplace_back("RowVectorSourceQueue::next");
      *future = consumerPromises_.back().getSemiFuture();
      return BlockingReason::kWaitForProducer;
    }

    data = data_.front();
    // advance to next batch.
    data_.pop_front();
    notifyProducers(notification);
    return BlockingReason::kNotBlocked;
  }

  BlockingReason enqueue(
      RowVectorPtr input,
      ContinueFuture* future,
      ScopedPromiseNotification& notification) {
    if (!input) {
      atEnd_ = true;
      notifyConsumers(notification);
      return BlockingReason::kNotBlocked;
    }
    VELOX_CHECK(!data_.full(), "RowVectorSourceQueue is full");

    for (auto& child : input->children()) {
      child->loadedVector();
    }

    data_.push_back(input);
    notifyConsumers(notification);

    if (data_.full()) {
      producerPromises_.emplace_back("RowVectorSourceQueue::enqueue");
      *future = producerPromises_.back().getSemiFuture();
      return BlockingReason::kWaitForConsumer;
    }
    return BlockingReason::kNotBlocked;
  }

 private:
  void notifyConsumers(ScopedPromiseNotification& notification) {
    notification.add(std::move(consumerPromises_));
    VELOX_CHECK(consumerPromises_.empty());
  }

  void notifyProducers(ScopedPromiseNotification& notification) {
    notification.add(std::move(producerPromises_));
    VELOX_CHECK(producerPromises_.empty());
  }

  bool atEnd_{false};
  boost::circular_buffer<RowVectorPtr> data_;
  std::vector<ContinuePromise> consumerPromises_;
  std::vector<ContinuePromise> producerPromises_;
};

class LocalRowVectorSource final : public RowVectorSource {
 public:
  explicit LocalRowVectorSource(int queueSize)
      : queue_(RowVectorSourceQueue(queueSize)) {}

  BlockingReason next(RowVectorPtr& data, ContinueFuture* future) override {
    ScopedPromiseNotification notification(1);
    return queue_.withWLock(
        [&](auto& queue) { return queue.next(data, future, notification); });
  }

  BlockingReason enqueue(RowVectorPtr input, ContinueFuture* future) override {
    ScopedPromiseNotification notification(1);
    return queue_.withWLock([&](auto& queue) {
      return queue.enqueue(input, future, notification);
    });
  }

  std::string name() override {
    return "LocalRowVectorSource";
  }

 private:
  folly::Synchronized<RowVectorSourceQueue> queue_;
};

} // namespace

// static
std::shared_ptr<RowVectorSource> RowVectorSource::createRowVectorSource(
    int queueSize) {
  return std::make_shared<LocalRowVectorSource>(queueSize);
}
} // namespace facebook::velox::exec
