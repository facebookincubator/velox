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

#include <folly/futures/Future.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MemoryUsageTracker.h"

namespace facebook::velox::memory {
class MemoryPool;
class ScopedMemoryPool;

class MemoryConsumer {
 public:
  virtual ~MemoryConsumer() {}

  // Returns the tracker with the current usage and limits.
  virtual memory::MemoryUsageTracker& tracker() const = 0;

  // Returns the number of bytes that may be reclaimable with
  // reclaim().
  virtual int64_t reclaimableBytes() const = 0;

  //  Reclaims memory. Implementations may for example spill or evict
  //  caches. 'size' specifies a target number of bytes to reclaim. The
  //  actual effect on memory usage is seen in tracker(). This does not
  //  guarantee any result. Implementations have additional
  //  requirements for using this method, e.g. a Task must be paused.
  virtual void reclaim(int64_t size) = 0;
};

class MemoryManagerStrategy {
 public:
  virtual ~MemoryManagerStrategy() {}

  // returns true if this can resize limits of consumers.
  virtual bool canResize() const = 0;

  // Registers a memory consumer with MemoryManager.
  virtual void registerConsumer(
      MemoryConsumer* consumer,
      const std::weak_ptr<MemoryConsumer>& consumerPtr) = 0;

  // Unregisters a memory consumer with MemoryManager.
  virtual void unregisterConsumer(MemoryConsumer* consumer) = 0;

  // Tries to reclaim memory so that 'requester' can allocate 'size'
  // bytes of new memory. Returns true if the user memory limit of
  // 'requester' was increased by at least 'size'.
  virtual bool reclaim(
      std::shared_ptr<MemoryConsumer> requester,
      int64_t size) = 0;

  static MemoryManagerStrategy* instance();

  using MemoryManagerStrategyFactory =
      std::function<std::unique_ptr<MemoryManagerStrategy>()>;

  static void registerFactory(MemoryManagerStrategyFactory factory) {
    VELOX_CHECK(
        !initialized_,
        "Registering factory after creating MemoryManagerStrategy object");
    factory_ = factory;
  }

 private:
  static std::unique_ptr<MemoryManagerStrategy> createDefault();

  static MemoryManagerStrategyFactory factory_;
  static bool initialized_;
};

class MemoryManagerStrategyBase : public MemoryManagerStrategy {
 public:
  MemoryManagerStrategyBase() = default;

  bool canResize() const override {
    return false;
  }

  void registerConsumer(
      MemoryConsumer* consumer,
      const std::weak_ptr<MemoryConsumer>& consumerPtr) override {
    std::lock_guard<std::mutex> l(mutex_);
    consumers_.emplace(consumer, consumerPtr);
  }

  void unregisterConsumer(MemoryConsumer* consumer) override {
    std::lock_guard<std::mutex> l(mutex_);
    consumers_.erase(consumer);
  }

 protected:
  using ConsumerMap =
      std::unordered_map<MemoryConsumer*, std::weak_ptr<MemoryConsumer>>;

  std::mutex mutex_;
  ConsumerMap consumers_;
};

class DefaultMemoryManagerStrategy : public MemoryManagerStrategyBase {
 public:
  // No op.
  bool reclaim(std::shared_ptr<MemoryConsumer> /*requester*/, int64_t /*size*/)
      override {
    return false;
  }
};

} // namespace facebook::velox::memory
