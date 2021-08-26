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
/*

#pragma once

#include <folly/futures/Future.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MemoryUsageTracker.h"

namespace facebook::velox::memory {
class MemoryPool;
class ScopedMemoryPool;

enum class UsageLevel { kDefault, kProcess, kSpiller, kCache, kQuery };

class MemoryConsumer {
 public:
  virtual ~MemoryConsumer() {}

  virtual void updateMemoryUsageConfig(const MemoryUsageConfig& config) = 0;

  // Returns the number of bytes that this memory consumer overcomitted.
  virtual int64_t getOvercommittedMemory() const = 0;

  // Returns the number of bytes that may be recoverable with
  // tryRecoverMemory().
  virtual int64_t getRecoverableMemory() const = 0;

  //  Recovers memory. Implementations may for example spill or evict
  //  caches. Returns the number of bytes actually freed.  size
  //  specifies a target number of bytes to recover.
  virtual int64_t recover(int64_t size) = 0;
};

class MemoryManagerStrategy {
 public:
  virtual ~MemoryManagerStrategy() {}

  // returns true if this can resize initial limits of consumers.
  virtual bool canResize() const = 0;

  // Returns true if the memory manager allows overcommit for
  // individual memory pool or mapped memory.
  virtual bool allowOvercommit() const = 0;

  // Returns true if the memory manager may change memory usage quota
  // dynamically, depending on the number of queries/tasks in the system.
  // We use this to implement a fair memory usage policy.
  virtual bool canChangeLimitsDynamically() const = 0;

  // Returns a shared pointer to the memory usage quota.
  // UsageType specifies whether it is for Process, Query, Task or Operators
  // If canChangeQuotaDynamically() returns true, quota may change dynamically
  // as the number of MemoryConsumers changes and registered MemoryConsumers
  // will be/ updated through 'updateMemoryUsageConfig()'.
  virtual MemoryUsageConfig getDefaultMemoryUsageConfig(
      UsageLevel Level) const = 0;

  // To register a memory consumer with MemoryManager.
  virtual void registerConsumer(
      MemoryConsumer* consumer,
      const std::weak_ptr<MemoryConsumer>& consumerPtr) = 0;

  // To unregister a memory consumer with MemoryManager.
  virtual void unregisterConsumer(MemoryConsumer* consumer) = 0;

  // Tries to recover memory so that 'requester' can allocate 'size'
  // bytes of new memory. Returns true if the limit of 'requester' was
  // increased by at least 'size'.
  virtual bool recover(
      std::shared_ptr<MemoryConsumer> requester,
      UsageType type,
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

  bool allowOvercommit() const override {
    return true;
  }

  bool canChangeLimitsDynamically() const override {
    return false;
  }

  MemoryUsageConfig getDefaultMemoryUsageConfig(
      UsageLevel /*level*/) const override {
  return MemoryUsageConfig();
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
}
;

class DefaultMemoryManagerStrategy : public MemoryManagerStrategyBase {
 public:
  // No op.
  bool recover(
      std::shared_ptr<MemoryConsumer> /*requester*/,
      UsageType /*type*/,
      int64_t /*size*/) override {
    return false;
  }
};

} // namespace facebook::velox::memory
