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

#include "velox/common/memory/MemoryManagerStrategy.h"
#include <unistd.h>

#include <folly/executors/task_queue/UnboundedBlockingQueue.h>
#include <folly/executors/thread_factory/InitThreadFactory.h>

namespace facebook::velox::memory {

bool MemoryManagerStrategy::initialized_;
MemoryManagerStrategy::MemoryManagerStrategyFactory
    MemoryManagerStrategy::factory_;

// static
MemoryManagerStrategy* MemoryManagerStrategy::instance() {
  static std::unique_ptr<MemoryManagerStrategy> strategy =
      (factory_) ? factory_() : createDefault();
  initialized_ = true;
  return strategy.get();
}

std::unique_ptr<MemoryManagerStrategy> MemoryManagerStrategy::createDefault() {
  return std::make_unique<DefaultMemoryManagerStrategy>();
}

} // namespace facebook::velox::memory
