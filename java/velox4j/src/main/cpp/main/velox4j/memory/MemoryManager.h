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

#include <arrow/memory_pool.h>
#include <velox/common/config/Config.h>
#include <velox/common/memory/Memory.h>
#include <velox/common/memory/MemoryPool.h>
#include <memory>
#include <string>
#include <unordered_map>

#include "velox4j/memory/AllocationListener.h"
#include "velox4j/memory/ArrowMemoryPool.h"

namespace facebook::velox4j {

/// A memory manager is supposed to manage all the memory allocations under one
/// or multiple certain JNI sessions in Velox4J.
class MemoryManager {
 public:
  explicit MemoryManager(std::unique_ptr<AllocationListener> listener);

  virtual ~MemoryManager();

  MemoryManager(const MemoryManager&) = delete;
  MemoryManager(MemoryManager&&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;
  MemoryManager& operator=(MemoryManager&&) = delete;

  facebook::velox::memory::MemoryPool* getVeloxPool(
      const std::string& name,
      const facebook::velox::memory::MemoryPool::Kind& kind);

  arrow::MemoryPool* getArrowPool(const std::string& name);

 private:
  // Destructs the members in this memory manager with a leakage-check
  // conducted.
  // If there are outstanding memory allocations (which means memory is leaked)
  // , returns false. Returns true otherwise.
  bool tryDestruct();

  const std::unique_ptr<AllocationListener> listener_;
  std::unique_ptr<MemoryAllocator> arrowAllocator_;
  std::unordered_map<std::string, std::unique_ptr<arrow::MemoryPool>>
      arrowPoolRefs_;
  std::unique_ptr<facebook::velox::memory::MemoryManager> veloxMemoryManager_;
  std::shared_ptr<facebook::velox::memory::MemoryPool> veloxRootPool_;
  std::unordered_map<
      std::string,
      std::shared_ptr<facebook::velox::memory::MemoryPool>>
      veloxPoolRefs_;
};
} // namespace facebook::velox4j
