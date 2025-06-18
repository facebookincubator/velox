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
#include <arrow/status.h>
#include <stdint.h>
#include <atomic>
#include <memory>
#include <string>

#include "velox4j/memory/AllocationListener.h"

namespace facebook::velox4j {

/// The allocator abstraction for constructing an ArrowMemoryPool. The pool
/// created with invoke this allocator for underlying allocations.
class MemoryAllocator {
 public:
  virtual ~MemoryAllocator() = default;

  // Allocates a memory region of size bytes.
  virtual bool allocate(int64_t size, void** out) = 0;

  /// Allocates a memory region of size bytes. The region should be
  /// initialized with zeros.
  virtual bool allocateZeroFilled(int64_t nmemb, int64_t size, void** out) = 0;

  // Allocates a memory region of size bytes with specific alignment.
  virtual bool
  allocateAligned(uint64_t alignment, int64_t size, void** out) = 0;

  // Reallocates a memory region of size bytes.
  virtual bool
  reallocate(void* p, int64_t size, int64_t newSize, void** out) = 0;

  // Reallocates a memory region of size bytes with specific alignment.
  virtual bool reallocateAligned(
      void* p,
      uint64_t alignment,
      int64_t size,
      int64_t newSize,
      void** out) = 0;

  // Free the memory region.
  virtual bool free(void* p, int64_t size) = 0;

  // Returns the number of bytes this allocator currently uses.
  virtual int64_t usedBytes() const = 0;

  // Returns the number of bytes this allocator mostly used.
  virtual int64_t peakBytes() const = 0;
};

/// An allocator decorator that listen on the allocations with a given
/// AllocationListener then dispatch the calls to the delegated allocator.
class ListenableMemoryAllocator final : public MemoryAllocator {
 public:
  explicit ListenableMemoryAllocator(
      MemoryAllocator* delegated,
      AllocationListener* listener)
      : delegated_(delegated), listener_(listener) {}

  bool allocate(int64_t size, void** out) override;

  bool allocateZeroFilled(int64_t nmemb, int64_t size, void** out) override;

  bool allocateAligned(uint64_t alignment, int64_t size, void** out) override;

  bool reallocate(void* p, int64_t size, int64_t newSize, void** out) override;

  bool reallocateAligned(
      void* p,
      uint64_t alignment,
      int64_t size,
      int64_t newSize,
      void** out) override;

  bool free(void* p, int64_t size) override;

  int64_t usedBytes() const override;

  int64_t peakBytes() const override;

 private:
  void updateUsage(int64_t size);

  MemoryAllocator* const delegated_;
  AllocationListener* const listener_;
  std::atomic_int64_t usedBytes_{0L};
  std::atomic_int64_t peakBytes_{0L};
};

/// A simple malloc allocator.
class StdMemoryAllocator final : public MemoryAllocator {
 public:
  bool allocate(int64_t size, void** out) override;

  bool allocateZeroFilled(int64_t nmemb, int64_t size, void** out) override;

  bool allocateAligned(uint64_t alignment, int64_t size, void** out) override;

  bool reallocate(void* p, int64_t size, int64_t newSize, void** out) override;

  bool reallocateAligned(
      void* p,
      uint64_t alignment,
      int64_t size,
      int64_t newSize,
      void** out) override;

  bool free(void* p, int64_t size) override;

  int64_t usedBytes() const override;

  int64_t peakBytes() const override;

 private:
  std::atomic_int64_t bytes_{0};
};

/// An Arrow memory pool implementation used by Velox4J that routes Arrow
/// allocation calls to a given MemoryAllocator.
class ArrowMemoryPool final : public arrow::MemoryPool {
 public:
  explicit ArrowMemoryPool(MemoryAllocator* allocator)
      : allocator_(allocator) {}

  arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out)
      override;

  arrow::Status Reallocate(
      int64_t oldSize,
      int64_t newSize,
      int64_t alignment,
      uint8_t** ptr) override;

  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;

  int64_t bytes_allocated() const override;

  int64_t total_bytes_allocated() const override;

  int64_t num_allocations() const override;

  std::string backend_name() const override;

 private:
  MemoryAllocator* allocator_;
};

std::shared_ptr<MemoryAllocator> defaultMemoryAllocator();

std::shared_ptr<arrow::MemoryPool> defaultArrowMemoryPool();

} // namespace facebook::velox4j
