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

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::io_sources {

/// Fixed-budget pool of pinned host buffers used to stage ABFS reads
/// before issuing `cudaMemcpyAsync` to device memory. Buffers are sized
/// on-demand. Pinned memory is expensive to allocate, so the pool reuses
/// allocations across reads. Acquisition blocks while the budget is
/// exhausted; this provides natural backpressure when many drivers race
/// for buffers.
class CudfAbfsPinnedBufferPool {
 public:
  /// `maxOutstandingBytes` is the maximum sum of buffer sizes that may
  /// be outstanding at any time. The pool itself does not pre-allocate.
  explicit CudfAbfsPinnedBufferPool(size_t maxOutstandingBytes);

  ~CudfAbfsPinnedBufferPool();

  CudfAbfsPinnedBufferPool(const CudfAbfsPinnedBufferPool&) = delete;
  CudfAbfsPinnedBufferPool& operator=(const CudfAbfsPinnedBufferPool&) = delete;

  /// Owning handle over a slice of pinned memory. Returns to the pool on
  /// destruction.
  class Handle {
   public:
    Handle() = default;
    Handle(CudfAbfsPinnedBufferPool* pool, uint8_t* data, size_t size)
        : pool_(pool), data_(data), size_(size) {}
    ~Handle() {
      release();
    }

    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    Handle(Handle&& other) noexcept
        : pool_(other.pool_), data_(other.data_), size_(other.size_) {
      other.pool_ = nullptr;
      other.data_ = nullptr;
      other.size_ = 0;
    }

    Handle& operator=(Handle&& other) noexcept {
      if (this != &other) {
        release();
        pool_ = other.pool_;
        data_ = other.data_;
        size_ = other.size_;
        other.pool_ = nullptr;
        other.data_ = nullptr;
        other.size_ = 0;
      }
      return *this;
    }

    [[nodiscard]] uint8_t* data() const {
      return data_;
    }

    [[nodiscard]] size_t size() const {
      return size_;
    }

   private:
    void release();

    CudfAbfsPinnedBufferPool* pool_{nullptr};
    uint8_t* data_{nullptr};
    size_t size_{0};
    friend class CudfAbfsPinnedBufferPool;
  };

  /// Reserves a pinned buffer of exactly `size` bytes. Blocks if the
  /// outstanding budget is exceeded until other handles return memory.
  /// `size == 0` is allowed and yields a non-owning handle.
  Handle acquire(size_t size);

  /// Process-wide shared pool, sized by the connector property
  /// `cudf.hive.abfs-pinned-buffer-bytes`. The first caller sets the
  /// budget; subsequent calls see the same pool regardless of the
  /// argument.
  static CudfAbfsPinnedBufferPool& shared(size_t budgetBytes);

 private:
  friend class Handle;

  struct Block {
    uint8_t* data{nullptr};
    size_t size{0};
    bool inUse{false};
  };

  void releaseBlock(uint8_t* data, size_t accountedSize);

  const size_t maxOutstandingBytes_;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<Block> blocks_;
  size_t outstandingBytes_{0};
};

} // namespace facebook::velox::cudf_velox::connector::hive::io_sources
