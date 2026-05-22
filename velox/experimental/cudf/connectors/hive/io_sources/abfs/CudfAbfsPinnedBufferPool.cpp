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

#include "velox/experimental/cudf/connectors/hive/io_sources/abfs/CudfAbfsPinnedBufferPool.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/utilities/error.hpp>

#include <algorithm>

namespace facebook::velox::cudf_velox::connector::hive::io_sources {

namespace {

void freePinned(uint8_t* ptr) noexcept {
  if (ptr != nullptr) {
    // Best-effort free; errors here only matter on shutdown.
    (void)cudaFreeHost(ptr);
  }
}

uint8_t* allocPinned(size_t size) {
  void* raw{nullptr};
  CUDF_CUDA_TRY(cudaHostAlloc(&raw, size, cudaHostAllocDefault));
  return static_cast<uint8_t*>(raw);
}

} // namespace

CudfAbfsPinnedBufferPool::CudfAbfsPinnedBufferPool(size_t maxOutstandingBytes)
    : maxOutstandingBytes_(maxOutstandingBytes) {}

CudfAbfsPinnedBufferPool::~CudfAbfsPinnedBufferPool() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& block : blocks_) {
    freePinned(block.data);
  }
}

void CudfAbfsPinnedBufferPool::Handle::release() {
  if (pool_ != nullptr and data_ != nullptr) {
    pool_->releaseBlock(data_, size_);
  }
  pool_ = nullptr;
  data_ = nullptr;
  size_ = 0;
}

CudfAbfsPinnedBufferPool::Handle CudfAbfsPinnedBufferPool::acquire(
    size_t size) {
  if (size == 0) {
    return Handle{this, nullptr, 0};
  }
  VELOX_CHECK_LE(
      size,
      maxOutstandingBytes_,
      "Requested ABFS pinned buffer ({} bytes) exceeds pool budget ({} bytes)",
      size,
      maxOutstandingBytes_);

  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(
      lock, [&]() { return outstandingBytes_ + size <= maxOutstandingBytes_; });

  // Prefer reusing an idle block that already fits.
  auto it =
      std::find_if(blocks_.begin(), blocks_.end(), [&](const Block& block) {
        return not block.inUse and block.size >= size;
      });
  if (it != blocks_.end()) {
    it->inUse = true;
    outstandingBytes_ += size;
    return Handle{this, it->data, size};
  }

  // Evict an idle, undersized block to keep the working set bounded.
  auto idleIt =
      std::find_if(blocks_.begin(), blocks_.end(), [](const Block& block) {
        return not block.inUse;
      });
  if (idleIt != blocks_.end()) {
    freePinned(idleIt->data);
    blocks_.erase(idleIt);
  }

  blocks_.push_back(Block{allocPinned(size), size, /*inUse=*/true});
  outstandingBytes_ += size;
  return Handle{this, blocks_.back().data, size};
}

void CudfAbfsPinnedBufferPool::releaseBlock(
    uint8_t* data,
    size_t accountedSize) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it =
        std::find_if(blocks_.begin(), blocks_.end(), [&](const Block& block) {
          return block.data == data;
        });
    if (it == blocks_.end()) {
      return;
    }
    it->inUse = false;
    outstandingBytes_ -= std::min<size_t>(outstandingBytes_, accountedSize);
  }
  cv_.notify_all();
}

CudfAbfsPinnedBufferPool& CudfAbfsPinnedBufferPool::shared(size_t budgetBytes) {
  static std::unique_ptr<CudfAbfsPinnedBufferPool> instance;
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    instance = std::make_unique<CudfAbfsPinnedBufferPool>(budgetBytes);
  });
  return *instance;
}

} // namespace facebook::velox::cudf_velox::connector::hive::io_sources
