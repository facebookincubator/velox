/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <optional>
#include <string_view>

#include "velox/buffer/Buffer.h"
#include "velox/buffer/BufferPool.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/compression/CompressionPolicy.h"

namespace facebook::nimble {

struct CompressionResult {
  CompressionType compressionType{CompressionType::Uncompressed};
  std::optional<Vector<char>> buffer;
};

struct ICompressor {
  virtual CompressionResult compress(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      DataType dataType,
      int bitWidth,
      const CompressionPolicy& compressionPolicy) = 0;

  virtual velox::BufferPtr uncompress(
      velox::memory::MemoryPool& pool,
      CompressionType compressionType,
      DataType dataType,
      std::string_view data,
      velox::BufferPool* bufferPool = nullptr) = 0;

  virtual std::optional<size_t> uncompressedSize(
      std::string_view data) const = 0;

  virtual CompressionType compressionType() = 0;

  virtual ~ICompressor() = default;
};

/// Allocates a buffer of at least 'bytes' capacity, trying the BufferPool
/// first if available, falling back to MemoryPool allocation.
inline velox::BufferPtr allocateBuffer(
    velox::memory::MemoryPool& memoryPool,
    velox::BufferPool* bufferPool,
    uint64_t bytes) {
  if (bufferPool != nullptr) {
    if (auto buf = bufferPool->get(bytes)) {
      buf->setSize(bytes);
      return buf;
    }
  }
  return velox::AlignedBuffer::allocate<char>(bytes, &memoryPool);
}

} // namespace facebook::nimble
