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

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "velox/buffer/Buffer.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/dwio/nimble/common/Types.h"

namespace facebook::nimble {

namespace test {
class MetadataBufferTestHelper;
} // namespace test

class MetadataBuffer {
 public:
  // Creates from decompressed data in a pool-tracked buffer.
  explicit MetadataBuffer(velox::BufferPtr buffer);

  // Creates from a cache pin with decompressed data.
  explicit MetadataBuffer(velox::cache::CachePin pin);

  ~MetadataBuffer();
  MetadataBuffer(const MetadataBuffer&) = delete;
  MetadataBuffer& operator=(const MetadataBuffer&) = delete;
  MetadataBuffer(MetadataBuffer&&) noexcept;
  MetadataBuffer& operator=(MetadataBuffer&&) noexcept;

  /// Creates a shallow copy sharing the same underlying buffer or cache pin.
  MetadataBuffer clone() const;

  /// Creates a zero-copy MetadataBuffer viewing a sub-range of this buffer's
  /// content. Shares the underlying buffer or cache pin.
  std::unique_ptr<MetadataBuffer> slice(uint64_t offset, uint64_t size) const;

  std::string_view content() const {
    const auto data = pin_.empty() ? bufferContent() : cacheContent();
    return data.substr(offset_, size_);
  }

  /// Decompresses metadata from a pool-tracked buffer. Returns the buffer
  /// directly for uncompressed data (zero copy), or decompresses for Zstd.
  static velox::BufferPtr decompress(
      velox::BufferPtr data,
      CompressionType type,
      velox::memory::MemoryPool* pool);

  /// Decompresses metadata from a string_view and returns a BufferPtr.
  static velox::BufferPtr decompress(
      std::string_view data,
      CompressionType type,
      velox::memory::MemoryPool* pool);

 private:
  MetadataBuffer(velox::BufferPtr buffer, uint64_t offset, uint64_t size);
  MetadataBuffer(velox::cache::CachePin pin, uint64_t offset, uint64_t size);

  inline std::string_view bufferContent() const {
    return {buffer_->as<char>(), buffer_->size()};
  }

  inline std::string_view cacheContent() const {
    auto* entry = pin_.checkedEntry();
    return {entry->contiguousData(), static_cast<size_t>(entry->size())};
  }

  // Exactly one of buffer_ or pin_ is set.
  // buffer_: holds decompressed data in a pool-tracked buffer (direct path).
  // pin_: holds decompressed data in a contiguous AsyncDataCache entry
  //   (cached path, keeps the cache entry pinned while this buffer is alive).
  velox::BufferPtr buffer_;
  velox::cache::CachePin pin_;
  // Sub-range within the underlying storage. Default (0, full size) covers
  // the entire content. Set by slice() for sub-range views.
  uint64_t offset_{0};
  uint64_t size_{0};

  friend class test::MetadataBufferTestHelper;
};

class Section {
 public:
  explicit Section(MetadataBuffer&& buffer) : buffer_{std::move(buffer)} {}

  std::string_view content() const {
    return buffer_.content();
  }
  explicit operator std::string_view() const {
    return content();
  }

  /// Returns a reference to the underlying MetadataBuffer.
  const MetadataBuffer& buffer() const {
    return buffer_;
  }

 private:
  MetadataBuffer buffer_;
};

class MetadataSection {
 public:
  /// 'uncompressedSize' is nullopt for files written before the
  /// uncompressed_size field was added to the FlatBuffers footer.
  /// When nullopt, readers must determine the size from the compressed
  /// data (e.g., reading the 4-byte Zstd size prefix).
  MetadataSection(
      uint64_t offset,
      uint32_t size,
      CompressionType compressionType,
      std::optional<uint32_t> uncompressedSize = std::nullopt);

  MetadataSection()
      : offset_{0}, size_{0}, compressionType_{CompressionType::Uncompressed} {}

  uint64_t offset() const {
    return offset_;
  }

  uint32_t size() const {
    return size_;
  }

  CompressionType compressionType() const {
    return compressionType_;
  }

  std::optional<uint32_t> uncompressedSize() const {
    return uncompressedSize_;
  }

 private:
  uint64_t offset_;
  uint32_t size_;
  CompressionType compressionType_;
  std::optional<uint32_t> uncompressedSize_;
};

/// Callback type for creating metadata sections in the file.
using CreateMetadataSectionFn =
    std::function<MetadataSection(std::string_view)>;

/// Callback type for writing raw data segments to the file.
/// Writes all segments contiguously and returns the (offset, totalSize).
using WriteDataFn = std::function<std::pair<uint64_t, uint32_t>(
    const std::vector<std::string_view>&)>;

/// Callback type for writing optional sections.
using WriteOptionalSectionFn =
    std::function<void(std::string, std::string_view)>;

} // namespace facebook::nimble
