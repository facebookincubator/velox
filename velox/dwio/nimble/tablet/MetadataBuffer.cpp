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

#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/Compression.h"

namespace facebook::nimble {

MetadataBuffer::MetadataBuffer(velox::BufferPtr buffer)
    : MetadataBuffer(buffer, 0, buffer->size()) {}

MetadataBuffer::MetadataBuffer(
    velox::BufferPtr buffer,
    uint64_t offset,
    uint64_t size)
    : buffer_{std::move(buffer)}, offset_{offset}, size_{size} {
  NIMBLE_CHECK_NOT_NULL(buffer_);
  NIMBLE_CHECK_LE(offset_ + size_, buffer_->size());
}

MetadataBuffer::MetadataBuffer(velox::cache::CachePin pin)
    : MetadataBuffer(
          pin,
          0,
          static_cast<uint64_t>(pin.checkedEntry()->size())) {}

MetadataBuffer::MetadataBuffer(
    velox::cache::CachePin pin,
    uint64_t offset,
    uint64_t size)
    : pin_{std::move(pin)}, offset_{offset}, size_{size} {
  NIMBLE_CHECK(!pin_.empty(), "CachePin must not be empty");
  NIMBLE_CHECK_LE(
      offset_ + size_, static_cast<uint64_t>(pin_.checkedEntry()->size()));
}

MetadataBuffer::~MetadataBuffer() = default;

MetadataBuffer::MetadataBuffer(MetadataBuffer&&) noexcept = default;

MetadataBuffer& MetadataBuffer::operator=(MetadataBuffer&&) noexcept = default;

MetadataBuffer MetadataBuffer::clone() const {
  if (buffer_ != nullptr) {
    return MetadataBuffer{buffer_, offset_, size_};
  }
  NIMBLE_CHECK(!pin_.empty(), "Cannot clone empty MetadataBuffer");
  return MetadataBuffer{pin_, offset_, size_};
}

MetadataSection::MetadataSection(
    uint64_t offset,
    uint32_t size,
    CompressionType compressionType,
    std::optional<uint32_t> uncompressedSize)
    : offset_{offset},
      size_{size},
      compressionType_{compressionType},
      uncompressedSize_{uncompressedSize} {
  if (uncompressedSize_.has_value()) {
    if (compressionType_ == CompressionType::Uncompressed) {
      NIMBLE_CHECK_EQ(
          uncompressedSize_.value(),
          size_,
          "Uncompressed size must equal size for uncompressed data");
    } else {
      NIMBLE_CHECK_GE(
          uncompressedSize_.value(),
          size_,
          "Uncompressed size must be >= compressed size");
    }
  }
}

namespace {

velox::BufferPtr
copyToBuffer(const char* data, size_t size, velox::memory::MemoryPool* pool) {
  auto buffer = velox::AlignedBuffer::allocateExact<char>(size, pool);
  std::memcpy(buffer->asMutable<char>(), data, size);
  return buffer;
}

velox::BufferPtr decompressZstd(
    std::string_view data,
    velox::memory::MemoryPool* pool) {
  return ZstdCompression::uncompress(data, pool);
}

} // namespace

velox::BufferPtr MetadataBuffer::decompress(
    velox::BufferPtr data,
    CompressionType type,
    velox::memory::MemoryPool* pool) {
  if (type == CompressionType::Uncompressed) {
    return data;
  }
  NIMBLE_CHECK_EQ(
      static_cast<int>(type),
      static_cast<int>(CompressionType::Zstd),
      "Unsupported metadata compression type");
  return decompressZstd({data->as<char>(), data->size()}, pool);
}

velox::BufferPtr MetadataBuffer::decompress(
    std::string_view data,
    CompressionType type,
    velox::memory::MemoryPool* pool) {
  if (type == CompressionType::Uncompressed) {
    return copyToBuffer(data.data(), data.size(), pool);
  }
  NIMBLE_CHECK_EQ(
      static_cast<int>(type),
      static_cast<int>(CompressionType::Zstd),
      "Unsupported metadata compression type");
  return decompressZstd(data, pool);
}

std::unique_ptr<MetadataBuffer> MetadataBuffer::slice(
    uint64_t offset,
    uint64_t size) const {
  NIMBLE_CHECK_LE(offset + size, size_, "Slice range exceeds buffer content");
  if (buffer_ != nullptr) {
    return std::unique_ptr<MetadataBuffer>(
        new MetadataBuffer(buffer_, offset_ + offset, size));
  }
  NIMBLE_CHECK(!pin_.empty(), "Cannot slice empty MetadataBuffer");
  return std::unique_ptr<MetadataBuffer>(
      new MetadataBuffer(pin_, offset_ + offset, size));
}

} // namespace facebook::nimble
