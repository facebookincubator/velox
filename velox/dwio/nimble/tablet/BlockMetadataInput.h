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

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <folly/Function.h>

#include "velox/common/file/Region.h"
#include "velox/dwio/nimble/tablet/BlockCache.h"
#include "velox/dwio/nimble/tablet/MetadataInput.h"

namespace facebook::nimble {

/// Reads metadata sections the way DataInput reads stripe data. Each load()
/// coalesces the sections it must read from the file, aligns the reads for
/// direct I/O, and submits them as one positioned vector read, which a file
/// backed by RocksDB serves with a single MultiRead. Each section then gets
/// its own allocation, so no read padding outlives the load. Ignores
/// Options::executor: a file without a native batched preadv() reads the
/// regions of a load one after another.
///
/// Thread-safe: load() keeps all per-call state local.
class BlockMetadataInput : public MetadataInput {
 public:
  /// Caps Options::maxCoalesceDistance, as in DirectDataInput, because the
  /// bytes of every merged gap are read and held until the load ends.
  static constexpr int32_t kMaxCoalesceDistance{256 << 10};

 protected:
  BlockMetadataInput(velox::ReadFile* file, const Options& options);

  // Receives the file bytes of one section, given the section's index in the
  // span passed to readSections(). The bytes are valid only during the call.
  using ReadCallback =
      folly::FunctionRef<void(uint32_t index, std::string_view data)>;

  // Returns the indices of 'sections' in file offset order. Fails when two
  // non-empty sections share an offset, or a compressed section is too small
  // to hold its uncompressed size.
  static std::vector<uint32_t> sortByOffset(
      std::span<const MetadataSection> sections);

  // Reads sections[i] for each i in 'sortedIndices', which must be in file
  // offset order, in one positioned vector read and passes each section's
  // bytes to 'callback'.
  void readSections(
      std::span<const MetadataSection> sections,
      std::span<const uint32_t> sortedIndices,
      ReadCallback callback) const;

  // Returns a section's decompressed bytes in a new pool buffer.
  std::shared_ptr<MetadataBuffer> decompress(
      const MetadataSection& section,
      std::string_view data) const;

 private:
  // A merged read covering the sections at positions [begin, end) of the
  // indices passed to coalesce().
  struct ReadGroup {
    // File range to read, aligned for direct I/O but clipped to the file size.
    velox::common::Region region;
    // Offset of this read's bytes in the shared read buffer.
    uint64_t bufferOffset{0};
    size_t begin{0};
    size_t end{0};
  };

  // Merges the reads of the sections at 'sortedIndices'. Reads that share an
  // aligned block always merge, so the returned reads never overlap.
  std::vector<ReadGroup> coalesce(
      std::span<const MetadataSection> sections,
      std::span<const uint32_t> sortedIndices) const;

  // Reads 'groups' into 'buffer', which holds 'bufferSize' bytes, in one
  // positioned vector read. 'payloadBytes' is the section bytes they cover.
  void read(
      std::span<const ReadGroup> groups,
      char* buffer,
      uint64_t bufferSize,
      uint64_t payloadBytes) const;

  velox::memory::MemoryPool* const pool_;
  // Alignment of read offsets, lengths, and buffers; 1 without direct I/O.
  const uint64_t alignment_;
  // Max gap in bytes between consecutive reads to merge into one, at most
  // kMaxCoalesceDistance.
  const uint64_t maxCoalesceDistance_;
  // Max section bytes per merged read.
  const uint64_t maxCoalesceBytes_;
};

/// Block metadata input without caching.
class DirectBlockMetadataInput final : public BlockMetadataInput {
 public:
  DirectBlockMetadataInput(velox::ReadFile* file, const Options& options);

  std::vector<std::shared_ptr<MetadataBuffer>> load(
      std::span<const MetadataSection> sections) override;
};

/// Block metadata input that caches decompressed sections in
/// Options::blockCache, keyed by (Options::blockCacheFileId, compressed file
/// offset). Sections whose uncompressed size the file does not record bypass
/// the cache.
class CachedBlockMetadataInput final : public BlockMetadataInput {
 public:
  CachedBlockMetadataInput(velox::ReadFile* file, const Options& options);

  bool cached() const override {
    return true;
  }

  std::vector<std::shared_ptr<MetadataBuffer>> load(
      std::span<const MetadataSection> sections) override;

  std::unique_ptr<MetadataBuffer> findCachedMetadata(uint64_t offset) override;

  void cacheMetadata(
      uint64_t cacheOffset,
      std::span<const std::string_view> ranges) override;

 private:
  // Decompresses a section read on a cache miss into 'entry', or into a pool
  // buffer when 'entry' is empty because the section bypasses the cache.
  std::shared_ptr<MetadataBuffer> fill(
      const MetadataSection& section,
      std::string_view data,
      BlockCache::Entry& entry);

  BlockCache* const cache_;
  // Identity of the file in cache keys.
  const uint64_t fileId_;
};

} // namespace facebook::nimble
