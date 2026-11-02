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
#include "velox/dwio/nimble/tablet/BlockMetadataInput.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>

#include <folly/ScopeGuard.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/MemoryAllocator.h"
#include "velox/common/time/Timer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/Compression.h"
#include "velox/dwio/nimble/tablet/IoAlignment.h"

namespace facebook::nimble {

// --- BlockMetadataInput ---

BlockMetadataInput::BlockMetadataInput(
    velox::ReadFile* file,
    const Options& options)
    : MetadataInput(file, options.ioStats),
      pool_{options.pool},
      alignment_{readAlignment(*file_)},
      maxCoalesceDistance_{static_cast<uint64_t>(
          std::min(options.maxCoalesceDistance, kMaxCoalesceDistance))},
      maxCoalesceBytes_{static_cast<uint64_t>(options.maxCoalesceBytes)} {
  NIMBLE_CHECK_NOT_NULL(pool_);
  NIMBLE_CHECK_GE(options.maxCoalesceDistance, 0);
  NIMBLE_CHECK_GE(options.maxCoalesceBytes, 0);
  NIMBLE_CHECK_GT(alignment_, 0);
  NIMBLE_CHECK(
      velox::bits::isPowerOfTwo(alignment_), "alignment must be a power of 2");
  NIMBLE_CHECK_LE(
      alignment_,
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
      "alignment exceeds the memory allocator limit");
}

std::vector<uint32_t> BlockMetadataInput::sortByOffset(
    std::span<const MetadataSection> sections) {
  std::vector<uint32_t> indices(sections.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](uint32_t lhs, uint32_t rhs) {
    return sections[lhs].offset() < sections[rhs].offset();
  });
  const MetadataSection* previous{nullptr};
  for (const auto index : indices) {
    const auto& section = sections[index];
    if (section.compressionType() != CompressionType::Uncompressed) {
      // Compressed bytes start with their uncompressed size.
      NIMBLE_CHECK_FILE_GE(
          section.size(),
          sizeof(uint32_t),
          "Compressed metadata section is too small");
    }
    if (section.size() == 0) {
      // An empty section takes no read and no cache entry, so it may share its
      // offset with the next section, which is how the writer lays it out.
      continue;
    }
    if (previous != nullptr) {
      NIMBLE_CHECK_NE(
          previous->offset(),
          section.offset(),
          "Metadata input must not contain duplicate regions");
    }
    previous = &section;
  }
  return indices;
}

std::vector<BlockMetadataInput::ReadGroup> BlockMetadataInput::coalesce(
    std::span<const MetadataSection> sections,
    std::span<const uint32_t> sortedIndices) const {
  const auto fileSize = file_->size();
  std::vector<ReadGroup> groups;
  uint64_t groupBytes{0};
  for (size_t i = 0; i < sortedIndices.size(); ++i) {
    const auto& section = sections[sortedIndices[i]];
    // Compared without adding offset and size, which a corrupt offset can
    // overflow.
    NIMBLE_CHECK_FILE_LE(
        section.offset(),
        fileSize,
        "Metadata section starts past the end of the file");
    NIMBLE_CHECK_FILE_LE(
        section.size(),
        fileSize - section.offset(),
        "Metadata section extends past the end of the file: offset {}, file size {}",
        section.offset(),
        fileSize);
    auto region = alignedRegion({section.offset(), section.size()}, alignment_);
    // A read past the end of the file comes back short, so the last read
    // stops at the end of the file even when that leaves it unaligned.
    region.length =
        std::min(region.offset + region.length, fileSize) - region.offset;
    if (!groups.empty()) {
      auto& group = groups.back();
      const auto groupEnd = group.region.offset + group.region.length;
      if (region.offset < groupEnd ||
          (region.offset - groupEnd <= maxCoalesceDistance_ &&
           groupBytes + section.size() <= maxCoalesceBytes_)) {
        group.region.length =
            std::max(groupEnd, region.offset + region.length) -
            group.region.offset;
        group.end = i + 1;
        groupBytes += section.size();
        continue;
      }
    }
    groups.push_back(ReadGroup{.region = region, .begin = i, .end = i + 1});
    groupBytes = section.size();
  }

  uint64_t bufferOffset{0};
  for (auto& group : groups) {
    group.bufferOffset = bufferOffset;
    bufferOffset += group.region.length;
  }
  return groups;
}

void BlockMetadataInput::read(
    std::span<const ReadGroup> groups,
    char* buffer,
    uint64_t bufferSize,
    uint64_t payloadBytes) const {
  std::vector<velox::common::Region> regions;
  std::vector<folly::Range<char*>> buffers;
  regions.reserve(groups.size());
  buffers.reserve(groups.size());
  for (const auto& group : groups) {
    regions.push_back(group.region);
    buffers.emplace_back(
        buffer + group.bufferOffset, static_cast<size_t>(group.region.length));
  }

  // Measure at nanosecond resolution so that fast reads still register a
  // non-zero scan time.
  uint64_t storageReadNs{0};
  uint64_t bytesRead{0};
  {
    velox::NanosecondTimer timer{&storageReadNs};
    bytesRead = file_->preadv(
        folly::Range<const velox::common::Region*>(
            regions.data(), regions.size()),
        folly::Range<const folly::Range<char*>*>(
            buffers.data(), buffers.size()));
  }
  NIMBLE_CHECK_EQ(bytesRead, bufferSize, "preadv returned a short read");
  const uint64_t storageReadUs = storageReadNs / 1'000;
  ioStats_->queryThreadIoLatencyUs().increment(storageReadUs);
  ioStats_->storageReadLatencyUs().increment(storageReadUs);
  ioStats_->incTotalScanTimeNs(storageReadNs);
  ioStats_->read().increment(bufferSize);
  // Overlapping sections, which only a corrupt file has, can make the payload
  // exceed the bytes read.
  const auto readPayloadBytes = std::min(payloadBytes, bufferSize);
  ioStats_->incRawBytesRead(readPayloadBytes);
  ioStats_->incRawOverreadBytes(bufferSize - readPayloadBytes);
}

void BlockMetadataInput::readSections(
    std::span<const MetadataSection> sections,
    std::span<const uint32_t> sortedIndices,
    ReadCallback callback) const {
  std::vector<uint32_t> readIndices;
  readIndices.reserve(sortedIndices.size());
  uint64_t payloadBytes{0};
  for (const auto index : sortedIndices) {
    // An empty section needs no read, and a read buffer cannot be empty. The
    // view still points at valid memory, as callers may pass it to memcpy.
    if (sections[index].size() == 0) {
      callback(index, std::string_view{"", 0});
      continue;
    }
    readIndices.push_back(index);
    payloadBytes += sections[index].size();
  }
  if (readIndices.empty()) {
    return;
  }

  const auto groups = coalesce(sections, readIndices);
  const auto bufferSize =
      groups.back().bufferOffset + groups.back().region.length;
  const auto allocationAlignment = static_cast<uint32_t>(std::max<uint64_t>(
      alignment_, velox::memory::MemoryAllocator::kMinAlignment));
  auto* buffer = static_cast<char*>(pool_->allocateAligned(
      static_cast<int64_t>(bufferSize), allocationAlignment));
  SCOPE_EXIT {
    pool_->freeAligned(
        buffer, static_cast<int64_t>(bufferSize), allocationAlignment);
  };
  read(groups, buffer, bufferSize, payloadBytes);

  for (const auto& group : groups) {
    for (auto i = group.begin; i < group.end; ++i) {
      const auto index = readIndices[i];
      const auto& section = sections[index];
      callback(
          index,
          {buffer + group.bufferOffset +
               (section.offset() - group.region.offset),
           section.size()});
    }
  }
}

std::shared_ptr<MetadataBuffer> BlockMetadataInput::decompress(
    const MetadataSection& section,
    std::string_view data) const {
  return std::make_shared<MetadataBuffer>(
      MetadataBuffer::decompress(data, section.compressionType(), pool_));
}

// --- DirectBlockMetadataInput ---

DirectBlockMetadataInput::DirectBlockMetadataInput(
    velox::ReadFile* file,
    const Options& options)
    : BlockMetadataInput(file, options) {}

std::vector<std::shared_ptr<MetadataBuffer>> DirectBlockMetadataInput::load(
    std::span<const MetadataSection> sections) {
  NIMBLE_CHECK(!sections.empty(), "No sections to load");
  std::vector<std::shared_ptr<MetadataBuffer>> buffers(sections.size());
  readSections(
      sections,
      sortByOffset(sections),
      [&](uint32_t index, std::string_view data) {
        buffers[index] = decompress(sections[index], data);
      });
  return buffers;
}

// --- CachedBlockMetadataInput ---

namespace {

// Moves a sealed entry holding 'size' bytes into bytes that keep it pinned.
PinnedBytes pinnedBytes(BlockCache::Entry entry, uint32_t size) {
  const std::string_view data{entry.data(), size};
  return PinnedBytes{
      std::make_shared<BlockCache::Entry>(std::move(entry)), data};
}

// Largest cache entry. velox's AsyncDataCache stores entry sizes as int32.
constexpr uint64_t kMaxCacheEntrySize{std::numeric_limits<int32_t>::max()};

// Returns the cache entry size for a section of 'uncompressedSize' bytes, or
// nullopt when the section bypasses the cache: an entry needs its size before
// the read, cannot be empty, and cannot exceed kMaxCacheEntrySize. The size
// comes from the file, so a corrupt one must not reach the cache.
std::optional<uint32_t> cacheEntrySize(
    std::optional<uint32_t> uncompressedSize) {
  if (!uncompressedSize.has_value() || uncompressedSize.value() == 0 ||
      uncompressedSize.value() > kMaxCacheEntrySize) {
    return std::nullopt;
  }
  return uncompressedSize;
}

// Writes a section's decompressed bytes to the 'size' bytes at 'destination'.
void decompressTo(
    const MetadataSection& section,
    std::string_view data,
    char* destination,
    uint32_t size) {
  if (section.compressionType() == CompressionType::Uncompressed) {
    NIMBLE_CHECK_EQ(
        data.size(), size, "Section size mismatch with uncompressed size");
    std::memcpy(destination, data.data(), size);
    return;
  }
  NIMBLE_CHECK_FILE_EQ(
      static_cast<int>(section.compressionType()),
      static_cast<int>(CompressionType::Zstd),
      "Unsupported metadata compression type");
  ZstdCompression::uncompress(data, destination, size);
}

} // namespace

CachedBlockMetadataInput::CachedBlockMetadataInput(
    velox::ReadFile* file,
    const Options& options)
    : BlockMetadataInput(file, options),
      cache_{options.blockCache},
      fileId_{options.blockCacheFileId} {
  NIMBLE_CHECK_NOT_NULL(cache_);
  NIMBLE_USER_CHECK_NE(
      fileId_, 0, "blockCacheFileId must be set together with blockCache");
}

std::vector<std::shared_ptr<MetadataBuffer>> CachedBlockMetadataInput::load(
    std::span<const MetadataSection> sections) {
  NIMBLE_CHECK(!sections.empty(), "No sections to load");
  std::vector<std::shared_ptr<MetadataBuffer>> buffers(sections.size());
  std::vector<BlockCache::Entry> entries(sections.size());
  std::vector<uint32_t> missIndices;
  // Entries are claimed in file offset order, the order of every load of this
  // file, so loads waiting on each other's unfilled entries cannot deadlock.
  for (const auto index : sortByOffset(sections)) {
    const auto& section = sections[index];
    const auto size = cacheEntrySize(resolveUncompressedSize(section));
    if (!size.has_value()) {
      missIndices.push_back(index);
      continue;
    }
    auto entry =
        cache_->getOrCreate(fileId_, section.offset(), size.value(), *ioStats_);
    if (entry.sealed()) {
      buffers[index] = std::make_shared<MetadataBuffer>(
          pinnedBytes(std::move(entry), size.value()));
      continue;
    }
    entries[index] = std::move(entry);
    missIndices.push_back(index);
  }

  if (!missIndices.empty()) {
    readSections(
        sections, missIndices, [&](uint32_t index, std::string_view data) {
          buffers[index] = fill(sections[index], data, entries[index]);
        });
  }
  return buffers;
}

std::shared_ptr<MetadataBuffer> CachedBlockMetadataInput::fill(
    const MetadataSection& section,
    std::string_view data,
    BlockCache::Entry& entry) {
  if (entry.empty()) {
    return decompress(section, data);
  }
  const auto size = resolveUncompressedSize(section).value();
  decompressTo(section, data, entry.data(), size);
  cache_->seal(entry);
  return std::make_shared<MetadataBuffer>(pinnedBytes(std::move(entry), size));
}

std::unique_ptr<MetadataBuffer> CachedBlockMetadataInput::findCachedMetadata(
    uint64_t offset) {
  auto [entry, size] = cache_->find(fileId_, offset, *ioStats_);
  if (entry.empty()) {
    return nullptr;
  }
  return std::make_unique<MetadataBuffer>(pinnedBytes(std::move(entry), size));
}

void CachedBlockMetadataInput::cacheMetadata(
    uint64_t cacheOffset,
    std::span<const std::string_view> ranges) {
  uint64_t totalSize{0};
  for (const auto& range : ranges) {
    totalSize += range.size();
  }
  if (totalSize == 0 || totalSize > kMaxCacheEntrySize) {
    return;
  }

  // An existing entry is not a RAM hit: this reader read the bytes from the
  // file. Only the time spent waiting on another filler is recorded.
  velox::io::IoStatistics probeStats;
  auto entry = cache_->getOrCreate(
      fileId_, cacheOffset, static_cast<uint32_t>(totalSize), probeStats);
  ioStats_->queryThreadIoLatencyUs().merge(probeStats.queryThreadIoLatencyUs());
  ioStats_->cacheWaitLatencyUs().merge(probeStats.cacheWaitLatencyUs());
  if (entry.sealed()) {
    return;
  }
  auto* destination = entry.data();
  for (const auto& range : ranges) {
    std::memcpy(destination, range.data(), range.size());
    destination += range.size();
  }
  cache_->seal(entry);
}

} // namespace facebook::nimble
