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
#include "velox/dwio/nimble/tablet/MetadataInput.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include <numeric>
#include "velox/common/caching/FileHandle.h"

#include "folly/Range.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/tablet/Compression.h"
#include "velox/dwio/nimble/tablet/IoAlignment.h"

#include "folly/ScopeGuard.h"
#include "folly/io/Cursor.h"
#include "velox/common/base/AsyncSource.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/CoalesceIo.h"
#include "velox/common/memory/MemoryAllocator.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/Timer.h"

namespace facebook::nimble {

// --- MetadataInput ---

std::unique_ptr<MetadataInput> MetadataInput::create(
    velox::ReadFile* file,
    const Options& options) {
  NIMBLE_CHECK_NOT_NULL(file);
  if (options.blockCache != nullptr) {
    return std::unique_ptr<MetadataInput>(
        new CachedMetadataInput(file, options));
  }
  return std::unique_ptr<MetadataInput>(new DirectMetadataInput(file, options));
}

MetadataInput::MetadataInput(velox::ReadFile* file, const Options& options)
    : file_{file},
      pool_{options.pool},
      maxCoalesceDistance_{options.maxCoalesceDistance},
      maxCoalesceBytes_{options.maxCoalesceBytes},
      executor_{options.executor},
      ioStats_{options.ioStats},
      fileIoContext_{options.fileIoContext} {
  NIMBLE_CHECK_NOT_NULL(file_);
  NIMBLE_CHECK_NOT_NULL(pool_);
  NIMBLE_CHECK_NOT_NULL(ioStats_.get());
}

std::vector<std::shared_ptr<MetadataBuffer>> MetadataInput::extractResults(
    std::vector<LoadedSection>&& sections) {
  std::vector<std::shared_ptr<MetadataBuffer>> results;
  results.reserve(sections.size());
  for (auto& loaded : sections) {
    NIMBLE_CHECK_NOT_NULL(loaded.buffer);
    results.emplace_back(std::move(loaded.buffer));
  }
  return results;
}

void MetadataInput::readRaw(uint64_t offset, uint64_t size, void* dest) {
  // Measure at nanosecond resolution so that fast (e.g. in-memory or cached)
  // reads still register a non-zero scan time; truncating to microseconds
  // would lose sub-microsecond reads entirely.
  uint64_t storageReadNs{0};
  {
    velox::NanosecondTimer timer{&storageReadNs};
    file_->pread(offset, size, static_cast<char*>(dest), fileIoContext_);
  }
  const uint64_t storageReadUs = storageReadNs / 1'000;
  ioStats_->queryThreadIoLatencyUs().increment(storageReadUs);
  ioStats_->storageReadLatencyUs().increment(storageReadUs);
  ioStats_->incTotalScanTimeNs(storageReadNs);
  ioStats_->read().increment(size);
  ioStats_->incRawBytesRead(size);
}

std::shared_ptr<MetadataBuffer> MetadataInput::decompressAndStore(
    const MetadataSection& section,
    velox::BufferPtr&& buffer) {
  NIMBLE_CHECK_EQ(
      section.size(), buffer->size(), "Buffer size mismatch with section size");
  if (section.compressionType() == CompressionType::Uncompressed) {
    return std::make_shared<MetadataBuffer>(toPinnedBuffer(std::move(buffer)));
  }
  NIMBLE_CHECK_EQ(
      static_cast<int>(section.compressionType()),
      static_cast<int>(CompressionType::Zstd),
      "Unsupported metadata compression type");
  std::string_view data{buffer->as<char>(), buffer->size()};
  return std::make_shared<MetadataBuffer>(
      toPinnedBuffer(ZstdCompression::uncompress(data, pool_)));
}

std::optional<uint32_t> MetadataInput::resolveUncompressedSize(
    const MetadataSection& section) {
  if (section.uncompressedSize().has_value()) {
    return section.uncompressedSize();
  }
  if (section.compressionType() == CompressionType::Uncompressed) {
    return section.size();
  }
  return std::nullopt;
}

std::vector<MetadataInput::IoGroup> MetadataInput::computeIoGroups(
    const std::vector<LoadedSection>& sections,
    const std::vector<uint32_t>& loadIndices,
    const std::vector<folly::Range<char*>>& readRanges) {
  std::vector<int32_t> items(loadIndices.size());
  std::iota(items.begin(), items.end(), 0);

  int64_t coalescedBytes = 0;
  std::vector<int32_t> groupEnds;

  const auto ioStats = velox::coalesceIo<int32_t, char>(
      items,
      /*maxGap=*/maxCoalesceDistance_,
      /*rangesPerIo=*/std::numeric_limits<int32_t>::max(),
      /*offsetFunc=*/
      [&](int32_t i) -> uint64_t {
        return sections[loadIndices[i]].section.offset();
      },
      /*sizeFunc=*/
      [&](int32_t i) -> int32_t {
        return sections[loadIndices[i]].section.size();
      },
      /*numRanges=*/
      [&](int32_t i) -> int32_t {
        const auto size = sections[loadIndices[i]].section.size();
        if (coalescedBytes + size > maxCoalesceBytes_) {
          coalescedBytes = 0;
          return velox::kNoCoalesce;
        }
        coalescedBytes += size;
        return 1;
      },
      /*addRanges=*/
      [&](const int32_t& /*i*/, std::vector<char>& ranges) {
        // Dummy range so coalesceIo sees non-empty ranges for kNoCoalesce
        // flush check.
        ranges.push_back(0);
      },
      /*skipRange=*/
      [&](int32_t /*gap*/, std::vector<char>& /*ranges*/) {},
      /*ioFunc=*/
      [&](const std::vector<int32_t>& /*items*/,
          int32_t /*begin*/,
          int32_t end,
          uint64_t /*offset*/,
          const std::vector<char>& /*ranges*/) {
        coalescedBytes = 0;
        groupEnds.push_back(end);
      });

  NIMBLE_CHECK_EQ(
      ioStats.duplicateRegions,
      0,
      "Metadata input must not contain duplicate regions");
  recordCoalescedIoStats(ioStats);
  velox::common::testutil::TestValue::adjust(
      "facebook::nimble::MetadataInput::computeIoGroups",
      const_cast<velox::CoalesceIoStats*>(&ioStats));

  std::vector<IoGroup> ioGroups;
  int32_t groupStart = 0;
  for (const auto groupEnd : groupEnds) {
    IoGroup group;
    group.offset = sections[loadIndices[groupStart]].section.offset();
    auto lastEnd = group.offset;
    for (int32_t i = groupStart; i < groupEnd; ++i) {
      const auto& section = sections[loadIndices[i]].section;
      if (section.offset() > lastEnd) {
        const auto gap = section.offset() - lastEnd;
        group.ranges.emplace_back(
            nullptr, reinterpret_cast<char*>(static_cast<uint64_t>(gap)));
      }
      group.ranges.emplace_back(readRanges[i]);
      lastEnd = section.offset() + section.size();
    }
    ioGroups.emplace_back(std::move(group));
    groupStart = groupEnd;
  }
  return ioGroups;
}

void MetadataInput::executeIoGroups(std::vector<IoGroup>& ioGroups) const {
  // Shared with the executor tasks below, which can outlive this call.
  std::vector<std::shared_ptr<velox::AsyncSource<folly::Unit>>> asyncIos;
  asyncIos.reserve(ioGroups.size());
  for (const auto& group : ioGroups) {
    asyncIos.emplace_back(
        std::make_shared<velox::AsyncSource<folly::Unit>>([this, &group]() {
          // Measure at nanosecond resolution so that fast reads still register
          // a non-zero scan time; truncating to microseconds would lose
          // sub-microsecond reads entirely.
          uint64_t storageReadNs{0};
          {
            velox::NanosecondTimer ioTimer(&storageReadNs);
            file_->preadv(group.offset, group.ranges, fileIoContext_);
          }
          ioStats_->storageReadLatencyUs().increment(storageReadNs / 1'000);
          ioStats_->incTotalScanTimeNs(storageReadNs);
          return std::make_unique<folly::Unit>();
        }));
  }

  QueryIoLatencyGuard queryIoGuard{ioStats_.get()};
  if (executor_ != nullptr && asyncIos.size() > 1) {
    // Queues every group but the first. This thread reads the first group
    // below, which gives the executor time to start the others.
    for (size_t i = 1; i < asyncIos.size(); ++i) {
      // move() below reads any group whose task has not started, so a task can
      // run after this call returns. It then finds its read done and exits
      // without touching 'group' or this object.
      executor_->add([asyncIo = asyncIos[i]]() { asyncIo->prepare(); });
    }
  }
  std::exception_ptr error;
  for (size_t i = 0; i < asyncIos.size(); ++i) {
    try {
      asyncIos[i]->move();
    } catch (...) {
      if (error == nullptr) {
        error = std::current_exception();
      }
    }
  }
  if (error != nullptr) {
    std::rethrow_exception(error);
  }
}

void MetadataInput::loadFromFile(
    const std::vector<LoadedSection>& sections,
    const std::vector<uint32_t>& loadIndices,
    const std::vector<folly::Range<char*>>& readRanges) {
  if (loadIndices.empty()) {
    return;
  }

  // Sort by file offset for IO coalescing. Reorder both loadIndices
  // and readRanges in lock-step. The caller's buffers[i] ↔
  // loadIndices[i] correspondence is preserved.
  std::vector<uint32_t> sortOrder(loadIndices.size());
  std::iota(sortOrder.begin(), sortOrder.end(), 0);
  std::sort(sortOrder.begin(), sortOrder.end(), [&](auto lhs, auto rhs) {
    return sections[loadIndices[lhs]].section.offset() <
        sections[loadIndices[rhs]].section.offset();
  });

  std::vector<uint32_t> sortedIndices(loadIndices.size());
  std::vector<folly::Range<char*>> sortedRanges(readRanges.size());
  for (size_t i = 0; i < sortOrder.size(); ++i) {
    sortedIndices[i] = loadIndices[sortOrder[i]];
    sortedRanges[i] = readRanges[sortOrder[i]];
  }

  auto ioGroups = computeIoGroups(sections, sortedIndices, sortedRanges);
  executeIoGroups(ioGroups);
}

void MetadataInput::loadFromFile(
    const std::vector<LoadedSection>& sections,
    const std::vector<folly::Range<char*>>& readRanges) {
  std::vector<uint32_t> loadIndices(sections.size());
  std::iota(loadIndices.begin(), loadIndices.end(), 0);
  loadFromFile(sections, loadIndices, readRanges);
}

void MetadataInput::recordCoalescedIoStats(
    const velox::CoalesceIoStats& ioStats) {
  ioStats_->read().increment(ioStats.payloadBytes + ioStats.extraBytes);
  ioStats_->incRawBytesRead(ioStats.payloadBytes);
  ioStats_->incRawOverreadBytes(ioStats.extraBytes);
}

// --- DirectMetadataInput ---

DirectMetadataInput::DirectMetadataInput(
    velox::ReadFile* file,
    const Options& options)
    : MetadataInput(file, options) {}

void DirectMetadataInput::prepareBuffers(
    const std::vector<LoadedSection>& sections,
    ReadBuffers& readBuffers) {
  readBuffers.buffers.resize(sections.size());
  readBuffers.readRanges.resize(sections.size());
  for (size_t i = 0; i < sections.size(); ++i) {
    const auto size = sections[i].section.size();
    readBuffers.buffers[i] =
        velox::AlignedBuffer::allocateExact<char>(size, pool_);
    readBuffers.readRanges[i] = {
        readBuffers.buffers[i]->asMutable<char>(), size};
  }
}

std::vector<std::shared_ptr<MetadataBuffer>> DirectMetadataInput::load(
    std::span<const MetadataSection> sections) {
  NIMBLE_CHECK(!sections.empty(), "No sections to load");

  std::vector<LoadedSection> loadSections(sections.begin(), sections.end());
  ReadBuffers readBuffers;
  prepareBuffers(loadSections, readBuffers);

  loadFromFile(loadSections, readBuffers.readRanges);

  for (size_t i = 0; i < loadSections.size(); ++i) {
    loadSections[i].buffer = decompressAndStore(
        loadSections[i].section, std::move(readBuffers.buffers[i]));
  }

  return extractResults(std::move(loadSections));
}

// --- CachedMetadataInput ---

namespace {

// Returns the indices of 'sections' in file offset order. Fails when two
// non-empty sections share an offset, or a compressed section is too small
// to hold its uncompressed size.
std::vector<uint32_t> sortByOffset(std::span<const MetadataSection> sections) {
  std::vector<uint32_t> sortIndices(sections.size());
  std::iota(sortIndices.begin(), sortIndices.end(), 0);
  std::sort(
      sortIndices.begin(), sortIndices.end(), [&](uint32_t lhs, uint32_t rhs) {
        return sections[lhs].offset() < sections[rhs].offset();
      });
  const MetadataSection* prev{nullptr};
  for (const auto index : sortIndices) {
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
    if (prev != nullptr) {
      NIMBLE_CHECK_GT(
          section.offset(),
          prev->offset(),
          "Metadata input must not contain duplicate regions");
    }
    prev = &section;
  }
  return sortIndices;
}

// Moves a sealed entry into bytes that keep it pinned.
PinnedBuffer pinnedBuffer(BlockCache::Entry entry) {
  const std::string_view data{entry.data(), entry.size()};
  return PinnedBuffer{
      std::make_shared<BlockCache::Entry>(std::move(entry)), data};
}

// Largest cache entry. velox's AsyncDataCache stores entry sizes as int32.
constexpr uint64_t kMaxCacheEntrySize{std::numeric_limits<int32_t>::max()};

// Returns true if an entry of 'size' bytes can be cached: it cannot be empty,
// and cannot exceed 'maxCacheEntrySize'. Sizes come from the file, and no valid
// metadata section exceeds kMaxCacheEntrySize, so a larger one means a corrupt
// file and fails.
bool isCacheableSize(uint64_t size, std::optional<uint32_t> maxCacheEntrySize) {
  NIMBLE_CHECK_FILE_LE(
      size, kMaxCacheEntrySize, "Metadata section is too large");
  return size > 0 &&
      (!maxCacheEntrySize.has_value() || size <= maxCacheEntrySize.value());
}

// Writes 'data', compressed with 'compressionType', decompressed to the 'size'
// bytes at 'destination'.
void decompressTo(
    std::string_view data,
    CompressionType compressionType,
    char* destination,
    uint32_t size) {
  if (compressionType == CompressionType::Uncompressed) {
    NIMBLE_CHECK_EQ(
        data.size(), size, "Section size mismatch with uncompressed size");
    std::memcpy(destination, data.data(), size);
    return;
  }
  NIMBLE_CHECK_FILE_EQ(
      static_cast<int>(compressionType),
      static_cast<int>(CompressionType::Zstd),
      "Unsupported metadata compression type");
  ZstdCompression::uncompress(data, destination, size);
}

} // namespace

CachedMetadataInput::CachedMetadataInput(
    velox::ReadFile* file,
    const Options& options)
    : MetadataInput(file, options),
      blockCache_{options.blockCache},
      fileId_{[&]() {
        NIMBLE_USER_CHECK_NOT_NULL(
            options.fileHandle, "blockCache requires fileHandle");
        return options.fileHandle->uuid;
      }()},
      maxCacheEntrySize_{options.maxCacheEntrySize},
      alignment_{readAlignment(*file_)} {
  NIMBLE_CHECK_NOT_NULL(blockCache_);
  NIMBLE_USER_CHECK(fileId_.hasValue(), "blockCache requires fileHandle->uuid");
  NIMBLE_USER_CHECK_GE(
      options.maxCoalesceDistance,
      0,
      "maxCoalesceDistance must not be negative");
  NIMBLE_USER_CHECK_GE(
      options.maxCoalesceBytes, 0, "maxCoalesceBytes must not be negative");
  NIMBLE_CHECK_GT(alignment_, 0);
  NIMBLE_CHECK(
      velox::bits::isPowerOfTwo(alignment_), "alignment must be a power of 2");
  NIMBLE_CHECK_LE(
      alignment_,
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
      "alignment exceeds the memory allocator limit");
}

std::vector<CachedMetadataInput::ReadGroup> CachedMetadataInput::coalesce(
    std::span<const MetadataSection> sections,
    std::span<const uint32_t> sortedIndices) const {
  const auto fileSize = file_->size();
  std::vector<ReadGroup> groups;
  uint64_t groupBytes{0};
  for (size_t i = 0; i < sortedIndices.size(); ++i) {
    const auto& section = sections[sortedIndices[i]];
    const auto region =
        alignedRegion({section.offset(), section.size()}, alignment_, fileSize);
    if (groups.empty()) {
      groups.push_back(ReadGroup{.region = region, .begin = i, .end = i + 1});
      groupBytes = section.size();
      continue;
    }
    auto& lastGroup = groups.back();
    const auto groupEnd = lastGroup.region.offset + lastGroup.region.length;
    if (region.offset < groupEnd ||
        (region.offset - groupEnd <=
             static_cast<uint64_t>(maxCoalesceDistance_) &&
         groupBytes + section.size() <=
             static_cast<uint64_t>(maxCoalesceBytes_))) {
      lastGroup.region.length =
          std::max(groupEnd, region.offset + region.length) -
          lastGroup.region.offset;
      lastGroup.end = i + 1;
      groupBytes += section.size();
      continue;
    }
    groups.push_back(ReadGroup{.region = region, .begin = i, .end = i + 1});
    groupBytes = section.size();
  }

  uint64_t destBufferOffset{0};
  for (auto& group : groups) {
    group.destBufferOffset = destBufferOffset;
    destBufferOffset += group.region.length;
  }
  return groups;
}

void CachedMetadataInput::readGroups(
    std::span<const ReadGroup> groups,
    char* buffer,
    uint64_t payloadBytes) const {
  std::vector<velox::common::Region> regions;
  std::vector<folly::Range<char*>> buffers;
  regions.reserve(groups.size());
  buffers.reserve(groups.size());
  for (const auto& group : groups) {
    regions.push_back(group.region);
    buffers.emplace_back(
        buffer + group.destBufferOffset,
        static_cast<size_t>(group.region.length));
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
            buffers.data(), buffers.size()),
        fileIoContext_);
  }

  // A read past the end of the file returns only the bytes before the end.
  const auto fileBytes = fileReadBytes(regions, file_->size());
  NIMBLE_CHECK_GE(bytesRead, fileBytes, "preadv returned a short read");
  recordReadStats(storageReadNs, fileBytes, payloadBytes);
}

void CachedMetadataInput::recordReadStats(
    uint64_t storageReadNs,
    uint64_t readBytes,
    uint64_t payloadBytes) const {
  const uint64_t storageReadUs = storageReadNs / 1'000;
  ioStats_->queryThreadIoLatencyUs().increment(storageReadUs);
  ioStats_->storageReadLatencyUs().increment(storageReadUs);
  ioStats_->incTotalScanTimeNs(storageReadNs);
  recordReadBytes(readBytes, payloadBytes);
}

void CachedMetadataInput::recordReadBytes(
    uint64_t readBytes,
    uint64_t payloadBytes) const {
  ioStats_->read().increment(readBytes);

  // Overlapping sections, which only a corrupt file has, can make the payload
  // exceed the bytes read.
  const auto readPayloadBytes = std::min(payloadBytes, readBytes);
  ioStats_->incRawBytesRead(readPayloadBytes);
  ioStats_->incRawOverreadBytes(readBytes - readPayloadBytes);
}

void CachedMetadataInput::loadDirectIo(
    std::span<const MetadataSection> sections,
    std::span<const uint32_t> missIndices,
    std::span<BlockCache::Entry> newEntries,
    std::span<std::shared_ptr<MetadataBuffer>> buffers) {
  std::vector<uint32_t> readIndices;
  readIndices.reserve(missIndices.size());
  uint64_t payloadBytes{0};
  for (const auto index : missIndices) {
    // An empty section needs no read, and a read buffer cannot be empty. The
    // view still points at valid memory, as cacheSection() may pass it to
    // memcpy.
    if (sections[index].size() == 0) {
      buffers[index] = cacheSection(
          sections[index],
          std::string_view{"", 0},
          newEntries[index],
          /*readBuffer=*/nullptr);
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
      groups.back().destBufferOffset + groups.back().region.length;
  const auto allocationAlignment = static_cast<uint32_t>(std::max<uint64_t>(
      alignment_, velox::memory::MemoryAllocator::kMinAlignment));
  auto* buffer = static_cast<char*>(pool_->allocateAligned(
      static_cast<int64_t>(bufferSize), allocationAlignment));
  SCOPE_EXIT {
    pool_->freeAligned(
        buffer, static_cast<int64_t>(bufferSize), allocationAlignment);
  };
  readGroups(groups, buffer, payloadBytes);

  for (const auto& group : groups) {
    for (auto i = group.begin; i < group.end; ++i) {
      const auto index = readIndices[i];
      const auto& section = sections[index];
      buffers[index] = cacheSection(
          section,
          {buffer + group.destBufferOffset +
               (section.offset() - group.region.offset),
           section.size()},
          newEntries[index],
          /*readBuffer=*/nullptr);
    }
  }
}

void CachedMetadataInput::loadBufferedIo(
    std::span<const MetadataSection> sections,
    std::span<const uint32_t> missIndices,
    std::span<BlockCache::Entry> newEntries,
    std::span<std::shared_ptr<MetadataBuffer>> buffers) {
  // Without direct I/O, each section is read into a buffer of its own size. An
  // uncompressed section with a new entry is read straight into the entry.
  std::vector<velox::BufferPtr> readBuffers(sections.size());
  std::vector<folly::Range<char*>> destinations(sections.size());
  for (const auto index : missIndices) {
    const auto& section = sections[index];
    const auto& entry = newEntries[index];
    if (!entry.empty() &&
        section.compressionType() == CompressionType::Uncompressed) {
      NIMBLE_CHECK_EQ(entry.size(), section.size());
      destinations[index] = {entry.data(), section.size()};
      continue;
    }
    readBuffers[index] =
        velox::AlignedBuffer::allocateExact<char>(section.size(), pool_);
    destinations[index] = {
        readBuffers[index]->asMutable<char>(), section.size()};
  }
  readSections(sections, missIndices, destinations);

  for (const auto index : missIndices) {
    const auto& section = sections[index];
    auto& entry = newEntries[index];
    if (readBuffers[index] == nullptr) {
      // The read filled the entry: an uncompressed section with a new entry is
      // read straight into it, with no read buffer.
      blockCache_->seal(entry);
      buffers[index] =
          std::make_shared<MetadataBuffer>(pinnedBuffer(std::move(entry)));
      continue;
    }
    const std::string_view data{readBuffers[index]->as<char>(), section.size()};
    buffers[index] =
        cacheSection(section, data, entry, std::move(readBuffers[index]));
  }
}

void CachedMetadataInput::readSections(
    std::span<const MetadataSection> sections,
    std::span<const uint32_t> sortedIndices,
    std::span<const folly::Range<char*>> destinations) const {
  NIMBLE_CHECK(!directIo(), "readSections() must not read a direct-I/O file");
  NIMBLE_CHECK_EQ(destinations.size(), sections.size());
  std::vector<uint32_t> readIndices;
  readIndices.reserve(sortedIndices.size());
  uint64_t payloadBytes{0};
  for (const auto index : sortedIndices) {
    NIMBLE_CHECK_EQ(
        destinations[index].size(),
        sections[index].size(),
        "Destination size mismatch with section size");

    // An empty section needs no read.
    if (sections[index].size() == 0) {
      continue;
    }
    readIndices.push_back(index);
    payloadBytes += sections[index].size();
  }
  if (readIndices.empty()) {
    return;
  }

  // The ranges of a scatter read are the destinations of its sections in file
  // order, with a null range over each gap, which the read skips. A section
  // that overlaps the one before it, which only a corrupt file has, starts a
  // new read, since a scatter read cannot go back.
  std::vector<IoGroup> reads;
  uint64_t readBytes{0};
  for (const auto& group : coalesce(sections, readIndices)) {
    uint64_t readEnd{0};
    for (auto i = group.begin; i < group.end; ++i) {
      const auto index = readIndices[i];
      const auto& section = sections[index];
      if (i == group.begin || section.offset() < readEnd) {
        reads.push_back(IoGroup{.offset = section.offset(), .ranges = {}});
      } else if (section.offset() > readEnd) {
        const auto gap = section.offset() - readEnd;
        reads.back().ranges.emplace_back(nullptr, reinterpret_cast<char*>(gap));
        readBytes += gap;
      }
      reads.back().ranges.push_back(destinations[index]);
      readBytes += section.size();
      readEnd = section.offset() + section.size();
    }
  }
  executeIoGroups(reads);
  recordReadBytes(readBytes, payloadBytes);
}

std::shared_ptr<MetadataBuffer> CachedMetadataInput::decompress(
    std::string_view data,
    CompressionType compressionType) const {
  return std::make_shared<MetadataBuffer>(
      toPinnedBuffer(MetadataBuffer::decompress(data, compressionType, pool_)));
}

std::shared_ptr<MetadataBuffer> CachedMetadataInput::decompress(
    velox::BufferPtr data,
    CompressionType compressionType) const {
  return std::make_shared<MetadataBuffer>(toPinnedBuffer(
      MetadataBuffer::decompress(std::move(data), compressionType, pool_)));
}

std::vector<std::shared_ptr<MetadataBuffer>> CachedMetadataInput::load(
    std::span<const MetadataSection> sections) {
  NIMBLE_CHECK(!sections.empty(), "No sections to load");
  std::vector<std::shared_ptr<MetadataBuffer>> buffers(sections.size());
  // Entries created before the read, for the read to fill.
  std::vector<BlockCache::Entry> newEntries(sections.size());
  std::vector<uint32_t> missIndices;

  // Entries are claimed in file offset order, the order of every load of this
  // file, so loads waiting on each other's unfilled entries cannot deadlock.
  // The claim that cacheSection() makes after the read is the exception.
  const auto sortedIndices = sortByOffset(sections);
  for (const auto index : sortedIndices) {
    auto entry = find(sections[index]);
    if (!entry.empty() && entry.sealed()) {
      ioStats_->incRawBytesRead(sections[index].size());
      buffers[index] =
          std::make_shared<MetadataBuffer>(pinnedBuffer(std::move(entry)));
      continue;
    }
    newEntries[index] = std::move(entry);
    missIndices.push_back(index);
  }

  if (missIndices.empty()) {
    return buffers;
  }

  if (directIo()) {
    loadDirectIo(sections, missIndices, newEntries, buffers);
  } else {
    loadBufferedIo(sections, missIndices, newEntries, buffers);
  }
  return buffers;
}

std::shared_ptr<MetadataBuffer> CachedMetadataInput::cacheSection(
    const MetadataSection& section,
    std::string_view data,
    BlockCache::Entry& entry,
    velox::BufferPtr readBuffer) {
  if (!resolveUncompressedSize(section).has_value()) {
    // The file does not record the uncompressed size, so no entry could be
    // created before the read: find() only probed, and missed. The compressed
    // bytes start with the size.
    return cacheSection(section, data);
  }
  if (!entry.empty()) {
    // The size is known, so find() created an entry for this read to fill.
    return fill(section, data, entry);
  }

  // The section is empty or too large to cache.
  if (readBuffer != nullptr) {
    return decompress(std::move(readBuffer), section.compressionType());
  }
  return decompress(data, section.compressionType());
}

BlockCache::Entry CachedMetadataInput::find(
    const MetadataSection& section) const {
  const auto uncompressedSize = resolveUncompressedSize(section);
  if (!uncompressedSize.has_value()) {
    // The entry size is known only once the section is read, so this probes
    // for an entry of any size.
    return blockCache_->find(fileId_.id(), section.offset(), *ioStats_);
  }
  const auto size = uncompressedSize.value();
  if (!isCacheableSize(size, maxCacheEntrySize_)) {
    return {};
  }
  auto entry =
      blockCache_->getOrCreate(fileId_.id(), section.offset(), size, *ioStats_);
  if (entry.sealed()) {
    NIMBLE_CHECK_EQ(
        entry.size(), size, "Cached entry size mismatch with expected size");
  }
  return entry;
}

std::shared_ptr<MetadataBuffer> CachedMetadataInput::fill(
    const MetadataSection& section,
    std::string_view data,
    BlockCache::Entry& entry) {
  decompressTo(data, section.compressionType(), entry.data(), entry.size());
  blockCache_->seal(entry);
  return std::make_shared<MetadataBuffer>(pinnedBuffer(std::move(entry)));
}

std::shared_ptr<MetadataBuffer> CachedMetadataInput::cacheSection(
    const MetadataSection& section,
    std::string_view data) {
  // Compressed bytes start with their uncompressed size.
  const char* sizePosition = data.data();
  const auto size = encoding::readUint32(sizePosition);
  if (!isCacheableSize(size, maxCacheEntrySize_)) {
    return decompress(data, section.compressionType());
  }

  // This claim is out of file offset order. It waits only on a reader that
  // holds this offset unfilled. A reader that claimed it here fills it before
  // it waits on any other entry. A reader that claimed it in find(),
  // which takes a section of known uncompressed size at this offset, can wait
  // on an entry this load holds and deadlock. Only a corrupt file describes one
  // offset both ways.
  auto entry = getOrCreateWithoutHit(section.offset(), size);
  if (entry.sealed()) {
    // Another reader cached the section while this load read it.
    NIMBLE_CHECK_EQ(
        entry.size(), size, "Cached entry size mismatch with expected size");
    return std::make_shared<MetadataBuffer>(pinnedBuffer(std::move(entry)));
  }
  return fill(section, data, entry);
}

BlockCache::Entry CachedMetadataInput::getOrCreateWithoutHit(
    uint64_t offset,
    uint32_t size) const {
  // An existing entry is not a RAM hit: this reader read the bytes from the
  // file. Only the time spent waiting on another filler, and an SSD read of
  // the entry, are recorded.
  velox::io::IoStatistics probeStats;
  auto entry = blockCache_->getOrCreate(fileId_.id(), offset, size, probeStats);
  ioStats_->queryThreadIoLatencyUs().merge(probeStats.queryThreadIoLatencyUs());
  ioStats_->cacheWaitLatencyUs().merge(probeStats.cacheWaitLatencyUs());
  ioStats_->ssdRead().merge(probeStats.ssdRead());
  ioStats_->ssdCacheReadLatencyUs().merge(probeStats.ssdCacheReadLatencyUs());
  return entry;
}

std::unique_ptr<MetadataBuffer> CachedMetadataInput::findCachedMetadata(
    uint64_t offset) {
  auto entry = blockCache_->find(fileId_.id(), offset, *ioStats_);
  if (entry.empty()) {
    return nullptr;
  }
  ioStats_->incRawBytesRead(entry.size());
  return std::make_unique<MetadataBuffer>(pinnedBuffer(std::move(entry)));
}

void CachedMetadataInput::cacheMetadata(
    uint64_t cacheOffset,
    std::span<const std::string_view> ranges) {
  uint64_t totalSize{0};
  for (const auto& range : ranges) {
    totalSize += range.size();
  }
  if (!isCacheableSize(totalSize, maxCacheEntrySize_)) {
    return;
  }

  auto entry =
      getOrCreateWithoutHit(cacheOffset, static_cast<uint32_t>(totalSize));
  if (entry.sealed()) {
    return;
  }
  auto* destination = entry.data();
  for (const auto& range : ranges) {
    std::memcpy(destination, range.data(), range.size());
    destination += range.size();
  }
  blockCache_->seal(entry);
}

} // namespace facebook::nimble
