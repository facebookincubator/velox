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

#include <numeric>
#include "velox/common/caching/FileHandle.h"

#include "folly/Range.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/Compression.h"

#include "folly/executors/QueuedImmediateExecutor.h"
#include "folly/io/Cursor.h"
#include "velox/common/base/AsyncSource.h"
#include "velox/common/base/CoalesceIo.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/Timer.h"

namespace facebook::nimble {

// --- MetadataInput ---

std::unique_ptr<MetadataInput> MetadataInput::create(
    velox::ReadFile* file,
    const Options& options) {
  NIMBLE_CHECK_NOT_NULL(file);
  if (options.fileHandle != nullptr) {
    NIMBLE_CHECK_NOT_NULL(
        options.cache, "cache required when fileHandle is set");
    return std::unique_ptr<MetadataInput>(new CachedMetadataInput(
        file, options.fileHandle->uuid, options.cache, options));
  }
  return std::unique_ptr<MetadataInput>(new DirectMetadataInput(file, options));
}

MetadataInput::MetadataInput(velox::ReadFile* file, const Options& options)
    : file_{file},
      pool_{options.pool},
      maxCoalesceDistance_{options.maxCoalesceDistance},
      maxCoalesceBytes_{options.maxCoalesceBytes},
      executor_{options.executor},
      ioStats_{options.ioStats} {
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
    file_->pread(offset, size, static_cast<char*>(dest));
  }
  const uint64_t storageReadUs = storageReadNs / 1'000;
  ioStats_->queryThreadIoLatencyUs().increment(storageReadUs);
  ioStats_->storageReadLatencyUs().increment(storageReadUs);
  ioStats_->incTotalScanTimeNs(storageReadNs);
  ioStats_->read().increment(size);
  ioStats_->incRawBytesRead(size);
}

std::unique_ptr<MetadataBuffer> MetadataInput::findCachedMetadata(
    uint64_t /*offset*/) {
  NIMBLE_UNREACHABLE("findCachedMetadata requires CachedMetadataInput");
}

void MetadataInput::cacheMetadata(
    uint64_t /*cacheOffset*/,
    std::span<const std::string_view> /*ranges*/) {
  NIMBLE_UNREACHABLE("cacheMetadata requires CachedMetadataInput");
}

std::shared_ptr<MetadataBuffer> MetadataInput::decompressAndStore(
    const MetadataSection& section,
    velox::BufferPtr&& buffer) {
  NIMBLE_CHECK_EQ(
      section.size(), buffer->size(), "Buffer size mismatch with section size");
  if (section.compressionType() == CompressionType::Uncompressed) {
    return store(section, std::move(buffer));
  }
  NIMBLE_CHECK_EQ(
      static_cast<int>(section.compressionType()),
      static_cast<int>(CompressionType::Zstd),
      "Unsupported metadata compression type");
  std::string_view data{buffer->as<char>(), buffer->size()};
  return store(section, ZstdCompression::uncompress(data, pool_));
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

void MetadataInput::executeIoGroups(std::vector<IoGroup>& ioGroups) {
  std::vector<std::unique_ptr<velox::AsyncSource<folly::Unit>>> asyncIos;
  asyncIos.reserve(ioGroups.size());
  for (const auto& group : ioGroups) {
    asyncIos.emplace_back(
        std::make_unique<velox::AsyncSource<folly::Unit>>([this, &group]() {
          // Measure at nanosecond resolution so that fast reads still register
          // a non-zero scan time; truncating to microseconds would lose
          // sub-microsecond reads entirely.
          uint64_t storageReadNs{0};
          {
            velox::NanosecondTimer ioTimer(&storageReadNs);
            file_->preadv(group.offset, group.ranges);
          }
          ioStats_->storageReadLatencyUs().increment(storageReadNs / 1'000);
          ioStats_->incTotalScanTimeNs(storageReadNs);
          return std::make_unique<folly::Unit>();
        }));
  }

  QueryIoLatencyGuard queryIoGuard{ioStats_.get()};
  if (executor_ != nullptr && asyncIos.size() > 1) {
    for (size_t i = 0; i + 1 < asyncIos.size(); ++i) {
      executor_->add([&asyncIo = asyncIos[i]]() { asyncIo->prepare(); });
    }
  }
  std::exception_ptr error;
  for (auto& asyncIo : asyncIos) {
    try {
      asyncIo->move();
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

std::shared_ptr<MetadataBuffer> DirectMetadataInput::store(
    const MetadataSection& /*section*/,
    velox::BufferPtr&& decompressed) {
  return std::make_shared<MetadataBuffer>(std::move(decompressed));
}

// --- CachedMetadataInput ---

CachedMetadataInput::CachedMetadataInput(
    velox::ReadFile* file,
    velox::StringIdLease fileId,
    velox::cache::AsyncDataCache* cache,
    const Options& options)
    : MetadataInput(file, options), cache_{cache}, fileId_{std::move(fileId)} {
  NIMBLE_CHECK_NOT_NULL(cache_);
}

velox::cache::CachePin CachedMetadataInput::acquireCachePin(
    const velox::cache::RawFileCacheKey& key,
    uint32_t size) {
  for (;;) {
    folly::SemiFuture<bool> waitFuture{false};
    auto pin =
        cache_->findOrCreate(key, size, /*contiguous=*/true, &waitFuture);
    if (!pin.empty()) {
      return pin;
    }
    uint64_t waitUs{0};
    {
      velox::MicrosecondTimer timer(&waitUs);
      auto& exec = folly::QueuedImmediateExecutor::instance();
      std::move(waitFuture).via(&exec).wait();
    }
    ioStats_->queryThreadIoLatencyUs().increment(waitUs);
    ioStats_->cacheWaitLatencyUs().increment(waitUs);
  }
}

std::shared_ptr<MetadataBuffer> CachedMetadataInput::tryCacheHit(
    uint32_t expectedSize,
    velox::cache::CachePin& pin) {
  auto* entry = pin.checkedEntry();
  if (!entry->isShared()) {
    return nullptr;
  }
  NIMBLE_CHECK(
      entry->hasContiguousData(), "Cached entry must have contiguous data");
  NIMBLE_CHECK_EQ(
      entry->size(),
      expectedSize,
      "Cached entry size mismatch with expected size");
  ioStats_->ramHit().increment(entry->size());
  return std::make_shared<MetadataBuffer>(std::move(pin));
}

std::shared_ptr<MetadataBuffer> CachedMetadataInput::promoteCachePin(
    const MetadataSection& section,
    velox::cache::CachePin pin,
    velox::io::IoCounter& ioCounter) {
  auto* entry = pin.checkedEntry();
  NIMBLE_CHECK(entry->isExclusive(), "Expected exclusive cache pin");
  NIMBLE_CHECK(
      entry->hasContiguousData(), "Cache entry must have contiguous data");
  NIMBLE_CHECK_GE(
      entry->size(), section.size(), "Cache entry smaller than section size");
  const auto uncompressedSize = resolveUncompressedSize(section);
  if (uncompressedSize.has_value()) {
    NIMBLE_CHECK_EQ(
        entry->size(),
        uncompressedSize.value(),
        "Cache entry size mismatch with uncompressed size");
  }
  ioCounter.increment(entry->size());
  entry->setExclusiveToShared();
  return std::make_shared<MetadataBuffer>(std::move(pin));
}

void CachedMetadataInput::loadFromSsd(
    std::vector<LoadedSection>& sections,
    std::vector<uint32_t>& loadIndices,
    std::vector<std::optional<velox::cache::CachePin>>& cachePins) {
  if (loadIndices.empty()) {
    return;
  }
  NIMBLE_CHECK_EQ(
      cachePins.size(),
      sections.size(),
      "cachePins size must match sections size");

  auto* ssdCache = cache_->ssdCache();
  if (ssdCache == nullptr) {
    return;
  }

  struct SsdHit {
    uint32_t sectionIndex;
    velox::cache::SsdPin ssdPin;
    velox::cache::CachePin cachePin;
  };
  std::vector<SsdHit> ssdHits;
  std::vector<uint32_t> remainingLoadIndices;
  remainingLoadIndices.reserve(loadIndices.size());

  auto& ssdFile = ssdCache->file(fileId_.id());
  for (const auto index : loadIndices) {
    auto& loaded = sections[index];
    const velox::cache::RawFileCacheKey key{
        fileId_.id(), loaded.section.offset()};

    auto ssdPin = ssdFile.find(key);
    if (ssdPin.empty()) {
      remainingLoadIndices.emplace_back(index);
      continue;
    }

    const auto ssdSize = ssdPin.run().size();
    const auto uncompressedSize = resolveUncompressedSize(loaded.section);
    if (uncompressedSize.has_value()) {
      NIMBLE_CHECK_EQ(
          ssdSize,
          uncompressedSize.value(),
          "SSD entry size mismatch with expected uncompressed size");
    }

    velox::cache::CachePin pin;
    if (cachePins[index].has_value()) {
      pin = std::move(*cachePins[index]);
      cachePins[index].reset();
    } else {
      pin = acquireCachePin(key, ssdSize);
    }
    auto buffer = tryCacheHit(ssdSize, pin);
    if (buffer != nullptr) {
      loaded.buffer = std::move(buffer);
      continue;
    }
    ssdHits.emplace_back(SsdHit{index, std::move(ssdPin), std::move(pin)});
  }

  loadIndices = std::move(remainingLoadIndices);

  if (ssdHits.empty()) {
    return;
  }

  std::vector<velox::cache::SsdPin> ssdPins;
  std::vector<velox::cache::CachePin> loadPins;
  ssdPins.reserve(ssdHits.size());
  loadPins.reserve(ssdHits.size());
  for (auto& hit : ssdHits) {
    ssdPins.emplace_back(std::move(hit.ssdPin));
    loadPins.emplace_back(std::move(hit.cachePin));
  }
  uint64_t ssdLoadUs{0};
  {
    velox::MicrosecondTimer timer(&ssdLoadUs);
    ssdFile.load(ssdPins, loadPins);
  }
  ioStats_->queryThreadIoLatencyUs().increment(ssdLoadUs);
  ioStats_->ssdCacheReadLatencyUs().increment(ssdLoadUs);
  for (uint32_t i = 0; i < ssdHits.size(); ++i) {
    const auto& section = sections[ssdHits[i].sectionIndex].section;
    sections[ssdHits[i].sectionIndex].buffer =
        promoteCachePin(section, std::move(loadPins[i]), ioStats_->ssdRead());
  }
}

std::vector<uint32_t> CachedMetadataInput::loadFromCache(
    std::vector<LoadedSection>& sections,
    std::vector<std::optional<velox::cache::CachePin>>& cachePins) {
  cachePins.resize(sections.size());
  std::vector<uint32_t> loadIndices;
  loadIndices.reserve(sections.size());
  for (uint32_t index = 0; index < sections.size(); ++index) {
    const auto& section = sections[index].section;
    const velox::cache::RawFileCacheKey key{fileId_.id(), section.offset()};
    const auto uncompressedSize = resolveUncompressedSize(section);

    if (uncompressedSize.has_value()) {
      auto pin = acquireCachePin(key, uncompressedSize.value());
      auto buffer = tryCacheHit(uncompressedSize.value(), pin);
      if (buffer != nullptr) {
        sections[index].buffer = std::move(buffer);
        continue;
      }
      cachePins[index] = std::move(pin);
      loadIndices.push_back(index);
    } else {
      loadIndices.push_back(index);
    }
  }
  return loadIndices;
}

void CachedMetadataInput::prepareBuffers(
    const std::vector<LoadedSection>& sections,
    const std::vector<uint32_t>& loadIndices,
    const std::vector<std::optional<velox::cache::CachePin>>& cachePins,
    ReadBuffers& readBuffers) {
  readBuffers.buffers.resize(loadIndices.size());
  readBuffers.readRanges.resize(loadIndices.size());
  for (size_t i = 0; i < loadIndices.size(); ++i) {
    const auto sectionIndex = loadIndices[i];
    const auto& section = sections[sectionIndex].section;
    const auto& pin = cachePins[sectionIndex];
    if (pin.has_value() &&
        section.compressionType() == CompressionType::Uncompressed) {
      NIMBLE_CHECK(
          pin->entry()->hasContiguousData(),
          "Cache entry must have contiguous data");
      readBuffers.readRanges[i] = {
          pin->entry()->contiguousData(), section.size()};
    } else {
      readBuffers.buffers[i] =
          velox::AlignedBuffer::allocate<char>(section.size(), pool_);
      readBuffers.readRanges[i] = {
          readBuffers.buffers[i]->asMutable<char>(), section.size()};
    }
  }
}

void CachedMetadataInput::processLoadedBuffers(
    const std::vector<uint32_t>& loadIndices,
    std::vector<LoadedSection>& sections,
    std::vector<std::optional<velox::cache::CachePin>>& cachePins,
    ReadBuffers& readBuffers) {
  for (size_t i = 0; i < loadIndices.size(); ++i) {
    const auto sectionIndex = loadIndices[i];
    auto& loaded = sections[sectionIndex];
    auto& pin = cachePins[sectionIndex];
    if (pin.has_value()) {
      if (loaded.section.compressionType() != CompressionType::Uncompressed) {
        NIMBLE_CHECK_NOT_NULL(readBuffers.buffers[i]);
        NIMBLE_CHECK(
            pin->entry()->hasContiguousData(),
            "Cache entry must have contiguous data");
        std::string_view compressed{
            readBuffers.buffers[i]->as<char>(), loaded.section.size()};
        ZstdCompression::uncompress(
            compressed, pin->entry()->contiguousData(), pin->entry()->size());
      }
      loaded.buffer =
          promoteCachePin(loaded.section, std::move(*pin), ioStats_->read());
    } else {
      NIMBLE_CHECK(
          !loaded.section.uncompressedSize().has_value(),
          "Section without cache pin should not have uncompressed size set");
      loaded.buffer =
          decompressAndStore(loaded.section, std::move(readBuffers.buffers[i]));
    }
  }
}

std::vector<std::shared_ptr<MetadataBuffer>> CachedMetadataInput::load(
    std::span<const MetadataSection> sections) {
  NIMBLE_CHECK(!sections.empty(), "No sections to load");

  std::vector<LoadedSection> loadSections(sections.begin(), sections.end());
  std::vector<std::optional<velox::cache::CachePin>> cachePins;
  auto missIndices = loadFromCache(loadSections, cachePins);
  loadFromSsd(loadSections, missIndices, cachePins);

  if (!missIndices.empty()) {
    ReadBuffers readBuffers;
    prepareBuffers(loadSections, missIndices, cachePins, readBuffers);

    loadFromFile(loadSections, missIndices, readBuffers.readRanges);
    processLoadedBuffers(missIndices, loadSections, cachePins, readBuffers);
  }

  return extractResults(std::move(loadSections));
}

std::shared_ptr<MetadataBuffer> CachedMetadataInput::store(
    const MetadataSection& section,
    velox::BufferPtr&& decompressed) {
  // NOTE: This path handles sections without pre-claimed cache pins (old files
  // without uncompressed_size). Sections with pins are handled directly in
  // loadFromFile. This will be deprecated once all nimble metadata sections
  // include uncompressed size.
  const velox::cache::RawFileCacheKey key{fileId_.id(), section.offset()};
  auto pin = acquireCachePin(key, decompressed->size());
  auto buffer = tryCacheHit(decompressed->size(), pin);
  if (buffer != nullptr) {
    return buffer;
  }

  auto* entry = pin.checkedEntry();
  NIMBLE_CHECK(
      entry->hasContiguousData(), "Cache entry must have contiguous data");
  NIMBLE_CHECK_EQ(
      entry->size(),
      decompressed->size(),
      "Cache entry size mismatch with decompressed size");
  auto* destination = entry->contiguousData();
  std::memcpy(destination, decompressed->as<char>(), decompressed->size());
  entry->setExclusiveToShared();
  return std::make_shared<MetadataBuffer>(std::move(pin));
}

std::unique_ptr<MetadataBuffer> CachedMetadataInput::findCachedMetadata(
    uint64_t offset) {
  for (;;) {
    const velox::cache::RawFileCacheKey key{fileId_.id(), offset};
    folly::SemiFuture<bool> waitFuture{false};
    auto pin = cache_->findOrCreate(key, 0, /*contiguous=*/true, &waitFuture);
    if (pin.empty()) {
      if (!waitFuture.valid()) {
        return nullptr;
      }
      uint64_t waitUs{0};
      {
        velox::MicrosecondTimer timer(&waitUs);
        auto& exec = folly::QueuedImmediateExecutor::instance();
        std::move(waitFuture).via(&exec).wait();
      }
      ioStats_->queryThreadIoLatencyUs().increment(waitUs);
      ioStats_->cacheWaitLatencyUs().increment(waitUs);
      continue;
    }
    auto* entry = pin.checkedEntry();
    if (!entry->isShared()) {
      return nullptr;
    }
    NIMBLE_CHECK(
        entry->hasContiguousData(), "Cached entry must have contiguous data");
    ioStats_->ramHit().increment(entry->size());
    return std::make_unique<MetadataBuffer>(std::move(pin));
  }
}

void CachedMetadataInput::cacheMetadata(
    uint64_t cacheOffset,
    std::span<const std::string_view> ranges) {
  uint64_t totalSize = 0;
  for (const auto& range : ranges) {
    totalSize += range.size();
  }
  const velox::cache::RawFileCacheKey key{fileId_.id(), cacheOffset};
  auto pin = acquireCachePin(key, totalSize);
  velox::common::testutil::TestValue::adjust(
      "facebook::nimble::CachedMetadataInput::cacheMetadata", &pin);
  auto* entry = pin.checkedEntry();
  if (entry->isShared()) {
    return;
  }
  NIMBLE_CHECK(
      entry->hasContiguousData(), "Cache entry must have contiguous data");
  NIMBLE_CHECK_EQ(entry->size(), totalSize);
  auto* dest = entry->contiguousData();
  for (const auto& range : ranges) {
    std::memcpy(dest, range.data(), range.size());
    dest += range.size();
  }
  entry->setExclusiveToShared();
}

} // namespace facebook::nimble
