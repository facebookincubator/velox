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

#include "velox/common/caching/SsdCache.h"
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/portability/SysUio.h>
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/GroupTracker.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <numeric>

DEFINE_bool(ssd_odirect, true, "use O_DIRECT for SSD cache IO");
DEFINE_bool(verify_ssd_write, false, "Read back data after writing to SSD");

namespace facebook::velox::cache {

SsdPin::SsdPin(SsdFile& file, SsdRun run) : file_(&file), run_(run) {
  file_->checkPinned(run_.offset());
}

SsdPin::~SsdPin() {
  if (file_) {
    file_->unpinRegion(run_.offset());
  }
}

void SsdPin::operator=(SsdPin&& other) {
  if (file_) {
    file_->unpinRegion(run_.offset());
  }
  file_ = other.file_;
  other.file_ = nullptr;
  run_ = other.run_;
}

SsdFile::SsdFile(
    const std::string& filename,
    SsdCache& cache,
    int32_t ordinal,
    int32_t maxRegions)
    : cache_(cache),
      ordinal_(ordinal),
      maxRegions_(maxRegions),
      filename_(filename) {
  fd_ = open(
      filename.c_str(),
      O_CREAT | O_RDWR | (FLAGS_ssd_odirect ? O_DIRECT : 0),
      S_IRUSR | S_IWUSR);
  if (fd_ < 0) {
    LOG(ERROR) << "Cannot open or create " << filename << " error " << errno;
    exit(1);
  }
  readFile_ = std::make_unique<LocalReadFile>(fd_);
  uint64_t size = lseek(fd_, 0, SEEK_END);
  numRegions_ = size / kRegionSize;
  if (size % kRegionSize > 0) {
    ftruncate(fd_, numRegions_ * kRegionSize);
  }
  // The existing regions in the file are writable.
  for (auto i = 0; i < numRegions_; ++i) {
    writableRegions_.push_back(i);
  }
  fileSize_ = numRegions_ * kRegionSize;
  regionScore_.resize(maxRegions_);
  regionSize_.resize(maxRegions_);
  regionPins_.resize(maxRegions_);
}

void SsdFile::pinRegion(uint64_t offset) {
  std::lock_guard<std::mutex> l(mutex_);
  pinRegionLocked(offset);
}

void SsdFile::unpinRegion(uint64_t offset) {
  std::lock_guard<std::mutex> l(mutex_);
  auto count = --regionPins_[regionIndex(offset)];
  if (suspended_ && count == 0) {
    evictLocked();
  }
}

namespace {
void addEntryToIovecs(AsyncDataCacheEntry& entry, std::vector<iovec>& iovecs) {
  if (entry.tinyData()) {
    iovecs.push_back({entry.tinyData(), static_cast<size_t>(entry.size())});
    return;
  }
  auto& data = entry.data();
  iovecs.reserve(iovecs.size() + data.numRuns());
  int64_t bytesLeft = entry.size();
  for (auto i = 0; i < data.numRuns(); ++i) {
    auto run = data.runAt(i);
    iovecs.push_back(
        {run.data<char>(), std::min<size_t>(bytesLeft, run.numBytes())});
    bytesLeft -= run.numBytes();
    if (bytesLeft <= 0) {
      break;
    };
  }
}
} // namespace

SsdPin SsdFile::find(RawFileCacheKey key) {
  SsdKey ssdKey{StringIdLease(fileIds(), key.fileNum), key.offset};
  SsdRun run;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (suspended_) {
      return SsdPin();
    }
    newEventLocked();
    auto it = entries_.find(ssdKey);
    if (it == entries_.end()) {
      return SsdPin();
    }
    run = it->second;
    pinRegionLocked(run.offset());
  }
  return SsdPin(*this, run);
}

void SsdFile::newEventLocked() {
  ++numEvents_;
  if (numEvents_ > kDecayInterval && numEvents_ > entries_.size() / 2) {
    numEvents_ = 0;
    for (auto i = 0; i < numRegions_; ++i) {
      int64_t score = regionScore_[i];
      regionScore_[i] = (score * 15) / 16;
    }
  }
}

void SsdFile::load(SsdRun run, AsyncDataCacheEntry& entry) {
  VELOX_CHECK_EQ(run.size(), entry.size());
  regionScore_[regionIndex(run.offset())] += run.size();
  ++stats_.entriesRead;
  stats_.bytesRead += run.size();
  if (entry.tinyData()) {
    auto rc = pread(fd_, entry.tinyData(), run.size(), run.offset());
    VELOX_CHECK_EQ(rc, run.size());
  } else {
    std::vector<struct iovec> iovecs;
    addEntryToIovecs(entry, iovecs);

    auto rc = folly::preadv(fd_, iovecs.data(), iovecs.size(), run.offset());
    VELOX_CHECK_EQ(rc, run.size());
  }
  entry.setSsdFile(this, run.offset());
}

void SsdFile::read(
    uint64_t offset,
    const std::vector<folly::Range<char*>> buffers,
    int32_t numEntries) {
  stats_.entriesRead += numEntries;
  uint64_t regionBytes = 0;
  uint64_t regionOffset = offset;
  for (auto range : buffers) {
    if (range.data()) {
      regionBytes += range.size();
    }
    regionOffset += range.size();
  }
  regionUsed(regionIndex(offset), regionBytes);
  stats_.bytesRead += regionBytes;
  readFile_->preadv(offset, buffers);
}

std::pair<uint64_t, int32_t> SsdFile::getSpace(
    const std::vector<CachePin>& pins,
    int32_t begin) {
  std::lock_guard<std::mutex> l(mutex_);
  for (;;) {
    if (writableRegions_.empty()) {
      if (!evictLocked()) {
        return {0, 0};
      }
    }
    auto region = writableRegions_[0];
    auto offset = regionSize_[region];
    auto available = kRegionSize - regionSize_[region];
    int64_t toWrite = 0;
    for (; begin < pins.size(); ++begin) {
      auto entry = pins[begin].entry();
      if (entry->size() > available) {
        break;
      }
      available -= entry->size();
      toWrite += entry->size();
    }
    if (toWrite) {
      regionSize_[region] += toWrite;
      return {region * kRegionSize + offset, toWrite};
    }
    // A region has been filled. Set its score to be at least the best
    // score + kRegionSize so that it gets time to live. Otherwise it
    // has had the least time to get hits and would be the first
    // evicted.
    uint64_t best = 0;
    for (auto& score : regionScore_) {
      best = std::max<uint64_t>(best, score);
    }
    regionScore_[region] = std::max(regionScore_[region], best + kRegionSize);
    writableRegions_.erase(writableRegions_.begin());
  }
}

bool SsdFile::evictLocked() {
  if (numRegions_ < maxRegions_) {
    auto newSize = (numRegions_ + 1) * kRegionSize;
    auto rc = ftruncate(fd_, newSize);
    if (rc >= 0) {
      fileSize_ = newSize;
      writableRegions_.push_back(numRegions_);
      regionSize_[numRegions_] = 0;
      ++numRegions_;
      return true;
    } else {
      LOG(ERROR) << "Failed to grow cache file " << filename_ << " to "
                 << newSize;
    }
  }
  std::vector<int32_t> candidates;
  int64_t scoreSum = 0;
  for (int i = 0; i < numRegions_; ++i) {
    if (regionPins_[i]) {
      continue;
    }
    if (candidates.empty() || regionScore_[i] < scoreSum / candidates.size()) {
      scoreSum += regionScore_[i];
      candidates.push_back(i);
    }
  }
  if (candidates.empty()) {
    suspended_ = true;
    return false;
  }
  std::sort(
      candidates.begin(), candidates.end(), [&](int32_t left, int32_t right) {
        return regionScore_[left] < regionScore_[right];
      });
  // Free up to 3 lowest score regions.
  if (candidates.size() > 3) {
    candidates.resize(3);
  }
  clearRegionEntriesLocked(candidates);
  writableRegions_ = std::move(candidates);
  suspended_ = false;
  return true;
}

void SsdFile::clearRegionEntriesLocked(const std::vector<int32_t>& toErase) {
  // Remove all 'entries_' where the dependent points one of 'toErase'.
  auto it = entries_.begin();
  while (it != entries_.end()) {
    auto region = regionIndex(it->second.offset());
    if (std::find(toErase.begin(), toErase.begin(), region) != toErase.end()) {
      it = entries_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto region : toErase) {
    // While the region is being filled it may get score from
    // hits. When it is full, it will get a score boost to be a little
    // ahead of the best.
    regionScore_[region] = 0;
    regionSize_[region] = 0;
  }
}

void SsdFile::store(std::vector<CachePin>& pins) {
  std::sort(pins.begin(), pins.end());
  uint64_t total = 0;
  for (auto& pin : pins) {
    total += pin.entry()->size();
  }
  int32_t storeIndex = 0;
  while (storeIndex < pins.size()) {
    auto [offset, available] = getSpace(pins, storeIndex);
    if (!available) {
      // No space can be reclaimed. The pins are freed when the caller is freed.
      return;
    }
    int32_t numWritten = 0;
    int32_t bytes = 0;
    std::vector<iovec> iovecs;
    for (auto i = storeIndex; i < pins.size(); ++i) {
      auto entrySize = pins[i].entry()->size();
      if (bytes + entrySize > available) {
        break;
      }
      pins[i].entry()->setSsdFile(this, offset);
      addEntryToIovecs(*pins[i].entry(), iovecs);
      bytes += entrySize;
      ++numWritten;
    }
    VELOX_CHECK_GE(fileSize_, offset + bytes);
    auto rc = folly::pwritev(fd_, iovecs.data(), iovecs.size(), offset);
    if (rc != bytes) {
      LOG(ERROR) << "Failed to write to SSD " << errno;
      return;
    }
    {
      std::lock_guard<std::mutex> l(mutex_);
      for (auto i = storeIndex; i < storeIndex + numWritten; ++i) {
        auto entry = pins[i].entry();
        entry->setSsdFile(this, offset);
        char first = entry->tinyData() ? entry->tinyData()[0]
                                       : entry->data().runAt(0).data<char>()[0];
        auto size = entry->size();
        SsdKey key = {
            entry->key().fileNum, static_cast<uint64_t>(entry->offset())};
        entries_[std::move(key)] = SsdRun(offset, size);
        if (FLAGS_verify_ssd_write) {
          verifyWrite(*entry, SsdRun(offset, size));
        }
        offset += size;
        ++stats_.entriesWritten;
        stats_.bytesWritten += size;
      }
    }
    storeIndex += numWritten;
  }
}

char* readBytes(int fd, int64_t offset, int32_t size) {
  char* data = (char*)malloc(size);
  pread(fd, data, size, offset);
  return data;
}

namespace {
int32_t indexOfFirstMismatch(char* x, char* y, int n) {
  for (auto i = 0; i < n; ++i) {
    if (x[i] != y[i]) {
      return i;
    }
  }
  return -1;
}
} // namespace

void SsdFile::verifyWrite(AsyncDataCacheEntry& entry, SsdRun ssdRun) {
  auto testData = std::make_unique<char[]>(entry.size());
  auto rc = pread(fd_, testData.get(), entry.size(), ssdRun.offset());
  VELOX_CHECK_EQ(rc, entry.size());
  if (entry.tinyData()) {
    if (0 != memcmp(testData.get(), entry.tinyData(), entry.size())) {
      VELOX_FAIL("bad read back");
    }
  } else {
    auto& data = entry.data();
    int64_t bytesLeft = entry.size();
    int64_t offset = 0;
    for (auto i = 0; i < data.numRuns(); ++i) {
      auto run = data.runAt(i);
      auto compareSize = std::min<int64_t>(bytesLeft, run.numBytes());
      auto badIndex = indexOfFirstMismatch(
          run.data<char>(), testData.get() + offset, compareSize);
      if (badIndex != -1) {
        VELOX_FAIL("Bad read back");
      }
      bytesLeft -= run.numBytes();
      offset += run.numBytes();
      if (bytesLeft <= 0) {
        break;
      };
    }
  }
}

void SsdFile::updateStats(SsdCacheStats& stats) {
  stats.entriesWritten += stats_.entriesWritten;
  stats.bytesWritten += stats_.bytesWritten;
  stats.entriesRead += stats_.entriesRead;
  stats.bytesRead += stats_.bytesRead;
  stats.entriesCached += entries_.size();
  for (auto& regionSize : regionSize_) {
    stats.bytesCached += regionSize;
  }
}

void SsdFile::clear() {
  std::lock_guard<std::mutex> l(mutex_);
  entries_.clear();
  std::fill(regionSize_.begin(), regionSize_.end(), 0);
  writableRegions_.resize(numRegions_);
  std::iota(writableRegions_.begin(), writableRegions_.end(), 0);
  std::fill(regionScore_.begin(), regionScore_.end(), 0);
}

SsdCache::SsdCache(
    std::string_view filePrefix,
    uint64_t maxBytes,
    int32_t numShards)
    : filePrefix_(filePrefix),
      numShards_(numShards),
      groupStats_(std::make_unique<FileGroupStats>()) {
  files_.reserve(numShards_);
  uint64_t kSizeQuantum = numShards_ * SsdFile::kRegionSize;
  int32_t fileMaxRegions = bits::roundUp(maxBytes, kSizeQuantum) / kSizeQuantum;
  for (auto i = 0; i < numShards_; ++i) {
    files_.push_back(std::make_unique<SsdFile>(
        fmt::format("{}{}", filePrefix_, i), *this, i, fileMaxRegions));
  }
}

SsdFile& SsdCache::file(uint64_t fileId) {
  auto index = fileId % numShards_;
  return *files_[index];
}

namespace {
folly::IOThreadPoolExecutor* ssdStoreExecutor() {
  static auto executor = std::make_unique<folly::IOThreadPoolExecutor>(4);
  return executor.get();
}
} // namespace

bool SsdCache::startStore() {
  if (0 == storesInProgress_.fetch_add(numShards_)) {
    return true;
  }
  storesInProgress_.fetch_sub(numShards_);
  return false;
}

void SsdCache::store(std::vector<CachePin> pins) {
  std::vector<std::vector<CachePin>> shards(numShards_);
  for (auto& pin : pins) {
    auto& target = file(pin.entry()->key().fileNum.id());
    shards[target.ordinal()].push_back(std::move(pin));
  }
  int32_t numNoStore = 0;
  for (auto i = 0; i < numShards_; ++i) {
    if (shards[i].empty()) {
      ++numNoStore;
      continue;
    }
    struct PinHolder {
      std::vector<CachePin> pins;

      PinHolder(std::vector<CachePin>&& _pins) : pins(std::move(_pins)) {}
    };

    // We move the mutable vector of pins to the executor. These must
    // be wrapped in a shared struct to be passed via lambda capture.
    auto pinHolder = std::make_shared<PinHolder>(std::move(shards[i]));
    ssdStoreExecutor()->add([this, i, pinHolder]() {
      files_[i]->store(pinHolder->pins);
      --storesInProgress_;
    });
  }
  storesInProgress_.fetch_sub(numNoStore);
}

SsdCacheStats SsdCache::stats() const {
  SsdCacheStats stats;
  for (auto& file : files_) {
    file->updateStats(stats);
  }
  return stats;
}

void SsdCache::clear() {
  for (auto& file : files_) {
    file->clear();
  }
}

std::string SsdCache::toString() const {
  auto data = stats();
  uint64_t capacity =
      files_.size() * files_[0]->maxRegions() * SsdFile::kRegionSize;
  std::stringstream out;
  out << "Ssd cache IO: Write " << (data.bytesWritten >> 20) << "MB read "
      << (data.bytesRead >> 20) << "MB Size " << (capacity >> 30)
      << "GB Occupied " << (data.bytesCached >> 30) << "GB";
  out << (data.entriesCached >> 10) << "K entries.";
  out << "\nGroupStats: " << groupStats_->toString(capacity);
  return out.str();
}

} // namespace facebook::velox::cache
