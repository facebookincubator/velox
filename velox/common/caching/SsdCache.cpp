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
#include <folly/Executor.h>
#include <folly/portability/SysUio.h>
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/FileInfoMap.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/time/Timer.h"

#include <fcntl.h>
#ifdef linux
#include <linux/fs.h>
#endif // linux

#include <filesystem>
#include <fstream>
#include <numeric>

namespace facebook::velox::cache {

SsdCache::SsdCache(
    std::string_view filePrefix,
    uint64_t maxBytes,
    int32_t numShards,
    folly::Executor* executor,
    int64_t checkpointIntervalBytes,
    bool disableFileCow)
    : filePrefix_(filePrefix),
      numShards_(numShards),
      makeCheckpoint_(checkpointIntervalBytes > 0),
      groupStats_(std::make_unique<FileGroupStats>()),
      executor_(executor) {
  // Make sure the given path of Ssd files has the prefix for local file system.
  // Local file system would be derived based on the prefix.
  VELOX_CHECK(
      filePrefix_.find("/") == 0,
      "Ssd path '{}' does not start with '/' that points to local file system.",
      filePrefix_);
  filesystems::getFileSystem(filePrefix_, nullptr)
      ->mkdir(std::filesystem::path(filePrefix).parent_path().string());

  files_.reserve(numShards_);
  // Cache size must be a multiple of this so that each shard has the same max
  // size.
  uint64_t sizeQuantum = numShards_ * SsdFile::kRegionSize;
  int32_t fileMaxRegions = bits::roundUp(maxBytes, sizeQuantum) / sizeQuantum;
  for (auto i = 0; i < numShards_; ++i) {
    files_.push_back(std::make_unique<SsdFile>(
        fmt::format("{}{}", filePrefix_, i),
        i,
        fileMaxRegions,
        this,
        checkpointIntervalBytes / numShards,
        disableFileCow));
  }
}

SsdFile& SsdCache::file(uint64_t fileId) {
  const auto index = fileId % numShards_;
  return *files_[index];
}

bool SsdCache::startWrite() {
  if (isShutdown_) {
    return false;
  }
  if (writesInProgress_.fetch_add(numShards_) == 0) {
    // No write was pending, so now all shards are counted as writing.
    return true;
  }
  // There were writes in progress, so compensate for the increment.
  writesInProgress_.fetch_sub(numShards_);
  return false;
}

void SsdCache::write(std::vector<CachePin> pins) {
  VELOX_CHECK_LE(numShards_, writesInProgress_);

  const auto startTimeUs = getCurrentTimeMicro();

  uint64_t bytes = 0;
  std::vector<std::vector<CachePin>> shards(numShards_);
  for (auto& pin : pins) {
    bytes += pin.checkedEntry()->size();
    const auto& target = file(pin.checkedEntry()->key().fileNum.id());
    shards[target.shardId()].push_back(std::move(pin));
  }

  int32_t numNoStore = 0;
  for (auto i = 0; i < numShards_; ++i) {
    if (shards[i].empty()) {
      ++numNoStore;
      continue;
    }
    struct PinHolder {
      std::vector<CachePin> pins;

      explicit PinHolder(std::vector<CachePin>&& _pins)
          : pins(std::move(_pins)) {}
    };

    // We move the mutable vector of pins to the executor. These must
    // be wrapped in a shared struct to be passed via lambda capture.
    auto pinHolder = std::make_shared<PinHolder>(std::move(shards[i]));
    executor_->add([this, i, pinHolder, bytes, startTimeUs]() {
      try {
        files_[i]->write(pinHolder->pins);
      } catch (const std::exception& e) {
        // Catch so as not to miss updating 'writesInProgress_'. Could
        // theoretically happen for std::bad_alloc or such.
        VELOX_SSD_CACHE_LOG(WARNING)
            << "Ignoring error in SsdFile::write: " << e.what();
      }
      if (--writesInProgress_ == 0) {
        // Typically occurs every few GB. Allows detecting unusually slow rates
        // from failing devices.
        VELOX_SSD_CACHE_LOG(INFO) << fmt::format(
            "Wrote {}MB, {} MB/s",
            bytes >> 20,
            static_cast<float>(bytes) / (getCurrentTimeMicro() - startTimeUs));
      }
    });
  }
  writesInProgress_.fetch_sub(numNoStore);
}

SsdCacheStats SsdCache::refreshStats() const {
  SsdCacheStats stats;
  for (auto& file : files_) {
    file->updateStats(stats);
  }
  return stats;
}

void SsdCache::applyTtl(int64_t maxFileOpenTime) {
  if (isShutdown_) {
    return;
  }

  // If cache has writes in progress, skip entry eviction but mark the entries
  // in cache.
  bool writesInProgress = writesInProgress_.fetch_add(numShards_) > 0;

  std::vector<folly::SemiFuture<bool>> waitFutures;
  waitFutures.reserve(numShards_);
  for (auto i = 0; i < numShards_; i++) {
    auto [promise, future] = folly::makePromiseContract<bool>();
    waitFutures.push_back(std::move(future));

    executor_->add([this,
                    i,
                    maxFileOpenTime,
                    writesInProgress,
                    promise = std::move(promise)]() mutable {
      try {
        files_[i]->applyTtl(maxFileOpenTime, writesInProgress);
      } catch (const std::exception& e) {
        LOG(ERROR) << "Error applying TTL to SSD shard "
                   << files_[i]->shardId();
      }
      --writesInProgress_;
      promise.setValue(false);
    });
  }

  for (auto& future : waitFutures) {
    auto& exec = folly::QueuedImmediateExecutor::instance();
    std::move(future).via(&exec).wait();
  }
}

void SsdCache::makeFileInfoMapCheckpoint() {
  if (!FileInfoMap::exists() || !makeCheckpoint_) {
    return;
  }

  folly::SharedMutex::ReadHolder l(FileInfoMap::getInstance()->mutex());

  const auto checkpointPath = filePrefix_ + kFileInfoMapCheckpointExtension;
  try {
    std::ofstream state;
    state.exceptions(std::ofstream::failbit);
    state.open(checkpointPath, std::ios_base::out | std::ios_base::trunc);

    state.write(kFileInfoMapCheckpointMagic, 7);
    FileInfoMap::getInstance()->forEach(
        [&state](uint64_t fileNum, RawFileInfo& rawFileInfo) {
          auto fileName = fileIds().string(fileNum);
          int32_t length = fileName.size();
          state.write(reinterpret_cast<char*>(&length), sizeof(length));
          state.write(fileName.data(), length);
          state.write(
              reinterpret_cast<const char*>(&rawFileInfo.openTimeSec),
              sizeof(rawFileInfo.openTimeSec));
        });
    int32_t endMarker = kFileInfoMapCheckpointEndMarker;
    state.write(reinterpret_cast<const char*>(&endMarker), sizeof(endMarker));

    if (state.bad()) {
      throw std::runtime_error(fmt::format(
          "Error in writing file info map checkpoint file {}.",
          checkpointPath));
    }
    state.close();

    auto fd = open(checkpointPath.c_str(), O_WRONLY);
    if (fd <= 0) {
      throw std::runtime_error(fmt::format(
          "Error in opening file info map checkpoint file for sync: {}.", fd));
    }
    auto rc = fsync(fd);
    if (rc < 0) {
      throw std::runtime_error(fmt::format(
          "Error in syncing info file map checkpoint file: {}.", rc));
    }
    close(fd);

    VELOX_SSD_CACHE_LOG(INFO)
        << FileInfoMap::getInstance()->size()
        << " entries from file info map are written to the checkpoint.";
  } catch (const std::exception& e) {
    VELOX_SSD_CACHE_LOG(ERROR)
        << "Error in making checkpoint for the file info map. Deleting the checkpoint file "
        << checkpointPath;
    auto rc = unlink(checkpointPath.c_str());
    if (rc) {
      VELOX_SSD_CACHE_LOG(ERROR)
          << "Error in deleting the file info map checkpoint file "
          << checkpointPath << " after write failure. RC: " << rc;
    }
    throw e;
  }
}

void SsdCache::readFileInfoMapCheckpoint() {
  if (!FileInfoMap::exists() || !makeCheckpoint_) {
    return;
  }

  folly::SharedMutex::WriteHolder l(FileInfoMap::getInstance()->mutex());

  VELOX_CHECK_EQ(
      FileInfoMap::getInstance()->size(),
      0,
      "File info map is not empty before recovering from checkpoint.");

  auto checkpointPath = filePrefix_ + kFileInfoMapCheckpointExtension;
  std::ifstream state(checkpointPath);
  if (!state.is_open()) {
    LOG(INFO) << "No file info map checkpoint file is found.";
    return;
  }

  try {
    state.exceptions(std::ifstream::failbit);

    char magic[7];
    state.read(magic, sizeof(magic));
    VELOX_CHECK_EQ(
        strncmp(magic, kFileInfoMapCheckpointMagic, sizeof(magic)), 0);

    int64_t numEntries = 0;
    for (;;) {
      int32_t length;
      state.read(reinterpret_cast<char*>(&length), sizeof(length));
      if (length == kFileInfoMapCheckpointEndMarker) {
        break;
      }

      std::string fileName;
      fileName.resize(length);
      state.read(fileName.data(), length);
      uint64_t fileNum = fileIds().id(fileName);
      bool skipEntry = fileNum == StringIdMap::kNoId;

      int64_t openTimeSec;
      state.read(reinterpret_cast<char*>(&openTimeSec), sizeof(openTimeSec));

      if (skipEntry) {
        continue;
      }

      FileInfoMap::getInstance()->addOpenFileInfo(fileNum, openTimeSec);
      numEntries++;
    }

    VELOX_SSD_CACHE_LOG(INFO)
        << "Recovered " << numEntries
        << " entries of the file info map from checkpoint.";
  } catch (const std::exception& e) {
    try {
      VELOX_SSD_CACHE_LOG(ERROR)
          << "Error recovering file info map from the checkpoint file "
          << checkpointPath << ": " << e.what()
          << ". Starting with the file info map reset and deleting the checkpoint file.";
      FileInfoMap::getInstance()->clear();
      auto rc = unlink(checkpointPath.c_str());
      if (rc) {
        VELOX_SSD_CACHE_LOG(ERROR)
            << "Error in deleting the file info map checkpoint file "
            << checkpointPath << " after recovery failure. RC: " << rc;
      }
    } catch (const std::exception& e) {
      VELOX_SSD_CACHE_LOG(ERROR)
          << "Error in deleting the file info map checkpoint file "
          << checkpointPath << " after recovery failure: " << e.what();
    }
  }
}

void SsdCache::clear() {
  for (auto& file : files_) {
    file->clear();
  }
}

std::string SsdCache::toString() const {
  auto data = refreshStats();
  uint64_t capacity = maxBytes();
  std::stringstream out;
  out << "[Ssd cache] IO: Write " << (data.bytesWritten) << " Bytes, Read "
      << (data.bytesRead) << " Bytes, Size " << (capacity)
      << " Bytes, Occupied " << (data.bytesCached) << " Bytes.\n";
  out << "Internal: " << (data.entriesCached) << " entries.\n";
  out << "GroupStats: " << groupStats_->toString(capacity) << ".\n";
  return out.str();
}

void SsdCache::testingDeleteFiles() {
  for (auto& file : files_) {
    file->deleteFile();
  }
}

void SsdCache::shutdown() {
  isShutdown_ = true;
  while (writesInProgress_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // NOLINT
  }
  for (auto& file : files_) {
    file->checkpoint(true);
  }
}

} // namespace facebook::velox::cache
