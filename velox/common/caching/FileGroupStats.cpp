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

#include "velox/common/caching/FileGroupStats.h"

#include "velox/common/base/BitUtil.h"

#include <gflags/gflags.h>

DEFINE_int32(
    max_tracked_columns,
    200000,
    "Max number of columns*partitions tracked for possible staging "
    "on SSD");

namespace facebook::velox::cache {

void GroupTracker::recordFile(uint64_t fileId, int32_t numStripes) {
  if (numFiles_.add(fileId)) {
    numStripes_ += numStripes;
  }
}

void GroupTracker::recordReference(
    uint64_t fileId,
    int32_t columnId,
    int32_t bytes) {
  auto& data = columns_[columnId];
  data.referencedBytes += bytes;
  ++data.numReferences;
}

void GroupTracker::recordRead(
    uint64_t fileId,
    int32_t columnId,
    int32_t bytes) {
  auto& data = columns_[columnId];
  data.readBytes += bytes;
  ++data.numReads;
}

namespace {
uint64_t ssdFilterHash(uint64_t groupId, int32_t columnId) {
  return bits::hashMix(
      folly::hasher<uint64_t>()(groupId), folly::hasher<uint64_t>()(columnId));
}

// Returns an arbitrary multiplier for score based on
// size.

float sizeFactor(float size) {
  // Number of bytes transferred as part of a large request in in
  // the time of a round trip with no data transfer.
  constexpr float kBytesPerLatency = 10000;
  return kBytesPerLatency / (kBytesPerLatency + size);
}

// Decayse count by decayPct%. 'count' always decreases by at least
// 1. bytes is scaled down pro rata so as to maintain bytes/count.
void decay(int32_t decayPct, int64_t& bytes, int32_t& count) {
  if (!count) {
    bytes = 0;
    return;
  }
  auto newCount = count * (100 - decayPct) / 100;
  bytes = bytes * newCount / count;
  count = newCount;
}

void decay(int32_t decayPct, TrackingData& data) {
  decay(decayPct, data.referencedBytes, data.numReferences);
  decay(decayPct, data.readBytes, data.numReads);
}
} // namespace

void GroupTracker::addColumnScores(
    int32_t decayPct,
    std::vector<SsdScore>& scores) {
  int32_t numFiles = numFiles_.count();
  if (!numFiles) {
    return;
  }
  std::vector<int32_t> toErase;
  auto stripesInFile = numStripes_ / numFiles;
  auto numStripes = numFiles * stripesInFile;
  for (auto& pair : columns_) {
    auto& data = pair.second;
    if (decayPct) {
      decay(decayPct, data);
    }
    if (!data.numReads || !data.numReferences) {
      toErase.push_back(pair.first);
      continue;
    }
    float size = (data.referencedBytes / data.numReferences) * numStripes;
    float readSize = data.readBytes / data.numReads;
    float readFraction = readSize / size;
    float score = data.numReads * sizeFactor(size) * readFraction;
    scores.push_back(SsdScore{
        score,
        static_cast<float>(size),
        static_cast<float>(data.readBytes),
        name_.id(),
        pair.first});
  }
  for (auto id : toErase) {
    columns_.erase(id);
  }
}

FileGroupStats::FileGroupStats() : maxColumns_(FLAGS_max_tracked_columns) {}

void FileGroupStats::recordFile(
    uint64_t fileId,
    uint64_t groupId,
    int32_t numStripes) {
  std::lock_guard<std::mutex> l(mutex_);
  groupLocked(groupId).recordFile(fileId, numStripes);
}

void FileGroupStats::recordReference(
    uint64_t fileId,
    uint64_t groupId,
    TrackingId trackingId,
    int32_t bytes) {
  std::lock_guard<std::mutex> l(mutex_);
  groupLocked(groupId).recordReference(fileId, trackingId.columnId(), bytes);
}

void FileGroupStats::recordRead(
    uint64_t fileId,
    uint64_t groupId,
    TrackingId trackingId,
    int32_t bytes) {
  std::lock_guard<std::mutex> l(mutex_);
  groupLocked(groupId).recordRead(fileId, trackingId.columnId(), bytes);
}

bool FileGroupStats::shouldSaveToSsd(uint64_t groupId, TrackingId trackingId)
    const {
  if (!ssdFilterInited_) {
    return false;
  }
  if (allFitOnSsd_) {
    return true;
  }
  uint64_t hash = ssdFilterHash(groupId, trackingId.columnId());
  return saveToSsd_.withRLock(
      [&](auto& set) { return set.find(hash) != set.end(); });
}

std::vector<SsdScore> FileGroupStats::ssdScoresLocked(
    uint64_t cacheBytes,
    int32_t decayPct) {
  std::vector<SsdScore> scores;
  std::vector<SsdScore> deleted;
  for (auto& pair : groups_) {
    pair.second->addColumnScores(decayPct, scores);
  }
  // Sort the scores, high score first.
  std::sort(
      scores.begin(),
      scores.end(),
      [](const SsdScore& left, const SsdScore& right) {
        return left.score > right.score;
      });
  if (scores.size() > maxColumns_) {
    for (auto i = maxColumns_; i < scores.size(); ++i) {
      eraseStatLocked(scores[i].groupId, scores[i].columnId);
    }
    scores.resize(maxColumns_);
  }
  float totalSize = 0;
  totalRead_ = 0;
  int32_t numCachable = -1;
  float cachableReads = 0;
  for (auto i = 0; i < scores.size(); ++i) {
    totalSize += scores[i].totalBytes;
    totalRead_ += scores[i].readBytes;
    if (totalSize < cacheBytes) {
      cachableReads += scores[i].readBytes;
    } else if (numCachable == -1) {
      numCachable = i;
    }
  }
  dataSize_ = totalSize;
  cachableDataPct_ = 100 * cacheBytes / totalSize;
  cachableReadPct_ = 100 * cachableReads / totalRead_;

  return scores;
}

void FileGroupStats::eraseStatLocked(uint64_t groupId, int32_t columnId) {
  auto it = groups_.find(groupId);
  if (it != groups_.end()) {
    if (it->second->eraseColumn(columnId)) {
      groups_.erase(it);
    }
  }
}

void FileGroupStats::updateSsdFilter(uint64_t ssdSize, int32_t decayPct) {
  std::lock_guard<std::mutex> l(mutex_);
  auto scores = ssdScoresLocked(ssdSize, decayPct);
  float size = 0;

  int32_t i = 0;
  for (; i < scores.size(); ++i) {
    size += scores[i].totalBytes;
    if (size > ssdSize) {
      break;
    }
  }
  if (i == scores.size()) {
    allFitOnSsd_ = true;
    ssdFilterInited_ = true;
  } else {
    folly::F14FastSet<uint64_t> newFilter;
    for (auto included = 0; included < i; ++included) {
      auto hash = ssdFilterHash(scores[i].groupId, scores[i].columnId);
      newFilter.insert(hash);
    }
    saveToSsd_.withWLock([&](auto& set) { set = std::move(newFilter); });
    ssdFilterInited_ = true;
    allFitOnSsd_ = false;
  }
}

void FileGroupStats::clear() {
  groups_.clear();
  dataSize_ = 0;
  cachableDataPct_ = 0;
  cachableReadPct_ = 0;
}

std::string FileGroupStats::toString(uint64_t cacheBytes) {
  std::stringstream out;
  std::vector<SsdScore> scores;
  {
    std::lock_guard<std::mutex> l(mutex_);
    scores = ssdScoresLocked(cacheBytes);
  }
  out << fmt::format(
      "Group tracking: {} groups, {} bytes, {} bytes read\n",
      scores.size(),
      dataSize_,
      totalRead_);
  out << fmt::format(
      "Cache covers {}% of data and {}% of reads\n",
      cachableDataPct_,
      cachableReadPct_);

  return out.str();
}

} // namespace facebook::velox::cache
