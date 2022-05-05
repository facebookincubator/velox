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

#pragma once

#include "velox/common/base/BloomFilter.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/caching/StringIdMap.h"

#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

namespace facebook::velox::cache {

// Counts distinct integers. 'range' is an estimate of expected distinct
class DistinctCounter {
 public:
  DistinctCounter(int32_t /*range*/) {}

  //  Adds a value. Returns true if the value is new.
  bool add(uint64_t value) {
    return values_.insert(value).second;
  }

  int32_t count() const {
    return values_.size();
  }

 private:
  folly::F14FastSet<uint64_t> values_;
};

// Represents a groupId, column and its size and score. These are
// sorted and as many are selected from the top as will fit on SSD.
struct SsdScore {
  // Number used for ranking. Correlates to access frequency and
  // inversely to size.
  float score;

  // Expected size in bytes for caching to SSD
  float totalBytes;
  // Recorded read activity, with older reads decayed.
  float readBytes;
  // The column and file group. Not const because this struct must be
  // move assignable as an operand to std::sort.
  uint64_t groupId;
  int32_t columnId;
};

// Tracks usage of different columns inside a file group. Tracks the number
// of distinct files in the group and their average number of splits. Produces
// estimates of the column size for each accessed column in the group.
class GroupTracker {
 public:
  static constexpr int32_t kExpectedNumFiles = 100;

  // Constructs a tracker for file group 'ame'. 'name' is for example
  // the directory path of a Hive partition.
  GroupTracker(const StringIdLease& name)
      : name_(name), numFiles_(kExpectedNumFiles) {}

  // Records that 'fileId' belongs to this group and has 'numStripes'
  // stripes. Used to scale up the column sizes reported by
  // recordReference().
  void recordFile(uint64_t fileId, int32_t numStripes);

  // Records a sample stripe size for 'columnId'.
  void recordReference(uint64_t fileId, int32_t columnId, int32_t bytes);

  // Records read of 'bytes' from 'columnId' This is compared to the
  // reference size to get read density.
  void recordRead(uint64_t fileId, int32_t columnId, int32_t bytes);

  // Adds the column scores to 'scores'. If 'decayPct' is non-0,
  // decays the recorded accesses by 'decayPct'% but at least by one
  // whole access.
  void addColumnScores(int32_t decayPct, std::vector<SsdScore>& scores);

  bool eraseColumn(int32_t columnId) {
    columns_.erase(columnId);
    return columns_.empty();
  }

 private:
  StringIdLease name_;

  // Map of column to access data.
  folly::F14FastMap<int32_t, TrackingData> columns_;

  // Count of distinct files seen in recordFile().
  DistinctCounter numFiles_;

  uint64_t numStripes_{0};
};

// Set of file group stats. There is one instance per SSD cache.
class FileGroupStats {
 public:
  FileGroupStats();

  // Records ScanTracker::recordReference at group level
  void recordReference(
      uint64_t /*fileId*/,
      uint64_t /*groupId*/,
      TrackingId /*trackingId*/,
      int32_t /*bytes*/);

  // Records ScanTracker::recordRead at group level
  void recordRead(
      uint64_t /*fileId*/,
      uint64_t /*groupId*/,
      TrackingId /*trackingId*/,
      int32_t /*bytes*/);

  // Records the existence of a distinct file inside 'groupId'
  void recordFile(
      uint64_t /*fileId*/,
      uint64_t /*groupId*/,
      int32_t /*numStripes*/);

  // Returns true if groupId, trackingId qualify the data to be cached to SSD.
  bool shouldSaveToSsd(uint64_t groupId, TrackingId trackingId) const;

  // Updates the SSD selection criteria. The group. trackingId pairs
  // that account for the top 'ssdSize' bytes of reported IO are
  // selected. If 'decayPct' is non-0, old stats are decayed and
  // removed if counts go to zero.
  void updateSsdFilter(uint64_t ssdSize, int32_t decayPct = 0);

  // Returns an estimate of the total size of the dataset based of the
  // groups, files and columns referenced to data.
  float dataSize() const {
    return dataSize_;
  }

  // Returns the percentage of historical reads that hit the currently SSD
  // cachable fraction of the data.
  float cachableReadPct() const {
    return cachableReadPct_;
  }

  // Returns percent of all seen data that fits in the SSD cachable fraction.
  float cachableDataPct() const {
    return cachableDataPct_;
  }

  // Clears the state to be as after default construction.
  void clear();

  // Recalculates the best groups and makes a human readable
  // summary. 'cacheBytes' is used to compute what fraction of the tracked
  // working set can be cached in 'cacheBytes'.
  std::string toString(uint64_t cacheBytes);

 private:
  GroupTracker& groupLocked(uint64_t id) {
    auto it = groups_.find(id);
    if (it == groups_.end()) {
      groups_[id] =
          std::make_unique<GroupTracker>(StringIdLease(fileIds(), id));
      return *groups_[id];
    }
    return *it->second;
  }

  // Returns the tracked group/column pairs best score first. Sets the
  // 'dataSize_', 'cachableReadPct_' and 'cachableDataPct_' according
  // to 'cacheBytes'. access counts by decayPct if decayPct% is
  // non-0. Trims away scores that fall to zero accesses by decay or
  // fall outside of the top FLAGS_max_group_stats top scores.
  std::vector<SsdScore> ssdScoresLocked(
      uint64_t cacheBytes,
      int32_t decayPct = 0);

  //  Removes the information on groupId/id.
  void eraseStatLocked(uint64_t groupId, int32_t columnId);
  // Serializes access to  all data members and private methods.
  std::mutex mutex_;

  folly::F14FastMap<uint64_t, std::unique_ptr<GroupTracker>> groups_;

  // Max number of columns tracked.
  const int32_t maxColumns_;

  // Set of groupId, columnId combinations for streams that should be saved
  // to SSD.
  folly::Synchronized<folly::F14FastSet<uint64_t>> saveToSsd_;
  bool ssdFilterInited_{false};
  bool allFitOnSsd_{false};
  double dataSize_{0};
  double totalRead_{0};
  float cachableDataPct_{0};
  float cachableReadPct_{0};
};

} // namespace facebook::velox::cache
