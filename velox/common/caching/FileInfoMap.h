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

#include "velox/common/time/Timer.h"

#include "folly/SharedMutex.h"
#include "folly/container/F14Map.h"

namespace facebook::velox::cache {
struct RawFileInfo {
  int64_t openTimeSec;
  bool inCache;

  bool operator==(const RawFileInfo& other) {
    return openTimeSec == other.openTimeSec && inCache == other.inCache;
  }
};

/// Return a process-wide map of file id to file info.
class FileInfoMap {
 public:
  /// Return the process-wide FileInfoMap singleton instance.
  static std::shared_ptr<FileInfoMap> getInstance() {
    return instance_;
  }

  /// Create a singleton instance of FileInfoMap.
  static void create();

  static void release();

  /// Whether an instance of FileInfoMap has been created or not.
  static bool exists() {
    return instance_ != nullptr;
  }

  /// Add file opening info for fileNum and return true if fileNum is not in the
  /// map. If the map already includes fileNum, no action will happen and return
  /// false.
  bool addOpenFileInfo(
      uint64_t fileNum,
      int64_t openTimeSec = getCurrentTimeSec());

  const RawFileInfo* find(uint64_t fileNum) const;

  RawFileInfo* find(uint64_t fileNum);

  /// Reset the RawFileInfo flag, inCache, to false for all entries.
  void resetInCache();

  /// Remove all entries with the RawFileInfo flag inCache false.
  void deleteNotInCache();

  int64_t size() {
    return fileMap_.size();
  }

  void clear() {
    fileMap_.clear();
  }

  /// For every map entry, run function. The map key fileNum and map value
  /// RawFileInfo are passed to the function.
  void forEach(std::function<void(uint64_t, RawFileInfo&)> function);

  /// Return a SharedMutex guarding the FileInfoMap instance. It can be used to
  /// acquire a read/write lock on the FileInfoMap instance.
  folly::SharedMutex& mutex();

  /// Run function with a read lock on the FileInfoMap instance.
  template <typename T>
  T withRLock(std::function<T(FileInfoMap&)> function) {
    folly::SharedMutex::ReadHolder rl(fileMapMutex_);
    return function(*this);
  }

  /// Run function with a write lock on the FileInfoMap instance.
  template <typename T>
  T withWLock(std::function<T(FileInfoMap&)> function) {
    folly::SharedMutex::WriteHolder wl(fileMapMutex_);
    return function(*this);
  }

  // For unit test only.
  const folly::F14FastMap<uint64_t, RawFileInfo>& getMap() const {
    return fileMap_;
  }

 private:
  // A singleton instance.
  static std::shared_ptr<FileInfoMap> instance_;

  folly::SharedMutex fileMapMutex_;
  folly::F14FastMap<uint64_t, RawFileInfo> fileMap_{};
};

} // namespace facebook::velox::cache
