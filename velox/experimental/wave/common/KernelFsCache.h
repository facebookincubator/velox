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

#include <atomic>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "velox/experimental/wave/common/Cuda.h"

namespace facebook::velox::wave {

/// Caches compiled CUDA kernels on the filesystem, keyed by source hash.
/// Stores .cu source, .cubin binaries, and .names (mangled entry points)
/// in a directory. On lookup, hashes the source text and compares against
/// cached entries to avoid recompilation across runs.
///
/// Safe to share one directory between concurrent processes. Two properties
/// carry that: an entry's file name is derived from its own source text, so
/// two processes compiling the same kernel converge on the same name instead
/// of racing for a counter; and every file is published by renaming a
/// process-private temporary into place, so a reader sees a whole file or no
/// file. Neither process needs a lock, and a duplicate compile wastes work
/// without corrupting anything.
///
/// Directories written by earlier versions, whose entries are named by an
/// ordinal rather than by content, are still read: a cached entry is found by
/// the text of its .cu, and the file name is not otherwise interpreted.
class KernelFsCache {
 public:
  explicit KernelFsCache(const std::string& cacheDir);

  /// Returns a compiled kernel for the given key. If a matching source is
  /// found in the cache directory, loads the pre-compiled CUBIN. Otherwise
  /// compiles via wave and stores the result for future runs.
  std::unique_ptr<CompiledKernel> getKernel(
      const std::string& key,
      KernelGenFunc genFunc);

  /// Returns the number of cache hits since construction.
  int64_t hits() const {
    return hits_;
  }

  /// Returns the number of cache misses since construction.
  int64_t misses() const {
    return misses_;
  }

  /// Returns the number of distinct successfully cached entries.
  int32_t size() const {
    return numEntries_.load();
  }

 private:
  struct CacheEntry {
    // File name of the entry without its extension. For an entry this version
    // wrote it is the content digest; for one an earlier version wrote it is
    // the ordinal it was allocated. Never parsed, only pasted into a path.
    std::string stem;
  };

  /// Scans the cache directory and populates hashToEntries_ on first call.
  void init();

  /// Returns the path for the .cu source file of the given entry.
  std::string cuPath(std::string_view stem) const;
  /// Returns the path for the .cubin binary of the given entry.
  std::string cubinPath(std::string_view stem) const;
  /// Returns the path for the .cubin.names sidecar of the given entry.
  std::string namesPath(std::string_view stem) const;

  // Stem of a temporary this process alone writes, unique across processes
  // sharing the directory. The compiler writes the cubin and names here and
  // they are renamed into place once whole.
  std::string tempStem();

  // True when all three files of 'stem' are present and neither the source nor
  // the cubin is empty, i.e. the entry was fully published. A partially visible
  // entry is reported as absent so the caller recompiles rather than loading a
  // torn cubin.
  bool entryComplete(std::string_view stem) const;

  // Directory holding the cached .cu, .cubin, and .cubin.names files.
  std::string cacheDir_;
  // True after the first call to init().
  bool initialized_{false};
  // Protects hashToEntries_ and initialized_.
  std::mutex mutex_;
  // Maps source-text hash to cache entries sharing that hash.
  // Using unordered_map because adding folly::F14FastMap as a dependency
  // causes dependency count regressions across all downstream wave targets.
  std::unordered_map<size_t, std::vector<CacheEntry>> hashToEntries_;
  // Serial number for compile operations, part of the temporary file name.
  std::atomic<int32_t> compileSerial_{0};
  // Number of distinct cached entries.
  std::atomic<int32_t> numEntries_{0};
  std::atomic<int64_t> hits_{0};
  std::atomic<int64_t> misses_{0};
};

} // namespace facebook::velox::wave
