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
#include <unordered_map>
#include <vector>
#include "velox/experimental/wave/common/Cuda.h"

namespace facebook::velox::wave {

/// Caches compiled CUDA kernels on the filesystem, keyed by source hash.
/// Stores .cu source, .cubin binaries, and .names (mangled entry points)
/// in a directory. On lookup, hashes the source text and compares against
/// cached entries to avoid recompilation across runs.
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
    int32_t ordinal;
  };

  /// Scans the cache directory and populates hashToEntries_ on first call.
  void init();

  /// Returns the path for the .cu source file at the given cache ordinal.
  std::string cuPath(int32_t ordinal) const;
  /// Returns the path for the .cubin binary at the given cache ordinal.
  std::string cubinPath(int32_t ordinal) const;
  /// Returns the path for the .cubin.names sidecar at the given cache ordinal.
  std::string namesPath(int32_t ordinal) const;

  // Directory holding the cached .cu, .cubin, and .cubin.names files.
  std::string cacheDir_;
  // True after the first call to init().
  bool initialized_{false};
  // Protects hashToEntries_, nextOrdinal_, and initialized_.
  std::mutex mutex_;
  // Maps source-text hash to cache entries sharing that hash.
  // Using unordered_map because adding folly::F14FastMap as a dependency
  // causes dependency count regressions across all downstream wave targets.
  std::unordered_map<size_t, std::vector<CacheEntry>> hashToEntries_;
  // Next ordinal for new cache files.
  std::atomic<int32_t> nextOrdinal_{0};
  // Serial number for compile operations.
  std::atomic<int32_t> compileSerial_{0};
  // Number of distinct cached entries.
  std::atomic<int32_t> numEntries_{0};
  std::atomic<int64_t> hits_{0};
  std::atomic<int64_t> misses_{0};
};

} // namespace facebook::velox::wave
