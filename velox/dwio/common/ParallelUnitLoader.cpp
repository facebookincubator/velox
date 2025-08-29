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

#include "ParallelUnitLoader.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::dwio::common {

class ParallelUnitLoader : public UnitLoader {
 public:
  /// Enables concurrent loading of multiple units (stripes, row groups, etc.)
  /// using asynchronous I/O to improve throughput and reduce read latency.
  ///
  /// **Loading Strategy:**
  /// - Initialization: Preloads up to `maxConcurrentLoads` units concurrently
  /// - Access pattern: On each getLoadedUnit() call, ensures the requested unit
  ///   is loaded and triggers loading of subsequent units within the window
  /// - Memory management: Unloads all previous units to control memory usage
  ///
  /// **Performance Characteristics:**
  /// - Best suited for sequential access patterns
  /// - Memory usage: O(maxConcurrentLoads * average_unit_size)
  /// - I/O parallelism: Up to `maxConcurrentLoads` concurrent load operations
  ///
  /// **Parameters:**
  /// @param units All units to be loaded
  /// @param ioExecutor Thread pool for asynchronous unit loading operations
  /// @param maxConcurrentLoads Maximum units to load concurrently (sliding
  /// window size)
  ///
  /// **Example with maxConcurrentLoads=3:**
  /// ```
  /// Units: [0,1,2,3,4,5,6,7,8,9]
  /// Init:  Load [0,1,2] concurrently
  /// Get(0): Wait for unit 0, trigger load of units [0,1,2], unload none
  /// Get(1): Wait for unit 1, trigger load of units [1,2,3], unload [0]
  /// Get(2): Wait for unit 2, trigger load of units [2,3,4], unload [0,1]
  /// ```
  ParallelUnitLoader(
      std::vector<std::unique_ptr<LoadUnit>> units,
      folly::Executor* ioExecutor,
      uint16_t maxConcurrentLoads)
      : units_(std::move(units)),
        ioExecutor_(ioExecutor),
        maxConcurrentLoads_(maxConcurrentLoads) {
    VELOX_CHECK_NOT_NULL(ioExecutor, "ParallelUnitLoader ioExecutor is null");
    VELOX_CHECK_GT(
        maxConcurrentLoads_,
        0,
        "ParallelUnitLoader maxConcurrentLoads should be larger than 0");
    futures_.resize(units_.size());
    unitsLoaded_.resize(units_.size());

    for (size_t i = 0; i < units_.size() && i < maxConcurrentLoads; ++i) {
      load(i);
    }
  }

  /// Destructor ensures all pending load operations are properly cancelled
  /// and waited for to prevent resource leaks and dangling references.
  ~ParallelUnitLoader() override {
    for (auto& future : futures_) {
      future.cancel();
      future.wait();
    }
  }

  LoadUnit& getLoadedUnit(uint32_t unitIndex) override {
    // Ensure sliding window of units [unitIndex, unitIndex +
    // maxConcurrentLoads_) is loading
    for (size_t i = unitIndex;
         i < units_.size() && i < unitIndex + maxConcurrentLoads_;
         ++i) {
      if (!unitsLoaded_[i]) {
        load(i);
      }
    }

    try {
      futures_[unitIndex].wait();
    } catch (const std::exception& e) {
      VELOX_FAIL("Failed to load unit {}: {}", unitIndex, e.what());
    }

    // Unload the previous units
    folly::via(ioExecutor_, [this, unitIndex]() { unloadUntil(unitIndex); });

    return *units_[unitIndex];
  }

  void onRead(uint32_t unit, uint64_t rowOffsetInUnit, uint64_t /* rowCount */)
      override {
    VELOX_CHECK_LT(unit, units_.size(), "Unit out of range");
    VELOX_CHECK_LT(
        rowOffsetInUnit, units_[unit]->getNumRows(), "Row out of range");
  }

  void onSeek(uint32_t unit, uint64_t rowOffsetInUnit) override {
    VELOX_CHECK_LT(unit, units_.size(), "Unit out of range");
    VELOX_CHECK_LE(
        rowOffsetInUnit, units_[unit]->getNumRows(), "Row out of range");
  }

 private:
  /// Submits the unit's load() to the I/O thread pool
  void load(uint32_t unitIndex) {
    VELOX_CHECK_LT(unitIndex, units_.size(), "Unit index out of bounds");
    VELOX_CHECK_NOT_NULL(ioExecutor_, "ParallelUnitLoader ioExecutor is null");

    futures_[unitIndex] = folly::via(
        ioExecutor_, [this, unitIndex]() { units_[unitIndex]->load(); });
    unitsLoaded_[unitIndex] = true;
  }

  /// Unloads all the units before 'unitIndex'
  void unloadUntil(uint32_t unitIndex) {
    for (size_t i = 0; i < unitIndex; ++i) {
      if (unitsLoaded_[i]) {
        units_[i]->unload();
        // Reset the future
        futures_[i] = folly::Future<folly::Unit>();
        unitsLoaded_[i] = false;
      }
    }
  }

  std::vector<bool> unitsLoaded_;
  std::vector<std::unique_ptr<LoadUnit>> units_;
  std::vector<folly::Future<folly::Unit>> futures_;
  folly::Executor* ioExecutor_;
  size_t maxConcurrentLoads_;
};

std::unique_ptr<UnitLoader> ParallelUnitLoaderFactory::create(
    std::vector<std::unique_ptr<LoadUnit>> units,
    uint64_t /* rowsToSkip */) {
  return std::make_unique<ParallelUnitLoader>(
      std::move(units), ioExecutor_, maxConcurrentLoads_);
}

} // namespace facebook::velox::dwio::common
