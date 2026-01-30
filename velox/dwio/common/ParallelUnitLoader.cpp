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

#include "velox/dwio/common/ParallelUnitLoader.h"
#include <numeric>
#include "velox/common/base/AsyncSource.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/Timer.h"

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
      : loadUnits_(
            std::make_move_iterator(units.begin()),
            std::make_move_iterator(units.end())),
        ioExecutor_(ioExecutor),
        maxConcurrentLoads_(maxConcurrentLoads) {
    VELOX_CHECK_NOT_NULL(ioExecutor, "ParallelUnitLoader ioExecutor is null");
    VELOX_CHECK_GT(
        maxConcurrentLoads_,
        0,
        "ParallelUnitLoader maxConcurrentLoads should be larger than 0");
    asyncSources_.resize(loadUnits_.size());
    unitsLoaded_.resize(loadUnits_.size());
  }

  /// Destructor ensures all pending load operations are properly cancelled
  /// and waited for to prevent resource leaks and dangling references.
  ~ParallelUnitLoader() override {
    for (auto& source : asyncSources_) {
      if (source) {
        source->cancel();
      }
    }
  }

  LoadUnit& getLoadedUnit(uint32_t unit) override {
    VELOX_CHECK_LT(unit, loadUnits_.size(), "Unit out of range");

    processedUnits_.insert(unit);
    // Ensure sliding window of units [unit, unit + maxConcurrentLoads_) is
    // loading
    for (size_t i = unit;
         i < loadUnits_.size() && i < unit + maxConcurrentLoads_;
         ++i) {
      if (!unitsLoaded_[i]) {
        load(i);
      }
    }

    uint64_t unitLoadNanos{0};
    try {
      NanosecondTimer timer{&unitLoadNanos};
      asyncSources_[unit]->move();
    } catch (const std::exception& e) {
      VELOX_FAIL("Failed to load unit {}: {}", unit, e.what());
    }
    waitForUnitReadyNanos_ += unitLoadNanos;

    // Unload the previous units
    unloadUntil(unit);

    return *loadUnits_[unit];
  }

  void onRead(uint32_t unit, uint64_t rowOffsetInUnit, uint64_t /* rowCount */)
      override {
    VELOX_CHECK_LT(unit, loadUnits_.size(), "Unit out of range");
    VELOX_CHECK_LT(
        rowOffsetInUnit, loadUnits_[unit]->getNumRows(), "Row out of range");
  }

  void onSeek(uint32_t unit, uint64_t rowOffsetInUnit) override {
    VELOX_CHECK_LT(unit, loadUnits_.size(), "Unit out of range");
    VELOX_CHECK_LE(
        rowOffsetInUnit, loadUnits_[unit]->getNumRows(), "Row out of range");
  }

  UnitLoaderStats stats() override {
    UnitLoaderStats stats;
    stats.addCounter("processedUnits", RuntimeCounter(processedUnits_.size()));
    stats.addCounter(
        "waitForUnitReadyNanos",
        RuntimeCounter(
            unsignedToSigned(waitForUnitReadyNanos_),
            RuntimeCounter::Unit::kNanos));
    return stats;
  }

 private:
  /// Submits the unit's load() to the I/O thread pool
  void load(uint32_t unitIndex) {
    VELOX_CHECK_LT(unitIndex, loadUnits_.size(), "Unit index out of bounds");
    VELOX_CHECK_NOT_NULL(ioExecutor_, "ParallelUnitLoader ioExecutor is null");
    VELOX_DCHECK(!loadUnits_.empty(), "loadUnits_ should not be empty");

    // Capture shared_ptr by value to prevent use-after-free if
    // ParallelUnitLoader is destroyed while async operation is running
    auto unit = loadUnits_[unitIndex];
    auto asyncSource = std::make_shared<AsyncSource<folly::Unit>>([unit] {
      unit->load();
      return std::make_unique<folly::Unit>();
    });
    asyncSources_[unitIndex] = asyncSource;
    ioExecutor_->add([asyncSource] {
      velox::common::testutil::TestValue::adjust(
          "facebook::velox::dwio::common::ParallelUnitLoader::load",
          asyncSource.get());
      asyncSource->prepare();
    });
    unitsLoaded_[unitIndex] = true;
  }

  /// Unloads all the units before 'unitIndex'
  void unloadUntil(uint32_t unitIndex) {
    for (size_t i = 0; i < unitIndex; ++i) {
      if (unitsLoaded_[i]) {
        loadUnits_[i]->unload();
        unitsLoaded_[i] = false;
      }
    }
  }

  std::vector<bool> unitsLoaded_;
  std::vector<std::shared_ptr<LoadUnit>> loadUnits_;
  std::vector<std::shared_ptr<AsyncSource<folly::Unit>>> asyncSources_;
  folly::Executor* ioExecutor_;
  size_t maxConcurrentLoads_;

  // Stats
  std::unordered_set<uint32_t> processedUnits_;
  uint64_t waitForUnitReadyNanos_{0};
};

std::unique_ptr<UnitLoader> ParallelUnitLoaderFactory::create(
    std::vector<std::unique_ptr<LoadUnit>> loadUnits,
    uint64_t rowsToSkip) {
  const auto totalRows = std::accumulate(
      loadUnits.cbegin(), loadUnits.cend(), 0UL, [](uint64_t sum, auto& unit) {
        return sum + unit->getNumRows();
      });
  VELOX_CHECK_LE(
      rowsToSkip,
      totalRows,
      "Can only skip up to the past-the-end row of the file.");
  return std::make_unique<ParallelUnitLoader>(
      std::move(loadUnits), ioExecutor_, maxConcurrentLoads_);
}

} // namespace facebook::velox::dwio::common
