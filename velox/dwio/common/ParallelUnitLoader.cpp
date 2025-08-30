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
  /// ParallelUnitLoader enables concurrent loading of multiple units (e.g.,
  /// stripes, row groups) to improve I/O throughput and reduce read latency.
  ///
  /// **Loading Strategy:**
  /// - On initialization: Starts loading up to 'maxConcurrentLoads' units
  /// concurrently
  /// - On consumption: As each unit is accessed, triggers loading of the next
  /// unit
  /// - Memory management: Unloads units before the current accessed unit to limit
  /// memory usage
  ///
  /// **Parameters:**
  /// @param units Vector of all units that need to be loaded
  /// @param ioExecutor Executor for running unit loading operations
  /// asynchronously
  /// @param maxConcurrentLoads Number of units to keep in the loading/loaded
  /// state simultaneously
  ///
  /// **Example:**
  /// With maxConcurrentLoads=3 and 10 total units:
  /// - Initially loads units [0,1,2]
  /// - When unit 0 is consumed, starts loading unit 3 and unloads unit 0
  /// - When unit 1 is consumed, starts loading unit 4 and unloads unit 1
  /// - And so on...
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

    for (size_t i = 0; i < units_.size() && i < maxConcurrentLoads; ++i) {
      load(i);
    }
  }

  ~ParallelUnitLoader() override {
    for (auto& future : futures_) {
      future.cancel();
      future.wait();
    }
  }

  LoadUnit& getLoadedUnit(uint32_t unitIndex) override {
    // Unload the previous units
    unloadUntil(unitIndex);

    // Triggers loading for units in [unitIndex, unitIndex + maxConcurrentLoads_]
    // that haven't been loaded
    for (size_t i = 0; i < maxConcurrentLoads_; ++i) {
      if (!futures_[unitIndex].valid()) {
        load(unitIndex + i);
      }
    }

    try {
      futures_[unitIndex].wait();
    } catch (const std::exception& e) {
      VELOX_FAIL("Failed to load unit {}: {}", unitIndex, e.what());
    }
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
    VELOX_CHECK_LT(
        unitIndex, units_.size(), "ParallelUnitLoader prefetch invalid index");
    VELOX_CHECK_NOT_NULL(ioExecutor_, "ParallelUnitLoader ioExecutor is null");

    futures_[unitIndex] = folly::via(
        ioExecutor_, [this, unitIndex]() { units_[unitIndex]->load(); });
  }

  /// Unloads all the units before 'unitIndex'
  /// TODO: have a smarter unload algorithm
  void unloadUntil(uint32_t unitIndex) {
    for (int i = 0; i < units_.size(); ++i) {
      units_[i]->unload();
      // Reset the future
      futures_[i] = folly::Future<folly::Unit>();
    }
  }

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
