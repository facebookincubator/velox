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

#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <atomic>
#include <numeric>
#include <optional>
#include "velox/common/base/AsyncSource.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/Timer.h"

namespace facebook::velox::dwio::common {

// Bounds speculative loads and their retained encoded bytes across readers.
// Foreground units are excluded so admission never blocks reader progress.
class BoundedPrefetchBudget {
 public:
  BoundedPrefetchBudget(size_t maxLoads, uint64_t maxBytes);

  // Reserves a load slot and bytes without waiting for other readers.
  bool tryReserve(uint64_t bytes);

  // Releases a slot when a load completes or is cancelled before starting.
  void finishLoad();

  // Releases bytes when the unit is consumed or discarded.
  void releaseBytes(uint64_t bytes);

 private:
  struct State {
    size_t loads{0};
    uint64_t bytes{0};
  };

  const size_t maxLoads_;
  const uint64_t maxBytes_;
  folly::Synchronized<State> state_;
};

BoundedPrefetchBudget::BoundedPrefetchBudget(size_t maxLoads, uint64_t maxBytes)
    : maxLoads_(maxLoads), maxBytes_(maxBytes) {
  VELOX_CHECK_GT(maxLoads_, 0);
  VELOX_CHECK_GT(maxBytes_, 0);
}

bool BoundedPrefetchBudget::tryReserve(uint64_t bytes) {
  auto state = state_.wlock();
  if (state->loads >= maxLoads_ || bytes > maxBytes_ - state->bytes) {
    return false;
  }
  ++state->loads;
  state->bytes += bytes;
  return true;
}

void BoundedPrefetchBudget::finishLoad() {
  auto state = state_.wlock();
  VELOX_CHECK_GT(state->loads, 0);
  --state->loads;
}

void BoundedPrefetchBudget::releaseBytes(uint64_t bytes) {
  auto state = state_.wlock();
  VELOX_CHECK_GE(state->bytes, bytes);
  state->bytes -= bytes;
}

namespace {

// Retain admission until preparation finishes and the reader consumes the unit.
class PrefetchReservation {
 public:
  PrefetchReservation(
      std::shared_ptr<BoundedPrefetchBudget> budget,
      ParallelUnitLoaderOptions options,
      uint64_t bytes)
      : budget_(std::move(budget)),
        options_(std::move(options)),
        bytes_(bytes) {
    report(UnitLoader::Event::kBytesReserved, bytes_);
  }

  PrefetchReservation(const PrefetchReservation&) = delete;
  PrefetchReservation& operator=(const PrefetchReservation&) = delete;
  PrefetchReservation(PrefetchReservation&&) = delete;
  PrefetchReservation& operator=(PrefetchReservation&&) = delete;

  ~PrefetchReservation() {
    finishLoad();
    budget_->releaseBytes(bytes_);
    report(UnitLoader::Event::kBytesReleased, bytes_);
  }

  void finishLoad() {
    if (loading_) {
      budget_->finishLoad();
      loading_ = false;
    }
  }

 private:
  void report(UnitLoader::Event event, uint64_t value) const {
    if (options_.metricCallback) {
      options_.metricCallback(event, value);
    }
  }

  const std::shared_ptr<BoundedPrefetchBudget> budget_;
  const ParallelUnitLoaderOptions options_;
  const uint64_t bytes_;
  bool loading_{true};
};

class ParallelUnitLoader : public UnitLoader {
 public:
  /// Enables concurrent loading of multiple units (stripes, row groups, etc.)
  /// using asynchronous I/O to improve throughput and reduce read latency.
  ///
  /// **Loading Strategy:**
  /// - Initialization: No units are loaded until the first getLoadedUnit() call
  /// - Access pattern: On each getLoadedUnit() call, ensures the requested unit
  ///   is loaded and triggers loading of subsequent units within the window
  /// - Memory management: Unloads units outside the current window
  /// - Admission: Optional shared limits bound speculative loads; required
  ///   foreground loads bypass these limits
  ///
  /// **Performance Characteristics:**
  /// - Best suited for sequential access patterns
  /// - Retained unit resources: O(maxConcurrentLoads * average_unit_size)
  /// - I/O parallelism: Up to `maxConcurrentLoads` loads within each window,
  ///   subject to shared admission limits for speculative loads
  ///
  /// **Parameters:**
  /// @param units All units to be loaded
  /// @param ioExecutor Thread pool for asynchronous unit loading operations
  /// @param maxConcurrentLoads Maximum units in the sliding window
  /// @param prefetchBudget Optional admission budget shared across loaders
  /// @param options Metrics and blocked-on-I/O callbacks
  ///
  /// **Example with maxConcurrentLoads=3 and no admission limits:**
  /// ```
  /// Units: [0,1,2,3,4,5,6,7,8,9]
  /// Init:  No loads
  /// Get(0): Load [0,1,2], wait for unit 0, unload none
  /// Get(1): Retain [1,2], load [3], wait for unit 1, unload [0]
  /// Get(2): Retain [2,3], load [4], wait for unit 2, unload [1]
  /// ```
  ParallelUnitLoader(
      std::vector<std::unique_ptr<LoadUnit>> units,
      folly::Executor* ioExecutor,
      size_t maxConcurrentLoads,
      std::shared_ptr<BoundedPrefetchBudget> prefetchBudget,
      ParallelUnitLoaderOptions options)
      : maxConcurrentLoads_(maxConcurrentLoads),
        prefetchBudget_(std::move(prefetchBudget)),
        options_(std::move(options)),
        ioExecutor_(ioExecutor),
        loadUnits_(
            std::make_move_iterator(units.begin()),
            std::make_move_iterator(units.end())),
        pendingUnits_(loadUnits_.size()) {
    VELOX_CHECK_NOT_NULL(ioExecutor_);
    VELOX_CHECK_GT(maxConcurrentLoads_, 0);
  }

  ParallelUnitLoader(const ParallelUnitLoader&) = delete;
  ParallelUnitLoader& operator=(const ParallelUnitLoader&) = delete;
  ParallelUnitLoader(ParallelUnitLoader&&) = delete;
  ParallelUnitLoader& operator=(ParallelUnitLoader&&) = delete;

  /// Cancels queued load operations and waits for active loads before releasing
  /// unit resources and reader-owned dependencies.
  ~ParallelUnitLoader() override {
    for (size_t index = windowBegin_; index < windowEnd_; ++index) {
      if (pendingUnits_[index]) {
        pendingUnits_[index]->source->cancel();
      }
    }
    for (size_t index = windowBegin_; index < windowEnd_; ++index) {
      unload(index);
    }
  }

  LoadUnit& getLoadedUnit(uint32_t unit) override {
    VELOX_CHECK_LT(unit, loadUnits_.size(), "Unit out of range");
    VELOX_DCHECK_LT(unit, pendingUnits_.size());
    processedUnits_.insert(unit);
    // Ensure the sliding window [unit, unit + maxConcurrentLoads_) is loading.
    updateWindow(unit);
    if (!pendingUnits_[unit]) {
      load(unit, /*prefetch=*/false);
    }
    prefetchAfter(unit);

    auto& pending = *pendingUnits_[unit];
    if (pending.consumed) {
      return *loadUnits_[unit];
    }
    const bool ready = pending.source->hasValue();
    if (pending.prefetched) {
      report(
          ready ? UnitLoader::Event::kPrefetchReady
                : UnitLoader::Event::kPrefetchNotReady);
    }
    uint64_t unitLoadNanos{0};
    {
      NanosecondWallTimer timer{&unitLoadNanos};
      pending.source->move();
    }
    pending.consumed = true;
    waitForUnitReadyNanos_ += unitLoadNanos;
    report(UnitLoader::Event::kWaitNanos, unitLoadNanos);
    if (!ready && options_.blockedOnIoCallback) {
      options_.blockedOnIoCallback(
          std::chrono::duration_cast<
              std::chrono::high_resolution_clock::duration>(
              std::chrono::nanoseconds(unitLoadNanos)));
    }
    pending.reservation.reset();
    pending.prefetched = false;
    prefetchAfter(unit);
    return *loadUnits_[unit];
  }

  void onRead(uint32_t unit, uint64_t rowOffsetInUnit, uint64_t /* rowCount */)
      override {
    VELOX_CHECK_LT(unit, loadUnits_.size(), "Unit out of range");
    VELOX_CHECK_LT(
        rowOffsetInUnit, loadUnits_[unit]->getNumRows(), "Row out of range");
    prefetchAfter(unit);
  }

  void onSeek(uint32_t unit, uint64_t rowOffsetInUnit) override {
    VELOX_CHECK_LT(unit, loadUnits_.size(), "Unit out of range");
    VELOX_CHECK_LE(
        rowOffsetInUnit, loadUnits_[unit]->getNumRows(), "Row out of range");
  }

  UnitLoaderStats stats() override {
    UnitLoaderStats stats;
    stats.addCounter(
        "processedUnits", RuntimeCounter(saturateCast(processedUnits_.size())));
    stats.addCounter(
        "waitForUnitReadyNanos",
        RuntimeCounter(
            saturateCast(waitForUnitReadyNanos_),
            RuntimeCounter::Unit::kNanos));
    return stats;
  }

 private:
  // Tracks a scheduled unit load, its budget reservation, and
  // whether the reader has consumed the load result.
  struct PendingUnit {
    std::shared_ptr<AsyncSource<folly::Unit>> source;
    std::shared_ptr<PrefetchReservation> reservation;
    std::shared_ptr<std::atomic<bool>> needsUnload;
    bool prefetched;
    bool consumed{false};
  };

  void report(UnitLoader::Event event, uint64_t value = 1) const {
    report(options_, event, value);
  }

  // Async tasks own their options without retaining the loader.
  static void report(
      const ParallelUnitLoaderOptions& options,
      UnitLoader::Event event,
      uint64_t value = 1) {
    if (options.metricCallback) {
      options.metricCallback(event, value);
    }
  }

  // Submits the unit's load() to the I/O thread pool, subject to shared
  // admission limits for speculative loads.
  bool load(size_t unitIndex, bool prefetch) {
    VELOX_DCHECK_LT(unitIndex, loadUnits_.size());
    VELOX_DCHECK_LT(unitIndex, pendingUnits_.size());
    auto needsUnload = std::make_shared<std::atomic<bool>>(false);
    SCOPE_EXIT {
      if (!pendingUnits_[unitIndex] && needsUnload->exchange(false)) {
        loadUnits_[unitIndex]->unload();
      }
    };
    std::shared_ptr<PrefetchReservation> reservation;
    if (prefetch && prefetchBudget_) {
      needsUnload->store(true);
      const auto bytes = loadUnits_[unitIndex]->getIoSize();
      if (!prefetchBudget_->tryReserve(bytes)) {
        report(UnitLoader::Event::kPrefetchRejected);
        return false;
      }
      SCOPE_EXIT {
        if (!reservation) {
          prefetchBudget_->finishLoad();
          prefetchBudget_->releaseBytes(bytes);
        }
      };
      reservation = std::make_shared<PrefetchReservation>(
          prefetchBudget_, options_, bytes);
    }
    // Capture the unit by shared_ptr so the asynchronous operation owns it.
    auto source = std::make_shared<AsyncSource<folly::Unit>>(
        [unit = loadUnits_[unitIndex],
         needsUnload,
         reservation,
         options = options_]() mutable {
          uint64_t loadNs{0};
          SCOPE_EXIT {
            report(options, UnitLoader::Event::kLoadWallNanos, loadNs);
            if (reservation) {
              reservation->finishLoad();
              // Drop the task's reference before AsyncSource notifies waiters.
              reservation.reset();
            }
          };
          NanosecondWallTimer timer{&loadNs};
          report(options, UnitLoader::Event::kLoadStarted);
          try {
            needsUnload->store(true);
            unit->load();
            report(options, UnitLoader::Event::kLoadFinished);
          } catch (...) {
            report(options, UnitLoader::Event::kLoadFailed);
            throw;
          }
          return std::make_unique<folly::Unit>();
        });
    pendingUnits_[unitIndex] =
        PendingUnit{source, reservation, needsUnload, prefetch};
    report(
        prefetch ? UnitLoader::Event::kPrefetchScheduled
                 : UnitLoader::Event::kForegroundLoad);
    try {
      ioExecutor_->add([source] {
        velox::common::testutil::TestValue::adjust(
            "facebook::velox::dwio::common::ParallelUnitLoader::load",
            source.get());
        source->prepare();
      });
    } catch (...) {
      unload(unitIndex);
      if (prefetch) {
        report(UnitLoader::Event::kPrefetchError);
        return false;
      }
      throw;
    }
    return true;
  }

  void prefetchAfter(uint32_t unit) {
    VELOX_DCHECK_LT(unit, loadUnits_.size());
    VELOX_DCHECK_LT(unit, pendingUnits_.size());
    const auto numUnits =
        std::min(maxConcurrentLoads_, loadUnits_.size() - unit);
    for (size_t distance = 1; distance < numUnits; ++distance) {
      const auto nextUnit = unit + distance;
      if (!pendingUnits_[nextUnit] && !load(nextUnit, /*prefetch=*/true)) {
        break;
      }
    }
  }

  // Unloads units outside the sliding window, including units before unitIndex.
  void updateWindow(size_t unitIndex) {
    const auto windowEnd = unitIndex +
        std::min(maxConcurrentLoads_, loadUnits_.size() - unitIndex);
    for (size_t index = windowBegin_; index < std::min(unitIndex, windowEnd_);
         ++index) {
      unload(index);
    }
    for (size_t index = std::max(windowEnd, windowBegin_); index < windowEnd_;
         ++index) {
      unload(index);
    }
    windowBegin_ = unitIndex;
    windowEnd_ = windowEnd;
  }

  void unload(size_t index) {
    velox::common::testutil::TestValue::adjust(
        "facebook::velox::dwio::common::ParallelUnitLoader::unload", &index);
    VELOX_DCHECK_LT(index, pendingUnits_.size());
    VELOX_DCHECK_LT(index, loadUnits_.size());
    if (pendingUnits_[index]) {
      // LoadUnit may reference file metadata owned by the enclosing reader.
      pendingUnits_[index]->source->close();
      if (pendingUnits_[index]->needsUnload->exchange(false)) {
        loadUnits_[index]->unload();
      }
      pendingUnits_[index].reset();
    }
  }

  const size_t maxConcurrentLoads_;
  const std::shared_ptr<BoundedPrefetchBudget> prefetchBudget_;
  const ParallelUnitLoaderOptions options_;
  folly::Executor* const ioExecutor_;
  std::vector<std::shared_ptr<LoadUnit>> loadUnits_;
  std::vector<std::optional<PendingUnit>> pendingUnits_;
  // Bound the indices that may hold pending loads, including rejected holes.
  size_t windowBegin_{0};
  size_t windowEnd_{0};
  // Stats
  std::unordered_set<uint32_t> processedUnits_;
  uint64_t waitForUnitReadyNanos_{0};
};

} // namespace

ParallelUnitLoaderFactory::ParallelUnitLoaderFactory(
    folly::Executor* ioExecutor,
    size_t maxConcurrentLoads,
    ParallelUnitLoaderOptions options)
    : ioExecutor_(ioExecutor),
      maxConcurrentLoads_(maxConcurrentLoads),
      options_(std::move(options)) {
  VELOX_CHECK_EQ(
      options_.maxPrefetchLoads == 0,
      options_.maxPrefetchBytes == 0,
      "Set both prefetch limits to zero or both to positive values");
  if (options_.maxPrefetchLoads != 0) {
    prefetchBudget_ = std::make_shared<BoundedPrefetchBudget>(
        options_.maxPrefetchLoads, options_.maxPrefetchBytes);
  }
}

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
      std::move(loadUnits),
      ioExecutor_,
      maxConcurrentLoads_,
      prefetchBudget_,
      options_);
}

} // namespace facebook::velox::dwio::common
