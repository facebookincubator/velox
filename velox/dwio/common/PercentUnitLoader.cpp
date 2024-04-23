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

#include "velox/dwio/common/PercentUnitLoader.h"

#include <cstdint>
#include <future>

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/ExecutorBarrier.h"
#include "velox/dwio/common/MeasureTime.h"
#include "velox/dwio/common/OnDemandUnitLoader.h"
#include "velox/dwio/common/UnitLoaderTools.h"

namespace facebook::velox::dwio::common {

namespace {

inline uint32_t getEntireUnits(uint32_t percent) {
  return percent / 100;
}

inline uint32_t getPercentRemainder(uint32_t percent) {
  return percent % 100;
}

inline bool hasPartialUnit(uint32_t percent) {
  return getPercentRemainder(percent) > 0;
}

inline uint32_t getPartialUnitCount(uint32_t percent) {
  return hasPartialUnit(percent) ? 1 : 0;
}

inline uint32_t getUnitsAheadToLoad(uint32_t percent, uint32_t unitCount) {
  return std::min(
      getEntireUnits(percent) + getPartialUnitCount(percent), unitCount);
}

inline uint64_t getPercentOfRows(
    uint32_t percentRemainder,
    uint64_t totalRows) {
  return ((totalRows * percentRemainder) / 100);
}

inline uint64_t getAtLeastOneRow(uint64_t rows) {
  return std::max(rows, 1UL);
}

inline uint64_t getTransitionRow(uint32_t percent, uint64_t totalRows) {
  VELOX_CHECK_GT(totalRows, 0UL);
  const auto percentRemainder = getPercentRemainder(percent);
  if (percentRemainder == 0) {
    return 0;
  }

  // We need to make sure that the minimal percent of rows is 1 row. Otherwise
  // the transition will be past the end of the rows and we'll never read that
  // row, and never schedule the load, blocking forever.
  return totalRows -
      getAtLeastOneRow(getPercentOfRows(percentRemainder, totalRows));
}

std::vector<uint64_t> getTransitionRows(
    const std::vector<std::unique_ptr<LoadUnit>>& loadUnits,
    uint32_t percent) {
  const auto unitsAheadToLoad = getUnitsAheadToLoad(percent, loadUnits.size());
  const auto transitionUnits = loadUnits.size() - unitsAheadToLoad;
  std::vector<uint64_t> result;
  result.reserve(transitionUnits);
  for (size_t i = 0; i < transitionUnits; ++i) {
    result.push_back(getTransitionRow(percent, loadUnits[i]->getNumRows()));
  }
  return result;
}

inline bool isTransitionRowBeingRead(
    uint64_t transitionRow,
    uint64_t fromRow,
    uint64_t toRow) {
  return fromRow <= transitionRow && transitionRow < toRow;
}

void ensureBudget(
    LoadUnit& unit,
    const std::shared_ptr<velox::dwio::common::Budget>& memoryBudget,
    const std::function<void(std::chrono::high_resolution_clock::duration)>&
        blockedOnBudgetCallback) {
  if (!memoryBudget) {
    return;
  }
  auto elapsed = memoryBudget->waitForBudget(unit.getIoSize());
  if (elapsed.has_value() && blockedOnBudgetCallback) {
    blockedOnBudgetCallback(elapsed.value());
  }
}

void releaseBudget(
    LoadUnit& unit,
    const std::shared_ptr<velox::dwio::common::Budget>& memoryBudget) {
  if (!memoryBudget) {
    return;
  }
  memoryBudget->releaseBudget(unit.getIoSize());
}

// This is more simple
class ZeroPercentUnitLoader : public UnitLoader {
 public:
  ZeroPercentUnitLoader(
      std::vector<std::unique_ptr<LoadUnit>> loadUnits,
      std::shared_ptr<velox::dwio::common::Budget> memoryBudget,
      std::function<void()> noMoreIoCallback,
      std::function<void(std::chrono::high_resolution_clock::duration)>
          blockedOnIoCallback,
      std::function<void(std::chrono::high_resolution_clock::duration)>
          blockedOnBudgetCallback)
      : loadUnits_{std::move(loadUnits)},
        blockedOnIoCallback_{std::move(blockedOnIoCallback)},
        blockedOnBudgetCallback_{std::move(blockedOnBudgetCallback)},
        memoryBudget_{std::move(memoryBudget)},
        noMoreIoCallback_{std::move(noMoreIoCallback)} {}

  LoadUnit& getLoadedUnit(uint32_t unit) override {
    VELOX_CHECK(unit < loadUnits_.size(), "Unit out of range");

    if (loadedUnit_) {
      if (*loadedUnit_ == unit) {
        return *loadUnits_[unit];
      } else {
        loadUnits_[*loadedUnit_]->unload();
        releaseBudget(*loadUnits_[*loadedUnit_], memoryBudget_);
        loadedUnit_.reset();
      }
    }

    ensureBudget(*loadUnits_[unit], memoryBudget_, blockedOnBudgetCallback_);
    {
      const auto measure = measureTimeIfCallback(blockedOnIoCallback_);
      loadUnits_[unit]->load();
    }
    loadedUnit_ = unit;

    return *loadUnits_[unit];
  }

  void onRead(uint32_t unit, uint64_t rowOffsetInUnit, uint64_t rowCount)
      override {
    VELOX_CHECK(unit < loadUnits_.size(), "Unit out of range");
    if (noMoreIoCallback_ && rowOffsetInUnit == 0 && isLastUnit(unit)) {
      noMoreIoCallback_();
    }
  }

 private:
  bool isLastUnit(uint32_t unit) const {
    return unit == (loadUnits_.size() - 1);
  }

  std::vector<std::unique_ptr<LoadUnit>> loadUnits_;
  std::function<void(std::chrono::high_resolution_clock::duration)>
      blockedOnIoCallback_;
  std::function<void(std::chrono::high_resolution_clock::duration)>
      blockedOnBudgetCallback_;
  std::optional<uint32_t> loadedUnit_;
  std::shared_ptr<velox::dwio::common::Budget> memoryBudget_;
  std::function<void()> noMoreIoCallback_;
};

class PercentUnitLoader : public UnitLoader {
 public:
  PercentUnitLoader(
      std::vector<std::unique_ptr<LoadUnit>> loadUnits,
      uint32_t percentAheadToLoad,
      std::shared_ptr<velox::dwio::common::Budget> memoryBudget,
      folly::Executor& firstLoadExecutor,
      folly::Executor& loadExecutor,
      std::function<void()> noMoreIoCallback,
      std::function<void(std::chrono::high_resolution_clock::duration)>
          blockedOnIoCallback,
      std::function<void(std::chrono::high_resolution_clock::duration)>
          blockedOnBudgetCallback)
      : loadUnits_{std::move(loadUnits)},
        isUnitLoaded_(loadUnits_.size()),
        exceptions_{loadUnits_.size()},
        memoryBudget_{std::move(memoryBudget)},
        transitionRows_{getTransitionRows(loadUnits_, percentAheadToLoad)},
        unitsAheadToLoad_{
            getUnitsAheadToLoad(percentAheadToLoad, loadUnits_.size())},
        firstLoadBarrier_{firstLoadExecutor},
        loadBarrier_{loadExecutor},
        noMoreIoCallback_{std::move(noMoreIoCallback)},
        blockedOnIoCallback_{std::move(blockedOnIoCallback)},
        blockedOnBudgetCallback_{std::move(blockedOnBudgetCallback)} {
    VELOX_CHECK_EQ(isUnitLoaded_.size(), loadUnits_.size());
    VELOX_CHECK_GT(
        percentAheadToLoad,
        0,
        "Doesn't work with 0%. Blocks forever. Use ZeroPercentUnitLoader instead.");
    for (uint32_t unit = 0; unit < unitsAheadToLoad_; ++unit) {
      firstLoadBarrier_.add([this, unit]() { this->loadUnit(unit); });
    }
  }

  ~PercentUnitLoader() override {
    waitUntilNoMoreScheduledLoads();
  }

  LoadUnit& getLoadedUnit(uint32_t unit) override {
    VELOX_CHECK(unit < loadUnits_.size(), "Unit out of range");

    unloadPreviousUnit(unit);

    // We'll wait forever if this isn't loaded by another thread.
    waitUntilLoaded(unit);

    return *loadUnits_[unit];
  }

  void onRead(uint32_t unit, uint64_t rowOffsetInUnit, uint64_t rowsToRead)
      override {
    VELOX_CHECK(unit < loadUnits_.size(), "Unit out of range");

    if (isFirstRead(unit, rowOffsetInUnit) && allUnitsLoadedInConstructor()) {
      emitNoMoreIo();
    }

    scheduleLoadIfTransitionRow(unit, rowOffsetInUnit, rowsToRead);
  }

 private:
  void waitUntilNoMoreScheduledLoads() {
    firstLoadBarrier_.waitAll();
    loadBarrier_.waitAll();
  }

  bool allUnitsLoadedInConstructor() {
    return unitsAheadToLoad_ == loadUnits_.size();
  }

  bool isLastUnit(uint32_t unit) {
    return unit == (loadUnits_.size() - 1);
  }

  bool isFirstRead(uint32_t unit, uint64_t rowOffsetInUnit) {
    return unit == 0 && rowOffsetInUnit == 0;
  }

  // Upper layers count on this being called from processing thread (caller of
  // onRead()).
  void emitNoMoreIo() {
    if (noMoreIoCallback_) {
      noMoreIoCallback_();
    }
  }

  void scheduleLoadIfTransitionRow(
      uint32_t unit,
      uint64_t rowOffsetInUnit,
      uint64_t rowsToRead) {
    if (unit < transitionRows_.size()) {
      if (isTransitionRowBeingRead(
              transitionRows_[unit],
              rowOffsetInUnit,
              rowOffsetInUnit + rowsToRead)) {
        const auto unitToLoad = (unit + unitsAheadToLoad_);
        loadBarrier_.add([this, unitToLoad]() { this->loadUnit(unitToLoad); });
        if (isLastUnit(unitToLoad)) {
          emitNoMoreIo();
        }
      }
    }
  }

  void unloadPreviousUnit(uint32_t unit) {
    if (unit == 0 || !isUnitLoaded_[unit - 1]) {
      return;
    }
    const auto unitToUnload = unit - 1;
    loadUnits_[unitToUnload]->unload();
    isUnitLoaded_[unitToUnload] = false;
    releaseBudget(*loadUnits_[unitToUnload], memoryBudget_);
  }

  void throwIfUnitHasException(uint32_t unit) {
    if (exceptions_[unit].has_exception_ptr()) {
      exceptions_[unit].throw_exception();
    }
  }

  void waitUntilLoaded(uint32_t unit) {
    if (isUnitLoaded_[unit]) {
      throwIfUnitHasException(unit);
      return;
    }
    {
      auto measure = measureTimeIfCallback(blockedOnIoCallback_);
      isUnitLoaded_[unit].wait(false);
    }
    throwIfUnitHasException(unit);
  }

  void loadUnit(uint32_t unit) {
    try {
      VELOX_CHECK(unit < loadUnits_.size(), "Unit out of range");
      if (isUnitLoaded_[unit]) {
        return;
      }
      ensureBudget(*loadUnits_[unit], memoryBudget_, blockedOnBudgetCallback_);
      loadUnits_[unit]->load();
      isUnitLoaded_[unit] = true;
      // Notify processing thread that may be waiting for this unit to load.
      isUnitLoaded_[unit].notify_one();
    } catch (...) {
      exceptions_[unit] = folly::exception_wrapper(std::current_exception());
      isUnitLoaded_[unit] = true;
      isUnitLoaded_[unit].notify_one();
    }
  }

  std::vector<std::unique_ptr<LoadUnit>> loadUnits_;
  std::vector<std::atomic_bool> isUnitLoaded_;
  std::vector<folly::exception_wrapper> exceptions_;
  std::shared_ptr<velox::dwio::common::Budget> memoryBudget_;

  std::vector<uint64_t> transitionRows_;
  uint32_t unitsAheadToLoad_;
  velox::dwio::common::ExecutorBarrier firstLoadBarrier_;
  velox::dwio::common::ExecutorBarrier loadBarrier_;
  std::function<void()> noMoreIoCallback_;
  std::function<void(std::chrono::high_resolution_clock::duration)>
      blockedOnIoCallback_;
  std::function<void(std::chrono::high_resolution_clock::duration)>
      blockedOnBudgetCallback_;
};

} // namespace

PercentUnitLoaderFactory::PercentUnitLoaderFactory(
    uint32_t percent,
    std::shared_ptr<velox::dwio::common::Budget> memoryBudget,
    folly::Executor& firstLoadExecutor,
    folly::Executor& loadExecutor,
    std::function<void()> noMoreIoCallback,
    std::function<void(std::chrono::high_resolution_clock::duration)>
        blockedOnIoCallback,
    std::function<void(std::chrono::high_resolution_clock::duration)>
        blockedOnBudgetCallback)
    : percent_{percent},
      memoryBudget_{std::move(memoryBudget)},
      firstLoadExecutor_{firstLoadExecutor},
      loadExecutor_{loadExecutor},
      blockedOnIoCallback_{std::move(blockedOnIoCallback)},
      blockedOnBudgetCallback_{std::move(blockedOnBudgetCallback)},
      noMoreIoCallbacksLeftToTrigger_{std::make_shared<std::atomic_size_t>(0)},
      callbackOnLastSignal_{std::move(noMoreIoCallback)} {}

std::unique_ptr<UnitLoader> PercentUnitLoaderFactory::create(
    std::vector<std::unique_ptr<LoadUnit>> loadUnits) {
  if (percent_ == 0) {
    return std::make_unique<ZeroPercentUnitLoader>(
        std::move(loadUnits),
        memoryBudget_,
        callbackOnLastSignal_.getCallback(),
        blockedOnIoCallback_,
        blockedOnBudgetCallback_);
  }
  return std::make_unique<PercentUnitLoader>(
      std::move(loadUnits),
      percent_,
      memoryBudget_,
      firstLoadExecutor_,
      loadExecutor_,
      callbackOnLastSignal_.getCallback(),
      blockedOnIoCallback_,
      blockedOnBudgetCallback_);
}

} // namespace facebook::velox::dwio::common
