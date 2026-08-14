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

#include "velox/dwio/parquet/reader/PagePruningPlan.h"

#include <algorithm>

#include <folly/ScopeGuard.h>

namespace {

using facebook::velox::common::Region;
using facebook::velox::parquet::PagePruningCostModelOptions;

struct PhysicalReadEstimate {
  uint64_t bytes{0};
  uint64_t loads{0};
};

PhysicalReadEstimate estimatePhysicalReads(
    const std::vector<Region>& regions,
    const PagePruningCostModelOptions& options) {
  if (regions.empty()) {
    return {};
  }

  const auto maxDistance = options.maxCoalesceDistance;
  const auto maxBytes = std::max<uint64_t>(options.maxCoalesceBytes, 1);
  const auto loadQuantum = std::max<uint64_t>(options.loadQuantum, 1);
  PhysicalReadEstimate estimate;
  uint64_t groupStart{0};
  uint64_t groupEnd{0};

  const auto flush = [&]() {
    estimate.bytes = facebook::velox::checkedPlus(
        estimate.bytes, groupEnd - groupStart, "physical page bytes");
    const auto groupBytes = groupEnd - groupStart;
    const auto numLoads =
        groupBytes / loadQuantum + (groupBytes % loadQuantum == 0 ? 0 : 1);
    estimate.loads = facebook::velox::checkedPlus(
        estimate.loads, numLoads, "physical page loads");
  };

  for (const auto& region : regions) {
    if (region.length == 0) {
      continue;
    }
    const auto regionEnd = facebook::velox::checkedPlus(
        region.offset, region.length, "physical page region end");
    if (groupEnd == 0) {
      groupStart = region.offset;
      groupEnd = regionEnd;
      continue;
    }

    const auto gap = region.offset >= groupEnd ? region.offset - groupEnd : 0;
    const auto nextPhysicalBytes = regionEnd - groupStart;
    if (region.offset < groupEnd ||
        (gap <= maxDistance && nextPhysicalBytes <= maxBytes)) {
      groupEnd = std::max(groupEnd, regionEnd);
      continue;
    }

    flush();
    groupStart = region.offset;
    groupEnd = regionEnd;
  }

  if (groupEnd != 0) {
    flush();
  }
  return estimate;
}

} // namespace

namespace facebook::velox::parquet {

PagePlanMemoryReservation::~PagePlanMemoryReservation() {
  if (bytes_ == 0) {
    return;
  }
  state_->usedBytes.fetch_sub(bytes_, std::memory_order_relaxed);
  state_->pool->reportExternalFree(bytes_);
}

PagePlanMemoryBudget::PagePlanMemoryBudget(
    memory::MemoryPool& pool,
    uint64_t maxBytes)
    : state_(std::make_shared<detail::PagePlanMemoryBudgetState>()) {
  state_->pool = &pool;
  state_->maxBytes = maxBytes;
}

uint64_t PagePlanMemoryBudget::usedBytes() const {
  return state_->usedBytes.load(std::memory_order_relaxed);
}

bool PagePlanMemoryBudget::canReserve(uint64_t bytes) const {
  return bytes <= state_->maxBytes && usedBytes() <= state_->maxBytes - bytes;
}

std::shared_ptr<PagePlanMemoryReservation> PagePlanMemoryBudget::tryReserve(
    uint64_t bytes) {
  auto current = state_->usedBytes.load(std::memory_order_relaxed);
  for (;;) {
    if (bytes > state_->maxBytes || current > state_->maxBytes - bytes) {
      return nullptr;
    }
    if (state_->usedBytes.compare_exchange_weak(
            current,
            current + bytes,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      bool poolAllocationReported{false};
      auto rollback = folly::makeGuard([&] {
        if (bytes > 0) {
          state_->usedBytes.fetch_sub(bytes, std::memory_order_relaxed);
          if (poolAllocationReported) {
            state_->pool->reportExternalFree(bytes);
          }
        }
      });
      if (bytes > 0) {
        state_->pool->reportExternalAllocation(bytes);
        poolAllocationReported = true;
      }
      auto reservation =
          std::make_shared<PagePlanMemoryReservation>(state_, bytes);
      rollback.dismiss();
      return reservation;
    }
  }
}

ColumnPageReadPlan buildColumnPageReadPlan(
    uint32_t column,
    const ValidatedOffsetIndex& offsetIndex,
    const dwio::common::RowIntervalSet& retainedRows,
    std::optional<common::Region> prefixRegion,
    uint64_t fullChunkBytes,
    uint64_t indexBytes,
    bool preloaded,
    PagePruningCostModelOptions costModel) {
  ColumnPageReadPlan result;
  result.column = column;
  result.prefixRegion = prefixRegion;
  result.fullChunkBytes = fullChunkBytes;
  result.dataPageToRun.assign(offsetIndex.pages.size(), -1);

  for (uint32_t ordinal = 0; ordinal < offsetIndex.pages.size(); ++ordinal) {
    const auto& page = offsetIndex.pages[ordinal];
    const auto pageEnd =
        checkedPlus(page.firstRow, page.numRows, "page row end");
    result.dataPages.push_back({
        page.firstRow,
        page.numRows,
        common::Region{page.offset, page.compressedSize},
    });
    if (!retainedRows.overlaps({page.firstRow, pageEnd})) {
      ++result.numSkippedPages;
      continue;
    }

    result.retainedPageBytes = checkedPlus(
        result.retainedPageBytes,
        static_cast<uint64_t>(page.compressedSize),
        "retained page bytes");
    if (result.retainedRuns.empty() ||
        checkedPlus(
            result.retainedRuns.back().region.offset,
            result.retainedRuns.back().region.length,
            "logical page run end") != page.offset) {
      result.retainedRuns.push_back(
          {common::Region{page.offset, page.compressedSize}, ordinal, 1});
    } else {
      auto& run = result.retainedRuns.back();
      run.region.length = checkedPlus(
          run.region.length,
          static_cast<uint64_t>(page.compressedSize),
          "logical page run length");
      run.numDataPages = checkedPlus(
          run.numDataPages, uint32_t{1}, "logical page run page count");
    }
    result.dataPageToRun[ordinal] =
        static_cast<int32_t>(result.retainedRuns.size() - 1);
  }

  const bool allPagesSkipped = !result.dataPages.empty() &&
      result.numSkippedPages == result.dataPages.size();
  std::vector<common::Region> physicalRegions;
  physicalRegions.reserve(
      result.retainedRuns.size() + (prefixRegion.has_value() ? 1 : 0));
  if (prefixRegion.has_value()) {
    physicalRegions.push_back(*prefixRegion);
  }
  for (const auto& run : result.retainedRuns) {
    physicalRegions.push_back(run.region);
  }
  const auto estimate = estimatePhysicalReads(physicalRegions, costModel);
  result.plannedPhysicalBytes = estimate.bytes;
  result.plannedPhysicalLoads = estimate.loads;
  const auto loadQuantum = std::max<uint64_t>(costModel.loadQuantum, 1);
  const auto wholeChunkLoads = fullChunkBytes == 0
      ? uint64_t{0}
      : fullChunkBytes / loadQuantum +
          (fullChunkBytes % loadQuantum == 0 ? 0 : 1);
  const bool noMaterialSavings = !preloaded && fullChunkBytes != 0 &&
      (estimate.bytes >= fullChunkBytes || estimate.loads >= wholeChunkLoads);
  result.allPagesSkipped = allPagesSkipped;
  result.useWholeChunkStream =
      result.numSkippedPages == 0 || (!allPagesSkipped && noMaterialSavings);
  result.costModelFallback =
      !allPagesSkipped && result.numSkippedPages != 0 && noMaterialSavings;
  if (result.useWholeChunkStream && fullChunkBytes != 0) {
    result.plannedPhysicalBytes = fullChunkBytes;
    result.plannedPhysicalLoads = fullChunkBytes / loadQuantum +
        (fullChunkBytes % loadQuantum == 0 ? 0 : 1);
  }
  return result;
}

RowGroupPagePruningPlanPtr buildRowGroupPagePruningPlan(
    uint32_t rowGroup,
    uint64_t rowGroupRows,
    const dwio::common::RowIntervalSet& retainedRows,
    const folly::F14FastMap<uint32_t, ValidatedPageIndexes>& pageIndexes,
    const folly::F14FastMap<uint32_t, std::optional<common::Region>>&
        prefixRegions,
    const folly::F14FastMap<uint32_t, uint64_t>& fullChunkBytes,
    const folly::F14FastMap<uint32_t, uint64_t>& indexBytes,
    bool preloaded,
    uint64_t filterGeneration,
    PagePruningCostModelOptions costModel,
    uint64_t decodedPageIndexBytes,
    std::shared_ptr<PagePlanMemoryBudget> memoryBudget) {
  auto result = std::make_shared<RowGroupPagePruningPlan>();
  result->rowGroup = rowGroup;
  result->decodedPageIndexBytes = decodedPageIndexBytes;
  if (memoryBudget) {
    result->decodedPageIndexMemory =
        memoryBudget->tryReserve(decodedPageIndexBytes);
  }
  result->retainedRows = retainedRows;
  result->filterGeneration = filterGeneration;

  for (const auto& [column, indexes] : pageIndexes) {
    const auto prefix = prefixRegions.find(column);
    const auto fullBytes = fullChunkBytes.find(column);
    const auto index = indexBytes.find(column);
    const auto fullChunk =
        fullBytes == fullChunkBytes.end() ? uint64_t{0} : fullBytes->second;
    const auto indexSize =
        index == indexBytes.end() ? uint64_t{0} : index->second;
    auto plan = buildColumnPageReadPlan(
        column,
        indexes.offset,
        retainedRows,
        prefix == prefixRegions.end() ? std::nullopt : prefix->second,
        fullChunk,
        indexSize,
        preloaded,
        costModel);
    result->stats.indexBytesPlanned = checkedPlus(
        result->stats.indexBytesPlanned, indexSize, "page-index bytes");
    result->stats.dataBytesPlanned = checkedPlus(
        result->stats.dataBytesPlanned,
        plan.plannedPhysicalBytes,
        "planned page bytes");
    result->stats.dataBytesAvoided = checkedPlus(
        result->stats.dataBytesAvoided,
        !preloaded && fullChunk > plan.plannedPhysicalBytes
            ? fullChunk - plan.plannedPhysicalBytes
            : 0,
        "avoided page bytes");
    result->stats.pagesRetained = checkedPlus(
        result->stats.pagesRetained,
        static_cast<uint32_t>(plan.dataPages.size() - plan.numSkippedPages),
        "retained page count");
    result->stats.pagesSkipped = checkedPlus(
        result->stats.pagesSkipped, plan.numSkippedPages, "skipped page count");
    result->stats.logicalRuns = checkedPlus(
        result->stats.logicalRuns,
        static_cast<uint32_t>(plan.retainedRuns.size()),
        "logical run count");
    if (plan.costModelFallback) {
      result->stats.fallbackReason = PageIndexFallbackReason::kCostModel;
    }
    result->columns.emplace(column, std::move(plan));
  }

  const auto invalidRows = dwio::common::RowIntervalSet::difference(
      retainedRows, dwio::common::RowIntervalSet::full(rowGroupRows));
  VELOX_CHECK(
      invalidRows.intervals().empty(),
      "Retained rows exceed row-group bounds: {}",
      invalidRows.toString());
  return result;
}

} // namespace facebook::velox::parquet
