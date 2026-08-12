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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <folly/container/F14Map.h>

#include "velox/common/base/CheckedArithmetic.h"
#include "velox/common/file/Region.h"
#include "velox/dwio/common/RowIntervalSet.h"
#include "velox/dwio/parquet/common/PageIndex.h"

namespace facebook::velox::parquet {

/// Describes one data page in an immutable row-group read plan.
struct PageDataSpan {
  uint32_t dataPageOrdinal{0};
  uint64_t firstRow{0};
  uint64_t numRows{0};
  common::Region region;
};

/// Describes one exact logical stream containing retained data pages.
struct LogicalPageRun {
  common::Region region;
  uint32_t firstDataPageOrdinal{0};
  uint32_t numDataPages{0};
};

/// Describes the bounded physical-read policy used to evaluate page plans.
struct PagePruningCostModelOptions {
  uint64_t maxCoalesceDistance{512ULL << 10};
  uint64_t maxCoalesceBytes{128ULL << 20};
  uint64_t loadQuantum{8ULL << 20};
};

/// Describes which pages and exact byte regions a column reader consumes.
struct ColumnPageReadPlan {
  uint32_t column{0};
  std::optional<common::Region> prefixRegion;
  std::vector<PageDataSpan> dataPages;
  std::vector<LogicalPageRun> retainedRuns;
  std::vector<int32_t> dataPageToRun;
  uint32_t numSkippedPages{0};
  uint64_t fullChunkBytes{0};
  uint64_t retainedPageBytes{0};
  uint64_t plannedPhysicalBytes{0};
  uint32_t plannedPhysicalLoads{0};
  bool allPagesSkipped{false};
  bool useWholeChunkStream{false};
  bool costModelFallback{false};
};

struct PagePruningStats {
  uint64_t indexBytesPlanned{0};
  uint64_t dataBytesPlanned{0};
  uint64_t dataBytesAvoided{0};
  uint32_t pagesRetained{0};
  uint32_t pagesSkipped{0};
  uint32_t logicalRuns{0};
  PageIndexFallbackReason fallbackReason{PageIndexFallbackReason::kNone};
};

/// Holds all row and page decisions for one row group.
///
/// The object is fully populated before it is published to column readers.
/// Readers may consume it independently, but must not mutate it or derive a
/// second page-selection state from it.
struct RowGroupPagePruningPlan {
  uint32_t rowGroup{0};
  dwio::common::RowIntervalSet retainedRows;
  folly::F14FastMap<uint32_t, ColumnPageReadPlan> columns;
  PagePruningStats stats;
  uint64_t filterGeneration{0};
};

using RowGroupPagePruningPlanPtr =
    std::shared_ptr<const RowGroupPagePruningPlan>;

/// Builds exact logical page streams from one validated offset index.
ColumnPageReadPlan buildColumnPageReadPlan(
    uint32_t column,
    const ValidatedOffsetIndex& offsetIndex,
    const dwio::common::RowIntervalSet& retainedRows,
    std::optional<common::Region> prefixRegion,
    uint64_t fullChunkBytes,
    uint64_t indexBytes,
    bool preloaded = false,
    PagePruningCostModelOptions costModel = {});

/// Builds a complete immutable row-group plan from validated page indexes.
RowGroupPagePruningPlanPtr buildRowGroupPagePruningPlan(
    uint32_t rowGroup,
    uint64_t rowGroupRows,
    const dwio::common::RowIntervalSet& retainedRows,
    const folly::F14FastMap<uint32_t, ValidatedPageIndexes>& pageIndexes,
    const folly::F14FastMap<uint32_t, std::optional<common::Region>>&
        prefixRegions,
    const folly::F14FastMap<uint32_t, uint64_t>& fullChunkBytes,
    const folly::F14FastMap<uint32_t, uint64_t>& indexBytes,
    bool preloaded = false,
    uint64_t filterGeneration = 0,
    PagePruningCostModelOptions costModel = {});

} // namespace facebook::velox::parquet
