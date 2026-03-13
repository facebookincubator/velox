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

#include "velox/connectors/hive/BucketSortingWriter.h"

#include "velox/connectors/hive/HiveWriterTypes.h"
#include "velox/dwio/common/SortingWriter.h"
#include "velox/exec/SortBuffer.h"

namespace facebook::velox::connector::hive {

BucketSortingWriter::BucketSortingWriter(
    RowTypePtr dataType,
    std::vector<column_index_t> sortColumnIndices,
    std::vector<CompareFlags> sortCompareFlags,
    uint64_t finishTimeSliceLimitMs,
    uint64_t maxOutputRows,
    uint64_t maxOutputBytes,
    common::PrefixSortConfig prefixSortConfig,
    const common::SpillConfig* spillConfig)
    : dataType_(std::move(dataType)),
      sortColumnIndices_(std::move(sortColumnIndices)),
      sortCompareFlags_(std::move(sortCompareFlags)),
      finishTimeSliceLimitMs_(finishTimeSliceLimitMs),
      maxOutputRows_(maxOutputRows),
      maxOutputBytes_(maxOutputBytes),
      prefixSortConfig_(prefixSortConfig),
      spillConfig_(spillConfig) {}

std::unique_ptr<dwio::common::Writer> BucketSortingWriter::wrap(
    HiveWriterInfo* writerInfo,
    std::unique_ptr<dwio::common::Writer> writer) {
  VELOX_CHECK_NOT_NULL(writerInfo);
  if (!enabled()) {
    return writer;
  }

  auto* sortPool = writerInfo->sortPool.get();
  VELOX_CHECK_NOT_NULL(sortPool);
  auto sortBuffer = std::make_unique<exec::SortBuffer>(
      dataType_,
      sortColumnIndices_,
      sortCompareFlags_,
      sortPool,
      writerInfo->nonReclaimableSectionHolder.get(),
      prefixSortConfig_,
      spillConfig_,
      writerInfo->spillStats.get());

  return std::make_unique<dwio::common::SortingWriter>(
      std::move(writer),
      std::move(sortBuffer),
      maxOutputRows_,
      maxOutputBytes_,
      finishTimeSliceLimitMs_);
}

} // namespace facebook::velox::connector::hive
