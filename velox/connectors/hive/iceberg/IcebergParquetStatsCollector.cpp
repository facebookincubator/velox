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

#include "velox/connectors/hive/iceberg/IcebergParquetStatsCollector.h"

#include "velox/common/Casts.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergDataFileStatistics.h"
#include "velox/dwio/common/FileMetadata.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/dwio/parquet/writer/arrow/Metadata.h"
#include "velox/dwio/parquet/writer/arrow/Statistics.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

void addAllRecursive(
    const parquet::ParquetFieldId& field,
    const TypePtr& type,
    std::unordered_set<int32_t>& fieldIds) {
  fieldIds.insert(field.fieldId);

  VELOX_CHECK_EQ(field.children.size(), type->size());
  for (auto i = 0; i < type->size(); ++i) {
    addAllRecursive(field.children[i], type->childAt(i), fieldIds);
  }
}

// Recursively collects field IDs that should skip bounds collection.
// Repeated fields (e.g. MAP and ARRAY) are not currently supported by Iceberg.
// These fields, along with all their descendants, should skip bounds
// collection.
// @param field The Parquet field ID structure to process.
// @param type The Velox type corresponding to this field.
// @param fieldIds Output set to populate with field IDs to skip.
void collectSkipBoundsFieldIds(
    const parquet::ParquetFieldId& field,
    const TypePtr& type,
    std::unordered_set<int32_t>& fieldIds) {
  VELOX_CHECK_NOT_NULL(type, "Input column type cannot be null.");

  if (type->isMap() || type->isArray()) {
    addAllRecursive(field, type, fieldIds);
    return;
  }

  VELOX_CHECK_EQ(field.children.size(), type->size());
  for (auto i = 0; i < type->size(); ++i) {
    collectSkipBoundsFieldIds(field.children[i], type->childAt(i), fieldIds);
  }
}

} // namespace

IcebergParquetStatsCollector::IcebergParquetStatsCollector(
    const std::vector<IcebergColumnHandlePtr>& inputColumns) {
  parquetFieldIds_.children.reserve(inputColumns.size());
  for (const auto& columnHandle : inputColumns) {
    parquetFieldIds_.children.emplace_back(columnHandle->field());
    collectSkipBoundsFieldIds(
        columnHandle->field(), columnHandle->dataType(), skipBoundsFieldIds_);
  }
}

IcebergDataFileStatisticsPtr IcebergParquetStatsCollector::aggregate(
    std::unique_ptr<dwio::common::FileMetadata> fileMetadata) {
  // Empty data file.
  if (!fileMetadata) {
    return std::make_shared<IcebergDataFileStatistics>(
        IcebergDataFileStatistics::empty());
  }

  auto parquetMetadata =
      checkedPointerCast<parquet::ParquetFileMetadata>(std::move(fileMetadata));
  auto metadata = parquetMetadata->arrowMetadata();
  auto dataFileStats = std::make_shared<IcebergDataFileStatistics>();
  dataFileStats->numRecords = metadata->numRows();
  const auto numRowGroups = metadata->numRowGroups();

  // Track global min/max statistics for each column across all row groups.
  // Key: Iceberg field ID.
  // Value: A pair of Statistics objects where:
  // - first: The statistics from the row group containing the global minimum
  // value.
  // - second: The statistics from the row group containing the global maximum
  // value. Two separate objects are stored because the global minimum and
  // global maximum for a single column may originate from different row groups.
  folly::F14FastMap<
      int32_t,
      std::pair<
          std::shared_ptr<parquet::arrow::Statistics>,
          std::shared_ptr<parquet::arrow::Statistics>>>
      globalMinMaxStats;

  std::unordered_set<int32_t> fieldIds;
  for (auto i = 0; i < numRowGroups; ++i) {
    const auto& rowGroup = metadata->rowGroup(i);

    for (auto j = 0; j < rowGroup->numColumns(); ++j) {
      const auto& columnChunk = rowGroup->columnChunk(j);
      const auto fieldId = columnChunk->fieldId();
      fieldIds.insert(fieldId);

      auto& stats = dataFileStats->columnStats[fieldId];
      stats.valueCount += columnChunk->numValues();
      stats.columnSize += columnChunk->totalCompressedSize();

      const auto& columnChunkStats = columnChunk->statistics();
      if (columnChunkStats) {
        stats.nullValueCount += columnChunkStats->nullCount();

        if (columnChunkStats->hasMinMax() && shouldStoreBounds(fieldId)) {
          auto [it, inserted] = globalMinMaxStats.emplace(
              fieldId, std::pair{columnChunkStats, columnChunkStats});

          if (!inserted) {
            auto& [minStats, maxStats] = it->second;

            if (columnChunkStats->maxGreaterThan(*maxStats)) {
              maxStats = columnChunkStats;
            }
            if (columnChunkStats->minLessThan(*minStats)) {
              minStats = columnChunkStats;
            }
          }
        }
      }
    }
  }

  for (const auto fieldId : fieldIds) {
    const auto& [nanCount, hasNanCount] = metadata->getNaNCount(fieldId);
    if (hasNanCount) {
      dataFileStats->columnStats[fieldId].nanValueCount = nanCount;
    }
  }

  for (const auto& [fieldId, stats] : globalMinMaxStats) {
    const auto& [minStats, maxStats] = stats;

    auto& columnStats = dataFileStats->columnStats[fieldId];
    const auto& lowerBound =
        minStats->icebergLowerBoundInclusive(kDefaultTruncateLength);
    columnStats.lowerBound =
        encoding::Base64::encode(lowerBound.data(), lowerBound.size());

    const auto upperBound =
        maxStats->icebergUpperBoundExclusive(kDefaultTruncateLength);
    if (upperBound.has_value()) {
      columnStats.upperBound =
          encoding::Base64::encode(upperBound->data(), upperBound->size());
    }
  }
  return dataFileStats;
}

} // namespace facebook::velox::connector::hive::iceberg
