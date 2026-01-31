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
#include "velox/connectors/hive/iceberg/IcebergFieldStatsCollector.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/encode/Base64.h"
#include "velox/dwio/parquet/writer/arrow/Metadata.h"
#include "velox/dwio/parquet/writer/arrow/Statistics.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

// Recursively counts leaf fields and collects field IDs that should skip
// bounds collection.
// @param field The field config to process.
// @param skipBoundsFields Output set to collect field IDs with skipBounds.
// @return The count of leaf fields under this field.
int32_t countLeafFieldsAndCollectSkipBounds(
    const IcebergFieldStatsConfig& field,
    std::unordered_set<int32_t>& skipBoundsFields) {
  if (field.skipBounds) {
    skipBoundsFields.insert(field.fieldId);
  }
  if (field.children.empty()) {
    return 1;
  }
  int32_t count = 0;
  for (const auto& child : field.children) {
    count += countLeafFieldsAndCollectSkipBounds(child, skipBoundsFields);
  }
  return count;
}

} // namespace

IcebergDataFileStatsCollector::IcebergDataFileStatsCollector(
    std::vector<IcebergFieldStatsConfig> configs)
    : icebergStatsConfig_(std::move(configs)) {}

void IcebergDataFileStatsCollector::collectStats(
    const void* metadata,
    const std::shared_ptr<dwio::common::DataFileStatistics>& dataFileStats) {
  const auto& fileMetadata =
      *static_cast<const std::shared_ptr<parquet::arrow::FileMetaData>*>(
          metadata);
  VELOX_CHECK_NOT_NULL(fileMetadata);

  // numFields is not the number of columns in Iceberg table's schema,
  // e.g., schema_->size(). It also contains the sub-fields when there are
  // nested data types in table's schema.
  std::unordered_set<int32_t> skipBoundsFields;
  int32_t numFields = 0;
  for (const auto& field : icebergStatsConfig_) {
    numFields += countLeafFieldsAndCollectSkipBounds(field, skipBoundsFields);
  }

  std::unordered_map<int32_t, std::shared_ptr<parquet::arrow::Statistics>>
      globalMinStats;
  std::unordered_map<int32_t, std::shared_ptr<parquet::arrow::Statistics>>
      globalMaxStats;

  dataFileStats->numRecords = fileMetadata->num_rows();
  const auto numRowGroups = fileMetadata->num_row_groups();
  for (auto i = 0; i < numRowGroups; ++i) {
    const auto& rgm = fileMetadata->RowGroup(i);
    VELOX_CHECK_EQ(numFields, rgm->num_columns());
    dataFileStats->splitOffsets.emplace_back(rgm->file_offset());

    for (auto j = 0; j < numFields; ++j) {
      const auto& columnChunkMetadata = rgm->ColumnChunk(j);
      const auto fieldId = columnChunkMetadata->field_id();
      const auto numValues = columnChunkMetadata->num_values();

      dataFileStats->valueCounts[fieldId] += numValues;
      dataFileStats->columnsSizes[fieldId] +=
          columnChunkMetadata->total_compressed_size();

      const auto& columnChunkStats = columnChunkMetadata->statistics();

      // TODO: Once https://github.com/facebookincubator/velox/pull/14725 is
      // landed, NaN is available.
      // if (columnChunkStats->nan_count() > 0) {
      //  dataFileStats->nanValueCounts[fieldId] +=
      //  columnChunkStats->nan_count();
      //}

      dataFileStats->nullValueCounts[fieldId] += columnChunkStats->null_count();

      if (columnChunkStats->HasMinMax() &&
          !skipBoundsFields.contains(fieldId)) {
        if (globalMaxStats.find(fieldId) == globalMaxStats.end()) {
          globalMinStats[fieldId] = columnChunkStats;
          globalMaxStats[fieldId] = columnChunkStats;
        } else {
          globalMaxStats[fieldId] =
              parquet::arrow::Statistics::CompareAndGetMax(
                  globalMaxStats[fieldId], columnChunkStats);
          globalMinStats[fieldId] =
              parquet::arrow::Statistics::CompareAndGetMin(
                  globalMinStats[fieldId], columnChunkStats);
        }
      }
    }
  }

  for (const auto& [fieldId, minStats] : globalMinStats) {
    const auto& lowerBound = minStats->MinValue();
    dataFileStats->lowerBounds[fieldId] =
        encoding::Base64::encode(lowerBound.data(), lowerBound.size());
  }
  for (const auto& [fieldId, maxStats] : globalMaxStats) {
    const auto& upperBound = maxStats->MaxValue();
    dataFileStats->upperBounds[fieldId] =
        encoding::Base64::encode(upperBound.data(), upperBound.size());
  }
}

} // namespace facebook::velox::connector::hive::iceberg
