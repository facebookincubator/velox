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
#include "velox/dwio/parquet/writer/arrow/Metadata.h"
#include "velox/dwio/parquet/writer/arrow/Statistics.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

// Recursively collects field IDs that should skip bounds collection.
// MAP and ARRAY types, along with all their descendants, should skip bounds.
// @param field The Parquet field ID structure to process.
// @param type The Velox type corresponding to this field.
// @param skipBoundsFieldIds Output set to populate with field IDs to skip.
// @param skipBounds Whether this field and its descendants should skip bounds.
void collectSkipBoundsFieldIds(
    const parquet::ParquetFieldId& field,
    const TypePtr& type,
    std::unordered_set<int32_t>& skipBoundsFieldIds,
    bool skipBounds) {
  VELOX_CHECK_NOT_NULL(type, "Input column type cannot be null.");

  // If this is a MAP or ARRAY type, or if we're already inside one,
  // add this field ID to the skip set.
  const bool shouldSkip = skipBounds || type->isMap() || type->isArray();
  if (shouldSkip) {
    skipBoundsFieldIds.insert(field.fieldId);
  }

  VELOX_CHECK_EQ(field.children.size(), type->size());
  for (auto i = 0; i < type->size(); ++i) {
    collectSkipBoundsFieldIds(
        field.children[i], type->childAt(i), skipBoundsFieldIds, shouldSkip);
  }
}

} // namespace

IcebergParquetStatsCollector::IcebergParquetStatsCollector(
    const std::vector<std::shared_ptr<const HiveColumnHandle>>& inputColumns) {
  parquetFieldIds_.reserve(inputColumns.size());
  for (const auto& columnHandle : inputColumns) {
    auto icebergColumnHandle =
        checkedPointerCast<const IcebergColumnHandle>(columnHandle);
    parquetFieldIds_.emplace_back(icebergColumnHandle->field());
    collectSkipBoundsFieldIds(
        icebergColumnHandle->field(),
        icebergColumnHandle->dataType(),
        skipBoundsFieldIds_,
        /*skipBounds=*/false);
  }
}

std::shared_ptr<dwio::common::IcebergDataFileStatistics>
IcebergParquetStatsCollector::aggregate(const ParquetFileMetadata& metadata) {
  VELOX_CHECK_NOT_NULL(metadata);

  auto dataFileStats =
      std::make_shared<dwio::common::IcebergDataFileStatistics>();

  std::unordered_map<
      int32_t,
      std::pair<
          std::shared_ptr<parquet::arrow::Statistics>,
          std::shared_ptr<parquet::arrow::Statistics>>>
      globalMinMaxStats;

  dataFileStats->numRecords = metadata->num_rows();
  const auto numRowGroups = metadata->num_row_groups();

  for (auto i = 0; i < numRowGroups; ++i) {
    const auto& rgm = metadata->RowGroup(i);
    dataFileStats->splitOffsets.emplace_back(rgm->file_offset());

    for (auto j = 0; j < rgm->num_columns(); ++j) {
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

      if (columnChunkStats->HasMinMax() && shouldStoreBounds(fieldId)) {
        auto [it, inserted] = globalMinMaxStats.emplace(
            fieldId, std::pair{columnChunkStats, columnChunkStats});

        if (!inserted) {
          auto& [minStats, maxStats] = it->second;

          if (columnChunkStats->MaxGreaterThan(*maxStats)) {
            maxStats = columnChunkStats;
          }
          if (columnChunkStats->MinLessThan(*minStats)) {
            minStats = columnChunkStats;
          }
        }
      }
    }
  }

  for (const auto& [fieldId, stats] : globalMinMaxStats) {
    const auto& [minStats, maxStats] = stats;

    const auto& lowerBound =
        minStats->IcebergLowerBoundInclusive(kDefaultTruncateLength);
    dataFileStats->lowerBounds[fieldId] =
        encoding::Base64::encode(lowerBound.data(), lowerBound.size());

    const auto upperBound =
        maxStats->IcebergUpperBoundExclusive(kDefaultTruncateLength);
    if (upperBound.has_value()) {
      dataFileStats->upperBounds[fieldId] =
          encoding::Base64::encode(upperBound->data(), upperBound->size());
    }
  }

  return dataFileStats;
}

} // namespace facebook::velox::connector::hive::iceberg
