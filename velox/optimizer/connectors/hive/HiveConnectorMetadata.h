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

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/Options.h"
#include "velox/optimizer/connectors/ConnectorMetadata.h"

namespace facebook::velox::connector::hive {

/// Describes a single partition of a Hive table. If the table is
/// bucketed, this resolves to a single file. If the table is
/// partitioned and not bucketed, this resolves to a leaf
/// directory. If the table is not bucketed and not partitioned,
/// this resolves to the directory corresponding to the table.
struct HivePartitionHandle : public PartitionHandle {
  HivePartitionHandle(
      const std::unordered_map<std::string, std::optional<std::string>>
          partitionKeys,
      std::optional<int32_t> tableBucketNumber)
      : partitionKeys(partitionKeys), tableBucketNumber(tableBucketNumber) {}

  const std::unordered_map<std::string, std::optional<std::string>>
      partitionKeys;
  const std::optional<int32_t> tableBucketNumber;
};

/// Describes a Hive table layout. Adds a file format and a list of
/// Hive partitioning columns and an optional bucket count to the base
/// TableLayout. The partitioning in TableLayout does not differentiate between
/// bucketing and Hive partitioning columns. The bucketing columns
/// are the 'partitioning' columns minus the
/// 'hivePartitioningColumns'. 'numBuckets' is the number of Hive buckets if
/// 'partitionColumns' differs from 'hivePartitionColumns'.
class HiveTableLayout : public TableLayout {
 public:
  HiveTableLayout(
      const std::string& name,
      const Table* table,
      connector::Connector* connector,
      std::vector<const Column*> columns,
      std::vector<const Column*> partitioning,
      std::vector<const Column*> orderColumns,
      std::vector<SortOrder> sortOrder,
      std::vector<const Column*> lookupKeys,
      std::vector<const Column*> hivePartitionColumns,
      dwio::common::FileFormat fileFormat,
      std::optional<int32_t> numBuckets = std::nullopt)
      : TableLayout(
            name,
            table,
            connector,
            columns,
            partitioning,
            orderColumns,
            sortOrder,
            lookupKeys,
            true),
        fileFormat_(fileFormat),
        hivePartitionColumns_(hivePartitionColumns),
        numBuckets_(numBuckets) {}

  dwio::common::FileFormat fileFormat() const {
    return fileFormat_;
  }

  const std::vector<const Column*>& hivePartitionColumns() const {
    return hivePartitionColumns_;
  }

  std::optional<int32_t> numBuckets() const {
    return numBuckets_;
  }

 protected:
  const dwio::common::FileFormat fileFormat_;
  const std::vector<const Column*> hivePartitionColumns_;
  std::optional<int32_t> numBuckets_;
};

class HiveConnectorMetadata : public ConnectorMetadata {
 public:
  explicit HiveConnectorMetadata(HiveConnector* hiveConnector)
      : hiveConnector_(hiveConnector) {}

  ColumnHandlePtr createColumnHandle(
      const TableLayout& layout,
      const std::string& columnName,
      std::vector<common::Subfield> subfields = {},
      std::optional<TypePtr> castToType = std::nullopt,
      SubfieldMapping subfieldMapping = {}) override;

  ConnectorTableHandlePtr createTableHandle(
      const TableLayout& layout,
      std::vector<ColumnHandlePtr> columnHandles,
      core::ExpressionEvaluator& evaluator,
      std::vector<core::TypedExprPtr> filters,
      std::vector<core::TypedExprPtr>& rejectedFilters,
      std::optional<LookupKeys> lookupKeys = std::nullopt) override;

 protected:
  HiveConnector* const hiveConnector_;
};

} // namespace facebook::velox::connector::hive
