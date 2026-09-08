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

#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergConfig.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/IcebergTableHandle.h"
#include "velox/connectors/hive/iceberg/tests/IcebergPlanBuilder.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#ifdef VELOX_ENABLE_PARQUET
#include "velox/common/file/LocalFile.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#endif

namespace facebook::velox::connector::hive::iceberg::test {

using TempDirectoryPath = common::testutil::TempDirectoryPath;

struct PartitionField {
  // 0-based column index.
  int32_t id;
  TransformType type;
  std::optional<int32_t> parameter;
};

class IcebergTestBase : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override;

  void TearDown() override;

  std::vector<RowVectorPtr> createTestData(
      RowTypePtr rowType,
      int32_t numBatches,
      vector_size_t rowsPerBatch,
      double nullRatio = 0.0);

  std::shared_ptr<IcebergDataSink> createDataSink(
      const RowTypePtr& rowType,
      const std::string& outputDirectoryPath,
      const std::vector<PartitionField>& partitionFields = {});

  std::shared_ptr<IcebergDataSink> createDataSinkAndAppendData(
      const std::vector<RowVectorPtr>& vectors,
      const std::string& dataPath,
      const std::vector<PartitionField>& partitionFields = {});

  std::vector<std::shared_ptr<ConnectorSplit>> createSplitsForDirectory(
      const std::string& directory);

  /// Returns the size of a test file.
  static uint64_t getFileSize(const std::string& path);

  /// Creates Iceberg connector splits for a data file. Tests can attach delete
  /// files, partition keys, info columns, and a data sequence number to each
  /// split.
  std::vector<std::shared_ptr<ConnectorSplit>> makeIcebergSplits(
      const std::string& dataFilePath,
      const std::vector<IcebergDeleteFile>& deleteFiles = {},
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      uint32_t splitCount = 1,
      const std::unordered_map<std::string, std::string>& infoColumns = {},
      int64_t dataSequenceNumber = 0,
      const std::unordered_map<int32_t, std::optional<std::string>>&
          identityPartitionKeys = {});

  /// Creates one Iceberg connector split for a full data file with info
  /// columns.
  std::shared_ptr<ConnectorSplit> makeIcebergSplitWithInfoColumns(
      const std::string& dataFilePath,
      const std::unordered_map<std::string, std::string>& infoColumns,
      const std::vector<IcebergDeleteFile>& deleteFiles = {},
      int64_t dataSequenceNumber = 0);

  /// Writes a DWRF data file with no iceberg.id footer attributes.
  /// The DWRF reader falls back to positional name mapping for these files.
  std::shared_ptr<common::testutil::TempFilePath> writeDataFile(
      const std::vector<RowVectorPtr>& data);

  /// Writes a DWRF file stamping "iceberg.id" footer attributes on each
  /// top-level column. 'icebergFieldIds[i]' is the Iceberg field ID for the
  /// i-th column; DWRF pre-order node IDs: 0=root, 1=first column, etc.
  std::shared_ptr<common::testutil::TempFilePath> writeDwrfFileWithFieldIds(
      const std::vector<RowVectorPtr>& data,
      const std::vector<int32_t>& icebergFieldIds);

#ifdef VELOX_ENABLE_PARQUET
  /// Writes a Parquet file. 'icebergFieldIds[i]' is stamped as the Parquet
  /// field ID for column i so the reader resolves columns by field ID under
  /// kParquetFieldId mode. Pass an empty vector to omit field IDs.
  std::shared_ptr<common::testutil::TempFilePath> writeParquetFile(
      const std::vector<RowVectorPtr>& data,
      const std::vector<int32_t>& icebergFieldIds = {});
#endif

  /// Builds an Iceberg table scan plan via IcebergPlanBuilder.
  ///
  /// IcebergColumnHandle assignments are auto-built from 'dataColumns' and
  /// 'dataColumnFieldIds'. Filter strings are parsed by the builder so there
  /// is no duplication of filter-parsing logic. Caller-supplied 'assignments'
  /// and 'filterColumnHandles' take precedence when provided.
  ///
  ///   outputType          — columns to project.
  ///   dataColumns         — full table schema (authoritative for field IDs).
  ///                         Defaults to outputType when nullptr.
  ///   subfieldFilters     — subfield filter expressions pushed into the scan.
  ///   remainingFilter     — remaining filter expression pushed into the scan.
  ///   assignments         — explicit column-handle map; auto-built when empty.
  ///   filterColumnHandles — filter-only column handles; auto-detected when
  ///                         empty.
  ///   dataColumnFieldIds  — Iceberg field IDs aligned to dataColumns;
  ///                         defaults to 1-based ordinals when empty.
  ///   postScanFilter      — SQL expression added as a FilterNode on top of the
  ///                         scan (not pushed down). Empty string = no filter.
  core::PlanNodePtr makeIcebergTableScanPlan(
      const RowTypePtr& outputType,
      const RowTypePtr& dataColumns = nullptr,
      const std::vector<std::string>& subfieldFilters = {},
      const std::string& remainingFilter = "",
      connector::ColumnHandleMap assignments = {},
      std::vector<IcebergColumnHandlePtr> filterColumnHandles = {},
      const std::vector<int32_t>& dataColumnFieldIds = {},
      const std::string& postScanFilter = {},
      common::SubfieldFilters subfieldFiltersMap = {});

  /// Creates Hive column handles for all columns in 'rowType', marking
  /// specified columns as partition keys.
  ColumnHandleMap makeColumnHandles(
      const RowTypePtr& rowType,
      const std::unordered_set<int>& partitionIndices = {});

  std::vector<std::string> listFiles(const std::string& dirPath);

  std::shared_ptr<IcebergPartitionSpec> createPartitionSpec(
      const RowTypePtr& rowType,
      const std::vector<PartitionField>& partitionFields);

  void setConnectorSessionProperty(
      const std::string& key,
      const std::string& value);

  /// Recreates the connector query context with the given session timezone
  /// and timestamp-adjustment flag. Tests use this to exercise non-UTC
  /// session configurations and verify timezone-sensitive behavior.
  void recreateConnectorQueryCtx(
      const std::string& sessionTimezone,
      bool adjustTimestampToTimezone);

  /// Extracts partition key-value pairs from a file path.
  /// Returns a map where keys are partition column names and values are
  /// partition values (std::nullopt for null values).
  /// Example: "/path/to/c1=10/c2=null/file.parquet" returns
  /// {{"c1", "10"}, {"c2", std::nullopt}}.
  static std::unordered_map<std::string, std::optional<std::string>>
  extractPartitionKeys(const std::string& filePath);

  dwio::common::FileFormat fileFormat_{dwio::common::FileFormat::PARQUET};
  std::shared_ptr<memory::MemoryPool> opPool_;
  std::unique_ptr<ConnectorQueryCtx> connectorQueryCtx_;

 private:
  IcebergInsertTableHandlePtr createInsertTableHandle(
      const RowTypePtr& rowType,
      const std::string& outputDirectoryPath,
      const std::vector<PartitionField>& partitionFields = {});

  std::vector<std::string> listPartitionDirectories(
      const std::string& dataPath);

  void setupMemoryPools();

  std::shared_ptr<memory::MemoryPool> root_;
  std::shared_ptr<memory::MemoryPool> connectorPool_;
  std::shared_ptr<config::ConfigBase> connectorSessionProperties_;
  std::shared_ptr<HiveConfig> hiveConfig_;
  std::shared_ptr<IcebergConfig> icebergConfig_;
  VectorFuzzer::Options fuzzerOptions_;
  std::unique_ptr<VectorFuzzer> fuzzer_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
};

} // namespace facebook::velox::connector::hive::iceberg::test
