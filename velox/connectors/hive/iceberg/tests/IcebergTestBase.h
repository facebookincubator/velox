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
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#endif

namespace facebook::velox::connector::hive::iceberg::test {

using TempDirectoryPath = common::testutil::TempDirectoryPath;

extern const std::string kIcebergConnectorId;

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

  uint64_t getFileSize(const std::string& path) const;

  std::vector<std::shared_ptr<ConnectorSplit>> makeIcebergSplits(
      const std::string& dataFilePath,
      const std::vector<IcebergDeleteFile>& deleteFiles = {},
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      uint32_t splitCount = 1,
      const std::unordered_map<std::string, std::string>& infoColumns = {},
      int64_t dataSequenceNumber = 0);

  std::shared_ptr<ConnectorSplit> makeIcebergSplitWithInfoColumns(
      const std::string& dataFilePath,
      const std::unordered_map<std::string, std::string>& infoColumns,
      const std::vector<IcebergDeleteFile>& deleteFiles = {},
      int64_t dataSequenceNumber = 0);

  std::shared_ptr<IcebergColumnHandle> makeIcebergHandle(
      const std::string& name,
      const TypePtr& type,
      int fieldId,
      const std::string& defaultValue);

  std::shared_ptr<IcebergColumnHandle> makeIcebergHandle(
      const std::string& name,
      const TypePtr& type,
      int fieldId,
      FileColumnHandle::ColumnType columnType =
          FileColumnHandle::ColumnType::kRegular);

  ColumnHandleMap makeColumnHandles(
      const RowTypePtr& rowType,
      const std::unordered_set<int>& partitionIndices = {});

  void assertTableScan(
      const RowTypePtr& outputType,
      const std::vector<std::shared_ptr<ConnectorSplit>>& splits,
      const std::vector<RowVectorPtr>& expected,
      const RowTypePtr& dataColumns = nullptr,
      const ColumnHandleMap& assignments = {},
      const std::string& filter = "",
      const std::string& remainingFilter = "",
      const std::string& subfieldFilter = "",
      const std::unordered_map<std::string, std::string>& sessionProperties =
          {});

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
