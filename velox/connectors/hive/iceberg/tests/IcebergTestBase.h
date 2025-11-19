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

#include "velox/connectors/hive/iceberg/IcebergDataSink.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#endif

namespace facebook::velox::connector::hive::iceberg::test {

constexpr std::string_view kDefaultTestIcebergFunctionNamePrefix{
    "$internal$.test_iceberg."};

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
      const RowTypePtr& rowType,
      const std::vector<RowVectorPtr>& vectors,
      const std::string& dataPath,
      const std::vector<PartitionField>& partitionFields = {});

  std::vector<std::shared_ptr<ConnectorSplit>> createSplitsForDirectory(
      const std::string& directory);

  std::vector<std::string> listFiles(const std::string& dirPath);

  std::shared_ptr<IcebergPartitionSpec> createPartitionSpec(
      const RowTypePtr& rowType,
      const std::vector<PartitionField>& partitionFields);

  dwio::common::FileFormat fileFormat_{dwio::common::FileFormat::DWRF};
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
  std::shared_ptr<HiveConfig> connectorConfig_;
  VectorFuzzer::Options fuzzerOptions_;
  std::unique_ptr<VectorFuzzer> fuzzer_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
};

} // namespace facebook::velox::connector::hive::iceberg::test
