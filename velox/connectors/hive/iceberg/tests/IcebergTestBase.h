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

#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
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

  std::shared_ptr<IcebergDataSink> createIcebergDataSink(
      const RowTypePtr& rowType,
      const std::string& outputDirectoryPath,
      const std::vector<PartitionField>& partitionTransforms = {},
      const std::vector<std::string>& sortedBy = {});

  std::vector<std::shared_ptr<ConnectorSplit>> createSplitsForDirectory(
      const std::string& directory);

  std::vector<std::string> listFiles(const std::string& dirPath);

  void setConnectorSessionProperty(
      const std::string& key,
      const std::string& value);

  std::shared_ptr<IcebergPartitionSpec> createPartitionSpec(
      const std::vector<PartitionField>& transformSpecs,
      const RowTypePtr& rowType);

  void setupMemoryPools();

  dwio::common::FileFormat fileFormat_{dwio::common::FileFormat::PARQUET};
  RowTypePtr rowType_;
  std::shared_ptr<memory::MemoryPool> opPool_;
  std::shared_ptr<config::ConfigBase> connectorSessionProperties_;

 private:
  IcebergInsertTableHandlePtr createIcebergInsertTableHandle(
      const RowTypePtr& rowType,
      const std::string& outputDirectoryPath,
      const std::vector<PartitionField>& partitionTransforms = {},
      const std::vector<std::string>& sortedBy = {});

  std::vector<std::string> listPartitionDirectories(
      const std::string& dataPath);

  std::shared_ptr<memory::MemoryPool> root_;
  std::shared_ptr<memory::MemoryPool> connectorPool_;
  std::shared_ptr<HiveConfig> connectorConfig_;
  std::unique_ptr<ConnectorQueryCtx> connectorQueryCtx_;
  VectorFuzzer::Options fuzzerOptions_;
  std::unique_ptr<VectorFuzzer> fuzzer_;
};

} // namespace facebook::velox::connector::hive::iceberg::test
