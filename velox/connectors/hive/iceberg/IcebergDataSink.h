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

#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/TransformFactory.h"
#include "velox/connectors/hive/iceberg/Transforms.h"

namespace facebook::velox::connector::hive::iceberg {

// Represents a request for Iceberg write.
class IcebergInsertTableHandle final : public HiveInsertTableHandle {
 public:
  IcebergInsertTableHandle(
      std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns,
      std::shared_ptr<const LocationHandle> locationHandle,
      std::shared_ptr<const IcebergPartitionSpec> partitionSpec,
      memory::MemoryPool* pool,
      dwio::common::FileFormat tableStorageFormat =
          dwio::common::FileFormat::PARQUET,
      std::shared_ptr<HiveBucketProperty> bucketProperty = nullptr,
      std::optional<common::CompressionKind> compressionKind = {},
      const std::unordered_map<std::string, std::string>& serdeParameters = {});

  ~IcebergInsertTableHandle() = default;

  std::shared_ptr<const IcebergPartitionSpec> partitionSpec() const {
    return partitionSpec_;
  }

  const std::vector<std::shared_ptr<Transform>>& columnTransforms() const {
    return columnTransforms_;
  }

 private:
  const std::shared_ptr<const IcebergPartitionSpec> partitionSpec_;
  const std::vector<std::shared_ptr<Transform>> columnTransforms_;
};

class IcebergDataSink : public HiveDataSink {
 public:
  IcebergDataSink(
      RowTypePtr inputType,
      const std::shared_ptr<const IcebergInsertTableHandle>& insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig);

  void appendData(RowVectorPtr input) override;

  const std::vector<std::shared_ptr<dwio::common::IcebergDataFileStatistics>>&
  dataFileStats() const {
    return dataFileStats_;
  }

 private:
  IcebergDataSink(
      RowTypePtr inputType,
      const std::shared_ptr<const IcebergInsertTableHandle>& insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const std::vector<column_index_t>& partitionChannels,
      const std::vector<column_index_t>& dataChannels);

  void splitInputRowsAndEnsureWriters(RowVectorPtr input) override;

  std::vector<std::string> commitMessage() const override;

  HiveWriterId getIcebergWriterId(size_t row) const;

  std::shared_ptr<dwio::common::WriterOptions> createWriterOptions()
      const override;

  std::optional<std::string> getPartitionName(
      const HiveWriterId& id) const override;

  void closeInternal() override;

  // Below are structures for partitions from all inputs. partitionData_
  // is indexed by partitionId.
  std::vector<std::vector<folly::dynamic>> partitionData_;

  std::vector<std::shared_ptr<dwio::common::IcebergDataFileStatistics>>
      dataFileStats_;
  std::shared_ptr<std::vector<dwio::common::IcebergStatsSettings>>
      statsSettings_;
};

} // namespace facebook::velox::connector::hive::iceberg
