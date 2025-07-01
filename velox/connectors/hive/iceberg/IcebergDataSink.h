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

#include "ColumnTransform.h"
#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/iceberg/TransformFactory.h"

namespace facebook::velox::connector::hive::iceberg {

class IcebergInsertFileNameGenerator : public FileNameGenerator {
 public:
  IcebergInsertFileNameGenerator() {}

  std::pair<std::string, std::string> gen(
      std::optional<uint32_t> bucketId,
      const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx& connectorQueryCtx,
      bool commitRequired) const override;

  folly::dynamic serialize() const override;

  std::string toString() const override;
};

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
      const std::unordered_map<std::string, std::string>& serdeParameters = {})
      : HiveInsertTableHandle(
            std::move(inputColumns),
            std::move(locationHandle),
            tableStorageFormat,
            std::move(bucketProperty),
            compressionKind,
            serdeParameters,
            nullptr,
            false,
            std::make_shared<const IcebergInsertFileNameGenerator>()),
        partitionSpec_(std::move(partitionSpec)),
        columnTransforms_(
            parsePartitionTransformSpecs(partitionSpec_->fields, pool)) {}

  ~IcebergInsertTableHandle() = default;

  std::shared_ptr<const IcebergPartitionSpec> partitionSpec() const {
    return partitionSpec_;
  }

  const std::vector<ColumnTransform>& columnTransforms() const {
    return columnTransforms_;
  }

 private:
  const std::shared_ptr<const IcebergPartitionSpec> partitionSpec_;
  const std::vector<ColumnTransform> columnTransforms_;
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

  std::optional<std::string> getPartitionName(
      const HiveWriterId& id) const override;

  // Below are structures for partitions from all inputs. partitionData_
  // is indexed by partitionId.
  std::vector<std::vector<folly::dynamic>> partitionData_;
};

} // namespace facebook::velox::connector::hive::iceberg
