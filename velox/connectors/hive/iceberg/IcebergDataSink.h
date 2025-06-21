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

namespace facebook::velox::connector::hive::iceberg {

enum TransformType { IDENTITY, YEAR, MONTH, DAY, HOUR, BUCKET, TRUNCATE };

struct IcebergPartitionField {
  // The column name of this partition field as it appears in the partition
  // spec.
  std::string name;

  // The transform type applied to the source field (e.g., IDENTITY, BUCKET,
  // TRUNCATE, etc.).
  TransformType transform;

  // Optional parameter for transforms that require configuration
  // (e.g., bucket count or truncate width).
  std::optional<int32_t> parameter;

  IcebergPartitionField(
      const std::string& _name,
      TransformType _transform,
      std::optional<int32_t> _parameter)
      : name(_name), transform(_transform), parameter(_parameter) {}
};

struct IcebergPartitionSpec {
  const int32_t specId;
  const std::vector<IcebergPartitionField> fields;

  IcebergPartitionSpec(
      int32_t _specId,
      const std::vector<IcebergPartitionField>& _fields)
      : specId(_specId), fields(_fields) {}
};

/// Represents a request for Iceberg write.
class IcebergInsertTableHandle final : public HiveInsertTableHandle {
 public:
  IcebergInsertTableHandle(
      std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns,
      std::shared_ptr<const LocationHandle> locationHandle,
      std::shared_ptr<const IcebergPartitionSpec> partitionSpec,
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
            serdeParameters),
        partitionSpec_(std::move(partitionSpec)) {}

  ~IcebergInsertTableHandle() = default;

  std::shared_ptr<const IcebergPartitionSpec> partitionSpec() const {
    return partitionSpec_;
  }

 private:
  std::shared_ptr<const IcebergPartitionSpec> partitionSpec_;
};

class IcebergDataSink : public HiveDataSink {
 public:
  IcebergDataSink(
      RowTypePtr inputType,
      std::shared_ptr<const IcebergInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig);

  std::vector<std::string> close() override;

 private:
  IcebergDataSink(
      RowTypePtr inputType,
      std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const std::vector<column_index_t>& dataChannels);

  void splitInputRowsAndEnsureWriters(RowVectorPtr input) override;

  std::string makePartitionDirectory(
      const std::string& tableDirectory,
      const std::optional<std::string>& partitionSubdirectory) const override;

  // Below are structures for partitions from all inputs. partitionData_
  // is indexed by partitionId.
  std::vector<std::vector<std::string>> partitionData_;
};

} // namespace facebook::velox::connector::hive::iceberg
