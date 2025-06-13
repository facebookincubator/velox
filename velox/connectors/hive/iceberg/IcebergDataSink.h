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
  // The ID of the source column in the table schema.
  int32_t sourceId_;

  // Optional parameter for transforms that require configuration
  // (e.g., bucket count or truncate width).
  std::optional<int32_t> parameter_;

  // The transform type applied to the source field (e.g., IDENTITY, BUCKET,
  // TRUNCATE, etc.).
  TransformType transform_;

  // The column name of this partition field as it appears in the partition
  // spec.
  std::string name_;

  IcebergPartitionField(
      const int32_t sourceId,
      const std::optional<int32_t> parameter,
      const TransformType transform,
      const std::string& name)
      : sourceId_(sourceId),
        parameter_(parameter),
        transform_(transform),
        name_(name) {}
};

struct IcebergPartitionSpec {
  const int32_t specId;
  const std::vector<IcebergPartitionField> fields;

  IcebergPartitionSpec(
      const int32_t _specId,
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
      const std::vector<column_index_t>& partitionChannels,
      const std::vector<column_index_t>& dataChannels);

  void splitInputRowsAndEnsureWriters(RowVectorPtr input) override;

  void extendBuffersForPartitionedTables() override;

  std::string makePartitionDirectory(
      const std::string& tableDirectory,
      const std::optional<std::string>& partitionSubdirectory) const override;

  class PartitionData {
   public:
    PartitionData(const std::vector<std::string>& partitionValues)
        : partitionValues_(partitionValues) {
      VELOX_CHECK(!partitionValues.empty(), "partitionValues is null or empty");
    }

    int size() const {
      return partitionValues_.size();
    }

    std::string toJson() const;

   private:
    const std::vector<std::string> partitionValues_;
  };

  // Below are structures for partitions from all inputs. partitionData_
  // is indexed by partitionId.
  std::vector<std::shared_ptr<PartitionData>> partitionData_;
};

} // namespace facebook::velox::connector::hive::iceberg
