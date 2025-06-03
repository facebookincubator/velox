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
struct IcebergPartitionField;

struct IcebergNestedField {
  const bool optional;
  const int32_t id;
  const std::string name;
  const TypePtr type;
  const std::shared_ptr<std::string> doc;

  IcebergNestedField(
      bool _optional,
      int32_t _id,
      const std::string& _name,
      TypePtr _prestoType,
      std::shared_ptr<std::string> _doc)
      : optional(_optional),
        id(_id),
        name(_name),
        type(std::move(_prestoType)),
        doc(std::move(_doc)) {}
};

struct IcebergSchema {
  const int32_t schemaId;
  const std::vector<std::shared_ptr<const IcebergNestedField>> columns;
  const std::unordered_map<std::string, std::int32_t> columnNameToIdMapping;
  const std::unordered_map<std::string, std::int32_t> aliases;
  const std::vector<int32_t> identifierFieldIds;

  IcebergSchema(
      const int32_t _schemaId,
      std::vector<std::shared_ptr<const IcebergNestedField>> _columns,
      const std::unordered_map<std::string, std::int32_t>&
          _columnNameToIdMapping,
      const std::unordered_map<std::string, std::int32_t>& _aliases,
      std::vector<int32_t> _identifierFieldIds)
      : schemaId(_schemaId),
        columns(_columns),
        columnNameToIdMapping(_columnNameToIdMapping),
        aliases(_aliases),
        identifierFieldIds(_identifierFieldIds) {}
};

struct IcebergPartitionSpec {
  const int32_t specId;
  const std::shared_ptr<const IcebergSchema> schema;
  const std::vector<IcebergPartitionField> fields;

  IcebergPartitionSpec(
      const int32_t _specId,
      std::shared_ptr<const IcebergSchema> _schema,
      std::vector<IcebergPartitionField> _fields)
      : specId(_specId), schema(_schema), fields(_fields) {}
};

enum TransformType { IDENTITY, YEAR, MONTH, DAY, HOUR, BUCKET, TRUNCATE };

struct IcebergPartitionField {
  int32_t sourceId;
  std::optional<int32_t> parameter;
  TransformType transform;
  std::string name;
  IcebergPartitionField(
      const int32_t sourceId,
      const std::optional<int32_t> parameter,
      const TransformType transform,
      const std::string name)
      : sourceId(sourceId),
        parameter(parameter),
        transform(transform),
        name(name) {}

  IcebergPartitionField(
      const std::shared_ptr<const IcebergSchema>& schema,
      const std::optional<int32_t> parameter,
      const TransformType transform,
      const std::string name)
      : sourceId(schema->columnNameToIdMapping.at(name)),
        parameter(parameter),
        transform(transform),
        name(name) {}
};

/// Represents a request for Iceberg write.
class IcebergInsertTableHandle : public HiveInsertTableHandle {
 public:
  IcebergInsertTableHandle(
      std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns,
      std::shared_ptr<const LocationHandle> locationHandle,
      std::shared_ptr<const IcebergSchema> schema,
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
        schema_(std::move(schema)),
        partitionSpec_(std::move(partitionSpec)) {}

  virtual ~IcebergInsertTableHandle() = default;

  std::shared_ptr<const IcebergPartitionSpec> partitionSpec() const {
    return partitionSpec_;
  }

 private:
  std::shared_ptr<const IcebergSchema> schema_;
  std::shared_ptr<const IcebergPartitionSpec> partitionSpec_;
};

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

class IcebergDataSink : public HiveDataSink {
 public:
  IcebergDataSink(
      RowTypePtr inputType,
      std::shared_ptr<const IcebergInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig);

  std::vector<std::string> close() override;

 protected:
  // Below are structures for partitions from all inputs. partitionData_
  // is indexed by partitionId.
  std::vector<std::shared_ptr<PartitionData>> partitionData_;

  std::vector<column_index_t> createDataChannels(
      const std::shared_ptr<const HiveInsertTableHandle>& tableHandle)
      const override {
    std::vector<column_index_t> channels(tableHandle->inputColumns().size());
    std::iota(channels.begin(), channels.end(), 0);
    return channels;
  }

 private:
  void splitInputRowsAndEnsureWriters(RowVectorPtr input) override;

  void extendBuffersForPartitionedTables() override;

  std::string makePartitionDirectory(
      const std::string& tableDirectory,
      const std::optional<std::string>& partitionSubdirectory) const override;
};

} // namespace facebook::velox::connector::hive::iceberg
