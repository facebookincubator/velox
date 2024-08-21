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

struct VeloxIcebergNestedField {
  bool optional;
  const int32_t id;
  const std::string name;
  TypePtr prestoType;
  std::shared_ptr<std::string> doc;

  VeloxIcebergNestedField(
      bool _optional,
      int32_t _id,
      const std::string& _name,
      TypePtr _prestoType,
      std::shared_ptr<std::string> _doc)
      : optional(_optional),
        id(_id),
        name(_name),
        prestoType(std::move(_prestoType)),
        doc(std::move(_doc)) {}
};

struct VeloxIcebergSchema {
  const int32_t schemaId;
  std::vector<std::shared_ptr<const VeloxIcebergNestedField>> columns;
  std::unordered_map<std::string, std::int32_t> columnNameToIdMapping;
  std::unordered_map<std::string, std::int32_t> aliases;
  std::vector<int32_t> identifierFieldIds;

  VeloxIcebergSchema(
      int32_t _schemaId,
      std::vector<std::shared_ptr<const VeloxIcebergNestedField>> _columns,
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

struct VeloxIcebergPartitionSpec {
  const int32_t specId;
  std::shared_ptr<const VeloxIcebergSchema> schema;
  std::vector<std::string> fields;

  VeloxIcebergPartitionSpec(
      int32_t _specId,
      std::shared_ptr<const VeloxIcebergSchema> _schema,
      std::vector<std::string> _fields)
      : specId(_specId), schema(_schema), fields(_fields) {}
};

/**
 * Represents a request for Iceberg write.
 */
class IcebergInsertTableHandle : public HiveInsertTableHandle {
 public:
  IcebergInsertTableHandle(
      std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns,
      std::shared_ptr<const LocationHandle> locationHandle,
      std::shared_ptr<const VeloxIcebergSchema> schema,
      std::shared_ptr<const VeloxIcebergPartitionSpec> partitionSpec,
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

  std::shared_ptr<const VeloxIcebergSchema> schema() const {
    return schema_;
  }

  std::shared_ptr<const VeloxIcebergPartitionSpec> partitionSpec() const {
    return partitionSpec_;
  }

 private:
  std::shared_ptr<const VeloxIcebergSchema> schema_;
  std::shared_ptr<const VeloxIcebergPartitionSpec> partitionSpec_;
};

class PartitionData {
 private:
  std::vector<std::string> partitionValues;
  const std::string PARTITION_VALUES_FIELD = "partitionValues";

 public:
  PartitionData(const std::vector<std::string>& partitionValues)
      : partitionValues(partitionValues) {
    if (partitionValues.empty()) {
      throw std::invalid_argument("partitionValues is null or empty");
    }
  }

  int size() const {
    return partitionValues.size();
  }

  // Convert to JSON
  std::string toJson() const {
    try {
      folly::dynamic jsonObject = folly::dynamic::object();
      folly::dynamic valuesArray = folly::dynamic::array();

      for (const auto& value : partitionValues) {
        valuesArray.push_back(value); // Directly use the string values
      }

      jsonObject[PARTITION_VALUES_FIELD] = valuesArray;
      return folly::toJson(jsonObject); // Convert dynamic object to JSON string
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "JSON conversion failed for PartitionData: " + std::string(e.what()));
    }
  }
};

class IcebergDataSink : public HiveDataSink {
 public:
  IcebergDataSink(
      RowTypePtr inputType,
      std::shared_ptr<const IcebergInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig);

  void appendData(RowVectorPtr input) override;

  std::vector<std::string> close() override;

 protected:
  void write(size_t index, RowVectorPtr input) override;

  // Below are structures for partitions from all inputs. partitionData_
  // is indexed by partitionId.
  std::vector<std::shared_ptr<PartitionData>> partitionData_;

 private:
  void splitInputRowsAndEnsureWriters(RowVectorPtr input) override;

  void extendBuffersForPartitionedTables() override;

  std::string makePartitionDirectory(
      const std::string& tableDirectory,
      const std::optional<std::string>& partitionSubdirectory) const override;

  std::vector<column_index_t> getDataChannels(
      const std::vector<column_index_t>& partitionChannels,
      const column_index_t childrenSize) const override;
};

} // namespace facebook::velox::connector::hive::iceberg
