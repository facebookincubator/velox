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

#include <optional>
#include <unordered_map>
#include "velox/connectors/Connector.h"
#include "velox/dwio/common/Options.h"
#include "velox/experimental/cudf/connectors/parquet/FileProperties.h"
#include "velox/experimental/cudf/connectors/parquet/TableHandle.h"

namespace facebook::velox::cudf_velox::connector::parquet {

struct ParquetConnectorSplit : public connector::ConnectorSplit {
  const std::string filePath;
  dwio::common::FileFormat fileFormat;
  const uint64_t start;
  const uint64_t length;

  /// Mapping from partition keys to values. Values are specified as strings
  /// formatted the same way as CAST(x as VARCHAR). Null values are specified as
  /// std::nullopt. Date values must be formatted using ISO 8601 as YYYY-MM-DD.
  /// All scalar types and date type are supported.
  const std::unordered_map<std::string, std::optional<std::string>>
      partitionKeys;

  /// These represent columns like $file_size, $file_modified_time that are
  /// associated with the ParquetSplit.
  std::unordered_map<std::string, std::string> infoColumns;

  /// These represent file properties like file size that are used while opening
  /// the file handle.
  std::optional<FileProperties> properties;

  ParquetConnectorSplit(
      const std::string& connectorId,
      const std::string& _filePath,
      dwio::common::FileFormat _fileFormat,
      uint64_t _start = 0,
      uint64_t _length = std::numeric_limits<uint64_t>::max(),
      const std::unordered_map<std::string, std::optional<std::string>>&
          _partitionKeys = {},
      const std::shared_ptr<std::string>& _extraFileInfo = {},
      int64_t _splitWeight = 0,
      const std::unordered_map<std::string, std::string>& _infoColumns = {},
      std::optional<FileProperties> _properties = std::nullopt)
      : ConnectorSplit(connectorId, _splitWeight),
        filePath(_filePath),
        fileFormat(_fileFormat),
        start(_start),
        length(_length),
        partitionKeys(_partitionKeys),
        extraFileInfo(_extraFileInfo),
        infoColumns(_infoColumns),
        properties(_properties) {}

  std::string toString() const override;

  std::string getFileName() const;
}
};

class ParquetConnectorSplitBuilder {
 public:
  explicit ParquetConnectorSplitBuilder(std::string filePath)
      : filePath_{std::move(filePath)} {
    infoColumns_["$path"] = filePath_;
  }

  ParquetConnectorSplitBuilder& start(uint64_t start) {
    start_ = start;
    return *this;
  }

  ParquetConnectorSplitBuilder& length(uint64_t length) {
    length_ = length;
    return *this;
  }

  ParquetConnectorSplitBuilder& splitWeight(int64_t splitWeight) {
    splitWeight_ = splitWeight;
    return *this;
  }

  ParquetConnectorSplitBuilder& fileFormat(dwio::common::FileFormat format) {
    fileFormat_ = format;
    return *this;
  }

  ParquetConnectorSplitBuilder& infoColumn(
      const std::string& name,
      const std::string& value) {
    infoColumns_.emplace(std::move(name), std::move(value));
    return *this;
  }

  ParquetConnectorSplitBuilder& partitionKey(
      std::string name,
      std::optional<std::string> value) {
    partitionKeys_.emplace(std::move(name), std::move(value));
    return *this;
  }

  ParquetConnectorSplitBuilder& tableBucketNumber(int32_t bucket) {
    tableBucketNumber_ = bucket;
    infoColumns_["$bucket"] = std::to_string(bucket);
    return *this;
  }

  ParquetConnectorSplitBuilder& customSplitInfo(
      const std::unordered_map<std::string, std::string>& customSplitInfo) {
    customSplitInfo_ = customSplitInfo;
    return *this;
  }

  ParquetConnectorSplitBuilder& extraFileInfo(
      const std::shared_ptr<std::string>& extraFileInfo) {
    extraFileInfo_ = extraFileInfo;
    return *this;
  }

  ParquetConnectorSplitBuilder& connectorId(const std::string& connectorId) {
    connectorId_ = connectorId;
    return *this;
  }

  ParquetConnectorSplitBuilder& fileProperties(FileProperties fileProperties) {
    fileProperties_ = fileProperties;
    return *this;
  }

  std::shared_ptr<connector::parquet::ParquetConnectorSplit> build() const {
    return std::make_shared<connector::parquet::ParquetConnectorSplit>(
        connectorId_,
        filePath_,
        fileFormat_,
        start_,
        length_,
        partitionKeys_,
        customSplitInfo_,
        extraFileInfo_,
        splitWeight_,
        infoColumns_,
        fileProperties_);
  }

 private:
  const std::string filePath_;
  dwio::common::FileFormat fileFormat_{dwio::common::FileFormat::PARQUET};
  uint64_t start_{0};
  uint64_t length_{std::numeric_limits<uint64_t>::max()};
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys_;
  std::unordered_map<std::string, std::string> customSplitInfo_ = {};
  std::shared_ptr<std::string> extraFileInfo_ = {};
  std::unordered_map<std::string, std::string> infoColumns_ = {};
  std::string connectorId_;
  int64_t splitWeight_{0};
  std::optional<FileProperties> fileProperties_;
};

} // namespace facebook::velox::cudf_velox::connector::parquet
