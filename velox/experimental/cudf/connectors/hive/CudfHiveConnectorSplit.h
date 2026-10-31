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

#include "velox/common/caching/FileProperties.h"
#include "velox/connectors/Connector.h"
#include "velox/dwio/common/Options.h"

namespace cudf {
namespace io {
struct source_info;
}
} // namespace cudf

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox::connector::hive {

struct CudfHiveConnectorSplit
    : public facebook::velox::connector::ConnectorSplit {
  const std::string filePath;
  const uint64_t start;
  const uint64_t length;
  const facebook::velox::dwio::common::FileFormat fileFormat{
      facebook::velox::dwio::common::FileFormat::PARQUET};
  const std::unique_ptr<cudf::io::source_info> cudfSourceInfo;

  /// These represent columns like $file_size, $file_modified_time that are
  /// associated with the CudfHiveConnectorSplit.
  std::unordered_map<std::string, std::string> infoColumns = {};

  /// Hive partition keys of the split, keyed by column name. A nullopt value
  /// is a NULL partition key. Partition columns are not stored in the data
  /// file; the data source materializes them from these values.
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys =
      {};

  /// File properties such as the size, used while opening the file. A known
  /// size lets an S3 or KvikIO open skip the request for it. Presto's Hive
  /// splits carry the size in the $file_size info column instead, and
  /// fileSize() reads either place.
  std::optional<FileProperties> properties;

  CudfHiveConnectorSplit(
      const std::string& connectorId,
      const std::string& _filePath,
      uint64_t _start = 0,
      uint64_t _length = std::numeric_limits<uint64_t>::max(),
      int64_t _splitWeight = 0,
      const std::unordered_map<std::string, std::string>& _infoColumns = {},
      const std::unordered_map<std::string, std::optional<std::string>>&
          _partitionKeys = {},
      std::optional<FileProperties> _properties = std::nullopt,
      bool _cacheable = true);

  std::string toString() const override;
  std::string getFileName() const;

  /// Returns the file size when the split carries a non-negative one, from
  /// 'properties' first and else from the $file_size info column.
  std::optional<int64_t> fileSize() const;

  const cudf::io::source_info& getCudfSourceInfo() const {
    return *cudfSourceInfo;
  }

  uint64_t size() const override;

  folly::dynamic serialize() const override;

  static std::shared_ptr<CudfHiveConnectorSplit> create(
      const folly::dynamic& obj);
};

class CudfHiveConnectorSplitBuilder {
 public:
  explicit CudfHiveConnectorSplitBuilder(std::string filePath)
      : filePath_{std::move(filePath)} {}

  CudfHiveConnectorSplitBuilder& start(uint64_t start) {
    start_ = start;
    return *this;
  }

  CudfHiveConnectorSplitBuilder& length(uint64_t length) {
    length_ = length;
    return *this;
  }

  CudfHiveConnectorSplitBuilder& infoColumn(
      const std::string& name,
      const std::string& value) {
    infoColumns_.emplace(std::move(name), std::move(value));
    return *this;
  }

  CudfHiveConnectorSplitBuilder& partitionKey(
      std::string name,
      std::optional<std::string> value) {
    partitionKeys_.emplace(std::move(name), std::move(value));
    return *this;
  }

  CudfHiveConnectorSplitBuilder& splitWeight(int64_t splitWeight) {
    splitWeight_ = splitWeight;
    return *this;
  }

  CudfHiveConnectorSplitBuilder& cacheable(bool cacheable) {
    cacheable_ = cacheable;
    return *this;
  }

  CudfHiveConnectorSplitBuilder& fileProperties(FileProperties fileProperties) {
    fileProperties_ = std::move(fileProperties);
    return *this;
  }

  CudfHiveConnectorSplitBuilder& connectorId(const std::string& connectorId) {
    connectorId_ = connectorId;
    return *this;
  }

  std::shared_ptr<CudfHiveConnectorSplit> build() const {
    return std::make_shared<CudfHiveConnectorSplit>(
        connectorId_,
        filePath_,
        start_,
        length_,
        splitWeight_,
        infoColumns_,
        partitionKeys_,
        fileProperties_,
        cacheable_);
  }

 private:
  const std::string filePath_;
  uint64_t start_{0};
  uint64_t length_{std::numeric_limits<uint64_t>::max()};
  std::string connectorId_;
  int64_t splitWeight_{0};
  bool cacheable_{true};
  std::unordered_map<std::string, std::string> infoColumns_ = {};
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys_ =
      {};
  std::optional<FileProperties> fileProperties_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
