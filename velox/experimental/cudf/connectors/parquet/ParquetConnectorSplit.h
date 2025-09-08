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

#include "velox/connectors/Connector.h"
#include "velox/dwio/common/Options.h"

#include <cudf/io/types.hpp>

#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox::connector::parquet {

struct ParquetConnectorSplit
    : public facebook::velox::connector::ConnectorSplit {
  const std::string filePath;
  const dwio::common::FileFormat fileFormat{dwio::common::FileFormat::PARQUET};
  const uint64_t start;
  const uint64_t length;
  const cudf::io::source_info cudfSourceInfo;

  /// These represent columns like $file_size, $file_modified_time that are
  /// associated with the HiveSplit.
  std::unordered_map<std::string, std::string> infoColumns;

  ParquetConnectorSplit(
      const std::string& connectorId,
      const std::string& _filePath,
      uint64_t _start = 0,
      uint64_t _length =
          static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max()),
      int64_t _splitWeight = 0,
      const std::unordered_map<std::string, std::string>& _infoColumns = {})
      : facebook::velox::connector::ConnectorSplit(connectorId, _splitWeight),
        filePath(_filePath),
        start(_start),
        length(_length),
        cudfSourceInfo({filePath}),
        infoColumns(_infoColumns) {
    VELOX_USER_CHECK(
        start <=
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "ParquetConnectorSplit `start` (rows to skip) must be less than or equal to INT64_MAX");
        VELOX_USER_CHECK(
        length <=
            static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max()),
        "ParquetConnectorSplit `length` (rows to read after skipping) must be less than or equal to 2^31");
  }

  std::string toString() const override;
  std::string getFileName() const;

  const cudf::io::source_info& getCudfSourceInfo() const {
    return cudfSourceInfo;
  }

  uint64_t getNumRows() const {
    return length;
  }

  uint64_t getSkipRows() const {
    return start;
  }

  static std::shared_ptr<ParquetConnectorSplit> create(
      const folly::dynamic& obj);
};

class ParquetConnectorSplitBuilder {
 public:
  explicit ParquetConnectorSplitBuilder(std::string filePath)
      : filePath_{std::move(filePath)} {
    infoColumns_["$path"] = filePath_;
  }

  ParquetConnectorSplitBuilder& splitWeight(int64_t splitWeight) {
    splitWeight_ = splitWeight;
    return *this;
  }

  ParquetConnectorSplitBuilder& connectorId(const std::string& connectorId) {
    connectorId_ = connectorId;
    return *this;
  }

  ParquetConnectorSplitBuilder& infoColumn(
      const std::string& name,
      const std::string& value) {
    infoColumns_.emplace(std::move(name), std::move(value));
    return *this;
  }

  ParquetConnectorSplitBuilder& start(uint64_t start) {
    VELOX_USER_CHECK(
        start <=
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "ParquetConnectorSplit `start` (rows to skip) must be less than or equal to INT64_MAX");
    start_ = start;
    return *this;
  }

  ParquetConnectorSplitBuilder& length(uint64_t length) {
    VELOX_USER_CHECK(
        length <=
            static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max()),
        "ParquetConnectorSplit `length` (rows to read after skipping) must be less than or equal to 2^31");
    length_ = length;
    return *this;
  }

  std::shared_ptr<ParquetConnectorSplit> build() const {
    return std::make_shared<ParquetConnectorSplit>(
        connectorId_, filePath_, start_, length_, splitWeight_, infoColumns_);
  }

 private:
  const std::string filePath_;
  std::string connectorId_;
  uint64_t start_{0};
  uint64_t length_{std::numeric_limits<cudf::size_type>::max()};
  int64_t splitWeight_{0};
  std::unordered_map<std::string, std::string> infoColumns_ = {};
};

} // namespace facebook::velox::cudf_velox::connector::parquet
