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

namespace cudf {
namespace io {
struct source_info;
}
} // namespace cudf

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

struct CudfHiveConnectorSplit
    : public facebook::velox::connector::ConnectorSplit {
  /// Paths in this split. Batched splits contain more than one full file.
  const std::vector<std::string> filePaths;

  /// First path, retained for source compatibility with single-file callers.
  const std::string filePath;
  const uint64_t start;
  const uint64_t length;
  const facebook::velox::dwio::common::FileFormat fileFormat{
      facebook::velox::dwio::common::FileFormat::PARQUET};
  const std::unique_ptr<cudf::io::source_info> cudfSourceInfo;

  /// These represent columns like $file_size, $file_modified_time that are
  /// associated with the CudfHiveConnectorSplit.
  std::unordered_map<std::string, std::string> infoColumns = {};

  /// Constructs a single-file (or byte-range) split.
  CudfHiveConnectorSplit(
      const std::string& connectorId,
      const std::string& _filePath,
      uint64_t _start = 0,
      uint64_t _length = std::numeric_limits<uint64_t>::max(),
      int64_t _splitWeight = 0,
      const std::unordered_map<std::string, std::string>& _infoColumns = {});

  /// Constructs a batch of full-file splits.
  static std::shared_ptr<CudfHiveConnectorSplit> makeBatch(
      const std::string& connectorId,
      const std::vector<std::string>& filePaths,
      int64_t splitWeight);

  std::string toString() const override;
  std::string getFileName() const;

  const cudf::io::source_info& getCudfSourceInfo() const {
    return *cudfSourceInfo;
  }

  uint64_t size() const override;

  folly::dynamic serialize() const override;

  static std::shared_ptr<CudfHiveConnectorSplit> create(
      const folly::dynamic& obj);

 private:
  struct BatchTag {};

  CudfHiveConnectorSplit(
      BatchTag,
      const std::string& connectorId,
      const std::vector<std::string>& filePaths,
      int64_t splitWeight);
};

/// Groups paths into native cuDF multi-file splits. A maximum of zero places
/// all paths in one split.
std::vector<std::shared_ptr<CudfHiveConnectorSplit>>
makeCudfHiveConnectorSplitBatches(
    const std::string& connectorId,
    const std::vector<std::string>& filePaths,
    size_t maxFilesPerBatch,
    int64_t splitWeight = 0);

class CudfHiveConnectorSplitBuilder {
 public:
  explicit CudfHiveConnectorSplitBuilder(std::string filePath)
      : filePaths_{std::move(filePath)} {}

  static CudfHiveConnectorSplitBuilder forFilePaths(
      std::vector<std::string> filePaths);

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

  CudfHiveConnectorSplitBuilder& splitWeight(int64_t splitWeight) {
    splitWeight_ = splitWeight;
    return *this;
  }

  CudfHiveConnectorSplitBuilder& connectorId(const std::string& connectorId) {
    connectorId_ = connectorId;
    return *this;
  }

  std::shared_ptr<CudfHiveConnectorSplit> build() const {
    if (auto* single = std::get_if<std::string>(&filePaths_)) {
      return std::make_shared<CudfHiveConnectorSplit>(
          connectorId_, *single, start_, length_, splitWeight_, infoColumns_);
    }
    VELOX_USER_CHECK_EQ(
        start_, 0, "Batched splits do not support a non-zero start offset");
    VELOX_USER_CHECK_EQ(
        length_,
        std::numeric_limits<uint64_t>::max(),
        "Batched splits do not support byte-range lengths");
    VELOX_USER_CHECK(
        infoColumns_.empty(),
        "Batched splits cannot share per-file info columns");
    return CudfHiveConnectorSplit::makeBatch(
        connectorId_,
        std::get<std::vector<std::string>>(filePaths_),
        splitWeight_);
  }

 private:
  struct BatchTag {};

  explicit CudfHiveConnectorSplitBuilder(
      BatchTag,
      std::vector<std::string> filePaths)
      : filePaths_{std::move(filePaths)} {}

  const std::variant<std::string, std::vector<std::string>> filePaths_;
  uint64_t start_{0};
  uint64_t length_{std::numeric_limits<uint64_t>::max()};
  std::string connectorId_;
  int64_t splitWeight_{0};
  std::unordered_map<std::string, std::string> infoColumns_ = {};
};

} // namespace facebook::velox::cudf_velox::connector::hive
