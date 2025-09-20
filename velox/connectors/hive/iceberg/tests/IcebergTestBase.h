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

#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::connector::hive::iceberg {

enum class NullParam { kNoNulls, kPartialNulls, kAllNulls };
class IcebergTestBase : public HiveConnectorTestBase {
 public:
  IcebergTestBase()
      : config_{std::make_shared<facebook::velox::dwrf::Config>()} {
    // Make the writers flush per batch so that we can create non-aligned
    // RowGroups between the base data files and delete files
    flushPolicyFactory_ = []() {
      return std::make_unique<dwrf::LambdaFlushPolicy>([]() { return true; });
    };
  }

 protected:
  dwio::common::FileFormat fileFormat_{dwio::common::FileFormat::DWRF};
  static constexpr int rowCount_ = 20000;
  std::shared_ptr<dwrf::Config> config_;
  std::function<std::unique_ptr<dwrf::DWRFFlushPolicy>()> flushPolicyFactory_;

  static std::vector<int64_t> makeRandomDeleteValues(int32_t maxRowNumber);

  template <class T>
  std::vector<T> makeSequenceValues(int32_t numRows, int8_t repeat = 1);

  core::PlanNodePtr tableScanNode(const RowTypePtr& outputRowType) const;

  std::vector<std::shared_ptr<ConnectorSplit>> makeIcebergSplits(
      const std::string& dataFilePath,
      const std::vector<IcebergDeleteFile>& deleteFiles = {},
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      const uint32_t splitCount = 1);

  /// Generate test data vectors with mixed column types and configurable null
  /// patterns per column
  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      const std::vector<TypeKind>& columnTypes,
      const std::vector<NullParam>& nullParams);

  std::string makeTypePredicates(
      const std::vector<RowVectorPtr>& deleteVectors,
      const std::vector<int32_t>& equalityFieldIds,
      const std::vector<TypeKind>& columnTypes);

  /// Unified writeDataFiles function that handles all use cases
  /// @param config Configuration for file structure and data generation
  /// @return Map of file names to their corresponding file paths (or indexed
  /// names for simple cases)
  struct WriteDataFilesConfig {
    // Basic parameters
    uint64_t numRows = 20000;
    int32_t numColumns = 1;
    int32_t splitCount = 1;

    // Advanced parameters for complex row group structures
    std::optional<std::map<std::string, std::vector<int64_t>>>
        rowGroupSizesForFiles;

    // Custom data vectors (takes precedence if provided)
    std::vector<RowVectorPtr> dataVectors;

    // File writing configuration
    bool useConfigAndFlushPolicy = false;
  };

  std::map<std::string, std::shared_ptr<TempFilePath>> writeDataFiles(
      const WriteDataFilesConfig& config);
};
} // namespace facebook::velox::connector::hive::iceberg
