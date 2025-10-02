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

/// Enumeration for configuring null value patterns in test data generation
enum class NullParam {
  kNoNulls, /// No null values in the column
  kPartialNulls, /// ~20% null values randomly distributed
  kAllNulls /// All null values in the column
};
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
  /// File format used for writing test data files (default: DWRF)
  dwio::common::FileFormat fileFormat_{dwio::common::FileFormat::DWRF};

  /// Default number of rows for test data generation
  static constexpr int rowCount_ = 20000;

  /// Configuration for DWRF file writers
  std::shared_ptr<dwrf::Config> config_;

  /// Factory function for creating flush policies that flush per batch
  std::function<std::unique_ptr<dwrf::DWRFFlushPolicy>()> flushPolicyFactory_;

  /// Generate random delete row indices for testing positional deletes
  /// @param maxRowNumber Maximum row number to generate values for (exclusive)
  /// @return Vector of randomly selected row indices, approximately 20% of the
  /// range
  static std::vector<int64_t> makeRandomDeleteValues(int32_t maxRowNumber);

  /// Generate sequence values with configurable repeat patterns for test data
  /// @tparam T Integral type for the generated values
  /// @param numRows Total number of values to generate
  /// @param repeat Number of times each value should be repeated (default: 1)
  /// @return Vector containing sequence values with specified repeat pattern
  template <class T>
  std::vector<T> makeSequenceValues(int32_t numRows, int8_t repeat = 1);

  /// Create a simple table scan plan node for testing
  /// @param outputRowType Row type schema for the scan operation
  /// @return Plan node configured for table scanning with the specified schema
  core::PlanNodePtr tableScanNode(const RowTypePtr& outputRowType) const;

  /// Create Iceberg connector splits for test data files
  /// @param dataFilePath Path to the data file to create splits for
  /// @param deleteFiles Vector of delete files to associate with the splits
  /// (optional)
  /// @param partitionKeys Map of partition column names to values (optional)
  /// @param splitCount Number of splits to create from the file (default: 1)
  /// @return Vector of connector splits configured for Iceberg testing
  std::vector<std::shared_ptr<ConnectorSplit>> makeIcebergSplits(
      const std::string& dataFilePath,
      const std::vector<IcebergDeleteFile>& deleteFiles = {},
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      const uint32_t splitCount = 1);

  /// Generate test data vectors with mixed column types and configurable null
  /// patterns
  /// @param count Number of row vectors to generate
  /// @param rowsPerVector Number of rows in each vector
  /// @param columnTypes Vector specifying the type for each column
  /// @param nullParams Vector specifying null behavior for each column
  /// @return Vector of row vectors with generated test data following specified
  /// patterns
  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      const std::vector<TypeKind>& columnTypes,
      const std::vector<NullParam>& nullParams);

  /// Generate SQL predicates for equality delete operations
  /// @param deleteVectors Vector containing delete data (expected size: 1)
  /// @param equalityFieldIds Field IDs for equality comparison columns
  /// @param columnTypes Types of the columns being compared
  /// @return SQL predicate string for filtering out deleted rows, or empty
  /// string if no deletes
  std::string makePredicates(
      const std::vector<RowVectorPtr>& deleteVectors,
      const std::vector<int32_t>& equalityFieldIds,
      const std::vector<TypeKind>& columnTypes);

  /// Convert a vector of values to a comma-separated string for SQL queries
  /// @tparam T Type of values in the vector (numeric types or StringView)
  /// @param deleteValues Vector of values to convert
  /// @return Comma-separated string representation, with quotes for string
  /// types
  template <typename T>
  std::string getListAsCSVString(const std::vector<T>& deleteValues);

  /// Configuration structure for writeDataFiles function
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

  /// Write test data files with configurable structure and content
  /// Supports multiple use cases: simple uniform files, complex row group
  /// structures, and custom data vectors. Automatically creates DuckDB table
  /// for validation.
  /// @param config Configuration specifying file structure, row counts, and
  /// options
  /// @return Map of file names to their temporary file paths for test access
  std::map<std::string, std::shared_ptr<TempFilePath>> writeDataFiles(
      const WriteDataFilesConfig& config);

 private:
  template <typename T>
  std::string makeSingleColumnNotInPredicate(
      const VectorPtr& deleteVector,
      int32_t numDeletedRows,
      const std::string& columnName);

  template <typename T>
  std::string makeSingleValueInequalityPredicate(
      const VectorPtr& deleteVector,
      int32_t row,
      const std::string& columnName);
};
} // namespace facebook::velox::connector::hive::iceberg
