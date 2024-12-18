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

#include "velox/common/config/Config.h"

#include <cudf/types.hpp>

#include <optional>
#include <string>

namespace facebook::velox::config {
class ConfigBase;
}

namespace facebook::velox::cudf_velox::connector::parquet {

class ParquetReaderConfig {
 public:
  enum class InsertExistingPartitionsBehavior {
    kError,
    kOverwrite,
  };

  static std::string insertExistingPartitionsBehaviorString(
      InsertExistingPartitionsBehavior behavior);

  /// Behavior on insert into existing partitions.
  static constexpr const char* kInsertExistingPartitionsBehaviorSession =
      "insert_existing_partitions_behavior";
  static constexpr const char* kInsertExistingPartitionsBehavior =
      "insert-existing-partitions-behavior";

  // Number of rows to skip from the start; Parquet stores the number of rows as
  // int64_t
  static constexpr const char* kSkipRows = "skip-rows";

  // Number of rows to read; `nullopt` is all
  static constexpr const char* kNumRows = "num-rows";

  static constexpr const char* kMaxChunkReadLimit = "chunk-read-limit";
  static constexpr const char* kMaxChunkReadLimitSession = "chunk_read_limit";

  static constexpr const char* kMaxPassReadLimit = "pass-read-limit";
  static constexpr const char* kMaxPassReadLimitSession = "pass_read_limit";

  // Whether to store string data as categorical type
  static constexpr const char* kConvertStringsToCategories =
      "convert-strings-to-categories";
  static constexpr const char* kConvertStringsToCategoriesSession =
      "convert_strings_to_categories";

  // Whether to use PANDAS metadata to load columns
  static constexpr const char* kUsePandasMetadata = "use-pandas-metadata";
  static constexpr const char* kUsePandasMetadataSession =
      "use_pandas_metadata";

  // Whether to read and use ARROW schema
  static constexpr const char* kUseArrowSchema = "use-arrow-schema";
  static constexpr const char* kUseArrowSchemaSession = "use_arrow_schema";

  // Whether to allow reading matching select columns from mismatched Parquet
  // files.
  static constexpr const char* kAllowMismatchedParquetSchemas =
      "allow-mismatched-parquet-schemas";
  static constexpr const char* kAllowMismatchedParquetSchemasSession =
      "allow_mismatched_parquet_schemas";

  // Cast timestamp columns to a specific type
  static constexpr const char* kTimestampType = "timestamp-type";
  static constexpr const char* kTimestampTypeSession = "timestamp_type";

  InsertExistingPartitionsBehavior insertExistingPartitionsBehavior(
      const config::ConfigBase* session) const;

  ParquetReaderConfig(std::shared_ptr<const config::ConfigBase> config) {
    VELOX_CHECK_NOT_NULL(
        config, "Config is null for ParquetReaderConfig initialization");
    config_ = std::move(config);
  }

  const std::shared_ptr<const config::ConfigBase>& config() const {
    return config_;
  }

  std::size_t maxChunkReadLimit() const;
  std::size_t maxChunkReadLimitSession(const config::ConfigBase* session) const;

  std::size_t maxPassReadLimit() const;
  std::size_t maxPassReadLimitSession(const config::ConfigBase* session) const;

  int64_t skipRows() const;
  std::optional<cudf::size_type> numRows() const;

  bool isConvertStringsToCategories() const;
  bool isConvertStringsToCategoriesSession(
      const config::ConfigBase* session) const;

  bool isUsePandasMetadata() const;
  bool isUsePandasMetadataSession(const config::ConfigBase* session) const;

  bool isUseArrowSchema() const;
  bool isUseArrowSchemaSession(const config::ConfigBase* session) const;

  bool isAllowMismatchedParquetSchemas() const;
  bool isAllowMismatchedParquetSchemasSession(
      const config::ConfigBase* session) const;

  cudf::data_type timestampType() const;
  cudf::data_type timestampTypeSession(const config::ConfigBase* session) const;

 private:
  std::shared_ptr<const config::ConfigBase> config_;
};
} // namespace facebook::velox::cudf_velox::connector::parquet
