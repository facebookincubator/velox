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

namespace facebook::velox::parquet {

/// Config constants for the Parquet writer.
///
/// IMPORTANT: These constants are kept in a separate header rather than in
/// Writer.h because Gluten's WholeStageResultIterator.cc needs access to these
/// configuration constants but cannot include Writer.h due to Arrow header
/// conflicts. This separation allows external code to reference these constants
/// without pulling in Arrow dependencies.
struct WriterConfig {
  // Parsing session and hive configs.

  // This isn't a typo; session and hive connector config names are different
  // ('_' vs '-').
  static constexpr const char* kParquetSessionWriteTimestampUnit =
      "hive.parquet.writer.timestamp_unit";
  static constexpr const char* kParquetHiveConnectorWriteTimestampUnit =
      "hive.parquet.writer.timestamp-unit";
  static constexpr const char* kParquetSessionEnableDictionary =
      "hive.parquet.writer.enable_dictionary";
  static constexpr const char* kParquetHiveConnectorEnableDictionary =
      "hive.parquet.writer.enable-dictionary";
  static constexpr const char* kParquetSessionDictionaryPageSizeLimit =
      "hive.parquet.writer.dictionary_page_size_limit";
  static constexpr const char* kParquetHiveConnectorDictionaryPageSizeLimit =
      "hive.parquet.writer.dictionary-page-size-limit";
  static constexpr const char* kParquetSessionDataPageVersion =
      "hive.parquet.writer.datapage_version";
  static constexpr const char* kParquetHiveConnectorDataPageVersion =
      "hive.parquet.writer.datapage-version";
  static constexpr const char* kParquetSessionWritePageSize =
      "hive.parquet.writer.page_size";
  static constexpr const char* kParquetHiveConnectorWritePageSize =
      "hive.parquet.writer.page-size";
  static constexpr const char* kParquetSessionWriteBatchSize =
      "hive.parquet.writer.batch_size";
  static constexpr const char* kParquetHiveConnectorWriteBatchSize =
      "hive.parquet.writer.batch-size";
  static constexpr const char* kParquetHiveConnectorCreatedBy =
      "hive.parquet.writer.created-by";

  // Use the same property name from HiveConfig::kMaxTargetFileSize.
  static constexpr const char* kParquetConnectorMaxTargetFileSize =
      "max-target-file-size";
  static constexpr const char* kParquetSessionMaxTargetFileSize =
      "max_target_file_size";
  // Serde parameter keys for timestamp settings. These can be set via
  // serdeParameters map to override the default timestamp behavior.
  // The timezone key accepts a timezone string or empty string to disable
  // timezone conversion.
  static constexpr const char* kParquetSerdeTimestampUnit =
      "parquet.writer.timestamp.unit";
  static constexpr const char* kParquetSerdeTimestampTimezone =
      "parquet.writer.timestamp.timezone";
};

} // namespace facebook::velox::parquet
