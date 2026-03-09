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

/// Lightweight config constants for the Parquet writer.
/// Separated from Writer.h to allow access without pulling in Arrow headers.
/// WriterOptions inherits from this struct, so all existing code using
/// WriterOptions::kParquet* constants continues to work.
struct WriterConfig {
  // Parsing session and hive configs.

  // Session and connector config names differ by '_' vs '-' separators.
  // Connector keys are inferred from session keys by replacing '_' with '-'.
  static constexpr const char* kParquetWriteTimestampUnit =
      "hive.parquet.writer.timestamp_unit";
  static constexpr const char* kParquetEnableDictionary =
      "hive.parquet.writer.enable_dictionary";
  static constexpr const char* kParquetDictionaryPageSizeLimit =
      "hive.parquet.writer.dictionary_page_size_limit";
  static constexpr const char* kParquetDataPageVersion =
      "hive.parquet.writer.datapage_version";
  static constexpr const char* kParquetWritePageSize =
      "hive.parquet.writer.page_size";
  static constexpr const char* kParquetWriteBatchSize =
      "hive.parquet.writer.batch_size";
  static constexpr const char* kParquetCreatedBy =
      "hive.parquet.writer.created_by";
  static constexpr const char* kParquetMaxTargetFileSize =
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
