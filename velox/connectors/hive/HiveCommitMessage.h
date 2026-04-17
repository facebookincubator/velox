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

namespace facebook::velox::connector::hive {

/// JSON field names for the partition update object produced by each writer
/// and consumed by the Presto coordinator to finalize files and update the
/// metastore.
///
/// JSON structure:
/// {
///   "name":                        "<partition key, e.g. ds=2024-01-01>",
///   "updateMode":                  "NEW" | "APPEND" | "OVERWRITE",
///   "writePath":                   "<staging directory>",
///   "targetPath":                  "<final directory>",
///   "fileWriteInfos": [
///     {
///       "writeFileName":           "<temp filename in writePath>",
///       "targetFileName":          "<final filename in targetPath>",
///       "fileSize":                <bytes>
///     }
///   ],
///   "rowCount":                    <total rows>,
///   "inMemoryDataSizeInBytes":     <uncompressed bytes>,
///   "onDiskDataSizeInBytes":       <compressed bytes on disk>,
///   "containsNumberedFileNames":   true | false
/// }
struct HiveCommitMessage {
  /// Partition directory name in Hive format (e.g., "ds=2024-01-01/region=us").
  /// Empty string for unpartitioned tables.
  static constexpr const char* kName = "name";
  /// Write mode: "NEW", "APPEND", or "OVERWRITE". Controls how the committer
  /// handles metastore updates and existing file conflicts.
  static constexpr const char* kUpdateMode = "updateMode";
  /// Staging directory where files were written during execution.
  static constexpr const char* kWritePath = "writePath";
  /// Final destination directory. Files are renamed from writePath to
  /// targetPath during commit.
  static constexpr const char* kTargetPath = "targetPath";
  /// Array of per-file metadata objects. One entry per file written, including
  /// rotated files.
  static constexpr const char* kFileWriteInfos = "fileWriteInfos";
  /// Temporary filename used during writing (in the staging directory).
  static constexpr const char* kWriteFileName = "writeFileName";
  /// Final filename after commit (in the target directory).
  static constexpr const char* kTargetFileName = "targetFileName";
  /// Size of individual file in bytes.
  static constexpr const char* kFileSize = "fileSize";
  /// Total rows written to this partition across all files.
  static constexpr const char* kRowCount = "rowCount";
  /// Uncompressed input data size in bytes.
  static constexpr const char* kInMemoryDataSizeInBytes =
      "inMemoryDataSizeInBytes";
  /// Compressed bytes written to disk.
  static constexpr const char* kOnDiskDataSizeInBytes = "onDiskDataSizeInBytes";
  /// Whether filenames follow a numbered sequence from file rotation.
  static constexpr const char* kContainsNumberedFileNames =
      "containsNumberedFileNames";
};

} // namespace facebook::velox::connector::hive
