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

#include <string>
#include <unordered_map>
#include <vector>

#include "velox/dwio/common/Options.h"

namespace facebook::velox::connector::hive::iceberg {

enum class FileContent {
  kData,
  kPositionalDeletes,
  kEqualityDeletes,
  /// Iceberg V3 deletion vector. A serialized roaring bitmap of deleted row
  /// positions stored as a blob inside a Puffin file. More compact than V2
  /// positional delete files and avoids sorted merge of multiple delete files.
  /// The coordinator extracts the blob offset and length from the Puffin
  /// footer and provides them via IcebergDeleteFile fields.
  kDeletionVector,
};

struct IcebergDeleteFile {
  FileContent content;
  const std::string filePath;
  dwio::common::FileFormat fileFormat;
  uint64_t recordCount;
  uint64_t fileSizeInBytes;
  // The field ids for the delete columns for equality delete files
  std::vector<int32_t> equalityFieldIds;
  // The lower bounds of the in-file positions for the deleted rows, identified
  // by each column's field id. E.g. The deleted rows for a column with field id
  // 1 is in range [10, 50], where 10 and 50 are the deleted row positions in
  // the data file, then lowerBounds would contain entry <1, "10">
  std::unordered_map<int32_t, std::string> lowerBounds;
  // The upper bounds of the in-file positions for the deleted rows, identified
  // by each column's field id. E.g. The deleted rows for a column with field id
  // 1 is in range [10, 50], then upperBounds will contain entry <1, "50">
  std::unordered_map<int32_t, std::string> upperBounds;

  /// Data sequence number of this delete file, assigned by the Iceberg snapshot
  /// that produced it. Per the Iceberg spec (V2+), an equality delete file must
  /// only be applied to data files whose data sequence number is strictly less
  /// than the delete file's data sequence number. A value of 0 means
  /// "unassigned" (legacy V1 tables) and disables sequence number filtering.
  int64_t dataSequenceNumber{0};

  IcebergDeleteFile(
      FileContent _content,
      const std::string& _filePath,
      dwio::common::FileFormat _fileFormat,
      uint64_t _recordCount,
      uint64_t _fileSizeInBytes,
      std::vector<int32_t> _equalityFieldIds = {},
      std::unordered_map<int32_t, std::string> _lowerBounds = {},
      std::unordered_map<int32_t, std::string> _upperBounds = {},
      int64_t _dataSequenceNumber = 0)
      : content(_content),
        filePath(_filePath),
        fileFormat(_fileFormat),
        recordCount(_recordCount),
        fileSizeInBytes(_fileSizeInBytes),
        equalityFieldIds(_equalityFieldIds),
        lowerBounds(_lowerBounds),
        upperBounds(_upperBounds),
        dataSequenceNumber(_dataSequenceNumber) {}
};

} // namespace facebook::velox::connector::hive::iceberg
