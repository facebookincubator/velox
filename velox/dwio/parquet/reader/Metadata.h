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

// Adapted from Apache Arrow.

#pragma once

#include "velox/dwio/common/compression/Compression.h"

namespace facebook::velox::parquet {

/// ColumnChunkMetaData is a proxy around thrift::ColumnChunk.
class ColumnChunkMetaData {
 public:
  ColumnChunkMetaData(const void* metadata);

  ~ColumnChunkMetaData();

  int64_t numValues() const;

  common::CompressionKind compression() const;

 private:
  const void* ptr_;
};

/// RowGroupMetaData is a proxy around thrift::RowGroup.
class RowGroupMetaData {
 public:
  RowGroupMetaData(const void* metadata);

  ~RowGroupMetaData();

  /// The number of columns in this row group. The order must match the
  /// parent's column ordering.
  int numColumns() const;

  /// Return the ColumnChunkMetaData of the corresponding column ordinal.
  /// The returned object references memory location in it's parent
  /// (RowGroupMetaData) object. Hence, the parent must outlive the returned
  /// object.
  ///
  std::unique_ptr<ColumnChunkMetaData> columnChunk(int index) const;

  /// Number of rows in this row group.
  int64_t numRows() const;

  /// Total byte size of all the uncompressed column data in this row
  /// group.
  int64_t totalByteSize() const;

  /// Total byte size of all the compressed (and potentially encrypted)
  /// column data in this row group.
  /// This information is optional and may be 0 if omitted.
  int64_t totalCompressedSize() const;

 private:
  const void* ptr_;
};

/// FileMetaData is a proxy around thrift::FileMetaData.
class FileMetaData {
 public:
  FileMetaData(const void* metadata);

  ~FileMetaData();

  /// The total number of rows.
  int64_t numRows() const;

  /// The number of row groups in the file.
  int numRowGroups() const;

  /// Return the RowGroupMetaData of the corresponding row group ordinal.
  /// WARNING, the returned object references memory location in it's parent
  /// (FileMetaData) object. Hence, the parent must outlive the returned object.
  std::unique_ptr<RowGroupMetaData> rowGroup(int index) const;

 private:
  const void* ptr_;
};

} // namespace facebook::velox::parquet
