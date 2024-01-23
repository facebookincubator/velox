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

#include "velox/dwio/parquet/reader/Metadata.h"

#include <vector>

#include "velox/dwio/parquet/reader/ParquetReaderUtil.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"

namespace facebook::velox::parquet {

ColumnChunkMetaData::ColumnChunkMetaData(const void* metadata)
    : ptr_(metadata) {}

ColumnChunkMetaData::~ColumnChunkMetaData() = default;

FOLLY_ALWAYS_INLINE const thrift::ColumnChunk* thriftColumnMetaData(
    const void* metadata) {
  return reinterpret_cast<const thrift::ColumnChunk*>(metadata);
}

int64_t ColumnChunkMetaData::numValues() const {
  return thriftColumnMetaData(ptr_)->meta_data.num_values;
}

common::CompressionKind ColumnChunkMetaData::compression() const {
  return thriftCodecToCompressionKind(
      thriftColumnMetaData(ptr_)->meta_data.codec);
}

FOLLY_ALWAYS_INLINE const thrift::RowGroup* thriftRGMetaData(
    const void* metadata) {
  return reinterpret_cast<const thrift::RowGroup*>(metadata);
}

RowGroupMetaData::RowGroupMetaData(const void* metadata) : ptr_(metadata) {}

RowGroupMetaData::~RowGroupMetaData() = default;

int RowGroupMetaData::numColumns() const {
  return thriftRGMetaData(ptr_)->columns.size();
}

int64_t RowGroupMetaData::numRows() const {
  return thriftRGMetaData(ptr_)->num_rows;
}

int64_t RowGroupMetaData::totalByteSize() const {
  return thriftRGMetaData(ptr_)->total_byte_size;
}

int64_t RowGroupMetaData::totalCompressedSize() const {
  return thriftRGMetaData(ptr_)->total_compressed_size;
}

std::unique_ptr<ColumnChunkMetaData> RowGroupMetaData::columnChunk(
    int i) const {
  return std::make_unique<ColumnChunkMetaData>(
      reinterpret_cast<const void*>(&thriftRGMetaData(ptr_)->columns[i]));
}

FOLLY_ALWAYS_INLINE const thrift::FileMetaData* thriftFileMetaData(
    const void* metadata) {
  return reinterpret_cast<const thrift::FileMetaData*>(metadata);
}

FileMetaData::FileMetaData(const void* metadata) : ptr_(metadata) {}

FileMetaData::~FileMetaData() = default;

std::unique_ptr<RowGroupMetaData> FileMetaData::rowGroup(int i) const {
  return std::make_unique<RowGroupMetaData>(
      reinterpret_cast<const void*>(&thriftFileMetaData(ptr_)->row_groups[i]));
}

int64_t FileMetaData::numRows() const {
  return thriftFileMetaData(ptr_)->num_rows;
}

int FileMetaData::numRowGroups() const {
  return thriftFileMetaData(ptr_)->row_groups.size();
}

} // namespace facebook::velox::parquet
