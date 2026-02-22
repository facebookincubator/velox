/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "arrow/io/interfaces.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"

namespace facebook::velox::parquet::arrow {

class InternalFileDecryptor;
class BloomFilter;
class FileMetaData;

class PARQUET_EXPORT RowGroupBloomFilterReader {
 public:
  virtual ~RowGroupBloomFilterReader() = default;

  /// \brief Read bloom filter of a column chunk.
  ///
  /// \param[in] i column ordinal of the column chunk.
  /// \returns bloom filter of the column or nullptr if it does not exist.
  /// \throws ParquetException if the index is out of bound, or read bloom.
  /// Filter failed.
  virtual std::unique_ptr<BloomFilter> getColumnBloomFilter(int i) = 0;
};

/// \brief Interface for reading the bloom filter for a Parquet file.
class PARQUET_EXPORT BloomFilterReader {
 public:
  virtual ~BloomFilterReader() = default;

  /// \brief Create a BloomFilterReader instance.
  /// \returns a BloomFilterReader instance.
  /// WARNING: The returned BloomFilterReader references to all the input.
  /// Parameters, so it must not outlive all of the input parameters. Usually.
  /// These input parameters come from the same ParquetFileReader object, so it.
  /// Must not outlive the reader that creates this BloomFilterReader.
  static std::unique_ptr<BloomFilterReader> make(
      std::shared_ptr<::arrow::io::RandomAccessFile> input,
      std::shared_ptr<FileMetaData> fileMetadata,
      const ReaderProperties& properties,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor = NULLPTR);

  /// \brief Get the bloom filter reader of a specific row group.
  /// \param[in] i row group ordinal to get bloom filter reader.
  /// \returns RowGroupBloomFilterReader of the specified row group. A nullptr.
  /// May or may.
  ///          Not be returned if the bloom filter for the row group is.
  ///          Unavailable. It is the caller's responsibility to check the.
  ///          Return value of follow-up calls to the RowGroupBloomFilterReader.
  /// \throws ParquetException if the index is out of bound.
  virtual std::shared_ptr<RowGroupBloomFilterReader> rowGroup(int i) = 0;
};

} // namespace facebook::velox::parquet::arrow
