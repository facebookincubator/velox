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

#include <arrow/buffer.h>

#include "velox/common/testutil/TempFilePath.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"

namespace facebook::velox::parquet::arrow::test {

/// Materializes a Parquet file produced in memory by the Arrow writer onto
/// disk and reads it back with the Velox ParquetReader. The reader holds raw
/// pointers into the memory pools and reads from the file on demand, so an
/// instance keeps the file and both pools alive alongside the reader.
class ParquetTestFile {
 public:
  /// Writes 'buffer' to 'path'. Use this instead of open() when the test needs
  /// the bytes on disk but constructs the reader itself, for example to assert
  /// that construction fails.
  static void write(
      const std::shared_ptr<common::testutil::TempFilePath>& path,
      const std::shared_ptr<::arrow::Buffer>& buffer);

  /// Writes 'buffer' to a new temporary file and opens a ParquetReader over
  /// it. 'poolName' names the root and leaf memory pools the reader allocates
  /// from. Pass the ScanSpec to createRowReader() through RowReaderOptions; the
  /// Parquet reader never reads the one on ReaderOptions.
  static ParquetTestFile open(
      const std::shared_ptr<::arrow::Buffer>& buffer,
      const std::string& poolName);

  ParquetReader& reader() {
    return *reader_;
  }

  /// Returns the pool the reader allocates from. Result vectors passed to the
  /// row reader must come from this pool.
  memory::MemoryPool* leafPool() const {
    return leafPool_.get();
  }

 private:
  ParquetTestFile() = default;

  // Temporary file holding the Parquet bytes, removed on destruction.
  std::shared_ptr<common::testutil::TempFilePath> path_;

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
  std::unique_ptr<ParquetReader> reader_;
};

} // namespace facebook::velox::parquet::arrow::test
