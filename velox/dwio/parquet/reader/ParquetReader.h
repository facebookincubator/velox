/*
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

#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/parquet/reader/duckdb/InputStreamFileSystem.h"
#include "velox/external/duckdb/parquet-amalgamation.hpp"

namespace facebook::velox::parquet {

class ParquetRowReader : public dwio::common::RowReader {
 public:
  ParquetRowReader(
      std::shared_ptr<::duckdb::ParquetReader> _reader,
      const dwio::common::RowReaderOptions& options,
      memory::MemoryPool& _pool);
  ~ParquetRowReader() override = default;

  uint64_t seekToRow(uint64_t rowNumber) override;

  uint64_t next(uint64_t size, velox::VectorPtr& result) override;

  size_t estimatedRowSize() const override;

 private:
  std::shared_ptr<::duckdb::ParquetReader> reader;
  ::duckdb::ParquetReaderScanState state;
  memory::MemoryPool& pool;
  std::shared_ptr<const velox::RowType> rowType;
  std::vector<::duckdb::LogicalType> duckdbRowType;
};

class ParquetReader : public dwio::common::Reader {
 public:
  ParquetReader(
      std::unique_ptr<dwio::common::InputStream> stream,
      const dwio::common::ReaderOptions& options);
  ~ParquetReader() override = default;

  std::optional<uint64_t> getNumberOfRows() const override;

  std::unique_ptr<velox::dwrf::ColumnStatistics> getColumnStatistics(
      uint32_t index) const override;

  const std::shared_ptr<const velox::RowType>& getType() const override;

  const std::shared_ptr<const dwio::common::TypeWithId>& getTypeWithId() const;

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const dwio::common::RowReaderOptions& options = {}) const override;

 private:
  static duckdb::InputStreamFileSystem fileSystem;

  ::duckdb::Allocator allocator;
  std::shared_ptr<::duckdb::ParquetReader> reader;
  memory::MemoryPool& pool;

  std::shared_ptr<const velox::RowType> type;
  mutable std::shared_ptr<const dwio::common::TypeWithId> typeWithId;
};

class ParquetReaderFactory : public dwio::common::ReaderFactory {
 public:
  ParquetReaderFactory() : ReaderFactory(dwio::common::FileFormat::PARQUET) {}

  std::unique_ptr<dwio::common::Reader> createReader(
      std::unique_ptr<dwio::common::InputStream> stream,
      const dwio::common::ReaderOptions& options) override {
    return std::make_unique<ParquetReader>(std::move(stream), options);
  }
};

} // namespace facebook::velox::parquet
