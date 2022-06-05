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

#include "ParquetTypeWithId.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"

namespace facebook::velox::parquet {

using TypePtr = std::shared_ptr<const velox::Type>;

using namespace dwio::common;

constexpr uint64_t DIRECTORY_SIZE_GUESS = 1024 * 1024;
constexpr uint64_t FILE_PRELOAD_THRESHOLD = 1024 * 1024 * 8;

////////////////////////////////////////////////////////////////////////////////

class ReaderBase {
 public:
  ReaderBase(std::unique_ptr<InputStream> stream, const ReaderOptions& options);

  virtual ~ReaderBase() = default;

  memory::MemoryPool& getMemoryPool() const;
  BufferedInput& getBufferedInput() const;

  const InputStream& getStream() const;
  const uint64_t getFileLength() const;
  const uint64_t getFileNumRows() const;
  const thrift::FileMetaData& getFileMetaData() const;
  const std::shared_ptr<const RowType>& getSchema() const;
  const std::shared_ptr<const TypeWithId>& getSchemaWithId();

 protected:
  void loadFileMetaData();
  void initializeSchema();
  std::shared_ptr<const ParquetTypeWithId> getParquetColumnInfo(
      uint32_t maxSchemaElementIdx,
      uint32_t maxRepeat,
      uint32_t maxDefine,
      uint32_t& schemaIdx,
      uint32_t& columnIdx) const;
  TypePtr convertType(const thrift::SchemaElement& schemaElement) const;
  static std::shared_ptr<const RowType> createRowType(
      std::vector<std::shared_ptr<const ParquetTypeWithId::TypeWithId>>
          children);

  memory::MemoryPool& pool_;
  const ReaderOptions& options_;
  const std::unique_ptr<InputStream> stream_;
  std::shared_ptr<BufferedInput> input_;

  uint64_t fileLength_;
  std::unique_ptr<thrift::FileMetaData> fileMetaData_;
  RowTypePtr schema_;
  std::shared_ptr<const TypeWithId> schemaWithId_;

  const bool binaryAsString = false;
};

////////////////////////////////////////////////////////////////////////////////

class ParquetReader : public Reader {
 public:
  ParquetReader(
      std::unique_ptr<InputStream> stream,
      const ReaderOptions& options);
  ~ParquetReader() override = default;

  std::optional<uint64_t> numberOfRows() const override {
    return readerBase_->getFileNumRows();
  }

  // TODO: Merge the stats for all RowGroups. Parquet column stats is per row
  // group It's only used in tests for DWRF
  std::unique_ptr<ColumnStatistics> columnStatistics(
      uint32_t index) const override {
    VELOX_NYI(
        "columnStatistics for native Parquet reader is not implemented yet");
  }

  const velox::RowTypePtr& rowType() const override {
    return readerBase_->getSchema();
  }

  const std::shared_ptr<const TypeWithId>& typeWithId() const override {
    return readerBase_->getSchemaWithId();
  }

  std::unique_ptr<RowReader> createRowReader(
      const RowReaderOptions& options = {}) const override {
    VELOX_NYI("ParquetRowReader will be merged later");
  }

 private:
  std::shared_ptr<ReaderBase> readerBase_;
};

////////////////////////////////////////////////////////////////////////////////

class ParquetReaderFactory : public ReaderFactory {
 public:
  ParquetReaderFactory() : ReaderFactory(FileFormat::PARQUET) {}

  std::unique_ptr<Reader> createReader(
      std::unique_ptr<InputStream> stream,
      const ReaderOptions& options) override {
    return std::make_unique<ParquetReader>(std::move(stream), options);
  }
};

} // namespace facebook::velox::parquet