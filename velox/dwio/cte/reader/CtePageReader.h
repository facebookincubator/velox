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
#include "velox/common/file/FileInputStream.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::pagefile {

class ReaderBase;
class CtePageReader : public dwio::common::Reader {
 public:
  /**
   * Constructor that lets the user specify reader options and input stream.
   */
  CtePageReader(
      const dwio::common::ReaderOptions& options,
      std::unique_ptr<dwio::common::BufferedInput> input);

  ~CtePageReader() override = default;

  std::optional<uint64_t> numberOfRows() const override;

  /**
   * Get statistics for a specified column.
   * @param index column index
   * @return column statisctics
   */
  std::unique_ptr<dwio::common::ColumnStatistics> columnStatistics(
      uint32_t nodeId) const override {
    return nullptr;
  }
  //
  const std::shared_ptr<const RowType>& rowType() const override {
    return options_.fileSchema();
  };

  const std::shared_ptr<const dwio::common::TypeWithId>& typeWithId()
      const override {
    if (!schemaWithId_) {
      if (options_.scanSpec()) {
        schemaWithId_ = dwio::common::TypeWithId::create(
            options_.fileSchema(), *options_.scanSpec());
      } else {
        schemaWithId_ = dwio::common::TypeWithId::create(options_.fileSchema());
      }
    }
    return schemaWithId_;
  }

  /**
   * Create row reader object to fetch the data.
   * @param options Row reader options describing the data to fetch
   * @return Row reader
   */
  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const dwio::common::RowReaderOptions& options = {}) const override;

  // /**
  //  * Create a reader to the for the pagefile.
  //  * @param input the stream to read
  //  * @param options the options for reading the file
  //  */
  static std::unique_ptr<CtePageReader> create(
      std::unique_ptr<dwio::common::BufferedInput> input,
      const dwio::common::ReaderOptions& options);

 private:
  std::shared_ptr<ReaderBase> readerBase_;
  const dwio::common::ReaderOptions options_;
  mutable std::shared_ptr<const dwio::common::TypeWithId> schemaWithId_;
};

/// Implements the RowReader interface for TEMP.
class CtePageRowReader : public dwio::common::RowReader {
 public:
  CtePageRowReader(
      const std::shared_ptr<ReaderBase>& readerBase,
      const dwio::common::RowReaderOptions& options);
  ~CtePageRowReader() override = default;

  int64_t nextRowNumber() override {
    return 0;
  };

  int64_t nextReadSize(uint64_t size) override {
    return 0;
  };

  uint64_t next(
      uint64_t size,
      velox::VectorPtr& result,
      const dwio::common::Mutation* = nullptr) override;

  void updateRuntimeStats(
      dwio::common::RuntimeStatistics& stats) const override{};

  void resetFilterCaches() override{};

  std::optional<size_t> estimatedRowSize() const override {
    return std::nullopt;
  };

  bool allPrefetchIssued() const override {
    //  Allow opening the next split while this is reading.
    return true;
  }

 protected:
  std::shared_ptr<ReaderBase> readerBase_;
  dwio::common::RowReaderOptions options_;
  RowTypePtr schema_;
  const serializer::presto::PrestoVectorSerde::PrestoOptions readOptions_;
  VectorSerde* serde_;
};

class CtePageReaderFactory : public dwio::common::ReaderFactory {
 public:
  CtePageReaderFactory() : ReaderFactory(dwio::common::FileFormat::PAGEFILE) {}

  std::unique_ptr<dwio::common::Reader> createReader(
      std::unique_ptr<dwio::common::BufferedInput> input,
      const dwio::common::ReaderOptions& options) override {
    return CtePageReader::create(std::move(input), options);
  }
};

} // namespace facebook::velox::pagefile
