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

#include "velox/dwio/cte/reader/CtePageReader.h"

namespace facebook::velox::pagefile {
using namespace facebook::velox::common;
using namespace dwio::common;

/// Metadata and options for reading Parquet.
class ReaderBase {
 public:
  ReaderBase(
      std::unique_ptr<dwio::common::BufferedInput>,
      const dwio::common::ReaderOptions& options);

  virtual ~ReaderBase() = default;

  memory::MemoryPool& getMemoryPool() const {
    return pool_;
  }

  FileInputStream* fileInputStream() const {
    return input_.get();
  }

  const dwio::common::ReaderOptions& getReaderOptions() const {
    return options_;
  }

 private:
  memory::MemoryPool& pool_;
  // Copy of options. Must be owned by 'this'.
  const dwio::common::ReaderOptions options_;
  std::unique_ptr<FileInputStream> input_;
};

ReaderBase::ReaderBase(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const dwio::common::ReaderOptions& options)
    : pool_{options.memoryPool()}, options_{options} {
  input_ = std::make_unique<facebook::velox::common::FileInputStream>(
      input->getReadFile(), 1 << 20, &(options.memoryPool()));
}

CtePageReader::CtePageReader(
    const dwio::common::ReaderOptions& options,
    std::unique_ptr<dwio::common::BufferedInput> input)
    : readerBase_(std::make_shared<ReaderBase>(std::move(input), options)),
      options_(options){};

std::optional<uint64_t> CtePageReader::numberOfRows() const {
  return readerBase_->fileInputStream()->size();
}

std::unique_ptr<RowReader> CtePageReader::createRowReader(
    const RowReaderOptions& opts) const {
  return std::make_unique<CtePageRowReader>(readerBase_, opts);
}

std::unique_ptr<CtePageReader> CtePageReader::create(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const dwio::common::ReaderOptions& options) {
  return std::make_unique<CtePageReader>(options, std::move(input));
}

CtePageRowReader::CtePageRowReader(
    const std::shared_ptr<ReaderBase>& readerBase,
    const dwio::common::RowReaderOptions& options)
    : readerBase_{readerBase},
      options_{options},
      schema_{readerBase_->getReaderOptions().fileSchema()},
      serde_(getNamedVectorSerde(VectorSerde::Kind::kPresto)){};

uint64_t CtePageRowReader::next(
    uint64_t size,
    velox::VectorPtr& result,
    const dwio::common::Mutation*) {
  if (readerBase_->fileInputStream()->atEnd()) {
    return 0;
  }
  auto rowVector = std::dynamic_pointer_cast<RowVector>(result);
  VectorStreamGroup::read(
      readerBase_->fileInputStream(),
      &(readerBase_->getMemoryPool()),
      schema_,
      serde_,
      &(rowVector),
      &readOptions_);
  return size;
}

void registerCtePageReaderFactory() {
  dwio::common::registerReaderFactory(std::make_shared<CtePageReaderFactory>());
}

void unregisterCtePageReaderFactory() {
  dwio::common::unregisterReaderFactory(dwio::common::FileFormat::PAGEFILE);
}
} // namespace facebook::velox::pagefile
