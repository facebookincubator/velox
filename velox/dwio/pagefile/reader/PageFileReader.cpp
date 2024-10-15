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

#include "velox/dwio/pagefile/reader/PageFileReader.h"

#include <chrono>

#include "velox/dwio/common/OnDemandUnitLoader.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/dwio/common/exception/Exception.h"
#include "velox/vector/FlatVector.h"
#include <arrow/c/bridge.h>


namespace facebook::velox::pagefile {

void registerPageFileReaderFactory() {
  dwio::common::registerReaderFactory(std::make_shared<PageFileReaderFactory>());
}



ReaderBase::ReaderBase(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const dwio::common::ReaderOptions& options)
    : pool_{options.memoryPool()},
      footerEstimatedSize_{options.footerEstimatedSize()},
      filePreloadThreshold_{options.filePreloadThreshold()},
      options_{options},
      input_{std::move(input)},
      fileLength_{input_->getReadFile()->size()} {
  VLOG(0)<<"Initialize";
}

PageFileReader::PageFileReader(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const dwio::common::ReaderOptions& options
   )
  :readerBase_(std::make_shared<ReaderBase>(std::move(input), options)) {}


std::unique_ptr<dwio::common::RowReader> PageFileReader::createRowReader(
    const dwio::common::RowReaderOptions& options) const {
  return std::make_unique<PageFileRowReader>(readerBase_, options);
}

PageFileRowReader::PageFileRowReader(
    const std::shared_ptr<ReaderBase> readerBase,
    const dwio::common::RowReaderOptions& options):
arrowRandomFile_(std::make_shared<ArrowRandomAccessFile>(readerBase->bufferedInput())),
options_(options),
readerBase_(readerBase) {
  initializeIpcReader();
  initializeReadRange();
}

void PageFileRowReader::initializeIpcReader() {
  // Initialize the Arrow IPC Reader from ArrowDataBufferSource
  auto result = arrow::ipc::RecordBatchFileReader::Open(arrowRandomFile_);
  if (!result.ok()) {
    throw std::runtime_error("Failed to initialize Arrow IPC reader");
  }

  ipcReader_ = result.ValueOrDie();
}

std::unique_ptr<dwio::common::Reader> PageFileReader::create(
     std::unique_ptr<dwio::common::BufferedInput> input,
     const dwio::common::ReaderOptions& options) {
  return std::make_unique<PageFileReader>(std::move(input), options);
}

uint64_t PageFileRowReader::next(
      uint64_t size,
      velox::VectorPtr& result,
      const dwio::common::Mutation* mutation) {
  // Implement the row-reading logic using RecordBatchFileReader
  if (currentBatch_ == nullptr || currentRowIndex_ >= currentBatch_->num_rows()) {
    if (!loadNextPage()) {
      return 0;  // No more rows to read
    }
  }

  // convert arrow batch to velox
  ArrowArray c_array;
  ArrowSchema c_schema;
  auto status = arrow::ExportRecordBatch(*currentBatch_, &c_array, &c_schema);
  if (!status.ok()) {
    LOG(ERROR) << "Error exporting to ArrowArray and ArrowSchema: " << status.ToString();
    return false;
  }
  result = std::dynamic_pointer_cast<RowVector>(
            importFromArrowAsOwner(c_schema, c_array,  &readerBase_->getMemoryPool()));

  currentRowIndex_+=currentBatch_->num_rows();
  return currentBatch_->num_rows();  // Return number of rows read
}

void  PageFileRowReader::initializeReadRange() {
  // Use the dataStart from RowReaderOptions to seek to the correct position
  if (options_.getOffset() > 0) {
    arrowRandomFile_->Seek(options_.getOffset());
  }
}

bool PageFileRowReader::loadNextPage() {
  VLOG(0) << "ee " << ipcReader_->num_record_batches();
  auto result = ipcReader_->CountRows();
  VLOG(0) << "ff " << static_cast<int>(result.ValueOrDie());
  // auto& test = *ipcReader_;
  if (currentPageIndex_ >= ipcReader_->num_record_batches()) {
    return false;  // No more pages
  }

  auto result1 = ipcReader_->ReadRecordBatch(currentPageIndex_);
  if (!result1.ok()) {
    throw std::runtime_error("Failed to read record batch");
  }

  currentBatch_ = result1.ValueOrDie();
  currentRowIndex_ = 0;
  ++currentPageIndex_;
  return true;
}

} // namespace facebook::velox::pagefile
