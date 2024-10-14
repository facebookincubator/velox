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


PageFileReader::PageFileReader(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const dwio::common::ReaderOptions& options
   )
  : options_(options),
    input_(std::move(input))// assuming 'input_' is a member variable
{
  // Constructor body, if needed
}


std::unique_ptr<dwio::common::RowReader> PageFileReader::createRowReader(
    const dwio::common::RowReaderOptions& options) const override {
  return std::make_unique<PageFileRowReader>(input_, options_.memoryPool(), options);
}

PageFileRowReader::PageFileRowReader(
   std::unique_ptr<dwio::common::BufferedInput> input_,
   velox::memory::MemoryPool& pool_,
    const dwio::common::RowReaderOptions& options):
arrowRandomFile_(std::make_unique<ArrowRandomAccessFile>(std::move(input_))),
pool_(pool_),
options_(options) {
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

uint64_t PageFileRowReader::next(
      uint64_t size,
      velox::VectorPtr& result,
      const dwio::common::Mutation* mutation) {
  // Implement the row-reading logic using RecordBatchFileReader
  if (currentRowIndex_ >= currentBatch_->num_rows()) {
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
            importFromArrowAsOwner(c_schema, c_array,  &pool_));

  currentRowIndex_++;
  return currentBatch_->num_rows();  // Return number of rows read
}

void  PageFileRowReader::initializeReadRange() {
  // Use the dataStart from RowReaderOptions to seek to the correct position
  if (options_.getOffset() > 0) {
    arrowRandomFile_->Seek(options_.getOffset());
  }
}

bool PageFileRowReader::loadNextPage() {
  if (currentPageIndex_ >= ipcReader_->num_record_batches()) {
    return false;  // No more pages
  }

  auto result = ipcReader_->ReadRecordBatch(currentPageIndex_);
  if (!result.ok()) {
    throw std::runtime_error("Failed to read record batch");
  }

  currentBatch_ = result.ValueOrDie();
  currentRowIndex_ = 0;
  ++currentPageIndex_;
  return true;
}

} // namespace facebook::velox::pagefile
