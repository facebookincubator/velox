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

#include "velox/connectors/hive/iceberg/IcebergSplitReader.h"

#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/dwio/common/BufferUtil.h"

using namespace facebook::velox::dwio::common;

namespace facebook::velox::connector::hive::iceberg {

IcebergSplitReader::IcebergSplitReader(
    std::shared_ptr<velox::connector::hive::HiveConnectorSplit> hiveSplit,
    std::shared_ptr<HiveTableHandle> hiveTableHandle,
    std::shared_ptr<common::ScanSpec> scanSpec,
    const RowTypePtr readerOutputType,
    std::unordered_map<std::string, std::shared_ptr<HiveColumnHandle>>*
        partitionKeys,
    FileHandleFactory* fileHandleFactory,
    ConnectorQueryCtx* connectorQueryCtx,
    folly::Executor* executor,
    std::shared_ptr<io::IoStatistics> ioStats)
    : SplitReader(
          hiveSplit,
          hiveTableHandle,
          scanSpec,
          readerOutputType,
          partitionKeys,
          fileHandleFactory,
          connectorQueryCtx,
          executor,
          ioStats) {}

void IcebergSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStatistics& runtimeStats) {
  SplitReader::prepareSplit(metadataFilter, runtimeStats);
  baseReadOffset_ = 0;
  deleteFileReaders_.clear();
  splitOffset_ = baseRowReader_->nextRowNumber();

  // TODO: Deserialize the std::vector<IcebergDeleteFile> deleteFiles. For now
  // we assume it's already deserialized.
  std::shared_ptr<HiveIcebergSplit> icebergSplit =
      std::dynamic_pointer_cast<HiveIcebergSplit>(hiveSplit_);

  const auto& deleteFiles = icebergSplit->deleteFiles;
  for (const auto& deleteFile : deleteFiles) {
    deleteFileReaders_.push_back(std::make_unique<DeleteFileReader>(
        deleteFile,
        hiveSplit_->filePath,
        fileHandleFactory_,
        executor_,
        connectorQueryCtx_,
        splitOffset_,
        hiveSplit_->connectorId));
  }
}

uint64_t IcebergSplitReader::next(int64_t size, VectorPtr& output) {
  Mutation mutation;
  mutation.deletedRows = nullptr;

  if (!deleteFileReaders_.empty()) {
    auto numBytes = bits::nbytes(size);
    dwio::common::ensureCapacity<int8_t>(
        deleteBitmap_, numBytes, connectorQueryCtx_->memoryPool());
    std::memset((void*)deleteBitmap_->as<int8_t>(), 0L, numBytes);

    for (auto iter = deleteFileReaders_.begin();
         iter != deleteFileReaders_.end();
         iter++) {
      (*iter)->readDeletePositions(
          baseReadOffset_, size, deleteBitmap_->asMutable<int8_t>());
      if ((*iter)->endOfFile()) {
        iter = deleteFileReaders_.erase(iter);
      }
    }

    deleteBitmap_->setSize(numBytes);
    mutation.deletedRows = deleteBitmap_->as<uint64_t>();
  }

  auto rowsScanned = baseRowReader_->next(size, output, &mutation);
  baseReadOffset_ += rowsScanned;

  return rowsScanned;
}

} // namespace facebook::velox::connector::hive::iceberg