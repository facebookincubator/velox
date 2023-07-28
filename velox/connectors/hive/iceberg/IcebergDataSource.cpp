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

#include "velox/connectors/hive/iceberg/IcebergDataSource.h"

#include "velox/connectors/hive/iceberg/IcebergMetadataColumn.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/dwio/common/BufferUtil.h"

using namespace facebook::velox::dwio::common;

namespace facebook::velox::connector::hive::iceberg {

HiveIcebergDataSource::HiveIcebergDataSource(
    const RowTypePtr& outputType,
    const std::shared_ptr<connector::ConnectorTableHandle>& tableHandle,
    const std::unordered_map<
        std::string,
        std::shared_ptr<connector::ColumnHandle>>&
        columnHandles, // output columns
    FileHandleFactory* fileHandleFactory,
    core::ExpressionEvaluator* expressionEvaluator,
    cache::AsyncDataCache* cache,
    const std::string& scanId,
    folly::Executor* executor,
    const dwio::common::ReaderOptions& options,
    std::shared_ptr<velox::connector::ConnectorSplit> split)
    : HiveDataSource(
          outputType,
          tableHandle,
          columnHandles,
          fileHandleFactory,
          expressionEvaluator,
          cache,
          scanId,
          executor,
          options,
          split),
      connectorId_(split->connectorId),
      deleteFileContentsOffset_(0),
      deleteRowsOffset_(0),
      numDeleteRowsInSplit_(0) {}

void HiveIcebergDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  readOffset_ = 0;
  deleteFileContents_.clear();
  deleteFileContentsOffset_ = 0;
  deleteRowsOffset_ = 0;
  numDeleteRowsInSplit_ = 0;

  HiveDataSource::addSplit(split);

  firstRowInSplit_ = rowReader_->nextRowNumber();

  // TODO: Deserialize the std::vector<IcebergDeleteFile> deleteFiles. For now
  // we assume it's already deserialized.
  std::shared_ptr<HiveIcebergSplit> icebergSplit =
      std::dynamic_pointer_cast<HiveIcebergSplit>(split);

  auto& deleteFiles = icebergSplit->deleteFiles;
  for (auto& deleteFile : deleteFiles) {
    if (deleteFile.content == FileContent::kEqualityDeletes) {
      readEqualityDeletes(deleteFile);
    } else if (deleteFile.content == FileContent::kPositionalDeletes) {
      readPositionalDeletes(deleteFile);
    }
  }
}

uint64_t HiveIcebergDataSource::readNext(uint64_t size) {
  // The equality deletes should have been put into filter in scanspec. Now we
  // just need to deal with positional deletes. The deleteFileContents_ contains
  // the deleted row numbers relative to the start of the file, and we need to
  // convert them into row numbers relative to the start of the split.
  Mutation mutation;
  mutation.deletedRows = nullptr;

  if (numDeleteRowsInSplit_ > 0) {
    auto numBytes = bits::nbytes(size);
    dwio::common::ensureCapacity<int8_t>(deleteBitmap_, numBytes, pool_);
    std::memset((void*)deleteBitmap_->as<int8_t>(), 0L, numBytes);

    while (deleteFileContentsOffset_ < deleteFileContents_.size()) {
      auto deleteFile = deleteFileContents_[deleteFileContentsOffset_];
      VectorPtr& posVector = deleteFile->childAt(1);
      const int64_t* deleteRows =
          posVector->as<FlatVector<int64_t>>()->rawValues();

      auto deleteRowsOffsetBefore = deleteRowsOffset_;
      while (deleteRowsOffset_ < posVector->size() &&
             deleteRows[deleteRowsOffset_] <
                 firstRowInSplit_ + readOffset_ + size) {
        bits::setBit(
            deleteBitmap_->asMutable<uint64_t>(),
            deleteRows[deleteRowsOffset_] - readOffset_ - firstRowInSplit_);
        deleteRowsOffset_++;
      }

      numDeleteRowsInSplit_ -= deleteRowsOffset_ - deleteRowsOffsetBefore;
      if (deleteRowsOffset_ == posVector->size()) {
        deleteFileContentsOffset_++;
        deleteRowsOffset_ = 0;
      } else {
        break;
      }
    }

    deleteBitmap_->setSize(numBytes);
    mutation.deletedRows = deleteBitmap_->as<uint64_t>();
  }

  auto rowsScanned = HiveDataSource::rowReader_->next(size, output_, &mutation);
  readOffset_ += rowsScanned;

  return rowsScanned;
}

void HiveIcebergDataSource::readPositionalDeletes(
    const IcebergDeleteFile& deleteFile) {
  auto filePathColumn = ICEBERG_DELETE_FILE_PATH_COLUMN();
  auto positionsColumn = ICEBERG_DELETE_FILE_POSITIONS_COLUMN();

  std::vector<std::string> deleteColumnNames(
      {filePathColumn->name, positionsColumn->name});
  std::vector<std::shared_ptr<const Type>> deleteColumnTypes(
      {filePathColumn->type, positionsColumn->type});

  RowTypePtr deleteRowType =
      ROW(std::move(deleteColumnNames), std::move(deleteColumnTypes));

  std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
      deleteColumnHandles;
  deleteColumnHandles[filePathColumn->name] =
      std::make_shared<HiveColumnHandle>(
          filePathColumn->name,
          HiveColumnHandle::ColumnType::kRegular,
          VARCHAR(),
          VARCHAR());
  deleteColumnHandles[positionsColumn->name] =
      std::make_shared<HiveColumnHandle>(
          positionsColumn->name,
          HiveColumnHandle::ColumnType::kRegular,
          BIGINT(),
          BIGINT());

  openDeletes(deleteFile, deleteRowType, deleteColumnHandles);
}

void HiveIcebergDataSource::readEqualityDeletes(
    const IcebergDeleteFile& deleteFile) {
  VELOX_NYI();
}

void HiveIcebergDataSource::openDeletes(
    const IcebergDeleteFile& deleteFile,
    const RowTypePtr& deleteRowType,
    const std::unordered_map<
        std::string,
        std::shared_ptr<connector::ColumnHandle>>& deleteColumnHandles) {
  size_t lastPathDelimiterPos = deleteFile.filePath.find_last_of('/');
  std::string deleteFileName = deleteFile.filePath.substr(
      lastPathDelimiterPos, deleteFile.filePath.size() - lastPathDelimiterPos);

  SubfieldFilters subfieldFilters = {};
  auto deleteTableHandle = std::make_shared<HiveTableHandle>(
      connectorId_, deleteFileName, false, std::move(subfieldFilters), nullptr);

  auto deleteSplit = std::make_shared<HiveConnectorSplit>(
      connectorId_,
      deleteFile.filePath,
      deleteFile.fileFormat,
      0,
      deleteFile.fileSizeInBytes);

  auto deleteFileDataSource = std::make_shared<HiveDataSource>(
      deleteRowType,
      deleteTableHandle,
      deleteColumnHandles,
      fileHandleFactory_,
      expressionEvaluator_,
      cache_,
      scanId_ + deleteFile.filePath,
      executor_,
      readerOpts_,
      deleteSplit);

  deleteFileDataSource->addSplit(deleteSplit);

  ContinueFuture blockingFuture(ContinueFuture::makeEmpty());
  uint64_t numDeleteRowsInFile = 0;
  while (true) {
    std::optional<RowVectorPtr> deletesResult =
        deleteFileDataSource->next(deleteFile.recordCount, blockingFuture);

    if (!deletesResult.has_value()) {
      return;
    }

    auto data = deletesResult.value();
    if (data) {
      if (data->size() > 0) {
        VELOX_CHECK(data->childrenSize() > 0);
        numDeleteRowsInFile += data->childAt(0)->size();

        data->loadedVector();
        deleteFileContents_.push_back(data);
      }
    } else {
      numDeleteRowsInSplit_ += numDeleteRowsInFile;
      VELOX_CHECK(
          numDeleteRowsInFile == deleteFile.recordCount,
          "Failed to read Iceberg delete file %s. Supposed to read %lld rows, actual read %lld rows",
          deleteFile.filePath,
          deleteFile.recordCount,
          numDeleteRowsInFile);

      return;
    }
  };
}

} // namespace facebook::velox::connector::hive::iceberg