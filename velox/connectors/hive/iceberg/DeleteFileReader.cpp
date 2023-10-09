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

#include "velox/connectors/hive/iceberg/DeleteFileReader.h"

#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"

namespace facebook::velox::connector::hive::iceberg {
DeleteFileReader::DeleteFileReader(
    const IcebergDeleteFile& deleteFile,
    const std::string& baseFilePath,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    ConnectorQueryCtx* connectorQueryCtx,
    uint64_t splitOffset,
    const std::string& connectorId)
    : deleteFile_(deleteFile),
      baseFilePath_(baseFilePath),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      splitOffset_(splitOffset),
      deletePositionsVector_(nullptr),
      deletePositionsOffset_(-1),
      endOfFile_(false) {
  if (deleteFile.content == FileContent::kPositionalDeletes) {
    createPositionalDeleteDataSource(deleteFile, connectorId);
  } else if (deleteFile.content == FileContent::kEqualityDeletes) {
    VELOX_NYI("Iceberg equality delete files are not supported yet.");
  } else {
    VELOX_FAIL("Unrecogonized Iceberg delete file type: ", deleteFile.content);
  }
}

void DeleteFileReader::createPositionalDeleteDataSource(
    const IcebergDeleteFile& deleteFile,
    const std::string& connectorId) {
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

  size_t lastPathDelimiterPos = deleteFile.filePath.find_last_of('/');
  std::string deleteFileName = deleteFile.filePath.substr(
      lastPathDelimiterPos, deleteFile.filePath.size() - lastPathDelimiterPos);

  // TODO: Build filters on the path column: filePathColumn = baseFilePath_
  // TODO: Build filters on the positionsColumn:
  //  positionsColumn >= baseReadOffset_ + splitOffsetInFile
  SubfieldFilters subfieldFilters = {};

  auto deleteTableHandle = std::make_shared<HiveTableHandle>(
      connectorId, deleteFileName, false, std::move(subfieldFilters), nullptr);

  auto deleteSplit = std::make_shared<HiveConnectorSplit>(
      connectorId,
      deleteFile.filePath,
      deleteFile.fileFormat,
      0,
      deleteFile.fileSizeInBytes);

  deleteDataSource_ = std::make_unique<HiveDataSource>(
      deleteRowType,
      deleteTableHandle,
      deleteColumnHandles,
      fileHandleFactory_,
      executor_,
      connectorQueryCtx_);

  deleteDataSource_->addSplit(deleteSplit);
}

void DeleteFileReader::readDeletePositions(
    uint64_t baseReadOffset,
    uint64_t size,
    int8_t* deleteBitmap) {
  ContinueFuture blockingFuture(ContinueFuture::makeEmpty());
  // We are going to read to the row number up to the end of the batch. For the
  // same base file, the deleted rows are in ascending order in the same delete
  // file
  int64_t rowNumberUpperBound = splitOffset_ + baseReadOffset + size;

  // Finish unused delete positions from last batch
  if (deletePositionsVector_ &&
      deletePositionsOffset_ < deletePositionsVector_->size()) {
    readDeletePositionsVec(baseReadOffset, rowNumberUpperBound, deleteBitmap);

    if (readFinishedForBatch(rowNumberUpperBound)) {
      return;
    }
  }

  uint64_t numRowsToRead = std::min(size, deleteFile_.recordCount);
  while (true) {
    std::optional<RowVectorPtr> deletesResult =
        deleteDataSource_->next(numRowsToRead, blockingFuture);

    if (!deletesResult.has_value()) {
      return;
    }

    auto data = deletesResult.value();
    if (data) {
      if (data->size() > 0) {
        VELOX_CHECK(data->childrenSize() > 0);
        data->loadedVector();

        deletePositionsVector_ = data->childAt(1);
        deletePositionsOffset_ = 0;

        readDeletePositionsVec(
            baseReadOffset, rowNumberUpperBound, deleteBitmap);

        if (readFinishedForBatch(rowNumberUpperBound)) {
          return;
        }
      }
      continue;
    } else {
      endOfFile_ = true;
      return;
    }
  }
}

bool DeleteFileReader::endOfFile() {
  return endOfFile_;
}

void DeleteFileReader::readDeletePositionsVec(
    uint64_t baseReadOffset,
    int64_t rowNumberUpperBound,
    int8_t* deleteBitmap) {
  // Convert the positions in file into positions relative to the start of the
  // split.
  const int64_t* deletePositions =
      deletePositionsVector_->as<FlatVector<int64_t>>()->rawValues();
  int64_t offset = baseReadOffset + splitOffset_;
  while (deletePositionsOffset_ < deletePositionsVector_->size() &&
         deletePositions[deletePositionsOffset_] < rowNumberUpperBound) {
    bits::setBit(
        deleteBitmap, deletePositions[deletePositionsOffset_] - offset);
    deletePositionsOffset_++;
  }
}

bool DeleteFileReader::readFinishedForBatch(int64_t rowNumberUpperBound) {
  const int64_t* deletePositions =
      deletePositionsVector_->as<FlatVector<int64_t>>()->rawValues();
  if (deletePositionsOffset_ < deletePositionsVector_->size() &&
      deletePositions[deletePositionsOffset_] >= rowNumberUpperBound) {
    return true;
  }
  return false;
}

} // namespace facebook::velox::connector::hive::iceberg