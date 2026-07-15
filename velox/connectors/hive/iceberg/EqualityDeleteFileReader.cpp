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

#include "velox/connectors/hive/iceberg/EqualityDeleteFileReader.h"

#include "velox/common/base/BitUtil.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/dwio/common/ReaderFactory.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

struct ResolvedValue {
  const BaseVector* vector;
  vector_size_t index;
  bool isNull;
};

ResolvedValue valueAtPath(
    const RowVectorPtr& row,
    vector_size_t index,
    const std::vector<column_index_t>& path) {
  const BaseVector* current = row.get();
  auto currentIndex = index;

  for (const auto childIndex : path) {
    current = current->loadedVector();
    if (current->isNullAt(currentIndex)) {
      return {nullptr, 0, true};
    }

    const auto wrappedIndex = current->wrappedIndex(currentIndex);
    const auto* rowVector = current->wrappedVector()->as<RowVector>();
    current = rowVector->childAt(childIndex).get();
    currentIndex = wrappedIndex;
  }

  current = current->loadedVector();
  return {current, currentIndex, current->isNullAt(currentIndex)};
}

uint64_t hashValue(const ResolvedValue& value) {
  return value.isNull ? BaseVector::kNullHash
                      : value.vector->hashValueAt(value.index);
}

bool compareValues(const ResolvedValue& left, const ResolvedValue& right) {
  if (left.isNull || right.isNull) {
    return left.isNull && right.isNull;
  }
  return left.vector->equalValueAt(right.vector, left.index, right.index);
}

} // namespace

EqualityDeleteFileReader::EqualityDeleteFileReader(
    const IcebergDeleteFile& deleteFile,
    const RowTypePtr& deleteFileSchema,
    const std::vector<EqualityDeleteFieldPath>& equalityFieldPaths,
    const std::string& /*baseFilePath*/,
    FileHandleFactory* fileHandleFactory,
    const ConnectorQueryCtx* connectorQueryCtx,
    folly::Executor* executor,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    dwio::common::RuntimeStatistics& runtimeStats,
    const std::string& connectorId)
    : equalityFieldPaths_(equalityFieldPaths),
      pool_(connectorQueryCtx->memoryPool()) {
  VELOX_CHECK(
      deleteFile.content == FileContent::kEqualityDeletes,
      "Expected equality delete file but got content type: {}",
      static_cast<int>(deleteFile.content));
  VELOX_CHECK_GT(deleteFile.recordCount, 0, "Empty equality delete file.");
  VELOX_CHECK(
      !equalityFieldPaths_.empty(),
      "Equality delete file must specify at least one field.");

  // Create a ScanSpec that reads only the equality delete columns.
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addAllChildFields(*deleteFileSchema);

  auto deleteSplit = std::make_shared<HiveConnectorSplit>(
      connectorId,
      deleteFile.filePath,
      deleteFile.fileFormat,
      0,
      deleteFile.fileSizeInBytes);

  dwio::common::ReaderOptions deleteReaderOpts(pool_);
  // TODO: Use separate IoStatistics for data and metadata.
  deleteReaderOpts.setDataIoStats(ioStatistics);
  deleteReaderOpts.setMetadataIoStats(ioStatistics);
  configureReaderOptions(
      fileConfig,
      connectorQueryCtx,
      deleteFileSchema,
      deleteSplit,
      /*tableParameters=*/{},
      deleteReaderOpts);

  const FileHandleKey fileHandleKey{
      .filename = deleteFile.filePath,
      .tokenProvider = connectorQueryCtx->fsTokenProvider()};
  auto deleteFileHandleCachePtr = fileHandleFactory->generate(fileHandleKey);
  auto deleteFileInput = BufferedInputBuilder::getInstance()->create(
      *deleteFileHandleCachePtr,
      deleteReaderOpts,
      connectorQueryCtx,
      ioStatistics,
      ioStats,
      executor);

  auto deleteReader =
      dwio::common::getReaderFactory(deleteReaderOpts.fileFormat())
          ->createReader(std::move(deleteFileInput), deleteReaderOpts);

  if (!testFilters(
          scanSpec.get(),
          deleteReader.get(),
          deleteSplit->filePath,
          deleteSplit->partitionKeys,
          {},
          fileConfig->readTimestampPartitionValueAsLocalTime(
              connectorQueryCtx->sessionProperties()))) {
    runtimeStats.skippedSplitBytes += static_cast<int64_t>(deleteSplit->length);
    return;
  }

  dwio::common::RowReaderOptions deleteRowReaderOpts;
  configureRowReaderOptions(
      {},
      scanSpec,
      nullptr,
      deleteFileSchema,
      deleteSplit,
      nullptr,
      nullptr,
      nullptr,
      deleteRowReaderOpts);

  auto deleteRowReader = deleteReader->createRowReader(deleteRowReaderOpts);

  // Read the entire equality delete file and build the hash set.
  VectorPtr output;
  output = BaseVector::create(deleteFileSchema, 0, pool_);

  while (true) {
    auto rowsRead = deleteRowReader->next(
        std::max(static_cast<uint64_t>(1'000), deleteFile.recordCount), output);
    if (rowsRead == 0) {
      break;
    }

    auto numRows = output->size();
    if (numRows == 0) {
      continue;
    }

    output->loadedVector();
    auto rowOutput = std::dynamic_pointer_cast<RowVector>(output);
    VELOX_CHECK_NOT_NULL(rowOutput);

    size_t batchIndex = deleteRows_.size();
    deleteRows_.push_back(rowOutput);

    // Resolve column indices on the first batch.
    if (deleteColumnIndices_.empty()) {
      deleteColumnIndices_ = resolveColumnIndices(rowOutput->type()->asRow());
    }

    // Hash each row and insert into the multimap.
    for (vector_size_t i = 0; i < numRows; ++i) {
      uint64_t hash = hashRow(rowOutput, i, deleteColumnIndices_);
      deleteKeyHashes_.emplace(hash, DeleteKeyEntry{batchIndex, i});
    }

    // Reset output for next batch.
    output = BaseVector::create(deleteFileSchema, 0, pool_);
  }
}

void EqualityDeleteFileReader::applyDeletes(
    const RowVectorPtr& output,
    BufferPtr deleteBitmap) {
  if (deleteKeyHashes_.empty() || output->size() == 0) {
    return;
  }

  auto* bitmap = deleteBitmap->asMutable<uint8_t>();

  // For each row in the output, compute its hash and probe the delete set.
  for (vector_size_t i = 0; i < output->size(); ++i) {
    // Skip rows already deleted by positional/DV deletes.
    if (bits::isBitSet(bitmap, i)) {
      continue;
    }

    uint64_t hash = hashRow(output, i, resolveOutputColumnIndices(output));
    auto range = deleteKeyHashes_.equal_range(hash);

    for (auto it = range.first; it != range.second; ++it) {
      auto& entry = it->second;
      if (equalRows(output, i, deleteRows_[entry.batchIndex], entry.rowIndex)) {
        bits::setBit(bitmap, i);
        break;
      }
    }
  }
}

std::vector<std::vector<column_index_t>>
EqualityDeleteFileReader::resolveColumnIndices(const RowType& rowType) const {
  std::vector<std::vector<column_index_t>> resolvedPaths;
  resolvedPaths.reserve(equalityFieldPaths_.size());

  for (const auto& fieldPath : equalityFieldPaths_) {
    const RowType* currentType = &rowType;
    std::vector<column_index_t> resolvedPath;
    resolvedPath.reserve(fieldPath.size());

    for (size_t i = 0; i < fieldPath.size(); ++i) {
      const auto childIndex = currentType->getChildIdx(fieldPath[i]);
      resolvedPath.push_back(static_cast<column_index_t>(childIndex));

      if (i + 1 < fieldPath.size()) {
        currentType = &currentType->childAt(childIndex)->asRow();
      }
    }
    resolvedPaths.push_back(std::move(resolvedPath));
  }
  return resolvedPaths;
}

const std::vector<std::vector<column_index_t>>&
EqualityDeleteFileReader::resolveOutputColumnIndices(
    const RowVectorPtr& row) const {
  if (outputColumnIndices_.empty()) {
    outputColumnIndices_ = resolveColumnIndices(row->type()->asRow());
  }
  return outputColumnIndices_;
}

uint64_t EqualityDeleteFileReader::hashRow(
    const RowVectorPtr& row,
    vector_size_t index,
    const std::vector<std::vector<column_index_t>>& colIndices) const {
  uint64_t hash = 0;

  for (const auto& colPath : colIndices) {
    auto colHash = hashValue(valueAtPath(row, index, colPath));
    hash ^= colHash + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  }
  return hash;
}

bool EqualityDeleteFileReader::equalRows(
    const RowVectorPtr& left,
    vector_size_t leftIndex,
    const RowVectorPtr& right,
    vector_size_t rightIndex) const {
  const auto& leftColIndices = resolveOutputColumnIndices(left);

  for (size_t i = 0; i < leftColIndices.size(); ++i) {
    if (!compareValues(
            valueAtPath(left, leftIndex, leftColIndices[i]),
            valueAtPath(right, rightIndex, deleteColumnIndices_[i]))) {
      return false;
    }
  }
  return true;
}

} // namespace facebook::velox::connector::hive::iceberg
