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

// Hashes a single value from a vector at the given index.
// Handles lazy vectors via loadedVector(). Returns 0 for null values.
uint64_t hashValue(const VectorPtr& vectorPtr, vector_size_t index) {
  const auto* vector = vectorPtr->loadedVector();
  if (vector->isNullAt(index)) {
    return 0;
  }

  auto type = vector->type();
  switch (type->kind()) { // NOLINT(clang-diagnostic-switch-enum)
    case TypeKind::BOOLEAN:
      return std::hash<bool>{}(
          vector->as<SimpleVector<bool>>()->valueAt(index));
    case TypeKind::TINYINT:
      return std::hash<int8_t>{}(
          vector->as<SimpleVector<int8_t>>()->valueAt(index));
    case TypeKind::SMALLINT:
      return std::hash<int16_t>{}(
          vector->as<SimpleVector<int16_t>>()->valueAt(index));
    case TypeKind::INTEGER:
      return std::hash<int32_t>{}(
          vector->as<SimpleVector<int32_t>>()->valueAt(index));
    case TypeKind::BIGINT:
      return std::hash<int64_t>{}(
          vector->as<SimpleVector<int64_t>>()->valueAt(index));
    case TypeKind::REAL:
      return std::hash<float>{}(
          vector->as<SimpleVector<float>>()->valueAt(index));
    case TypeKind::DOUBLE:
      return std::hash<double>{}(
          vector->as<SimpleVector<double>>()->valueAt(index));
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY: {
      auto stringView = vector->as<SimpleVector<StringView>>()->valueAt(index);
      return folly::hasher<std::string_view>{}(
          std::string_view(stringView.data(), stringView.size()));
    }
    case TypeKind::TIMESTAMP: {
      auto ts = vector->as<SimpleVector<Timestamp>>()->valueAt(index);
      return std::hash<int64_t>{}(ts.toNanos());
    }
    default:
      VELOX_NYI(
          "Equality delete hash not implemented for type: {}",
          type->toString());
  }
}

// Compares two values from vectors at given indices.
// Handles lazy vectors via loadedVector().
bool compareValues(
    const VectorPtr& leftPtr,
    vector_size_t leftIndex,
    const VectorPtr& rightPtr,
    vector_size_t rightIndex) {
  const auto* left = leftPtr->loadedVector();
  const auto* right = rightPtr->loadedVector();
  bool leftNull = left->isNullAt(leftIndex);
  bool rightNull = right->isNullAt(rightIndex);
  if (leftNull && rightNull) {
    return true;
  }
  if (leftNull || rightNull) {
    return false;
  }

  auto type = left->type();
  switch (type->kind()) { // NOLINT(clang-diagnostic-switch-enum)
    case TypeKind::BOOLEAN:
      return left->as<SimpleVector<bool>>()->valueAt(leftIndex) ==
          right->as<SimpleVector<bool>>()->valueAt(rightIndex);
    case TypeKind::TINYINT:
      return left->as<SimpleVector<int8_t>>()->valueAt(leftIndex) ==
          right->as<SimpleVector<int8_t>>()->valueAt(rightIndex);
    case TypeKind::SMALLINT:
      return left->as<SimpleVector<int16_t>>()->valueAt(leftIndex) ==
          right->as<SimpleVector<int16_t>>()->valueAt(rightIndex);
    case TypeKind::INTEGER:
      return left->as<SimpleVector<int32_t>>()->valueAt(leftIndex) ==
          right->as<SimpleVector<int32_t>>()->valueAt(rightIndex);
    case TypeKind::BIGINT:
      return left->as<SimpleVector<int64_t>>()->valueAt(leftIndex) ==
          right->as<SimpleVector<int64_t>>()->valueAt(rightIndex);
    case TypeKind::REAL:
      return left->as<SimpleVector<float>>()->valueAt(leftIndex) ==
          right->as<SimpleVector<float>>()->valueAt(rightIndex);
    case TypeKind::DOUBLE:
      return left->as<SimpleVector<double>>()->valueAt(leftIndex) ==
          right->as<SimpleVector<double>>()->valueAt(rightIndex);
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY: {
      auto leftValue = left->as<SimpleVector<StringView>>()->valueAt(leftIndex);
      auto rightValue =
          right->as<SimpleVector<StringView>>()->valueAt(rightIndex);
      return std::string_view(leftValue.data(), leftValue.size()) ==
          std::string_view(rightValue.data(), rightValue.size());
    }
    case TypeKind::TIMESTAMP:
      return left->as<SimpleVector<Timestamp>>()->valueAt(leftIndex) ==
          right->as<SimpleVector<Timestamp>>()->valueAt(rightIndex);
    default:
      VELOX_NYI(
          "Equality delete comparison not implemented for type: {}",
          type->toString());
  }
}

} // namespace

EqualityDeleteFileReader::EqualityDeleteFileReader(
    const IcebergDeleteFile& deleteFile,
    const std::vector<std::string>& equalityColumnNames,
    const std::vector<TypePtr>& equalityColumnTypes,
    const std::string& /*baseFilePath*/,
    FileHandleFactory* fileHandleFactory,
    const ConnectorQueryCtx* connectorQueryCtx,
    folly::Executor* executor,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    dwio::common::RuntimeStatistics& runtimeStats,
    const std::string& connectorId)
    : equalityColumnNames_(equalityColumnNames),
      equalityColumnTypes_(equalityColumnTypes),
      pool_(connectorQueryCtx->memoryPool()) {
  VELOX_CHECK(
      deleteFile.content == FileContent::kEqualityDeletes,
      "Expected equality delete file but got content type: {}",
      static_cast<int>(deleteFile.content));
  VELOX_CHECK_GT(deleteFile.recordCount, 0, "Empty equality delete file.");
  VELOX_CHECK(
      !equalityColumnNames_.empty(),
      "Equality delete file must specify at least one column.");
  VELOX_CHECK_EQ(
      equalityColumnNames_.size(),
      equalityColumnTypes_.size(),
      "Equality column names and types must have the same size.");

  // Build the file schema for the equality delete columns only.
  auto deleteFileSchema =
      ROW(std::vector<std::string>(equalityColumnNames_),
          std::vector<TypePtr>(equalityColumnTypes_));

  // Create a ScanSpec that reads only the equality delete columns.
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  for (size_t i = 0; i < equalityColumnNames_.size(); ++i) {
    scanSpec->addField(equalityColumnNames_[i], static_cast<int>(i));
  }

  auto deleteSplit = std::make_shared<HiveConnectorSplit>(
      connectorId,
      deleteFile.filePath,
      deleteFile.fileFormat,
      0,
      deleteFile.fileSizeInBytes);

  dwio::common::ReaderOptions deleteReaderOpts(pool_);
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
      for (const auto& colName : equalityColumnNames_) {
        auto idx = rowOutput->type()->as<TypeKind::ROW>().getChildIdx(colName);
        deleteColumnIndices_.push_back(static_cast<column_index_t>(idx));
      }
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

const std::vector<column_index_t>&
EqualityDeleteFileReader::resolveOutputColumnIndices(
    const RowVectorPtr& row) const {
  if (outputColumnIndices_.empty()) {
    const auto& rowType = row->type()->asRow();
    outputColumnIndices_.reserve(equalityColumnNames_.size());
    for (const auto& colName : equalityColumnNames_) {
      auto colIdx = rowType.getChildIdxIfExists(colName);
      VELOX_CHECK(colIdx.has_value(), "Column not found in row: {}", colName);
      outputColumnIndices_.push_back(static_cast<column_index_t>(*colIdx));
    }
  }
  return outputColumnIndices_;
}

uint64_t EqualityDeleteFileReader::hashRow(
    const RowVectorPtr& row,
    vector_size_t index,
    const std::vector<column_index_t>& colIndices) const {
  uint64_t hash = 0;

  for (auto colIdx : colIndices) {
    auto colHash = hashValue(row->childAt(colIdx), index);
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
            left->childAt(leftColIndices[i]),
            leftIndex,
            right->childAt(deleteColumnIndices_[i]),
            rightIndex)) {
      return false;
    }
  }
  return true;
}

} // namespace facebook::velox::connector::hive::iceberg
