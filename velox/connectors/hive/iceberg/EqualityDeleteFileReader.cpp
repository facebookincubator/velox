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

#include <algorithm>

#include <folly/container/F14Map.h>

#include "velox/common/base/BitUtil.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/iceberg/IcebergGeometryConverter.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/dwio/common/ReaderFactory.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

std::optional<std::pair<const BaseVector*, vector_size_t>> valueAtSubfield(
    const RowVectorPtr& row,
    vector_size_t index,
    const common::Subfield& subfield) {
  const BaseVector* current = row.get();
  auto currentIndex = index;

  for (const auto& element : subfield.path()) {
    if (current->encoding() != VectorEncoding::Simple::ROW) {
      current = current->loadedVector();
    }
    if (current->isNullAt(currentIndex)) {
      return std::nullopt;
    }

    const auto wrappedIndex = current->wrappedIndex(currentIndex);
    const auto* rowVector = current->wrappedVector()->asChecked<RowVector>();
    const auto* nestedField =
        element->asChecked<common::Subfield::NestedField>();
    const auto childIndex =
        rowVector->type()->asRow().getChildIdx(nestedField->name());
    current = rowVector->childAt(childIndex).get();
    currentIndex = wrappedIndex;
  }

  current = current->loadedVector();
  if (current->isNullAt(currentIndex)) {
    return std::nullopt;
  }
  return std::pair{current, currentIndex};
}

} // namespace

EqualityDeleteFileReader::EqualityDeleteFileReader(
    const IcebergDeleteFile& deleteFile,
    const RowTypePtr& tableSchema,
    const std::vector<dwio::common::ParquetFieldId>& tableFieldIds,
    const std::vector<common::Subfield>& equalityFields,
    const std::string& /*baseFilePath*/,
    FileHandleFactory* fileHandleFactory,
    const ConnectorQueryCtx* connectorQueryCtx,
    folly::Executor* executor,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    dwio::common::RuntimeStats& runtimeStats,
    const std::string& connectorId)
    : pool_(connectorQueryCtx->memoryPool()) {
  equalityFields_.reserve(equalityFields.size());
  for (const auto& field : equalityFields) {
    equalityFields_.push_back(field.clone());
  }

  VELOX_CHECK(
      deleteFile.content == FileContent::kEqualityDeletes,
      "Expected equality delete file but got content type: {}",
      static_cast<int>(deleteFile.content));
  VELOX_CHECK_GT(deleteFile.recordCount, 0, "Empty equality delete file.");
  VELOX_CHECK(
      !equalityFields_.empty(),
      "Equality delete file must specify at least one field.");

  std::vector<std::string> deleteColumnNames;
  std::vector<TypePtr> deleteColumnTypes;
  std::vector<dwio::common::ParquetFieldId> deleteFieldIds;
  std::vector<const common::Subfield*> geometryFields;
  folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
      equalitySubfields;
  // Iceberg geometry support in this connector is Parquet-only, and an equality
  // delete file carries its own format independent of the base data file. The
  // WKB re-encoding below is format-agnostic, but no non-Parquet geometry
  // fixture exists to verify it against, so refuse rather than convert values
  // from an unverified path. This mirrors the base-file check in
  // IcebergSplitReader::prepareSplit().
  for (const auto& field : equalityFields_) {
    auto& subfields = equalitySubfields[field.baseName()];
    if (subfields.empty()) {
      deleteColumnNames.push_back(field.baseName());
      deleteColumnTypes.push_back(tableSchema->findChild(field.baseName()));
      if (!tableFieldIds.empty()) {
        deleteFieldIds.push_back(
            tableFieldIds.at(tableSchema->getChildIdx(field.baseName())));
      }
    }
    subfields.push_back(&field);

    TypePtr type = tableSchema;
    for (const auto& element : field.path()) {
      type = type->asRow().findChild(
          element->asChecked<common::Subfield::NestedField>()->name());
    }
    if (isGeometryType(type)) {
      VELOX_USER_CHECK_EQ(
          deleteFile.fileFormat,
          dwio::common::FileFormat::PARQUET,
          "Reading Iceberg geometry equality delete columns is only supported for Parquet delete files; column '{}'",
          field.toString());
      if (std::none_of(
              geometryFields.begin(),
              geometryFields.end(),
              [&](const auto* existing) { return *existing == field; })) {
        geometryFields.push_back(&field);
      }
    }
  }
  // Build the file schema for the equality delete columns.
  auto deleteFileSchema =
      ROW(std::move(deleteColumnNames), std::move(deleteColumnTypes));

  // Create a ScanSpec that reads only the equality delete columns.
  auto scanSpec = makeScanSpec(
      deleteFileSchema,
      equalitySubfields,
      /*subfieldFilters=*/{},
      /*dataColumns=*/tableSchema,
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      /*disableStatsBasedFilterReorder=*/false,
      pool_);

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
  deleteReaderOpts.setColumnMappingMode(dwio::common::ColumnMappingMode::kName);
  if (!deleteFieldIds.empty()) {
    deleteReaderOpts.setFieldIds(std::move(deleteFieldIds));
    if (deleteFile.fileFormat == dwio::common::FileFormat::PARQUET) {
      deleteReaderOpts.setColumnMappingMode(
          dwio::common::ColumnMappingMode::kParquetFieldId);
    } else if (
        deleteFile.fileFormat == dwio::common::FileFormat::ORC ||
        deleteFile.fileFormat == dwio::common::FileFormat::DWRF) {
      deleteReaderOpts.setColumnMappingMode(
          dwio::common::ColumnMappingMode::kFieldId);
      deleteReaderOpts.setUseColumnNamesForMissingFieldIds(true);
    }
  }

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

    // A geometry equality-delete column arrives from the delete file as the
    // ISO WKB the Iceberg spec mandates, but the base rows this set is probed
    // with have already been re-encoded into Velox's internal geometry
    // encoding by IcebergSplitReader. Hashing the two representations would
    // never collide, so geometry equality deletes would silently never match.
    // Re-encode the delete keys with the same converter the base path uses, so
    // both sides hash the same logical encoding.
    convertGeometryColumns(rowOutput, geometryFields);

    size_t batchIndex = deleteRows_.size();
    deleteRows_.push_back(rowOutput);

    // Hash each row and insert into the multimap.
    for (vector_size_t i = 0; i < numRows; ++i) {
      uint64_t hash = hashRow(rowOutput, i);
      deleteKeyHashes_.emplace(hash, DeleteKeyEntry{batchIndex, i});
    }

    // Reset output for next batch.
    output = BaseVector::create(deleteFileSchema, 0, pool_);
  }
}

void EqualityDeleteFileReader::convertGeometryColumns(
    const RowVectorPtr& deleteRows,
    const std::vector<const common::Subfield*>& geometryFields) const {
  for (const auto* field : geometryFields) {
#ifdef VELOX_ENABLE_GEO
    auto* row = deleteRows.get();
    SelectivityVector rows(row->size());
    const auto& path = field->path();
    for (size_t i = 0; i < path.size(); ++i) {
      if (row->rawNulls()) {
        rows.deselectNulls(row->rawNulls(), 0, row->size());
      }
      const auto& name =
          path[i]->asChecked<common::Subfield::NestedField>()->name();
      auto& child = row->childAt(row->type()->asRow().getChildIdx(name));
      if (i + 1 == path.size()) {
        child = convertIcebergGeometry(
            child, GEOMETRY(), rows, deleteRows->pool(), field->toString());
      } else {
        if (child->encoding() != VectorEncoding::Simple::ROW) {
          BaseVector::flattenVector(child);
        }
        row = child->asChecked<RowVector>();
      }
    }
#else
    // Matches the read-side guard in IcebergSplitReader::prepareSplit(): a
    // geospatial-free build cannot re-encode WKB, and hashing the two sides in
    // different encodings would silently drop every delete.
    VELOX_USER_FAIL(
        "Applying an Iceberg equality delete on the geometry column '{}' requires a build with geospatial support (VELOX_ENABLE_GEO=ON)",
        field->toString());
#endif
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

    uint64_t hash = hashRow(output, i);
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

uint64_t EqualityDeleteFileReader::hashRow(
    const RowVectorPtr& row,
    vector_size_t index) const {
  uint64_t hash = 0;

  for (const auto& field : equalityFields_) {
    const auto value = valueAtSubfield(row, index, field);
    const auto colHash = value.has_value()
        ? value->first->hashValueAt(value->second)
        : BaseVector::kNullHash;
    hash ^= colHash + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  }
  return hash;
}

bool EqualityDeleteFileReader::equalRows(
    const RowVectorPtr& left,
    vector_size_t leftIndex,
    const RowVectorPtr& right,
    vector_size_t rightIndex) const {
  for (const auto& field : equalityFields_) {
    const auto leftValue = valueAtSubfield(left, leftIndex, field);
    const auto rightValue = valueAtSubfield(right, rightIndex, field);
    if (leftValue.has_value() != rightValue.has_value()) {
      return false;
    }
    if (leftValue.has_value() &&
        !leftValue->first->equalValueAt(
            rightValue->first, leftValue->second, rightValue->second)) {
      return false;
    }
  }
  return true;
}

} // namespace facebook::velox::connector::hive::iceberg
