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
#include "velox/connectors/paimon/PaimonSplitReader.h"

#include <limits>

namespace facebook::velox::connector::paimon {
namespace {

// Indexed by target-table column; nullopt means its field ID is absent from
// the file's historical schema. No physical-file ID attributes are involved.
std::vector<std::optional<column_index_t>> mapTableColumnsToFile(
    const std::vector<int32_t>& tableFieldIds,
    const std::vector<int32_t>& fileFieldIds) {
  std::unordered_map<int32_t, column_index_t> fileColumns;
  for (column_index_t i = 0; i < fileFieldIds.size(); ++i) {
    fileColumns.emplace(fileFieldIds[i], i);
  }

  std::vector<std::optional<column_index_t>> mapping(tableFieldIds.size());
  for (column_index_t i = 0; i < tableFieldIds.size(); ++i) {
    if (const auto it = fileColumns.find(tableFieldIds[i]);
        it != fileColumns.end()) {
      mapping[i] = it->second;
    }
  }
  return mapping;
}

} // namespace

PaimonSplitReader::PaimonSplitReader(
    const std::shared_ptr<const connector::FileConnectorSplit>& fileSplit,
    std::shared_ptr<const PaimonConnectorSplit> paimonSplit,
    const std::shared_ptr<const PaimonTableHandle>& tableHandle,
    const std::unordered_map<std::string, connector::FileColumnHandlePtr>*
        partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const connector::FileConfig>& fileConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const std::shared_ptr<common::ScanSpec>& scanSpec)
    : connector::FileSplitReader(
          fileSplit,
          tableHandle,
          partitionKeys,
          connectorQueryCtx,
          fileConfig,
          readerOutputType,
          dataIoStats,
          metadataIoStats,
          ioStats,
          fileHandleFactory,
          executor,
          scanSpec),
      paimonSplit_(std::move(paimonSplit)) {
  VELOX_CHECK_NOT_NULL(paimonSplit_);

  // Build FileConnectorSplits for all data files so we can switch between
  // them during multi-file iteration.
  const auto& dataFiles = paimonSplit_->dataFiles();
  fileSplits_.reserve(dataFiles.size());
  for (const auto& dataFile : dataFiles) {
    fileSplits_.emplace_back(makeFileConnectorSplit(dataFile));
    if (!fileReadSchemas_.count(dataFile.schemaId)) {
      fileReadSchemas_.emplace(
          dataFile.schemaId, makeFileReadSchema(dataFile.schemaId));
    }
  }
}

PaimonSplitReader::FileReadSchema PaimonSplitReader::makeFileReadSchema(
    int64_t fileSchemaId) const {
  const auto& table = static_cast<const PaimonTableHandle&>(*tableHandle_);
  const auto& fileType = fileSchemaId == table.schemaId()
      ? table.rowType()
      : table.tableSchemas().at(fileSchemaId).rowType;
  const auto& fileFieldIds = fileSchemaId == table.schemaId()
      ? table.fieldIds()
      : table.tableSchemas().at(fileSchemaId).fieldIds;
  const auto tableToFileColumns =
      mapTableColumnsToFile(table.fieldIds(), fileFieldIds);
  std::vector<TypePtr> types = table.rowType()->children();
  std::unordered_map<std::string, std::string> columnNames;

  // Only top-level field evolution is handled here; nested fields are treated
  // as unchanged.
  // TODO: Support nested field evolution using historical nested field IDs.
  for (column_index_t i = 0; i < tableToFileColumns.size(); ++i) {
    const auto fileColumn = tableToFileColumns[i];
    if (!fileColumn) {
      continue;
    }
    const auto& name = table.rowType()->nameOf(i);
    columnNames.emplace(fileType->nameOf(*fileColumn), name);
    types[i] = fileType->childAt(*fileColumn);
    if (const auto* spec = scanSpec_->childByName(name);
        spec && spec->columnType() == common::ScanSpec::ColumnType::kRegular) {
      const auto& type = table.rowType()->childAt(i);
      VELOX_USER_CHECK(
          *type == *fileType->childAt(*fileColumn),
          "Paimon type evolution is not supported for field {} ('{}'), "
          "file schema {}, target schema {}: {} to {}",
          table.fieldIds()[i],
          name,
          fileSchemaId,
          table.schemaId(),
          fileType->childAt(*fileColumn)->toString(),
          type->toString());
    }
  }
  return {
      ROW(table.rowType()->names(), std::move(types)), std::move(columnNames)};
}

void PaimonSplitReader::configureBaseReaderOptions() {
  connector::FileSplitReader::configureBaseReaderOptions();
  const auto id = paimonSplit_->dataFiles()[currentFileIndex_].schemaId;
  const auto& schema = fileReadSchemas_.at(id);
  baseReaderOpts_.setFileSchema(schema.rowType);
  baseReaderOpts_.setFieldIdColumnNameMapping(schema.columnNameMapping);
  baseReaderOpts_.setColumnMappingMode(
      dwio::common::ColumnMappingMode::kFieldId);
}

std::vector<TypePtr> PaimonSplitReader::adaptColumns(
    const RowTypePtr& fileType,
    const RowTypePtr& /*tableSchema*/) const {
  return connector::FileSplitReader::adaptColumns(
      fileType, tableHandle_->dataColumns());
}

void PaimonSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStats& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& /*fileReadOps*/) {
  // Save for re-use when advancing to subsequent files.
  metadataFilter_ = std::move(metadataFilter);
  runtimeStats_ = &runtimeStats;
}

uint64_t PaimonSplitReader::next(uint64_t size, VectorPtr& output) {
  while (ensureFileSplitReader()) {
    // TODO: Apply deletion vectors using original physical file row positions.
    // TODO: Decode internal columns and row kinds for primary-key/changelog
    // files.
    const auto rowsRead = connector::FileSplitReader::next(size, output);
    if (rowsRead > 0) {
      return rowsRead;
    }
    // Current file exhausted, try the next one.
    finishFileSplitReader();
  }
  return 0;
}

bool PaimonSplitReader::ensureFileSplitReader() {
  if (baseReader_ != nullptr) {
    VELOX_CHECK_NOT_NULL(baseRowReader_);
    return true;
  }
  VELOX_CHECK_NULL(baseRowReader_);

  while (currentFileIndex_ < fileSplits_.size()) {
    // Point to the current file.
    fileSplit_ = fileSplits_[currentFileIndex_];

    configureReaderOptions(baseReaderOpts_.randomSkip());
    connector::FileSplitReader::prepareSplit(metadataFilter_, *runtimeStats_);
    if (!emptySplit_) {
      return true;
    }
    // Empty preparation may leave a file reader but no row reader. Reset
    // the per-file state before advancing so the next file is prepared.
    baseReader_.reset();
    emptySplit_ = false;
    ++currentFileIndex_;
  }
  return false; // All files exhausted.
}

void PaimonSplitReader::finishFileSplitReader() {
  VELOX_CHECK_LT(currentFileIndex_, fileSplits_.size());
  VELOX_CHECK_NOT_NULL(baseReader_);
  VELOX_CHECK_NOT_NULL(baseRowReader_);
  ++currentFileIndex_;
  baseReader_ = nullptr;
  baseRowReader_ = nullptr;
  emptySplit_ = false;
}

std::shared_ptr<connector::FileConnectorSplit>
PaimonSplitReader::makeFileConnectorSplit(
    const PaimonDataFile& dataFile) const {
  return std::make_shared<connector::FileConnectorSplit>(
      paimonSplit_->connectorId,
      dataFile.path,
      paimonSplit_->fileFormat(),
      /*_start=*/0,
      /*_length=*/std::numeric_limits<uint64_t>::max(),
      /*splitWeight=*/0,
      /*cacheable=*/true,
      /*_properties=*/std::nullopt,
      paimonSplit_->partitionKeys());
}

} // namespace facebook::velox::connector::paimon
