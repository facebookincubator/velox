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

// Nimble exposes physical types; logical annotations such as DATE use the
// query type. Nested fields are unchanged in this top-level evolution reader.
bool hasSamePhysicalType(const Type& fileType, const Type& queryType) {
  if (fileType.kind() != queryType.kind() ||
      fileType.size() != queryType.size()) {
    return false;
  }
  if (fileType.isRow() &&
      fileType.asRow().names() != queryType.asRow().names()) {
    return false;
  }
  for (column_index_t i = 0; i < fileType.size(); ++i) {
    if (!hasSamePhysicalType(*fileType.childAt(i), *queryType.childAt(i))) {
      return false;
    }
  }
  return true;
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
  }
}

void PaimonSplitReader::configureBaseReaderOptions() {
  connector::FileSplitReader::configureBaseReaderOptions();
  if (fileSplit_->fileFormat != dwio::common::FileFormat::NIMBLE) {
    return;
  }
  const auto& table = static_cast<const PaimonTableHandle&>(*tableHandle_);
  std::vector<dwio::common::ParquetFieldId> fieldIds;
  fieldIds.reserve(table.fieldIds().size());
  for (auto id : table.fieldIds()) {
    fieldIds.push_back({id, {}});
  }
  baseReaderOpts_.setFileSchema(table.rowType());
  baseReaderOpts_.setFieldIds(std::move(fieldIds));
  baseReaderOpts_.setFieldIdAttributeKey("paimon.id");
  baseReaderOpts_.setColumnMappingMode(
      dwio::common::ColumnMappingMode::kTopLevelFieldId);
}

std::vector<TypePtr> PaimonSplitReader::adaptColumns(
    const RowTypePtr& fileType,
    const RowTypePtr& /*tableSchema*/) const {
  if (fileSplit_->fileFormat == dwio::common::FileFormat::NIMBLE) {
    // The reader has bound file IDs to query names while preserving physical
    // types. Validate before the common adapter substitutes output types.
    // TODO: Support nested field evolution and type conversions.
    const auto& table = static_cast<const PaimonTableHandle&>(*tableHandle_);
    for (const auto& spec : scanSpec_->children()) {
      if (spec->columnType() != common::ScanSpec::ColumnType::kRegular ||
          fileSplit_->partitionKeys.count(spec->fieldName())) {
        continue;
      }
      const auto index = fileType->getChildIdxIfExists(spec->fieldName());
      if (!index.has_value()) {
        // A missing field ID is NULL, not a name- or position-based fallback.
        continue;
      }
      const auto& physical = fileType->childAt(*index);
      const auto& query = table.rowType()->findChild(spec->fieldName());
      VELOX_USER_CHECK(
          hasSamePhysicalType(*physical, *query),
          "Paimon type evolution is not supported for field '{}': {} to {}",
          spec->fieldName(),
          physical->toString(),
          query->toString());
    }
  }
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
  while (true) {
    const bool newFile = baseReader_ == nullptr;
    if (!ensureFileSplitReader()) {
      return 0;
    }
    if (newFile && fileSplit_->fileFormat == dwio::common::FileFormat::NIMBLE) {
      // A previous batch may retain reader-private vectors in physical file
      // order. Do not reuse them for a file with a different column layout.
      output.reset();
    }
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
