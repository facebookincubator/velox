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
#include "velox/connectors/hive/paimon/PaimonSplitReader.h"

#include "velox/connectors/hive/FileConfig.h"

namespace facebook::velox::connector::hive::paimon {

class PaimonSplitReader::PhysicalReader final : public FileSplitReader {
 public:
  PhysicalReader(const PaimonSplitReader& owner, const PaimonDataFile& file)
      : FileSplitReader(
            owner.fileSplit_,
            owner.table_,
            owner.partitionKeys_,
            owner.ctx_,
            owner.config_,
            owner.outputType_,
            owner.dataIoStats_,
            owner.metadataIoStats_,
            owner.ioStats_,
            owner.fileHandleFactory_,
            owner.executor_,
            owner.state_.scanSpec),
        file_(file),
        table_(owner.table_) {}

  void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStats& stats,
      const folly::F14FastMap<std::string, std::string>& ops) override {
    createReader(ops);
    VELOX_USER_CHECK_NOT_NULL(
        baseReader_, "Invalid Paimon data file: {}", file_.path);
    if (const auto rows = baseReader_->numberOfRows()) {
      VELOX_USER_CHECK_EQ(
          *rows,
          file_.rowCount,
          "Paimon rowCount disagrees with file footer: {}",
          file_.path);
    }
    const auto& schema = table_->targetSchema();
    const auto& physical = baseReader_->rowType();
    for (auto i = 0; i < schema.fieldIds.size(); ++i) {
      const auto& name = schema.rowType->nameOf(i);
      const auto index = physical->getChildIdxIfExists(name);
      const auto& partitionIds = table_->partitionFieldIds();
      const auto isPartition =
          std::find(
              partitionIds.begin(), partitionIds.end(), schema.fieldIds[i]) !=
          partitionIds.end();
      VELOX_USER_CHECK(
          index.has_value() || isPartition,
          "Paimon field '{}' missing from same-schema file '{}'",
          name,
          file_.path);
      if (index) {
        VELOX_USER_CHECK(
            physical->childAt(*index)->equivalent(*schema.rowType->childAt(i)),
            "Paimon physical type disagrees with schema for '{}' in '{}'",
            name,
            file_.path);
      }
    }
    auto adapted = getAdaptedRowType();
    if (!checkIfSplitIsEmpty(stats)) {
      createRowReader(
          std::move(metadataFilter), std::move(adapted), std::nullopt);
    }
  }

 protected:
  void configureBaseReaderOptions() override {
    FileSplitReader::configureBaseReaderOptions();
    // Schema identity, spelling and mapping are table-format semantics; session
    // settings for a generic file scan cannot silently change them.
    baseReaderOpts_.setColumnMappingMode(
        dwio::common::ColumnMappingMode::kName);
    baseReaderOpts_.setFileColumnNamesReadAsLowerCase(false);
    baseReaderOpts_.setAllowEmptyFile(false);
  }

  void validateFileSize(uint64_t actualSize) const override {
    VELOX_USER_CHECK_EQ(
        actualSize,
        file_.size,
        "Paimon fileSize disagrees with opened file: {}",
        file_.path);
  }

 private:
  const PaimonDataFile& file_;
  const std::shared_ptr<const PaimonTableHandle> table_;
};

PaimonSplitReader::PaimonSplitReader(
    std::shared_ptr<const PaimonConnectorSplit> split,
    std::shared_ptr<const PaimonTableHandle> table,
    std::unordered_map<std::string, std::optional<std::string>> partitionValues,
    const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
    const ConnectorQueryCtx* ctx,
    std::shared_ptr<const FileConfig> config,
    RowTypePtr outputType,
    std::shared_ptr<io::IoStatistics> dataIoStats,
    std::shared_ptr<io::IoStatistics> metadataIoStats,
    std::shared_ptr<IoStats> ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    dwio::common::RuntimeStats& stats,
    std::function<FileScanState()> makeScanState,
    folly::F14FastSet<std::string> remainingFilterColumns)
    : split_(std::move(split)),
      table_(std::move(table)),
      partitionValues_(std::move(partitionValues)),
      partitionKeys_(partitionKeys),
      ctx_(ctx),
      config_(std::move(config)),
      outputType_(std::move(outputType)),
      dataIoStats_(std::move(dataIoStats)),
      metadataIoStats_(std::move(metadataIoStats)),
      ioStats_(std::move(ioStats)),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      stats_(stats),
      makeScanState_(std::move(makeScanState)),
      remainingFilterColumns_(std::move(remainingFilterColumns)) {}

void PaimonSplitReader::openFile() {
  const auto& file = split_->dataFiles()[fileIndex_];
  fileSplit_ = std::make_shared<FileConnectorSplit>(
      split_->connectorId,
      file.path,
      file.fileFormat.value_or(split_->fileFormat()),
      0,
      file.size,
      0,
      split_->cacheable,
      file.properties,
      partitionValues_,
      dwio::common::ColumnMappingMode::kName);
  fileSplit_->physicalFilePath = file.physicalFilePath;
  state_ = makeScanState_();
  reader_ = std::make_unique<PhysicalReader>(*this, file);
  reader_->configureReaderOptions(nullptr);
  reader_->setRemainingFilterColumns(remainingFilterColumns_);
  const auto ops = file.properties
      ? file.properties->fileReadOps
      : folly::F14FastMap<std::string, std::string>{};
  reader_->prepareSplit(state_.metadataFilter, stats_, ops);
}

void PaimonSplitReader::finishFile() {
  if (reader_) {
    reader_->updateRuntimeStats(stats_);
    output_.reset();
    reader_.reset();
    state_ = {};
  }
  ++fileIndex_;
}

ScanReadResult PaimonSplitReader::next(
    uint64_t maxRows,
    ContinueFuture& /*future*/) {
  VELOX_CHECK_GT(maxRows, 0);
  if (fileIndex_ == split_->dataFiles().size()) {
    return {ScanReadResult::State::kEnd, nullptr};
  }
  if (!reader_) {
    openFile();
  }
  if (!reader_->emptySplit()) {
    if (!output_) {
      output_ = BaseVector::create(outputType_, 0, ctx_->memoryPool());
    }
    const auto scanned = reader_->next(maxRows, output_);
    if (scanned > 0) {
      return {
          ScanReadResult::State::kData,
          std::static_pointer_cast<RowVector>(output_),
          scanned};
    }
  }
  finishFile();
  // Bounded progress even if a split contains many empty or safely pruned
  // files.
  if (!emptyOutput_) {
    emptyOutput_ = RowVector::createEmpty(outputType_, ctx_->memoryPool());
  }
  return {ScanReadResult::State::kData, emptyOutput_};
}

void PaimonSplitReader::resetFilterCaches() {
  if (reader_) {
    reader_->resetFilterCaches();
  }
}

void PaimonSplitReader::addDynamicFilter(
    column_index_t channel,
    const std::shared_ptr<common::Filter>& filter) {
  if (reader_) {
    state_.scanSpec->getChildByChannel(channel).setFilter(filter);
    state_.scanSpec->resetCachedValues(true);
    reader_->resetFilterCaches();
  }
}

void PaimonSplitReader::updateRuntimeStats(
    dwio::common::RuntimeStats& stats) const {
  if (reader_) {
    reader_->updateRuntimeStats(stats);
  }
}

int64_t PaimonSplitReader::estimatedRowSize() const {
  return reader_ ? reader_->estimatedRowSize() : DataSource::kUnknownRowSize;
}

bool PaimonSplitReader::allPrefetchIssued() const {
  return fileIndex_ == split_->dataFiles().size() ||
      (fileIndex_ + 1 == split_->dataFiles().size() && reader_ &&
       reader_->allPrefetchIssued());
}

void PaimonSplitReader::resetSplit() {
  VELOX_CHECK_EQ(fileIndex_, split_->dataFiles().size());
  fileSplit_.reset();
}

void PaimonSplitReader::cancel() {
  // FileDataSource archives the active reader's statistics before cancellation.
  output_.reset();
  reader_.reset();
  state_ = {};
  fileIndex_ = split_->dataFiles().size();
  fileSplit_.reset();
}

void PaimonSplitReader::setConnectorQueryCtx(const ConnectorQueryCtx* /*ctx*/) {
  VELOX_UNSUPPORTED("Paimon split preload/state takeover is not supported");
}

const FileConnectorSplit* PaimonSplitReader::currentFileSplit() const {
  return fileSplit_.get();
}

} // namespace facebook::velox::connector::hive::paimon
