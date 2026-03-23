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

#include "velox/connectors/hive/HiveConnectorSplit.h"

namespace facebook::velox::connector::hive::paimon {

PaimonSplitReader::PaimonSplitReader(
    const std::shared_ptr<const HiveConnectorSplit>& hiveSplit,
    const std::shared_ptr<const PaimonConnectorSplit>& paimonSplit,
    const std::shared_ptr<const HiveTableHandle>& hiveTableHandle,
    const std::unordered_map<
        std::string,
        std::shared_ptr<const HiveColumnHandle>>* partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const std::shared_ptr<common::ScanSpec>& scanSpec)
    : SplitReader(
          hiveSplit,
          hiveTableHandle,
          partitionKeys,
          connectorQueryCtx,
          hiveConfig,
          readerOutputType,
          ioStatistics,
          ioStats,
          fileHandleFactory,
          executor,
          scanSpec),
      paimonSplit_(paimonSplit) {}

/* static */ std::shared_ptr<HiveConnectorSplit>
PaimonSplitReader::toHiveConnectorSplit(
    const PaimonConnectorSplit& paimonSplit,
    const PaimonDataFile& dataFile) {
  return std::make_shared<HiveConnectorSplit>(
      paimonSplit.connectorId,
      dataFile.path,
      paimonSplit.fileFormat(),
      /*_start=*/0,
      /*_length=*/dataFile.size,
      paimonSplit.partitionKeys(),
      paimonSplit.tableBucketNumber());
}

void PaimonSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStatistics& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& fileReadOps) {
  validateDataFile(paimonSplit_->dataFiles()[currentFileIndex_]);
  SplitReader::prepareSplit(
      std::move(metadataFilter), runtimeStats, fileReadOps);
}

uint64_t PaimonSplitReader::next(uint64_t size, VectorPtr& output) {
  auto numRows = SplitReader::next(size, output);
  if (numRows > 0) {
    return numRows;
  }

  // Current file exhausted. Advance to the next file if available.
  ++currentFileIndex_;
  if (currentFileIndex_ >= paimonSplit_->dataFiles().size()) {
    return 0;
  }

  openNextFile();
  return SplitReader::next(size, output);
}

/* static */ void PaimonSplitReader::validateDataFile(
    const PaimonDataFile& dataFile) {
  if (dataFile.deletionFile.has_value()) {
    VELOX_NYI("Paimon deletion vector support is not yet implemented.");
  }
  if (dataFile.type == PaimonDataFile::Type::kChangelog) {
    VELOX_NYI("Paimon changelog file reading is not yet implemented.");
  }
}

void PaimonSplitReader::openNextFile() {
  const auto& dataFile = paimonSplit_->dataFiles()[currentFileIndex_];
  validateDataFile(dataFile);

  // Replace the base class hiveSplit_ and reset readers for the new file.
  hiveSplit_ = toHiveConnectorSplit(*paimonSplit_, dataFile);
  baseRowReader_.reset();
  baseReader_.reset();
  emptySplit_ = false;

  createReader();
  if (emptySplit_) {
    return;
  }

  auto rowType = getAdaptedRowType();
  createRowReader(/*metadataFilter=*/nullptr, std::move(rowType), std::nullopt);
}

} // namespace facebook::velox::connector::hive::paimon
