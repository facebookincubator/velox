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
#include "velox/connectors/hive/paimon/PaimonDataSource.h"

#include "velox/connectors/hive/paimon/PaimonSplitReader.h"

namespace facebook::velox::connector::hive::paimon {

PaimonDataSource::PaimonDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& assignments,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<HiveConfig>& hiveConfig)
    : HiveDataSource(
          outputType,
          tableHandle,
          assignments,
          fileHandleFactory,
          ioExecutor,
          connectorQueryCtx,
          hiveConfig) {}

void PaimonDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  paimonSplit_ = checkedPointerCast<PaimonConnectorSplit>(split);

  VELOX_CHECK(
      !paimonSplit_->dataFiles().empty(),
      "Paimon split must contain at least one data file.");

  if (!paimonSplit_->rawConvertible()) {
    VELOX_NYI(
        "Merge-on-read is not yet supported for Paimon splits with "
        "rawConvertible=false.");
  }

  // Convert the first data file to a HiveConnectorSplit for the parent.
  // HiveDataSource::addSplit() requires a HiveConnectorSplit — it sets split_,
  // calls createSplitReader() (which we override), and prepares the reader.
  // Per-file validation is handled by PaimonSplitReader.
  auto hiveSplit = PaimonSplitReader::toHiveConnectorSplit(
      *paimonSplit_, paimonSplit_->dataFiles()[0]);
  HiveDataSource::addSplit(std::move(hiveSplit));
}

std::unique_ptr<SplitReader> PaimonDataSource::createSplitReader() {
  return std::make_unique<PaimonSplitReader>(
      split_,
      paimonSplit_,
      hiveTableHandle_,
      &partitionKeys_,
      connectorQueryCtx_,
      hiveConfig_,
      readerOutputType_,
      ioStatistics_,
      ioStats_,
      fileHandleFactory_,
      ioExecutor_,
      scanSpec_);
}

} // namespace facebook::velox::connector::hive::paimon
