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
#pragma once

#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"

namespace facebook::velox::connector::hive::paimon {

/// Paimon-specific data source that extends HiveDataSource.
///
/// Handles PaimonConnectorSplit which may contain multiple data files
/// in a single logical split (one per LSM-tree level in a bucket).
/// Currently only supports append-only tables with rawConvertible=true.
///
/// Converts PaimonConnectorSplit to HiveConnectorSplit for the first data
/// file so that HiveDataSource machinery works. Multi-file iteration is
/// handled by PaimonSplitReader.
class PaimonDataSource : public HiveDataSource {
 public:
  PaimonDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& assignments,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<HiveConfig>& hiveConfig);

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

 protected:
  std::unique_ptr<SplitReader> createSplitReader() override;

 private:
  std::shared_ptr<PaimonConnectorSplit> paimonSplit_;
};

} // namespace facebook::velox::connector::hive::paimon
