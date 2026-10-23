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

#include "velox/connectors/hive/FileDataSource.h"
#include "velox/connectors/hive/paimon/PaimonConfig.h"
#include "velox/connectors/hive/paimon/PaimonTableHandle.h"

namespace facebook::velox::connector::hive::paimon {

class PaimonConnectorSplit;

/// Paimon-specific data source that extends FileDataSource.
///
/// Consumes PaimonTableHandle's fixed table definition and enforces the current
/// same-schema append reader limits. Generic query binding stays in
/// FileDataSource.
class PaimonDataSource : public FileDataSource {
 public:
  PaimonDataSource(
      const RowTypePtr& outputType,
      const std::shared_ptr<const PaimonTableHandle>& tableHandle,
      const ColumnHandleMap& assignments,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<PaimonConfig>& paimonConfig);

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

  void setFromDataSource(std::unique_ptr<DataSource> sourceUnique) override;

 protected:
  /// Creates a PaimonSplitReader with all data files from the Paimon split.
  /// Called by FileDataSource::addSplit() during split initialization.
  std::unique_ptr<FileSplitReader> createSplitReader() override;

 private:
  const std::shared_ptr<const PaimonTableHandle> paimonTableHandle_;

  // The original Paimon split. Stored so createSplitReader() can access
  // the full list of data files.
  std::shared_ptr<PaimonConnectorSplit> paimonSplit_;
};

} // namespace facebook::velox::connector::hive::paimon
