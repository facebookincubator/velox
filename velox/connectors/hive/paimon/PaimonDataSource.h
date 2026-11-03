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

/// Data source for reading Paimon tables. Reuses FileDataSource for projection
/// and filtering, and PaimonSplitReader to read the files in each logical
/// split.
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
  /// Creates a reader for the complete logical Paimon split.
  std::unique_ptr<FileSplitReader> createSplitReader() override;

 private:
  const std::shared_ptr<const PaimonTableHandle> paimonTableHandle_;

  // Retains the complete split while FileDataSource uses its first file.
  std::shared_ptr<PaimonConnectorSplit> paimonSplit_;
};

} // namespace facebook::velox::connector::hive::paimon
