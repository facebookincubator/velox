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

/// Executes a complete, caller-planned Paimon split. The common data source
/// retains responsibility for expression evaluation and final projection.
class PaimonDataSource : public FileDataSource {
 public:
  PaimonDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& assignments,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<PaimonConfig>& paimonConfig);

  void setFromDataSource(std::unique_ptr<DataSource> source) override;

 protected:
  std::unique_ptr<FileScanReader> createScanReader() override;

 private:
  std::shared_ptr<const PaimonTableHandle> paimonTable_;
  std::optional<int64_t> snapshotId_;
};

} // namespace facebook::velox::connector::hive::paimon
