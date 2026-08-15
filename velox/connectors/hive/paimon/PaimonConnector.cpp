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
#include "velox/connectors/hive/paimon/PaimonConnector.h"

#include "velox/connectors/hive/paimon/PaimonDataSource.h"

namespace facebook::velox::connector::hive::paimon {

PaimonConnector::PaimonConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config,
    folly::Executor* ioExecutor)
    : HiveConnector(id, config, ioExecutor),
      paimonConfig_(std::make_shared<PaimonConfig>(connectorConfig())) {}

std::unique_ptr<DataSource> PaimonConnector::createDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) {
  return std::make_unique<PaimonDataSource>(
      outputType,
      tableHandle,
      columnHandles,
      &fileHandleFactory_,
      ioExecutor_,
      connectorQueryCtx,
      paimonConfig_);
}

} // namespace facebook::velox::connector::hive::paimon
