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

#include "clp_s/TimestampPattern.hpp"

#include "velox/connectors/clp/ClpConnector.h"
#include "velox/connectors/clp/ClpDataSource.h"

namespace facebook::velox::connector::clp {

ClpConnector::ClpConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config)
    : Connector(id), config_(std::make_shared<ClpConfig>(std::move(config))) {}

std::unique_ptr<DataSource> ClpConnector::createDataSource(
    const RowTypePtr& outputType,
    const std::shared_ptr<ConnectorTableHandle>& tableHandle,
    const std::unordered_map<
        std::string,
        std::shared_ptr<connector::ColumnHandle>>& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) {
  return std::make_unique<ClpDataSource>(
      outputType,
      tableHandle,
      columnHandles,
      connectorQueryCtx->memoryPool(),
      config_);
}

std::unique_ptr<DataSink> ClpConnector::createDataSink(
    RowTypePtr inputType,
    std::shared_ptr<ConnectorInsertTableHandle> connectorInsertTableHandle,
    ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy) {
  VELOX_NYI("createDataSink for ClpConnector is not implemented!");
}

ClpConnectorFactory::ClpConnectorFactory()
    : ConnectorFactory(kClpConnectorName) {
  clp_s::TimestampPattern::init();
}

ClpConnectorFactory::ClpConnectorFactory(const char* connectorName)
    : ConnectorFactory(connectorName) {
  clp_s::TimestampPattern::init();
}

} // namespace facebook::velox::connector::clp
