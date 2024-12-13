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

#include "velox/experimental/cudf/connectors/parquet/ParquetConnector.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"

namespace facebook::velox::cudf_velox::connector::parquet {

ParquetConnector::ParquetConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config,
    folly::Executor* executor)
    : Connector(id),
      parquetConfig_(std::make_shared<ParquetConfig>(config)),
      executor_(executor) {
  LOG(INFO) << "cudf::Parquet connector " << connectorId() << " created.";
}

std::unique_ptr<DataSource> ParquetConnector::createDataSource(
    const std::shared_ptr<const RowType>& outputType,
    const std::shared_ptr<ConnectorTableHandle>& tableHandle,
    const std::unordered_map<
        std::string,
        std::shared_ptr<connector::ColumnHandle>>& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) {
  return std::make_unique<ParquetDataSource>(
      outputType,
      tableHandle,
      columnHandles,
      parquetConfig_,
      executor,
      connectorQueryCtx->memoryPool());
}

std::shared_ptr<Connector> ParquetConnectorFactory::newConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config,
    folly::Executor* executor = nullptr) {
  return std::make_shared<ParquetConnector>(id, config, executor);
}

} // namespace facebook::velox::cudf_velox::connector::parquet
