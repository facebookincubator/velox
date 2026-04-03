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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergConnector.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDataSource.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/iceberg/IcebergConfig.h"
#include "velox/connectors/hive/iceberg/IcebergDataSource.h"
#include "velox/functions/iceberg/Register.h"

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

using namespace facebook::velox::connector;

namespace {

void registerIcebergInternalFunctions(const std::string& prefix) {
  static std::once_flag registerFlag;
  std::call_once(registerFlag, [prefix]() {
    functions::iceberg::registerFunctions(prefix);
  });
}

} // namespace

CudfIcebergConnector::CudfIcebergConnector(
    const std::string& id,
    std::shared_ptr<const facebook::velox::config::ConfigBase> config,
    folly::Executor* executor)
    : ::facebook::velox::connector::hive::HiveConnector(id, config, executor),
      cudfHiveConfig_(std::make_shared<CudfHiveConfig>(config)),
      icebergConfig_(
          std::make_shared<
              ::facebook::velox::connector::hive::iceberg::IcebergConfig>(
              connectorConfig())) {
  registerIcebergInternalFunctions(icebergConfig_->functionPrefix());
  VLOG(1) << "cuDF Iceberg connector created";
}

std::unique_ptr<DataSource> CudfIcebergConnector::createDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) {
  if (cudfIsRegistered()) {
    return std::make_unique<CudfIcebergDataSource>(
        outputType,
        tableHandle,
        columnHandles,
        &fileHandleFactory_,
        ioExecutor_,
        connectorQueryCtx,
        cudfHiveConfig_,
        hiveConfig_);
  }

  // Fall back to the CPU-based Iceberg data source.
  return std::make_unique<
      ::facebook::velox::connector::hive::iceberg::IcebergDataSource>(
      outputType,
      tableHandle,
      columnHandles,
      &fileHandleFactory_,
      ioExecutor_,
      connectorQueryCtx,
      hiveConfig_);
}

std::shared_ptr<Connector> CudfIcebergConnectorFactory::newConnector(
    const std::string& id,
    std::shared_ptr<const facebook::velox::config::ConfigBase> config,
    folly::Executor* ioExecutor,
    folly::Executor* /*cpuExecutor*/) {
  return std::make_shared<CudfIcebergConnector>(id, config, ioExecutor);
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
