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

#include "velox/connectors/hive/iceberg/IcebergConnector.h"

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"

namespace facebook::velox::connector::hive::iceberg {

const std::string_view kIcebergFunctionPrefixConfig{"presto.iceberg-namespace"};
const std::string_view kDefaultIcebergFunctionPrefix{"$internal$.iceberg."};

namespace {

// Registers Iceberg partition transform functions with prefix.
// NOTE: These functions are registered for internal transform usage only.
// Upstream engines such as Prestissimo and Gluten should register the same
// functions with different prefixes to avoid conflicts.
void registerIcebergInternalFunctions(const std::string& prefix) {
  static std::once_flag registerFlag;

  std::call_once(registerFlag, [prefix]() {
    functions::iceberg::registerFunctions(prefix);
  });
}

} // namespace

IcebergConnector::IcebergConnector(
    const std::string& id,
    config::ConfigPtr config,
    folly::Executor* ioExecutor)
    : HiveConnector(id, config, ioExecutor),
      functionPrefix_(config->get<std::string>(
          std::string(kIcebergFunctionPrefixConfig),
          std::string(kDefaultIcebergFunctionPrefix))) {
  registerIcebergInternalFunctions(functionPrefix_);
}

std::unique_ptr<DataSink> IcebergConnector::createDataSink(
    RowTypePtr inputType,
    ConnectorInsertTableHandlePtr connectorInsertTableHandle,
    ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy) {
  auto icebergInsertHandle = checkedPointerCast<const IcebergInsertTableHandle>(
      connectorInsertTableHandle);

  return std::make_unique<IcebergDataSink>(
      inputType,
      icebergInsertHandle,
      connectorQueryCtx,
      commitStrategy,
      hiveConfig_,
      functionPrefix_);
}

} // namespace facebook::velox::connector::hive::iceberg
