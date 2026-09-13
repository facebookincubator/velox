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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/Expressions.h"

#include <string_view>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;

namespace {

constexpr std::string_view kDateFormatName{"date_format"};

bool isSparkDateFormatCall(std::string_view functionName) {
  return functionName.size() >= kDateFormatName.size() &&
      functionName.compare(
          functionName.size() - kDateFormatName.size(),
          kDateFormatName.size(),
          kDateFormatName) == 0;
}

bool containsSparkDateFormatCall(const core::TypedExprPtr& expression) {
  if (expression == nullptr) {
    return false;
  }
  const auto call =
      std::dynamic_pointer_cast<const core::CallTypedExpr>(expression);
  if (call && isSparkDateFormatCall(call->name())) {
    return true;
  }
  for (const auto& input : expression->inputs()) {
    if (containsSparkDateFormatCall(input)) {
      return true;
    }
  }
  return false;
}

} // namespace

bool isCudfTableScanSupported(const ConnectorTableHandlePtr& tableHandle) {
  const auto hiveTableHandle = std::dynamic_pointer_cast<
      const facebook::velox::connector::hive::HiveTableHandle>(tableHandle);
  return hiveTableHandle != nullptr &&
      !containsSparkDateFormatCall(hiveTableHandle->remainingFilter());
}

CudfHiveConnector::CudfHiveConnector(
    const std::string& id,
    std::shared_ptr<const facebook::velox::config::ConfigBase> config,
    folly::Executor* executor)
    : ::facebook::velox::connector::hive::HiveConnector(id, config, executor),
      cudfHiveConfig_(std::make_shared<CudfHiveConfig>(config)) {
  VLOG(1) << "cuDF Hive connector created";
}

std::unique_ptr<DataSource> CudfHiveConnector::createDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) {
  // If it's parquet then return CudfHiveDataSource
  // If it's not parquet then return HiveDataSource
  // TODO (dm): Make this ^^^ happen
  // Problem: this information is in split, not table handle

  if (cudfIsRegistered() && isCudfTableScanSupported(tableHandle)) {
    return std::make_unique<CudfHiveDataSource>(
        outputType,
        tableHandle,
        columnHandles,
        &fileHandleFactory_,
        ioExecutor_,
        connectorQueryCtx,
        cudfHiveConfig_);
  }

  return std::make_unique<::facebook::velox::connector::hive::HiveDataSource>(
      outputType,
      tableHandle,
      columnHandles,
      &fileHandleFactory_,
      ioExecutor_,
      connectorQueryCtx,
      hiveConfig_);
}

// TODO (dm): Re-add data sink

std::shared_ptr<Connector> CudfHiveConnectorFactory::newConnector(
    const std::string& id,
    std::shared_ptr<const facebook::velox::config::ConfigBase> config,
    folly::Executor* ioExecutor,
    folly::Executor* cpuExecutor) {
  return std::make_shared<CudfHiveConnector>(id, config, ioExecutor);
}

} // namespace facebook::velox::cudf_velox::connector::hive
