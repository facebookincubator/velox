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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/tests/utils/CudfTpcdsQueryBuilder.h"

#include "velox/connectors/hive/HiveConnector.h"

// Forward-declare registerCudf/unregisterCudf to avoid pulling in ToCudf.h
// which has namespace issues when included from a non-CUDA translation unit
// inside the cudf_velox::exec::test namespace.
namespace facebook::velox::cudf_velox {
void registerCudf();
void unregisterCudf();
} // namespace facebook::velox::cudf_velox

namespace facebook::velox::cudf_velox::exec::test {

CudfTpcdsQueryBuilder::CudfTpcdsQueryBuilder(
    dwio::common::FileFormat format,
    folly::Executor* ioExecutor)
    : TpcdsQueryBuilder(format), ioExecutor_(ioExecutor) {}

void CudfTpcdsQueryBuilder::enableCudf() {
  cudf_velox::registerCudf();
  cudfEnabled_ = true;
}

void CudfTpcdsQueryBuilder::registerHiveConnector(
    const std::string& connectorId,
    folly::Executor* /*ioExecutor*/) {
  auto configValues = std::unordered_map<std::string, std::string>();
  configValues
      [connector::hive::CudfHiveConfig::kAllowMismatchedCudfHiveSchemas] =
          "true";
  auto properties =
      std::make_shared<const config::ConfigBase>(std::move(configValues));

  connector::hive::CudfHiveConnectorFactory factory;
  auto c = factory.newConnector(connectorId, properties, ioExecutor_);
  facebook::velox::connector::registerConnector(c);

  LOG(INFO) << "CudfTpcdsQueryBuilder: registered CudfHiveConnector under ID '"
            << connectorId << "'";
}

void CudfTpcdsQueryBuilder::shutdown() {
  if (cudfEnabled_) {
    cudf_velox::unregisterCudf();
    cudfEnabled_ = false;
  }
  // Call base to unregister the connector.
  TpcdsQueryBuilder::shutdown();
}

} // namespace facebook::velox::cudf_velox::exec::test
