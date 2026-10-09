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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergConnector.h"
#include "velox/experimental/cudf/exec/CudfScanUtils.h"

#include "velox/connectors/ConnectorRegistry.h"

namespace facebook::velox::cudf_velox {

bool isGpuTableScan(const core::TableScanNode& node) {
  auto scanConnector = ::facebook::velox::connector::ConnectorRegistry::tryGet(
      node.tableHandle()->connectorId());
  return dynamic_cast<const connector::hive::CudfHiveConnector*>(
             scanConnector.get()) != nullptr ||
      dynamic_cast<const connector::hive::iceberg::CudfIcebergConnector*>(
          scanConnector.get()) != nullptr;
}

} // namespace facebook::velox::cudf_velox
