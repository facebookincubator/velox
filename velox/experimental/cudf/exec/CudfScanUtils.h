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

#include "velox/connectors/Connector.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"

namespace facebook::velox::cudf_velox {

/// Returns whether the scan of 'node' produces CudfVectors: its connector is a
/// cuDF Hive or Iceberg connector, cuDF supports its output type and the cuDF
/// data source accepts its table handle. Sets 'reason' when the scan stays on
/// the CPU. Shared by physical planning and the transitional driver adapters.
///
/// 'connectorQueryCtx' is the context the scan gives the connector, so that
/// the answer matches the data source the connector picks.
///
/// Throws if the scan runs on the GPU, 'queryConfig' sets legacy_cast and the
/// remaining filter has a cast that legacy_cast may change. The cuDF data
/// source folds the remaining filter without legacy_cast, and the connector
/// cannot see the setting to fall back.
bool isGpuTableScan(
    const core::TableScanNode& node,
    const velox::connector::ConnectorQueryCtx& connectorQueryCtx,
    const core::QueryConfig& queryConfig,
    std::string* reason = nullptr);

} // namespace facebook::velox::cudf_velox
