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

#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"

#include "velox/connectors/hive/HiveConfig.h"

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_connector = ::facebook::velox::connector;
namespace velox_hive = ::facebook::velox::connector::hive;

/// Iceberg-specific data source that extends CudfHiveDataSource.
///
/// Delegates all base Hive/Parquet handling to CudfHiveDataSource and
/// only overrides addSplit() and next() to handle Iceberg-specific
/// split types and the CudfIcebergSplitReader.
class CudfIcebergDataSource : public CudfHiveDataSource {
 public:
  CudfIcebergDataSource(
      const RowTypePtr& outputType,
      const velox_connector::ConnectorTableHandlePtr& tableHandle,
      const velox_connector::ColumnHandleMap& columnHandles,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const velox_connector::ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
      const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig);

  ~CudfIcebergDataSource() override;

  void addSplit(
      std::shared_ptr<velox_connector::ConnectorSplit> split) override;

  std::optional<RowVectorPtr> next(uint64_t size, velox::ContinueFuture& future)
      override;

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override;

 private:
  std::shared_ptr<const velox_hive::HiveConfig> hiveConfig_;
  std::unique_ptr<class CudfIcebergSplitReader> splitReader_;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
