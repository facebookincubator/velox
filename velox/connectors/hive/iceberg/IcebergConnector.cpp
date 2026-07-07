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

#include <numeric>

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/iceberg/IcebergConfig.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"
#include "velox/connectors/hive/iceberg/IcebergDataSource.h"
#include "velox/connectors/hive/iceberg/IcebergDeletionVectorSink.h"
#include "velox/connectors/hive/iceberg/IcebergMergeSink.h"

namespace facebook::velox::connector::hive::iceberg {

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
    std::shared_ptr<const config::ConfigBase> config,
    folly::Executor* ioExecutor)
    : HiveConnector(id, config, ioExecutor),
      icebergConfig_(std::make_shared<IcebergConfig>(connectorConfig())) {
  registerIcebergInternalFunctions(icebergConfig_->functionPrefix());
}

std::unique_ptr<DataSource> IcebergConnector::createDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) {
  return std::make_unique<IcebergDataSource>(
      outputType,
      tableHandle,
      columnHandles,
      &fileHandleFactory_,
      ioExecutor_,
      connectorQueryCtx,
      hiveConfig_);
}

std::unique_ptr<DataSink> IcebergConnector::createDataSink(
    RowTypePtr inputType,
    ConnectorInsertTableHandlePtr connectorInsertTableHandle,
    ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy) {
  auto icebergInsertHandle = checkedPointerCast<const IcebergInsertTableHandle>(
      connectorInsertTableHandle);

  switch (icebergInsertHandle->writeKind()) {
    case IcebergInsertTableHandle::WriteKind::kData:
      return std::make_unique<IcebergDataSink>(
          inputType,
          icebergInsertHandle,
          connectorQueryCtx,
          commitStrategy,
          hiveConfig_,
          icebergConfig_);
    case IcebergInsertTableHandle::WriteKind::kDeletionVector:
      return std::make_unique<IcebergDeletionVectorSink>(
          inputType,
          icebergInsertHandle,
          connectorQueryCtx,
          commitStrategy,
          hiveConfig_);
    case IcebergInsertTableHandle::WriteKind::kMerge: {
      // The IcebergMergeProcessor (Layer 1) emits the convention:
      //   [target cols 0..N-1, operation TINYINT @N, row_id ROW @N+1,
      //    insert_from_update TINYINT @N+2].
      // Derive the channel indices from the handle's inputColumns count.
      const auto numTargetColumns = static_cast<column_index_t>(
          icebergInsertHandle->inputColumns().size());
      std::vector<column_index_t> targetColumnChannels(numTargetColumns);
      std::iota(targetColumnChannels.begin(), targetColumnChannels.end(), 0);
      const column_index_t operationChannel = numTargetColumns;
      const column_index_t rowIdChannel = numTargetColumns + 1;
      return std::make_unique<IcebergMergeSink>(
          inputType,
          icebergInsertHandle,
          connectorQueryCtx,
          commitStrategy,
          hiveConfig_,
          icebergConfig_,
          std::move(targetColumnChannels),
          operationChannel,
          rowIdChannel);
    }
    case IcebergInsertTableHandle::WriteKind::kPositionDelete:
      // V2 position-delete sink: not yet implemented in Velox. V2 DELETE
      // currently runs through the Java row-id-rewrite path on the
      // coordinator, so this dispatch arm is never hit in production
      // today. Wired up so the WriteKind enum is exhaustive and a future
      // V2 native port has a clear plug-in point. See
      // ~/.llms/plans/iceberg_v2_native_positional_delete_sink.plan.md.
      VELOX_NYI(
          "Iceberg V2 native position-delete sink is not implemented. "
          "V2 DELETE flows through the Java row-id-rewrite path; if "
          "this NYI fires, the planner unexpectedly routed a V2 delete "
          "through the native bridge.");
  }
  VELOX_UNREACHABLE(
      "Unhandled IcebergInsertTableHandle::WriteKind: {}",
      static_cast<int32_t>(icebergInsertHandle->writeKind()));
}

void IcebergConnector::registerSerDe() {
  IcebergFileNameGenerator::registerSerDe();
}

} // namespace facebook::velox::connector::hive::iceberg
