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

#include "velox/connectors/hive/iceberg/IcebergDataSource.h"

#include <folly/String.h>

#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergFieldIdUtils.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/IcebergSplitReader.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

IcebergDataSource::IcebergDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& assignments,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<HiveConfig>& hiveConfig)
    : HiveDataSource(
          outputType,
          tableHandle,
          assignments,
          fileHandleFactory,
          ioExecutor,
          connectorQueryCtx,
          hiveConfig),
      columnHandles_(std::make_shared<ColumnHandleMap>(assignments)) {
  for (const auto& [name, columnHandle] : assignments) {
    if (!columnHandle) {
      continue;
    }

    auto icebergHandle =
        std::dynamic_pointer_cast<const IcebergColumnHandle>(columnHandle);
    if (!icebergHandle) {
      continue;
    }

    std::string lowerName = icebergHandle->name();
    folly::toLowerAscii(lowerName);
    const auto& field = icebergHandle->field();
    nameToFieldId_[lowerName] = field.fieldId;

    const auto dataType = icebergHandle->dataType();
    if (!dataType || field.children.empty()) {
      continue;
    }

    auto rowType = asRowType(dataType);
    if (!rowType) {
      continue;
    }

    extractNestedFieldIds(field, rowType, nameToFieldId_);
  }
}

std::unique_ptr<FileSplitReader> IcebergDataSource::createSplitReader() {
  prepareSplit();
  auto icebergSplit = checkedPointerCast<const HiveIcebergSplit>(split_);

  auto reader = std::make_unique<IcebergSplitReader>(
      icebergSplit,
      tableHandle_,
      &partitionKeys_,
      connectorQueryCtx_,
      fileConfig_,
      readerOutputType_,
      dataIoStats_,
      metadataIoStats_,
      ioStats_,
      fileHandleFactory_,
      ioExecutor_,
      scanSpec_,
      columnHandles_);

  if (!nameToFieldId_.empty()) {
    reader->setNameToFieldId(nameToFieldId_);
  }

  return reader;
}

} // namespace facebook::velox::connector::hive::iceberg
