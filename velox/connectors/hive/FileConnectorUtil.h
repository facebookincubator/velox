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

#include <folly/Executor.h>

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/dwio/common/Reader.h"

namespace facebook::velox::connector::hive {

class FileColumnHandle;
using FileColumnHandlePtr = std::shared_ptr<const FileColumnHandle>;
class FileTableHandle;
using FileTableHandlePtr = std::shared_ptr<const FileTableHandle>;
class FileConfig;
struct FileConnectorSplit;

/// Configures reader options for reading a data file. This is the generic
/// version that does not apply serde (serialization/deserialization) options.
/// For Hive tables that need serde options, use the overload in
/// HiveConnectorUtil.h that accepts serdeParameters.
void configureReaderOptions(
    const std::shared_ptr<const FileConfig>& config,
    const ConnectorQueryCtx* connectorQueryCtx,
    const FileTableHandlePtr& tableHandle,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    dwio::common::ReaderOptions& readerOptions);

void configureReaderOptions(
    const std::shared_ptr<const FileConfig>& fileConfig,
    const ConnectorQueryCtx* connectorQueryCtx,
    const RowTypePtr& fileSchema,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const std::unordered_map<std::string, std::string>& tableParameters,
    dwio::common::ReaderOptions& readerOptions);

/// Configures row reader options for reading rows from a data file. This is the
/// generic version that does not set serde parameters. For Hive tables that
/// need serde parameters, use the overload in HiveConnectorUtil.h.
void configureRowReaderOptions(
    const std::unordered_map<std::string, std::string>& tableParameters,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    const RowTypePtr& rowType,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const config::ConfigBase* sessionProperties,
    folly::Executor* ioExecutor,
    dwio::common::RowReaderOptions& rowReaderOptions);

/// Tests whether a file should be read based on partition key values and
/// column statistics. Returns true if the file passes all filters, false
/// if it can be skipped.
bool testFilters(
    const common::ScanSpec* scanSpec,
    const dwio::common::Reader* reader,
    const std::string& filePath,
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKey,
    const std::unordered_map<std::string, FileColumnHandlePtr>&
        partitionKeysHandle,
    bool asLocalTime);

} // namespace facebook::velox::connector::hive
