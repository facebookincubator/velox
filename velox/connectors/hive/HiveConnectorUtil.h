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

#include "velox/connectors/common/FileConnectorUtil.h"

namespace facebook::velox::connector::hive {

void configureReaderOptions(
    const std::shared_ptr<const file::FileConfig>& config,
    const ConnectorQueryCtx* connectorQueryCtx,
    const file::FileTableHandlePtr& tableHandle,
    const std::shared_ptr<const file::FileConnectorSplit>& fileSplit,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    dwio::common::ReaderOptions& readerOptions);

void configureReaderOptions(
    const std::shared_ptr<const file::FileConfig>& fileConfig,
    const ConnectorQueryCtx* connectorQueryCtx,
    const RowTypePtr& fileSchema,
    const std::shared_ptr<const file::FileConnectorSplit>& fileSplit,
    const std::unordered_map<std::string, std::string>& tableParameters,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    dwio::common::ReaderOptions& readerOptions);

void configureRowReaderOptions(
    const std::unordered_map<std::string, std::string>& tableParameters,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    const RowTypePtr& rowType,
    const std::shared_ptr<const file::FileConnectorSplit>& fileSplit,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    const std::shared_ptr<const file::FileConfig>& fileConfig,
    const config::ConfigBase* sessionProperties,
    folly::Executor* ioExecutor,
    dwio::common::RowReaderOptions& rowReaderOptions);

} // namespace facebook::velox::connector::hive
