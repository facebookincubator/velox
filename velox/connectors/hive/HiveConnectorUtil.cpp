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

#include "velox/connectors/hive/HiveConnectorUtil.h"

#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/FileTableHandle.h"

namespace facebook::velox::connector::hive {
namespace {

inline uint8_t parseDelimiter(const std::string& delim) {
  for (char const& ch : delim) {
    if (!std::isdigit(ch)) {
      return delim[0];
    }
  }
  return stoi(delim);
}

} // namespace

std::unique_ptr<dwio::common::SerDeOptions> parseSerdeParameters(
    const std::unordered_map<std::string, std::string>& serdeParameters,
    const std::unordered_map<std::string, std::string>& tableParameters) {
  auto fieldIt = serdeParameters.find(dwio::common::SerDeOptions::kFieldDelim);
  if (fieldIt == serdeParameters.end()) {
    fieldIt = serdeParameters.find("serialization.format");
  }
  auto collectionIt =
      serdeParameters.find(dwio::common::SerDeOptions::kCollectionDelim);
  if (collectionIt == serdeParameters.end()) {
    // For collection delimiter, Hive 1.x, 2.x uses "colelction.delim", but
    // Hive 3.x uses "collection.delim".
    // See: https://issues.apache.org/jira/browse/HIVE-16922)
    collectionIt = serdeParameters.find("colelction.delim");
  }
  auto mapKeyIt =
      serdeParameters.find(dwio::common::SerDeOptions::kMapKeyDelim);

  auto escapeCharIt =
      serdeParameters.find(dwio::common::SerDeOptions::kEscapeChar);

  auto nullStringIt = tableParameters.find(
      dwio::common::TableParameter::kSerializationNullFormat);

  if (fieldIt == serdeParameters.end() &&
      collectionIt == serdeParameters.end() &&
      mapKeyIt == serdeParameters.end() &&
      escapeCharIt == serdeParameters.end() &&
      nullStringIt == tableParameters.end()) {
    return nullptr;
  }

  uint8_t fieldDelim = '\1';
  uint8_t collectionDelim = '\2';
  uint8_t mapKeyDelim = '\3';
  // Treat an empty value as absent so parseDelimiter never sees "".
  if (fieldIt != serdeParameters.end() && !fieldIt->second.empty()) {
    fieldDelim = parseDelimiter(fieldIt->second);
  }
  if (collectionIt != serdeParameters.end() && !collectionIt->second.empty()) {
    collectionDelim = parseDelimiter(collectionIt->second);
  }
  if (mapKeyIt != serdeParameters.end() && !mapKeyIt->second.empty()) {
    mapKeyDelim = parseDelimiter(mapKeyIt->second);
  }

  // If escape character is specified then we use it, unless it is empty - in
  // which case we default to '\\'.
  // If escape character is not specified (not in the map) we turn escaping off.
  // Logic is based on apache hive java code:
  // https://github.com/apache/hive/blob/3f6f940af3f60cc28834268e5d7f5612e3b13c30/serde/src/java/org/apache/hadoop/hive/serde2/lazy/LazySerDeParameters.java#L105-L108
  uint8_t escapeChar = '\\';
  const bool hasEscapeChar = (escapeCharIt != serdeParameters.end());
  if (hasEscapeChar) {
    if (!escapeCharIt->second.empty()) {
      // If delim is convertible to uint8_t then we use it as character code,
      // otherwise we use the 1st character of the string.
      escapeChar = folly::tryTo<uint8_t>(escapeCharIt->second)
                       .value_or(escapeCharIt->second[0]);
    }
  }

  auto serDeOptions = hasEscapeChar
      ? std::make_unique<dwio::common::SerDeOptions>(
            fieldDelim, collectionDelim, mapKeyDelim, escapeChar, true)
      : std::make_unique<dwio::common::SerDeOptions>(
            fieldDelim, collectionDelim, mapKeyDelim);
  if (nullStringIt != tableParameters.end()) {
    serDeOptions->nullString = nullStringIt->second;
  }
  return serDeOptions;
}

void configureReaderOptions(
    const std::shared_ptr<const FileConfig>& fileConfig,
    const ConnectorQueryCtx* connectorQueryCtx,
    const FileTableHandlePtr& tableHandle,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    dwio::common::ReaderOptions& readerOptions) {
  configureReaderOptions(
      fileConfig,
      connectorQueryCtx,
      tableHandle->dataColumns(),
      fileSplit,
      tableHandle->tableParameters(),
      serdeParameters,
      readerOptions);
}

void configureReaderOptions(
    const std::shared_ptr<const FileConfig>& fileConfig,
    const ConnectorQueryCtx* connectorQueryCtx,
    const RowTypePtr& fileSchema,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const std::unordered_map<std::string, std::string>& tableParameters,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    dwio::common::ReaderOptions& readerOptions) {
  // Check if format was UNKNOWN before the generic call, since the generic
  // version will set it. Serde options should only be applied on the first
  // call (when format transitions from UNKNOWN).
  const bool formatWasUnknown =
      readerOptions.fileFormat() == dwio::common::FileFormat::UNKNOWN;

  // Call the generic version which handles everything except serde.
  // Use fully qualified call to the generic overload (6 params, no serde).
  configureReaderOptions(
      fileConfig,
      connectorQueryCtx,
      fileSchema,
      fileSplit,
      tableParameters,
      readerOptions);

  // Apply serde options on top (Hive-specific).
  if (formatWasUnknown) {
    auto serDeOptions = parseSerdeParameters(serdeParameters, tableParameters);
    if (serDeOptions) {
      readerOptions.setSerDeOptions(*serDeOptions);
    }
  }
}

void configureRowReaderOptions(
    const std::unordered_map<std::string, std::string>& tableParameters,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    const RowTypePtr& rowType,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const config::ConfigBase* sessionProperties,
    folly::Executor* const ioExecutor,
    dwio::common::RowReaderOptions& rowReaderOptions) {
  // Call the generic version which handles everything except serde.
  // Use fully qualified call to the generic overload (9 params, no serde).
  configureRowReaderOptions(
      tableParameters,
      scanSpec,
      std::move(metadataFilter),
      rowType,
      fileSplit,
      fileConfig,
      sessionProperties,
      ioExecutor,
      rowReaderOptions);

  // Apply serde parameters on top (Hive-specific).
  rowReaderOptions.setSerdeParameters(serdeParameters);
}

} // namespace facebook::velox::connector::hive
