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

#include "velox/experimental/cudf/connectors/parquet/ParquetConnectorSplit.h"

namespace facebook::velox::cudf_velox::connector::parquet {

std::string ParquetConnectorSplit::toString() const {
  return fmt::format("Parquet: {} {} - {}", filePath, start, length);
}

std::string ParquetConnectorSplit::getFileName() const {
  const auto i = filePath.rfind('/');
  return i == std::string::npos ? filePath : filePath.substr(i + 1);
}

// static
std::shared_ptr<ParquetConnectorSplit> ParquetConnectorSplit::create(
    const folly::dynamic& obj) {
  const auto connectorId = obj["connectorId"].asString();
  const auto filePath = obj["filePath"].asString();
  const auto fileFormat =
      dwio::common::toFileFormat(obj["fileFormat"].asString());
  const auto start = static_cast<uint64_t>(obj["start"].asInt());
  const auto length = static_cast<uint64_t>(obj["length"].asInt());

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
  for (const auto& [key, value] : obj["partitionKeys"].items()) {
    partitionKeys[key.asString()] = value.isNull()
        ? std::nullopt
        : std::optional<std::string>(value.asString());
  }

  std::unordered_map<std::string, std::string> customSplitInfo;
  for (const auto& [key, value] : obj["customSplitInfo"].items()) {
    customSplitInfo[key.asString()] = value.asString();
  }

  std::shared_ptr<std::string> extraFileInfo = obj["extraFileInfo"].isNull()
      ? nullptr
      : std::make_shared<std::string>(obj["extraFileInfo"].asString());

  std::unordered_map<std::string, std::string> infoColumns;
  for (const auto& [key, value] : obj["infoColumns"].items()) {
    infoColumns[key.asString()] = value.asString();
  }

  std::optional<FileProperties> properties = std::nullopt;
  const auto& propertiesObj = obj.getDefault("properties", nullptr);
  if (propertiesObj != nullptr) {
    properties = FileProperties{
        propertiesObj["fileSize"].isNull()
            ? std::nullopt
            : std::optional(propertiesObj["fileSize"].asInt()),
        propertiesObj["modificationTime"].isNull()
            ? std::nullopt
            : std::optional(propertiesObj["modificationTime"].asInt())};
  }

  return std::make_shared<ParquetConnectorSplit>(
      connectorId,
      filePath,
      fileFormat,
      start,
      length,
      customSplitInfo,
      extraFileInfo,
      splitWeight,
      infoColumns,
      properties);
}

} // namespace facebook::velox::cudf_velox::connector::parquet
