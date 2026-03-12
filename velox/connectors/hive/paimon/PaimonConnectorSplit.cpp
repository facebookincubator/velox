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
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"

#include <fmt/format.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::connector::hive::paimon {

std::string paimonTableTypeString(PaimonTableType type) {
  switch (type) {
    case PaimonTableType::kAppendOnly:
      return "APPEND_ONLY";
    case PaimonTableType::kPrimaryKey:
      return "PRIMARY_KEY";
    default:
      VELOX_FAIL("Unknown PaimonTableType: {}", static_cast<int>(type));
  }
}

PaimonTableType paimonTableTypeFromString(const std::string& str) {
  if (str == "APPEND_ONLY") {
    return PaimonTableType::kAppendOnly;
  }
  if (str == "PRIMARY_KEY") {
    return PaimonTableType::kPrimaryKey;
  }
  VELOX_FAIL("Unknown PaimonTableType: {}", str);
}

PaimonConnectorSplit::PaimonConnectorSplit(
    const std::string& connectorId,
    int64_t snapshotId,
    PaimonTableType tableType,
    dwio::common::FileFormat fileFormat,
    const std::vector<PaimonDataFile>& dataFiles,
    std::unordered_map<std::string, std::optional<std::string>> partitionKeys,
    std::optional<int32_t> tableBucketNumber,
    bool rawConvertible)
    : ConnectorSplit(connectorId),
      snapshotId_(snapshotId),
      tableType_(tableType),
      fileFormat_(fileFormat),
      dataFiles_(dataFiles),
      partitionKeys_(std::move(partitionKeys)),
      tableBucketNumber_(tableBucketNumber),
      rawConvertible_(rawConvertible) {
  VELOX_CHECK(
      !dataFiles_.empty(), "PaimonConnectorSplit requires non-empty dataFiles");

  if (rawConvertible_) {
    for (const auto& file : dataFiles_) {
      VELOX_CHECK_EQ(
          file.deleteRowCount,
          0,
          "rawConvertible split cannot have files with deleteRowCount > 0: {}",
          file.toString());
    }
  }
}

std::string PaimonConnectorSplit::toString() const {
  std::string dataFilesStr;
  for (const auto& file : dataFiles_) {
    if (!dataFilesStr.empty()) {
      dataFilesStr += ", ";
    }
    dataFilesStr += file.toString();
  }

  return fmt::format(
      "PaimonConnectorSplit[snapshot {}, type {}, rawConvertible {}, "
      "connector '{}', dataFiles=[{}]]",
      snapshotId_,
      paimonTableTypeString(tableType_),
      rawConvertible_,
      connectorId,
      dataFilesStr);
}

folly::dynamic PaimonConnectorSplit::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "PaimonConnectorSplit";
  obj["connectorId"] = connectorId;
  obj["snapshotId"] = snapshotId_;
  obj["tableType"] = paimonTableTypeString(tableType_);
  obj["rawConvertible"] = rawConvertible_;

  folly::dynamic filesArray = folly::dynamic::array;
  for (const auto& file : dataFiles_) {
    filesArray.push_back(file.serialize());
  }
  obj["dataFiles"] = filesArray;

  folly::dynamic partitionKeysObj = folly::dynamic::object;
  for (const auto& [key, value] : partitionKeys_) {
    partitionKeysObj[key] =
        value.has_value() ? folly::dynamic(value.value()) : nullptr;
  }
  obj["partitionKeys"] = partitionKeysObj;

  obj["tableBucketNumber"] = tableBucketNumber_.has_value()
      ? folly::dynamic(tableBucketNumber_.value())
      : nullptr;

  obj["fileFormat"] = dwio::common::toString(fileFormat_);

  return obj;
}

// static
std::shared_ptr<PaimonConnectorSplit> PaimonConnectorSplit::create(
    const folly::dynamic& obj) {
  const auto connectorId = obj["connectorId"].asString();
  const auto snapshotId = obj["snapshotId"].asInt();
  const auto tableType = paimonTableTypeFromString(obj["tableType"].asString());
  const auto rawConvertible = obj["rawConvertible"].asBool();

  std::vector<PaimonDataFile> dataFiles;
  for (const auto& fileObj : obj["dataFiles"]) {
    dataFiles.emplace_back(PaimonDataFile::create(fileObj));
  }

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
  for (const auto& [key, value] : obj["partitionKeys"].items()) {
    partitionKeys[key.asString()] = value.isNull()
        ? std::nullopt
        : std::optional<std::string>(value.asString());
  }

  const auto tableBucketNumber = obj["tableBucketNumber"].isNull()
      ? std::nullopt
      : std::optional<int32_t>(obj["tableBucketNumber"].asInt());

  const auto fileFormat =
      dwio::common::toFileFormat(obj["fileFormat"].asString());

  return std::make_shared<PaimonConnectorSplit>(
      connectorId,
      snapshotId,
      tableType,
      fileFormat,
      dataFiles,
      std::move(partitionKeys),
      tableBucketNumber,
      rawConvertible);
}

// static
void PaimonConnectorSplit::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("PaimonConnectorSplit", PaimonConnectorSplit::create);
}

// --- Builder ---

PaimonConnectorSplitBuilder& PaimonConnectorSplitBuilder::addFile(
    std::string filePath,
    uint64_t fileSize,
    int32_t level) {
  PaimonDataFile meta;
  meta.path = std::move(filePath);
  meta.size = fileSize;
  meta.level = level;
  dataFiles_.emplace_back(std::move(meta));
  return *this;
}

PaimonConnectorSplitBuilder& PaimonConnectorSplitBuilder::partitionKey(
    std::string name,
    std::optional<std::string> value) {
  partitionKeys_.emplace(std::move(name), std::move(value));
  return *this;
}

PaimonConnectorSplitBuilder& PaimonConnectorSplitBuilder::tableBucketNumber(
    int32_t bucketId) {
  tableBucketNumber_ = bucketId;
  return *this;
}

PaimonConnectorSplitBuilder& PaimonConnectorSplitBuilder::rawConvertible(
    bool value) {
  rawConvertible_ = value;
  return *this;
}

std::shared_ptr<PaimonConnectorSplit> PaimonConnectorSplitBuilder::build() {
  return std::make_shared<PaimonConnectorSplit>(
      connectorId_,
      snapshotId_,
      tableType_,
      fileFormat_,
      dataFiles_,
      partitionKeys_,
      tableBucketNumber_,
      rawConvertible_);
}

} // namespace facebook::velox::connector::hive::paimon
