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
#include "velox/connectors/hive/paimon/PaimonMetadata.h"

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
    bool rawConvertible,
    bool cacheable,
    PaimonPartition partitionValues,
    std::string readMode,
    int32_t wireVersion)
    : ConnectorSplit(connectorId, 0, cacheable),
      snapshotId_(snapshotId),
      tableType_(tableType),
      fileFormat_(fileFormat),
      dataFiles_(dataFiles),
      partitionKeys_(std::move(partitionKeys)),
      tableBucketNumber_(tableBucketNumber),
      rawConvertible_(rawConvertible),
      partitionValues_(std::move(partitionValues)),
      readMode_(std::move(readMode)),
      wireVersion_(wireVersion) {
  VELOX_CHECK(
      !dataFiles_.empty(), "PaimonConnectorSplit requires non-empty dataFiles");

  VELOX_USER_CHECK(
      wireVersion_ == 0 || wireVersion_ == 1,
      "Unsupported Paimon split wireVersion: {}",
      wireVersion_);
  VELOX_USER_CHECK_EQ(
      readMode_, "SNAPSHOT", "Paimon supports only SNAPSHOT reads");
  VELOX_USER_CHECK_GE(snapshotId_, 0, "Paimon snapshotId out of range");
  if (tableBucketNumber_) {
    VELOX_USER_CHECK_GE(
        *tableBucketNumber_,
        -1,
        "Paimon bucket out of range (only -1 is a sentinel)");
  }
  VELOX_USER_CHECK(
      partitionKeys_.empty() || partitionValues_.empty(),
      "Paimon partition must have exactly one encoding");
  for (const auto& [fieldId, value] : partitionValues_) {
    VELOX_USER_CHECK_GE(fieldId, 0, "Paimon partition fieldId out of range");
    VELOX_USER_CHECK(
        value.kind() == TypeKind::VARCHAR ||
            value.kind() == TypeKind::BOOLEAN ||
            value.kind() == TypeKind::INTEGER ||
            value.kind() == TypeKind::BIGINT,
        "Unsupported Paimon typed partition value");
  }
  for (const auto& file : dataFiles_) {
    file.validate();
    VELOX_USER_CHECK_LE(
        file.size,
        std::numeric_limits<int64_t>::max() - size_,
        "Paimon split size overflow");
    size_ += file.size;
    if (rawConvertible_ && file.deleteRowCount) {
      VELOX_USER_CHECK_EQ(
          *file.deleteRowCount,
          0,
          "rawConvertible split cannot have files with deleteRowCount > 0");
    }
    // Raw convertibility is a planner hint. Unknown deletion counts must reach
    // the capability check, where a KV fallback can later be selected.
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
  obj["wireVersion"] = wireVersion_;
  obj["splitType"] = "DATA_FILES";
  obj["readMode"] = readMode_;
  obj["cacheable"] = cacheable;
  folly::dynamic partitionValues = folly::dynamic::array;
  for (const auto& [id, value] : partitionValues_) {
    partitionValues.push_back(
        folly::dynamic::object("fieldId", id)("value", value.serialize()));
  }
  obj["partitionValues"] = std::move(partitionValues);

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

  obj["fileFormat"] = dwio::common::FileFormatName::toName(fileFormat_);

  return obj;
}

// static
std::shared_ptr<PaimonConnectorSplit> PaimonConnectorSplit::create(
    const folly::dynamic& obj) {
  const auto connectorId = obj["connectorId"].asString();
  const auto snapshotId = paimonInt(obj, "snapshotId");
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
      : std::optional<int32_t>(paimonInt(
            obj, "tableBucketNumber", -1, std::numeric_limits<int32_t>::max()));

  const auto fileFormat =
      dwio::common::toFileFormat(obj["fileFormat"].asString());

  PaimonPartition partitionValues;
  if (obj.count("partitionValues")) {
    for (const auto& item : obj["partitionValues"]) {
      const auto id =
          paimonInt(item, "fieldId", 0, std::numeric_limits<int32_t>::max());
      const auto& value = item["value"];
      const auto type = value["type"].asString();
      VELOX_USER_CHECK(
          type == "VARCHAR" || type == "BOOLEAN" || type == "INTEGER" ||
              type == "BIGINT",
          "Unsupported Paimon typed partition value: {}",
          type);
      if (value["type"] == "INTEGER" && !value["value"].isNull()) {
        paimonInt(
            value,
            "value",
            std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::max());
      }
      if (type == "BIGINT" && !value["value"].isNull()) {
        paimonInt(value, "value", std::numeric_limits<int64_t>::min());
      }
      if (type == "BOOLEAN" && !value["value"].isNull()) {
        VELOX_USER_CHECK(
            value["value"].isBool(),
            "Paimon BOOLEAN partition must be a boolean");
      }
      if (type == "VARCHAR" && !value["value"].isNull()) {
        VELOX_USER_CHECK(
            value["value"].isString(),
            "Paimon VARCHAR partition must be a string");
      }
      VELOX_USER_CHECK(
          partitionValues.emplace(id, variant::create(value)).second,
          "Duplicate Paimon partition fieldId: {}",
          id);
    }
  }
  const auto version =
      obj.count("wireVersion") ? paimonInt(obj, "wireVersion", 0, 1) : 0;
  VELOX_USER_CHECK(
      version == 0 || obj.count("readMode"),
      "Paimon split readMode is required");
  if (version == 1 || obj.count("splitType")) {
    VELOX_USER_CHECK(
        obj.count("splitType") && obj["splitType"] == "DATA_FILES",
        "Unsupported or missing Paimon splitType");
  }

  return std::make_shared<PaimonConnectorSplit>(
      connectorId,
      snapshotId,
      tableType,
      fileFormat,
      dataFiles,
      std::move(partitionKeys),
      tableBucketNumber,
      rawConvertible,
      obj.count("cacheable") ? obj["cacheable"].asBool() : true,
      std::move(partitionValues),
      obj.count("readMode") ? obj["readMode"].asString() : "SNAPSHOT",
      version);
}

// static
void PaimonConnectorSplit::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("PaimonConnectorSplit", PaimonConnectorSplit::create);
}

// --- Builder ---

PaimonConnectorSplitBuilder& PaimonConnectorSplitBuilder::addFile(
    PaimonDataFile file) {
  dataFiles_.emplace_back(std::move(file));
  return *this;
}

PaimonConnectorSplitBuilder& PaimonConnectorSplitBuilder::addFile(
    std::string filePath,
    uint64_t fileSize,
    int32_t level) {
  PaimonDataFile meta;
  meta.path = std::move(filePath);
  meta.size = fileSize;
  meta.level = level;
  return addFile(std::move(meta));
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
