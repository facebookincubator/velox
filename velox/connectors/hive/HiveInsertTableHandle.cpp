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
#include "velox/connectors/hive/HiveInsertTableHandle.h"

#include <folly/container/F14Map.h>
#include "velox/common/EnumDefine.h"

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <re2/re2.h>

namespace facebook::velox::connector::hive {
namespace {

// Replaces the characters of 'name' that are unsafe in a file name with '_'.
void sanitizeFileName(std::string& name) {
  static const re2::RE2 re("[^a-zA-Z0-9._-]");
  re2::RE2::GlobalReplace(&name, re, "_");
}

std::string makeUuid() {
  return boost::lexical_cast<std::string>(boost::uuids::random_generator()());
}

const folly::F14FastMap<LocationHandle::TableType, std::string_view>&
tableTypeNames() {
  static const folly::F14FastMap<LocationHandle::TableType, std::string_view>
      kNames = {
          {LocationHandle::TableType::kNew, "kNew"},
          {LocationHandle::TableType::kExisting, "kExisting"},
      };
  return kNames;
}

std::string computeBucketedFileName(
    const std::string& queryId,
    uint32_t maxNumBuckets,
    uint32_t bucket) {
  const uint32_t kMaxBucketCountPadding =
      std::to_string(maxNumBuckets - 1).size();
  const std::string bucketValueStr = std::to_string(bucket);
  return fmt::format(
      "0{:0>{}}_0_{}", bucketValueStr, kMaxBucketCountPadding, queryId);
}

std::vector<column_index_t> computePartitionChannels(
    const std::vector<std::shared_ptr<const HiveColumnHandle>>& inputColumns) {
  std::vector<column_index_t> channels;
  for (auto i = 0; i < inputColumns.size(); i++) {
    if (inputColumns[i]->isPartitionKey()) {
      channels.push_back(i);
    }
  }
  return channels;
}

std::vector<column_index_t> computeNonPartitionChannels(
    const std::vector<std::shared_ptr<const HiveColumnHandle>>& inputColumns) {
  std::vector<column_index_t> channels;
  for (auto i = 0; i < inputColumns.size(); i++) {
    if (!inputColumns[i]->isPartitionKey()) {
      channels.push_back(i);
    }
  }
  return channels;
}

} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(LocationHandle, TableType, tableTypeNames)

HiveInsertTableHandle::HiveInsertTableHandle(
    std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns,
    std::shared_ptr<const LocationHandle> locationHandle,
    HiveConfig::InsertExistingPartitionsBehavior
        insertExistingPartitionsBehavior,
    dwio::common::FileFormat storageFormat,
    HiveBucketPropertyPtr bucketProperty,
    std::optional<common::CompressionKind> compressionKind,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    bool writeEmptyFile,
    FileNameGeneratorPtr fileNameGenerator,
    const std::unordered_map<std::string, std::string>& storageParameters)
    : HiveInsertTableHandle(
          std::move(inputColumns),
          std::move(locationHandle),
          std::optional(insertExistingPartitionsBehavior),
          storageFormat,
          std::move(bucketProperty),
          compressionKind,
          serdeParameters,
          writeEmptyFile,
          std::move(fileNameGenerator),
          storageParameters) {}

HiveInsertTableHandle::HiveInsertTableHandle(
    std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns,
    std::shared_ptr<const LocationHandle> locationHandle,
    std::optional<HiveConfig::InsertExistingPartitionsBehavior>
        insertExistingPartitionsBehavior,
    dwio::common::FileFormat storageFormat,
    HiveBucketPropertyPtr bucketProperty,
    std::optional<common::CompressionKind> compressionKind,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    bool writeEmptyFile,
    FileNameGeneratorPtr fileNameGenerator,
    const std::unordered_map<std::string, std::string>& storageParameters)
    : inputColumns_(std::move(inputColumns)),
      locationHandle_(std::move(locationHandle)),
      insertExistingPartitionsBehavior_(insertExistingPartitionsBehavior),
      storageFormat_(storageFormat),
      bucketProperty_(std::move(bucketProperty)),
      compressionKind_(compressionKind),
      serdeParameters_(serdeParameters),
      writeEmptyFile_(writeEmptyFile),
      fileNameGenerator_(std::move(fileNameGenerator)),
      storageParameters_(storageParameters),
      partitionChannels_(computePartitionChannels(inputColumns_)),
      nonPartitionChannels_(computeNonPartitionChannels(inputColumns_)) {
  if (compressionKind.has_value()) {
    VELOX_CHECK(
        compressionKind.value() != common::CompressionKind_MAX,
        "Unsupported compression type: CompressionKind_MAX");
  }

  if (writeEmptyFile_) {
    // If writeEmptyFile is set and either the bucketProperty is set or some
    // partition keys are in the data, there is not a 1:1 mapping from Task to
    // files so we can't proactively create writers.
    VELOX_CHECK(
        bucketProperty_ == nullptr || bucketProperty_->bucketCount() == 0,
        "writeEmptyFile is not supported with bucketing");

    for (const auto& inputColumn : inputColumns_) {
      VELOX_CHECK(
          !inputColumn->isPartitionKey(),
          "writeEmptyFile is not supported with partition keys in the data");
    }
  }
}

std::pair<std::string, std::string> HiveInsertFileNameGenerator::gen(
    std::optional<uint32_t> bucketId,
    const HiveInsertTableHandlePtr& insertTableHandle,
    const ConnectorQueryCtx& connectorQueryCtx,
    uint32_t maxNumBuckets,
    bool commitRequired) const {
  auto targetFileName = insertTableHandle->locationHandle()->targetFileName();
  const bool generateFileName = targetFileName.empty();
  if (bucketId.has_value()) {
    VELOX_CHECK(generateFileName);
    // TODO: add hive.file_renaming_enabled support.
    targetFileName = computeBucketedFileName(
        connectorQueryCtx.queryId(), maxNumBuckets, bucketId.value());
    // queryId may contain unsafe characters.
    sanitizeFileName(targetFileName);
  } else if (generateFileName) {
    // targetFileName includes planNodeId and Uuid. As a result, different
    // table writers run by the same task driver or the same table writer
    // run in different task tries would have different targetFileNames.
    targetFileName = fmt::format(
        "{}_{}_{}_{}",
        connectorQueryCtx.taskId(),
        connectorQueryCtx.driverId(),
        connectorQueryCtx.planNodeId(),
        makeUuid());
    // taskId, planNodeId may contain unsafe characters.
    sanitizeFileName(targetFileName);
  }
  // do not try to sanitize user provided targetFileName
  VELOX_CHECK(!targetFileName.empty());
  const std::string writeFileName = commitRequired
      ? fmt::format(".tmp.velox.{}_{}", targetFileName, makeUuid())
      : targetFileName;
  if (generateFileName &&
      insertTableHandle->storageFormat() == dwio::common::FileFormat::PARQUET) {
    return {
        fmt::format("{}{}", targetFileName, ".parquet"),
        fmt::format("{}{}", writeFileName, ".parquet")};
  }
  return {targetFileName, writeFileName};
}

folly::dynamic HiveInsertFileNameGenerator::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "HiveInsertFileNameGenerator";
  return obj;
}

std::shared_ptr<HiveInsertFileNameGenerator>
HiveInsertFileNameGenerator::deserialize(
    const folly::dynamic& /* obj */,
    void* /* context */) {
  return std::make_shared<HiveInsertFileNameGenerator>();
}

void HiveInsertFileNameGenerator::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register(
      "HiveInsertFileNameGenerator", HiveInsertFileNameGenerator::deserialize);
}

std::string HiveInsertFileNameGenerator::toString() const {
  return "HiveInsertFileNameGenerator";
}

bool HiveInsertTableHandle::isPartitioned() const {
  return std::any_of(
      inputColumns_.begin(), inputColumns_.end(), [](auto column) {
        return column->isPartitionKey();
      });
}

const HiveBucketProperty* HiveInsertTableHandle::bucketProperty() const {
  return bucketProperty_.get();
}

bool HiveInsertTableHandle::isBucketed() const {
  return bucketProperty() != nullptr;
}

bool HiveInsertTableHandle::isExistingTable() const {
  return locationHandle_->tableType() == LocationHandle::TableType::kExisting;
}

folly::dynamic HiveInsertTableHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "HiveInsertTableHandle";
  folly::dynamic arr = folly::dynamic::array;
  for (const auto& ic : inputColumns_) {
    arr.push_back(ic->serialize());
  }

  obj["inputColumns"] = arr;
  obj["locationHandle"] = locationHandle_->serialize();
  obj["tableStorageFormat"] =
      dwio::common::FileFormatName::toName(storageFormat_);

  if (bucketProperty_) {
    obj["bucketProperty"] = bucketProperty_->serialize();
  }

  if (compressionKind_.has_value()) {
    obj["compressionKind"] = common::compressionKindToString(*compressionKind_);
  }

  folly::dynamic params = folly::dynamic::object;
  for (const auto& [key, value] : serdeParameters_) {
    params[key] = value;
  }
  obj["serdeParameters"] = params;

  folly::dynamic storageParams = folly::dynamic::object;
  for (const auto& [key, value] : storageParameters_) {
    storageParams[key] = value;
  }
  obj["storageParameters"] = storageParams;

  obj["writeEmptyFile"] = writeEmptyFile_;
  obj["fileNameGenerator"] = fileNameGenerator_->serialize();
  if (insertExistingPartitionsBehavior_.has_value()) {
    obj["insertExistingPartitionsBehavior"] =
        HiveConfig::toName(*insertExistingPartitionsBehavior_);
  }
  return obj;
}

HiveInsertTableHandlePtr HiveInsertTableHandle::create(
    const folly::dynamic& obj) {
  auto inputColumns = ISerializable::deserialize<std::vector<HiveColumnHandle>>(
      obj["inputColumns"]);
  auto locationHandle =
      ISerializable::deserialize<LocationHandle>(obj["locationHandle"]);
  auto storageFormat =
      dwio::common::toFileFormat(obj["tableStorageFormat"].asString());

  std::optional<common::CompressionKind> compressionKind = std::nullopt;
  if (obj.count("compressionKind") > 0) {
    compressionKind =
        common::stringToCompressionKind(obj["compressionKind"].asString());
  }

  HiveBucketPropertyPtr bucketProperty;
  if (obj.count("bucketProperty") > 0) {
    bucketProperty =
        ISerializable::deserialize<HiveBucketProperty>(obj["bucketProperty"]);
  }

  std::unordered_map<std::string, std::string> serdeParameters;
  for (const auto& pair : obj["serdeParameters"].items()) {
    serdeParameters.emplace(pair.first.asString(), pair.second.asString());
  }

  std::unordered_map<std::string, std::string> storageParameters;
  if (obj.count("storageParameters") > 0) {
    for (const auto& pair : obj["storageParameters"].items()) {
      storageParameters.emplace(pair.first.asString(), pair.second.asString());
    }
  }

  bool writeEmptyFile = obj["writeEmptyFile"].asBool();

  auto fileNameGenerator =
      ISerializable::deserialize<FileNameGenerator>(obj["fileNameGenerator"]);

  std::optional<HiveConfig::InsertExistingPartitionsBehavior>
      insertExistingPartitionsBehavior;
  if (obj.count("insertExistingPartitionsBehavior") > 0) {
    insertExistingPartitionsBehavior =
        HiveConfig::toInsertExistingPartitionsBehavior(
            obj["insertExistingPartitionsBehavior"].asString());
  }
  return std::shared_ptr<HiveInsertTableHandle>(new HiveInsertTableHandle(
      inputColumns,
      locationHandle,
      insertExistingPartitionsBehavior,
      storageFormat,
      bucketProperty,
      compressionKind,
      serdeParameters,
      writeEmptyFile,
      fileNameGenerator,
      storageParameters));
}

void HiveInsertTableHandle::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("HiveInsertTableHandle", HiveInsertTableHandle::create);
}

std::string HiveInsertTableHandle::toString() const {
  std::ostringstream out;
  out << "HiveInsertTableHandle ["
      << dwio::common::FileFormatName::toName(storageFormat_);
  if (compressionKind_.has_value()) {
    out << " " << common::compressionKindToString(compressionKind_.value());
  } else {
    out << " none";
  }
  out << "], [inputColumns: [";
  for (const auto& i : inputColumns_) {
    out << " " << i->toString();
  }
  out << " ], locationHandle: " << locationHandle_->toString();
  if (bucketProperty_) {
    out << ", bucketProperty: " << bucketProperty_->toString();
  }

  if (serdeParameters_.size() > 0) {
    std::map<std::string, std::string> sortedSerdeParams(
        serdeParameters_.begin(), serdeParameters_.end());
    out << ", serdeParameters: ";
    for (const auto& [key, value] : sortedSerdeParams) {
      out << "[" << key << ", " << value << "] ";
    }
  }
  if (insertExistingPartitionsBehavior_.has_value()) {
    out << ", insertExistingPartitionsBehavior: "
        << HiveConfig::toName(*insertExistingPartitionsBehavior_);
  }
  out << ", fileNameGenerator: " << fileNameGenerator_->toString();
  out << "]";
  return out.str();
}

std::string LocationHandle::toString() const {
  return fmt::format(
      "LocationHandle [targetPath: {}, writePath: {}, tableType: {}, tableFileName: {}]",
      targetPath_,
      writePath_,
      tableType_,
      targetFileName_);
}

void LocationHandle::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("LocationHandle", LocationHandle::create);
}

folly::dynamic LocationHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "LocationHandle";
  obj["targetPath"] = targetPath_;
  obj["writePath"] = writePath_;
  obj["tableType"] = toName(tableType_);
  obj["targetFileName"] = targetFileName_;
  return obj;
}

LocationHandlePtr LocationHandle::create(const folly::dynamic& obj) {
  auto targetPath = obj["targetPath"].asString();
  auto writePath = obj["writePath"].asString();
  auto tableType = toTableType(obj["tableType"].asString());
  auto targetFileName = obj["targetFileName"].asString();
  return std::make_shared<LocationHandle>(
      targetPath, writePath, tableType, targetFileName);
}

} // namespace facebook::velox::connector::hive
