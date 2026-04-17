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

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/TableHandle.h"

#include <algorithm>
#include <map>
#include <sstream>

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <re2/re2.h>

namespace facebook::velox::connector::hive {
namespace {

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

std::string makeUuid() {
  return boost::lexical_cast<std::string>(boost::uuids::random_generator()());
}

std::unordered_map<LocationHandle::TableType, std::string> tableTypeNames() {
  return {
      {LocationHandle::TableType::kNew, "kNew"},
      {LocationHandle::TableType::kExisting, "kExisting"},
  };
}

template <typename K, typename V>
std::unordered_map<V, K> invertMap(const std::unordered_map<K, V>& mapping) {
  std::unordered_map<V, K> inverted;
  for (const auto& [key, value] : mapping) {
    inverted.emplace(value, key);
  }
  return inverted;
}

std::string computeBucketedFileName(
    const std::string& queryId,
    uint32_t maxBucketCount,
    uint32_t bucket) {
  const uint32_t kMaxBucketCountPadding =
      std::to_string(maxBucketCount - 1).size();
  const std::string bucketValueStr = std::to_string(bucket);
  return fmt::format(
      "0{:0>{}}_0_{}", bucketValueStr, kMaxBucketCountPadding, queryId);
}

} // namespace

const std::string LocationHandle::tableTypeName(LocationHandle::TableType type) {
  static const auto tableTypes = tableTypeNames();
  return tableTypes.at(type);
}

LocationHandle::TableType LocationHandle::tableTypeFromName(
    const std::string& name) {
  static const auto nameTableTypes = invertMap(tableTypeNames());
  return nameTableTypes.at(name);
}

std::string LocationHandle::toString() const {
  return fmt::format(
      "LocationHandle [targetPath: {}, writePath: {}, tableType: {}, tableFileName: {}]",
      targetPath_,
      writePath_,
      tableTypeName(tableType_),
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
  obj["tableType"] = tableTypeName(tableType_);
  obj["targetFileName"] = targetFileName_;
  return obj;
}

LocationHandlePtr LocationHandle::create(const folly::dynamic& obj) {
  auto targetPath = obj["targetPath"].asString();
  auto writePath = obj["writePath"].asString();
  auto tableType = tableTypeFromName(obj["tableType"].asString());
  auto targetFileName = obj["targetFileName"].asString();
  return std::make_shared<LocationHandle>(
      targetPath, writePath, tableType, targetFileName);
}

HiveSortingColumn::HiveSortingColumn(
    const std::string& sortColumn,
    const core::SortOrder& sortOrder)
    : sortColumn_(sortColumn), sortOrder_(sortOrder) {
  VELOX_USER_CHECK(!sortColumn_.empty(), "hive sort column must be set");

  if (FOLLY_UNLIKELY(
          (sortOrder_.isAscending() && !sortOrder_.isNullsFirst()) ||
          (!sortOrder_.isAscending() && sortOrder_.isNullsFirst()))) {
    VELOX_USER_FAIL("Bad hive sort order: {}", toString());
  }
}

folly::dynamic HiveSortingColumn::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "HiveSortingColumn";
  obj["columnName"] = sortColumn_;
  obj["sortOrder"] = sortOrder_.serialize();
  return obj;
}

std::shared_ptr<HiveSortingColumn> HiveSortingColumn::deserialize(
    const folly::dynamic& obj,
    void* /*context*/) {
  const std::string columnName = obj["columnName"].asString();
  const auto sortOrder = core::SortOrder::deserialize(obj["sortOrder"]);
  return std::make_shared<HiveSortingColumn>(columnName, sortOrder);
}

std::string HiveSortingColumn::toString() const {
  return fmt::format(
      "[COLUMN[{}] ORDER[{}]]", sortColumn_, sortOrder_.toString());
}

void HiveSortingColumn::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("HiveSortingColumn", HiveSortingColumn::deserialize);
}

HiveBucketProperty::HiveBucketProperty(
    Kind kind,
    int32_t bucketCount,
    const std::vector<std::string>& bucketedBy,
    const std::vector<TypePtr>& bucketTypes,
    const std::vector<std::shared_ptr<const HiveSortingColumn>>& sortedBy)
    : kind_(kind),
      bucketCount_(bucketCount),
      bucketedBy_(bucketedBy),
      bucketTypes_(bucketTypes),
      sortedBy_(sortedBy) {
  validate();
}

void HiveBucketProperty::validate() const {
  VELOX_USER_CHECK_GT(bucketCount_, 0, "Hive bucket count can't be zero");
  VELOX_USER_CHECK(!bucketedBy_.empty(), "Hive bucket columns must be set");
  VELOX_USER_CHECK_EQ(
      bucketedBy_.size(),
      bucketTypes_.size(),
      "The number of hive bucket columns and types do not match {}",
      toString());
}

std::string HiveBucketProperty::kindString(Kind kind) {
  switch (kind) {
    case Kind::kHiveCompatible:
      return "HIVE_COMPATIBLE";
    case Kind::kPrestoNative:
      return "PRESTO_NATIVE";
    default:
      return fmt::format("UNKNOWN {}", static_cast<int>(kind));
  }
}

folly::dynamic HiveBucketProperty::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "HiveBucketProperty";
  obj["kind"] = static_cast<int64_t>(kind_);
  obj["bucketCount"] = bucketCount_;
  obj["bucketedBy"] = ISerializable::serialize(bucketedBy_);
  obj["bucketedTypes"] = ISerializable::serialize(bucketTypes_);
  obj["sortedBy"] = ISerializable::serialize(sortedBy_);
  return obj;
}

std::shared_ptr<HiveBucketProperty> HiveBucketProperty::deserialize(
    const folly::dynamic& obj,
    void* context) {
  const Kind kind = static_cast<Kind>(obj["kind"].asInt());
  const int32_t bucketCount = obj["bucketCount"].asInt();
  const auto buckectedBy =
      ISerializable::deserialize<std::vector<std::string>>(obj["bucketedBy"]);
  const auto bucketedTypes = ISerializable::deserialize<std::vector<Type>>(
      obj["bucketedTypes"], context);
  const auto sortedBy =
      ISerializable::deserialize<std::vector<HiveSortingColumn>>(
          obj["sortedBy"], context);
  return std::make_shared<HiveBucketProperty>(
      kind, bucketCount, buckectedBy, bucketedTypes, sortedBy);
}

bool HiveBucketProperty::operator==(const HiveBucketProperty& other) const {
  if (kind_ != other.kind_ || bucketCount_ != other.bucketCount_ ||
      bucketedBy_ != other.bucketedBy_) {
    return false;
  }

  if (bucketTypes_.size() != other.bucketTypes_.size()) {
    return false;
  }
  for (auto i = 0; i < bucketTypes_.size(); ++i) {
    if (bucketTypes_[i] == nullptr || other.bucketTypes_[i] == nullptr) {
      if (bucketTypes_[i] != other.bucketTypes_[i]) {
        return false;
      }
      continue;
    }
    if (!bucketTypes_[i]->equivalent(*other.bucketTypes_[i])) {
      return false;
    }
  }

  if (sortedBy_.size() != other.sortedBy_.size()) {
    return false;
  }
  for (auto i = 0; i < sortedBy_.size(); ++i) {
    const auto& lhs = sortedBy_[i];
    const auto& rhs = other.sortedBy_[i];
    if (lhs == nullptr || rhs == nullptr) {
      if (lhs != rhs) {
        return false;
      }
      continue;
    }
    if (lhs->sortColumn() != rhs->sortColumn() ||
        lhs->sortOrder() != rhs->sortOrder()) {
      return false;
    }
  }
  return true;
}

void HiveBucketProperty::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("HiveBucketProperty", HiveBucketProperty::deserialize);
}

std::string HiveBucketProperty::toString() const {
  std::stringstream out;
  out << "\nHiveBucketProperty[<" << kind_ << " " << bucketCount_ << ">\n";
  out << "\tBucket Columns:\n";
  for (const auto& column : bucketedBy_) {
    out << "\t\t" << column << "\n";
  }
  out << "\tBucket Types:\n";
  for (const auto& type : bucketTypes_) {
    out << "\t\t" << type->toString() << "\n";
  }
  if (!sortedBy_.empty()) {
    out << "\tSortedBy Columns:\n";
    for (const auto& sortColum : sortedBy_) {
      out << "\t\t" << sortColum->toString() << "\n";
    }
  }
  out << "]\n";
  return out.str();
}

HiveInsertTableHandle::HiveInsertTableHandle(
    std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns,
    std::shared_ptr<const LocationHandle> locationHandle,
    dwio::common::FileFormat storageFormat,
    std::shared_ptr<const HiveBucketProperty> bucketProperty,
    std::optional<common::CompressionKind> compressionKind,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    const std::shared_ptr<dwio::common::WriterOptions>& writerOptions,
    // When this option is set the HiveDataSink will always write a file even
    // if there's no data. This is useful when the table is bucketed, but the
    // engine handles ensuring a 1 to 1 mapping from task to bucket.
    const bool ensureFiles,
    std::shared_ptr<const FileNameGenerator> fileNameGenerator,
    const std::unordered_map<std::string, std::string>& storageParameters)
    : inputColumns_(std::move(inputColumns)),
      locationHandle_(std::move(locationHandle)),
      storageFormat_(storageFormat),
      bucketProperty_(std::move(bucketProperty)),
      compressionKind_(compressionKind),
      serdeParameters_(serdeParameters),
      writerOptions_(writerOptions),
      ensureFiles_(ensureFiles),
      fileNameGenerator_(std::move(fileNameGenerator)),
      storageParameters_(storageParameters),
      partitionChannels_(computePartitionChannels(inputColumns_)),
      nonPartitionChannels_(computeNonPartitionChannels(inputColumns_)) {
  if (compressionKind.has_value()) {
    VELOX_CHECK(
        compressionKind.value() != common::CompressionKind_MAX,
        "Unsupported compression type: CompressionKind_MAX");
  }

  if (ensureFiles_) {
    // If ensureFiles is set and either the bucketProperty is set or some
    // partition keys are in the data, there is not a 1:1 mapping from Task to
    // files so we can't proactively create writers.
    VELOX_CHECK(
        bucketProperty_ == nullptr || bucketProperty_->bucketCount() == 0,
        "ensureFiles is not supported with bucketing");

    for (const auto& inputColumn : inputColumns_) {
      VELOX_CHECK(
          !inputColumn->isPartitionKey(),
          "ensureFiles is not supported with partition keys in the data");
    }
  }
}

std::pair<std::string, std::string> HiveInsertFileNameGenerator::gen(
    std::optional<uint32_t> bucketId,
    const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx& connectorQueryCtx,
    bool commitRequired) const {
  auto defaultHiveConfig =
      std::make_shared<const HiveConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));

  return this->gen(
      bucketId,
      insertTableHandle,
      connectorQueryCtx,
      defaultHiveConfig,
      commitRequired);
}

std::pair<std::string, std::string> HiveInsertFileNameGenerator::gen(
    std::optional<uint32_t> bucketId,
    const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx& connectorQueryCtx,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    bool commitRequired) const {
  auto targetFileName = insertTableHandle->locationHandle()->targetFileName();
  const bool generateFileName = targetFileName.empty();
  if (bucketId.has_value()) {
    VELOX_CHECK(generateFileName);
    // TODO: add hive.file_renaming_enabled support.
    targetFileName = computeBucketedFileName(
        connectorQueryCtx.queryId(),
        hiveConfig->maxBucketCount(connectorQueryCtx.sessionProperties()),
        bucketId.value());
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

void HiveInsertFileNameGenerator::sanitizeFileName(std::string& name) {
  static const re2::RE2 re("[^a-zA-Z0-9._-]");
  re2::RE2::GlobalReplace(&name, re, "_");
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
  obj["tableStorageFormat"] = dwio::common::toString(storageFormat_);

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

  obj["ensureFiles"] = ensureFiles_;
  obj["fileNameGenerator"] = fileNameGenerator_->serialize();
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

  std::shared_ptr<const HiveBucketProperty> bucketProperty;
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

  bool ensureFiles = obj["ensureFiles"].asBool();

  auto fileNameGenerator =
      ISerializable::deserialize<FileNameGenerator>(obj["fileNameGenerator"]);
  return std::make_shared<HiveInsertTableHandle>(
      inputColumns,
      locationHandle,
      storageFormat,
      bucketProperty,
      compressionKind,
      serdeParameters,
      nullptr, // writerOptions is not serializable
      ensureFiles,
      fileNameGenerator,
      storageParameters);
}

void HiveInsertTableHandle::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("HiveInsertTableHandle", HiveInsertTableHandle::create);
}

std::string HiveInsertTableHandle::toString() const {
  std::ostringstream out;
  out << "HiveInsertTableHandle [" << dwio::common::toString(storageFormat_);
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
  out << ", fileNameGenerator: " << fileNameGenerator_->toString();
  out << "]";
  return out.str();
}

} // namespace facebook::velox::connector::hive
