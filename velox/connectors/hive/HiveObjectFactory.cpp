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
#include "velox/connectors/hive/HiveObjectFactory.h"

#include <string>

#include <folly/dynamic.h>

#include "velox/connectors/ConnectorNames.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSink.h"

namespace facebook::velox::connector::hive {

using namespace velox::common;
using namespace facebook::velox::connector;

std::shared_ptr<ConnectorSplit> HiveObjectFactory::makeConnectorSplit(
    const std::string& connectorId,
    const std::string& filePath,
    uint64_t start,
    uint64_t length,
    const folly::dynamic& options) const {
  auto builder = HiveConnectorSplitBuilder(filePath)
                     .start(start)
                     .length(length)
                     .connectorId(connectorId);

  dwio::common::FileFormat fileFormat =
      (!options.isNull() && options.count("fileFormat"))
      ? static_cast<dwio::common::FileFormat>(options["fileFormat"].asInt())
      : defaultFileFormat_;
  builder.fileFormat(fileFormat);

  int64_t splitWeight = (!options.isNull() && options.count("splitWeight"))
      ? options["splitWeight"].asInt()
      : 0;
  builder.splitWeight(splitWeight);

  bool cacheable = (!options.isNull() && options.count("cacheable"))
      ? options["cacheable"].asBool()
      : true;
  builder.cacheable(cacheable);

  if (!options.isNull() && options.count("infoColumns")) {
    for (auto& kv : options["infoColumns"].items()) {
      builder.infoColumn(kv.first.asString(), kv.second.asString());
    }
  }

  if (!options.isNull() && options.count("partitionKeys")) {
    for (auto& kv : options["partitionKeys"].items()) {
      builder.partitionKey(
          kv.first.asString(),
          kv.second.isNull()
              ? std::nullopt
              : std::optional<std::string>(kv.second.asString()));
    }
  }

  if (!options.isNull() && options.count("tableBucketNumber")) {
    builder.tableBucketNumber(options["tableBucketNumber"].asInt());
  }

  if (!options.isNull() && options.count("bucketConversion")) {
    HiveBucketConversion bucketConversion;
    const auto& bucketConversionOption = options["bucketConversion"];
    bucketConversion.tableBucketCount =
        bucketConversionOption["tableBucketCount"].asInt();
    bucketConversion.partitionBucketCount =
        bucketConversionOption["partitionBucketCount"].asInt();
    for (auto& bucketColumnHandlesOption :
         bucketConversionOption["bucketColumnHandles"]) {
      bucketConversion.bucketColumnHandles.push_back(
          std::const_pointer_cast<HiveColumnHandle>(
              ISerializable::deserialize<HiveColumnHandle>(
                  bucketColumnHandlesOption)));
    }
    builder.bucketConversion(bucketConversion);
  }

  if (!options.isNull() && options.count("customSplitInfo")) {
    std::unordered_map<std::string, std::string> info;
    for (auto& kv : options["customSplitInfo"].items()) {
      info[kv.first.asString()] = kv.second.asString();
    }
    builder.customSplitInfo(info);
  }

  if (!options.isNull() && options.count("extraFileInfo")) {
    auto extra = options["extraFileInfo"].isNull()
        ? std::shared_ptr<std::string>()
        : std::make_shared<std::string>(options["extraFileInfo"].asString());
    builder.extraFileInfo(extra);
  }

  if (!options.isNull() && options.count("serdeParameters")) {
    std::unordered_map<std::string, std::string> serde;
    for (auto& kv : options["serdeParameters"].items()) {
      serde[kv.first.asString()] = kv.second.asString();
    }
    builder.serdeParameters(serde);
  }

  if (!options.isNull() && options.count("fileProperties")) {
    FileProperties props;
    const auto& propertiesOption = options["fileProperties"];
    if (propertiesOption.count("fileSize") &&
        !propertiesOption["fileSize"].isNull()) {
      props.fileSize = propertiesOption["fileSize"].asInt();
    }
    if (propertiesOption.count("modificationTime") &&
        !propertiesOption["modificationTime"].isNull()) {
      props.modificationTime = propertiesOption["modificationTime"].asInt();
    }
    builder.fileProperties(props);
  }

  if (!options.isNull() && options.count("rowIdProperties")) {
    RowIdProperties rowIdProperties;
    const auto& rowIdPropertiesOption = options["rowIdProperties"];
    rowIdProperties.metadataVersion =
        rowIdPropertiesOption["metadataVersion"].asInt();
    rowIdProperties.partitionId = rowIdPropertiesOption["partitionId"].asInt();
    rowIdProperties.tableGuid = rowIdPropertiesOption["tableGuid"].asString();
    builder.rowIdProperties(rowIdProperties);
  }

  return builder.build();
}

std::shared_ptr<connector::ColumnHandle> HiveObjectFactory::makeColumnHandle(
    const std::string& connectorId,
    const std::string& name,
    const TypePtr& dataType,
    const folly::dynamic& options) const {
  using HiveColumnType = hive::HiveColumnHandle::ColumnType;
  HiveColumnType hiveColumnType = HiveColumnType::kRegular;

  if (options.isNull()) {
    return std::make_shared<HiveColumnHandle>(
        name, hiveColumnType, dataType, dataType);
  }

  if (options.count("columnType") && options["columnType"].isString()) {
    std::string columnType = options["columnType"].asString();
    // Accept a few friendly spellings, case-insensitive.
    folly::toLowerAscii(columnType);

    if (columnType == "kpartitionkey" || columnType == "partition_key" ||
        columnType == "partitionkey") {
      hiveColumnType = HiveColumnType::kPartitionKey;
    } else if (columnType == "kregular" || columnType == "regular") {
      hiveColumnType = HiveColumnType::kRegular;
    } else if (
        columnType == "ksynthesized" || columnType == "synthesized" ||
        columnType == "synthetic") {
      hiveColumnType = HiveColumnType::kSynthesized;
    } else if (
        columnType == "krowindex" || columnType == "row_index" ||
        columnType == "rowindex") {
      hiveColumnType = HiveColumnType::kRowIndex;
    } else if (
        columnType == "krowid" || columnType == "row_id" ||
        columnType == "rowid") {
      hiveColumnType = HiveColumnType::kRowId;
    }
  }

  TypePtr hiveType =
      options.count("hiveType") ? Type::create(options["hiveType"]) : dataType;

  //  subfields would be serialized as a vector of strings;
  std::vector<common::Subfield> subfields;
  if (auto rs = options.get_ptr("requiredSubfields")) {
    subfields.reserve(rs->size());
    for (auto& v : *rs) {
      subfields.emplace_back(v.asString());
    }
  }

  HiveColumnHandle::ColumnParseParameters parseParams{
      HiveColumnHandle::ColumnParseParameters::kISO8601};
  if (auto cp = options.get_ptr("columnParseParameters")) {
    auto formatName =
        cp->getDefault("partitionDateValueFormat", "ISO8601").asString();
    if (formatName == "DaysSinceEpoch" || formatName == "kDaysSinceEpoch") {
      parseParams.partitionDateValueFormat =
          HiveColumnHandle::ColumnParseParameters::kDaysSinceEpoch;
    } else {
      parseParams.partitionDateValueFormat =
          HiveColumnHandle::ColumnParseParameters::kISO8601;
    }
  }

  return std::make_shared<HiveColumnHandle>(
      name,
      hiveColumnType,
      dataType,
      hiveType,
      std::move(subfields),
      parseParams);
}

std::shared_ptr<ConnectorTableHandle> HiveObjectFactory::makeTableHandle(
    const std::string& connectorId,
    const std::string& tableName,
    std::vector<std::shared_ptr<const connector::ColumnHandle>> columnHandles,
    const folly::dynamic& options) const {
  bool filterPushdownEnabled =
      options.getDefault("filterPushdownEnabled", true).asBool();

  common::SubfieldFilters subfieldFilters;
  if (auto sf = options.get_ptr("subfieldFilters")) {
    subfieldFilters.reserve(sf->size());

    for (auto& kv : sf->items()) {
      // 1) Parse the key string into a Subfield
      //    (uses Subfield(const std::string&) and default separators)
      Subfield subfield(kv.first.asString());

      // 2) Deserialize the Filter from its dynamic form.
      //    Assumes every Filter subclass registered a SerDe entry in
      //    Filter::registerSerDe().
      auto filter = ISerializable::deserialize<Filter>(
          kv.second, /* context = */ nullptr);

      subfieldFilters.emplace(
          std::move(subfield), std::const_pointer_cast<Filter>(filter));
    }
  }

  core::TypedExprPtr remainingFilter = nullptr;
  if (auto rf = options.get_ptr("remainingFilter")) {
    // assuming rf["expr"] holds the serialized expression
    remainingFilter = ISerializable::deserialize<core::ITypedExpr>(*rf);
  }

  std::unordered_map<std::string, std::string> tableParameters;
  if (auto tp = options.get_ptr("tableParameters")) {
    for (auto& kv : tp->items()) {
      tableParameters.emplace(kv.first.asString(), kv.second.asString());
    }
  }

  // build RowTypePtr from columnHandles
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  names.reserve(columnHandles.size());
  types.reserve(columnHandles.size());
  for (auto& col : columnHandles) {
    auto hiveCol = std::static_pointer_cast<const HiveColumnHandle>(col);
    names.push_back(hiveCol->name());
    types.push_back(hiveCol->dataType());
  }
  auto dataColumns = ROW(std::move(names), std::move(types));

  return std::make_shared<HiveTableHandle>(
      connectorId,
      tableName,
      filterPushdownEnabled,
      std::move(subfieldFilters),
      std::move(remainingFilter),
      dataColumns,
      std::move(tableParameters));
}

std::shared_ptr<ConnectorInsertTableHandle>
HiveObjectFactory::makeInsertTableHandle(
    const std::string& connectorId,
    std::vector<std::shared_ptr<const connector::ColumnHandle>> inputColumns,
    std::shared_ptr<const ConnectorLocationHandle> locationHandle,
    const folly::dynamic& options) const {
  // Convert inputColumns
  std::vector<std::shared_ptr<const HiveColumnHandle>> inputHiveColumns;
  inputHiveColumns.reserve(inputColumns.size());
  for (const auto& handle : inputColumns) {
    inputHiveColumns.push_back(
        std::static_pointer_cast<const HiveColumnHandle>(handle));
  }

  auto hiveLoc =
      std::dynamic_pointer_cast<const hive::LocationHandle>(locationHandle);
  VELOX_CHECK(
      hiveLoc,
      "HiveObjectFactory::makeInsertTableHandle: "
      "expected HiveLocationHandle");

  auto fmt =
      options
          .getDefault(
              "storageFormat", static_cast<int>(dwio::common::FileFormat::DWRF))
          .asInt();
  auto storageFormat = static_cast<dwio::common::FileFormat>(fmt);

  std::shared_ptr<const HiveBucketProperty> bucketProperty = nullptr;
  if (auto bp = options.get_ptr("bucketProperty")) {
    bucketProperty = HiveBucketProperty::deserialize(*bp, nullptr);
  }

  std::optional<common::CompressionKind> compressionKind;
  if (auto ck = options.get_ptr("compressionKind")) {
    compressionKind = static_cast<common::CompressionKind>(ck->asInt());
  }

  std::unordered_map<std::string, std::string> serdeParameters;
  if (auto sp = options.get_ptr("serdeParameters")) {
    for (auto& kv : sp->items()) {
      serdeParameters.emplace(kv.first.asString(), kv.second.asString());
    }
  }

  std::shared_ptr<dwio::common::WriterOptions> writerOptions = nullptr;
  if (auto wo = options.get_ptr("writerOptions")) {
    writerOptions = dwio::common::WriterOptions::deserialize(*wo);
  }

  bool ensureFiles = options.getDefault("ensureFiles", false).asBool();

  auto fileNameGen = HiveInsertFileNameGenerator::deserialize(
      *options.get_ptr("fileNameGenerator"), nullptr);

  return std::make_shared<HiveInsertTableHandle>(
      std::move(inputHiveColumns),
      hiveLoc,
      storageFormat,
      std::move(bucketProperty),
      compressionKind,
      std::move(serdeParameters),
      std::move(writerOptions),
      ensureFiles,
      std::move(fileNameGen));
}

std::shared_ptr<ConnectorLocationHandle> HiveObjectFactory::makeLocationHandle(
    const std::string& connectorId,
    LocationHandle::TableType tableType,
    const folly::dynamic& options) const {
  VELOX_CHECK(options.isObject(), "Expected options to be a dynamic object");

  // Required fields
  auto targetPath = options.at("targetPath").asString();
  auto writePath = options.at("writePath").asString();

  // Optional field: targetFileName
  std::string targetFileName;
  if (options.count("targetFileName") &&
      !options.at("targetFileName").isNull()) {
    targetFileName = options.at("targetFileName").asString();
  }

  return std::make_shared<LocationHandle>(
      std::move(targetPath),
      std::move(writePath),
      tableType,
      std::move(targetFileName),
      connectorId);
}

core::PartitionFunctionSpecPtr HiveObjectFactory::makePartitionFunctionSpec(
    const std::string& /*connectorId*/,
    const folly::dynamic& options) const {
  VELOX_CHECK(options.isObject(), "Expected options to be a dynamic object");

  // Required: number of buckets
  const int numBuckets = options.at("numBuckets").asInt();

  // Optional: explicit bucket-to-partition mapping
  std::vector<int> bucketToPartition;
  if (options.count("bucketToPartition") &&
      !options["bucketToPartition"].isNull()) {
    bucketToPartition = ISerializable::deserialize<std::vector<int>>(
        options["bucketToPartition"]);
  }

  // Required: key channels (by column index) for the bucket function
  // NOTE: Keep the key name consistent with HashPartitionFunctionSpec
  // ("keyChannels")
  std::vector<column_index_t> channels =
      ISerializable::deserialize<std::vector<column_index_t>>(
          options.at("keyChannels"));

  // Optional: constants used as key(s) (serialized ConstantTypedExpr[])
  std::vector<VectorPtr> constValues;
  if (options.count("constants") && !options["constants"].isNull()) {
    const auto typedConsts =
        ISerializable::deserialize<std::vector<velox::core::ConstantTypedExpr>>(
            options["constants"]);
    constValues.reserve(typedConsts.size());

    for (const auto& c : typedConsts) {
      VELOX_CHECK_NOT_NULL(c);
      constValues.emplace_back(c->toConstantVector(pool_.get()));
    }
  }

  if (bucketToPartition.empty()) {
    // Spec form where mapping is derived at create() time (round-robin and
    // optional local shuffle).
    return std::make_shared<velox::connector::hive::HivePartitionFunctionSpec>(
        numBuckets, std::move(channels), std::move(constValues));
  }

  // Spec with explicit mapping.
  return std::make_shared<velox::connector::hive::HivePartitionFunctionSpec>(
      numBuckets,
      std::move(bucketToPartition),
      std::move(channels),
      std::move(constValues));
}

} // namespace facebook::velox::connector::hive
