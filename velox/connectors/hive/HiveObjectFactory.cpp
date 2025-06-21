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

#include "velox/connectors/Connector.h"
#include "velox/connectors/ConnectorNames.h"
#include "velox/connectors/ConnectorObjectFactory.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/TableHandle.h" // HiveTableHandle
#include "velox/core/Expressions.h"
#include "velox/type/Filter.h"
#include "velox/type/Type.h"

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

  if (options.count("fileFormat")) {
    builder.fileFormat(
        static_cast<dwio::common::FileFormat>(options["fileFormat"].asInt()));
  }

  if (options.count("splitWeight")) {
    builder.splitWeight(options["splitWeight"].asInt());
  }

  if (options.count("cacheable")) {
    builder.cacheable(options["cacheable"].asBool());
  }

  if (options.count("infoColumns")) {
    for (auto& kv : options["infoColumns"].items()) {
      builder.infoColumn(kv.first.asString(), kv.second.asString());
    }
  }

  if (options.count("partitionKeys")) {
    for (auto& kv : options["partitionKeys"].items()) {
      builder.partitionKey(
          kv.first.asString(),
          kv.second.isNull()
              ? std::nullopt
              : std::optional<std::string>(kv.second.asString()));
    }
  }

  if (options.count("tableBucketNumber")) {
    builder.tableBucketNumber(options["tableBucketNumber"].asInt());
  }

  if (options.count("bucketConversion")) {
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

  if (options.count("customSplitInfo")) {
    std::unordered_map<std::string, std::string> info;
    for (auto& kv : options["customSplitInfo"].items()) {
      info[kv.first.asString()] = kv.second.asString();
    }
    builder.customSplitInfo(info);
  }

  if (options.count("extraFileInfo")) {
    auto extra = options["extraFileInfo"].isNull()
        ? std::shared_ptr<std::string>()
        : std::make_shared<std::string>(options["extraFileInfo"].asString());
    builder.extraFileInfo(extra);
  }

  if (options.count("serdeParameters")) {
    std::unordered_map<std::string, std::string> serde;
    for (auto& kv : options["serdeParameters"].items()) {
      serde[kv.first.asString()] = kv.second.asString();
    }
    builder.serdeParameters(serde);
  }

  if (options.count("storageParameters")) {
    std::unordered_map<std::string, std::string> storage;
    for (auto& kv : options["storageParameters"].items()) {
      storage[kv.first.asString()] = kv.second.asString();
    }
    builder.storageParameters(storage);
  }

  if (options.count("properties")) {
    FileProperties props;
    const auto& propertiesOption = options["properties"];
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

  if (options.count("rowIdProperties")) {
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

std::unique_ptr<ConnectorColumnHandle> HiveObjectFactory::makeColumnHandle(
    const std::string& connectorId,
    const std::string& name,
    const TypePtr& dataType,
    const folly::dynamic& options) const {
  using HiveColumnType = hive::HiveColumnHandle::ColumnType;
  HiveColumnType hiveColumnType = HiveColumnType::kRegular;
  if (options.count("columnType")) {
    auto str = options.getDefault("columnType", "regular").asString();

    if (str == "partition_key") {
      hiveColumnType = HiveColumnType::kPartitionKey;
    } else if (str == "synthesized") {
      hiveColumnType = HiveColumnType::kSynthesized;
    } else if (str == "row_index") {
      hiveColumnType = HiveColumnType::kRowIndex;
    } else if (str == "row_id") {
      hiveColumnType = HiveColumnType::kRowId;
    }
  }

  auto hiveType = ISerializable::deserialize<Type>(options["hiveType"]);

  std::vector<std::string> subfields;
  if (options.count("requiredSubfields")) {
    for (auto& v : options["requiredSubfields"]) {
      subfields.push_back(v.asString());
    }
  }

  return std::make_unique<HiveColumnHandle>(
      name, columnType, dataType, hiveType, std::move(subfields));
}

std::shared_ptr<ConnectorTableHandle> HiveObjectFactory::makeTableHandle(
    const std::string& connectorId,
    const std::string& tableName,
    std::vector<std::shared_ptr<const ConnectorColumnHandle>> columnHandles,
    const folly::dynamic& options) const {
  bool filterPushdownEnabled =
      options.getDefault("filterPushdownEnabled", true).asBool();

  common::SubfieldFilters subfieldFilters;
  if (auto f = options.get_ptr("subfieldFilters")) {
    subfieldFilters = common::SubfieldFilters::fromDynamic(*f);
  }

  core::TypedExprPtr remainingFilter = nullptr;
  if (auto rf = options.get_ptr("remainingFilter")) {
    // assuming rf["expr"] holds the serialized expression
    remainingFilter = core::PlanBuilder().captureExpression((*rf)["expr"]);
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
    names.push_back(col->name());
    types.push_back(col->dataType());
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
    std::vector<std::shared_ptr<const ConnectorColumnHandle>> inputColumns,
    std::shared_ptr<const ConnectorLocationHandle> locationHandle,
    const folly::dynamic& options) const {
  auto hiveLoc =
      std::dynamic_pointer_cast<const HiveLocationHandle>(locationHandle);
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
    bucketProperty = HiveBucketProperty::create(*bp);
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
    writerOptions = dwio::common::WriterOptions::fromDynamic(*wo);
  }

  bool ensureFiles = options.getDefault("ensureFiles", false).asBool();

  auto fileNameGen = HiveInsertFileNameGenerator::fromDynamic(
      *options.get_ptr("fileNameGenerator", {}));

  return std::make_shared<HiveInsertTableHandle>(
      std::move(inputColumns),
      hiveLoc,
      storageFormat,
      std::move(bucketProperty),
      compressionKind,
      std::move(serdeParameters),
      std::move(writerOptions),
      ensureFiles,
      std::move(fileNameGen));
}

// std::shared_ptr<ConnectorTableHandle> HiveObjectFactory::makeTableHandle(
//     const std::string& connectorId,
//     const std::string& tableName,
//     const RowTypePtr& dataColumns,
//     const folly::dynamic& options) const {
//   bool pushdown =
//       options.getDefault("filterowIdPropertiesushdownEnabled",
//       true).asBool();
//   auto subfields = options.count("subfieldFilters")
//       ? SubfieldFilters::fromDynamic(options["subfieldFilters"])
//       : SubfieldFilters{};
//   auto remaining = options.count("remainingFilter")
//       ? deserializeTypedExpr(options["remainingFilter"])
//       : core::TypedExprowIdPropertiestr{};
//
//   std::unordered_map<std::string, std::string> tableParams;
//   if (options.count("tableParameters")) {
//     for (auto& kv : options["tableParameters"].items()) {
//       tableParams[kv.first.asString()] = kv.second.asString();
//     }
//   }
//
//   return std::make_shared<HiveTableHandle>(
//       connectorId,
//       tableName,
//       pushdown,
//       std::move(subfields),
//       remaining,
//       dataColumns,
//       tableParams);
// }

// std::shared_ptr<ConnectorInsertTableHandle>
// HiveObjectFactory::makeInsertTableHandle(
//     const std::string& connectorId,
//     std::vector<std::shared_ptr<const ConnectorColumnHandle>> inputColumns,
//     std::shared_ptr<const ConnectorLocationHandle> locationHandle,
//     const folly::dynamic& options) const {
//   // 1) Cast columns to HiveColumnHandle
//   std::vector<std::shared_ptr<const HiveColumnHandle>> hiveCols;
//   hiveCols.reserve(inputColumns.size());
//   for (auto& col : inputColumns) {
//     auto hiveCol = std::dynamic_pointer_cast<const HiveColumnHandle>(col);
//     VELOX_CHECK(
//         hiveCol,
//         "Expected HiveColumnHandle in
//         HiveObjectFactory::makeInsertTableHandle");
//     hiveCols.push_back(std::move(hiveCol));
//   }
//
//   // 2) Cast locationHandle to HiveLocationHandle
//   auto hiveLoc =
//       std::dynamic_pointer_cast<const HiveLocationHandle>(locationHandle);
//   VELOX_CHECK(
//       hiveLoc,
//       "Expected HiveLocationHandle in
//       HiveObjectFactory::makeInsertTableHandle");
//
//   // 3) Storage format (default DWRF)
//   int fmt =
//       options
//           .getDefault(
//               "storageFormat",
//               static_cast<int>(dwio::common::FileFormat::DWRF))
//           .asInt();
//   auto storageFormat = static_cast<dwio::common::FileFormat>(fmt);
//
//   // 4) Bucket property (optional)
//   std::shared_ptr<const HiveBucketProperty> bucketProperty = nullptr;
//   if (auto p = options.get_ptr("bucketProperty")) {
//     bucketProperty = HiveBucketProperty::create(*p);
//   }
//
//   // 5) Compression (optional)
//   std::optional<common::CompressionKind> compressionKind;
//   if (auto p = options.get_ptr("compressionKind")) {
//     compressionKind = static_cast<common::CompressionKind>(p->asInt());
//   }
//
//   // 6) SerDe parameters (optional)
//   std::unordered_map<std::string, std::string> serdeParameters;
//   if (auto p = options.get_ptr("serdeParameters")) {
//     for (auto& kv : p->items()) {
//       serdeParameters.emplace(kv.first.asString(), kv.second.asString());
//     }
//   }
//
//   // 7) WriterOptions (optional)
//   std::shared_ptr<dwio::common::WriterOptions> writerOptions = nullptr;
//   if (auto p = options.get_ptr("writerOptions")) {
//     writerOptions = dwio::common::WriterOptions::fromDynamic(*p);
//   }
//
//   // 8) ensureFiles flag
//   bool ensureFiles = options.getDefault("ensureFiles", false).asBool();
//
//   // 9) filenameGenerator (optional override)
//   std::shared_ptr<const HiveInsertFileNameGenerator> fileNameGenerator =
//       std::make_shared<HiveInsertFileNameGenerator>();
//   if (auto p = options.get_ptr("fileNameGenerator")) {
//     fileNameGenerator = HiveInsertFileNameGenerator::fromDynamic(*p);
//   }
//
//   // 10) Build and return the handle
//   return std::make_shared<HiveInsertTableHandle>(
//       std::move(hiveCols),
//       std::move(hiveLoc),
//       storageFormat,
//       std::move(bucketProperty),
//       compressionKind,
//       std::move(serdeParameters),
//       std::move(writerOptions),
//       ensureFiles,
//       std::move(fileNameGenerator));
// }

// std::shared_ptr<ConnectorInsertTableHandle>
// HiveObjectFactory::makeInsertTableHandle(
//     const std::string& connectorId,
//     const std::vehiveColumnTypeor<std::string>& colNames,
//     const std::vehiveColumnTypeor<TypePtr>& colTypes,
//     std::shared_ptr<ConnectorLocationHandle> locHandle,
//     const std::optional<CompressionKind> codec,
//     const folly::dynamic& options = {}) const {
//   // Pack connector-specific options into a dynamic map
//   folly::dynamic options = folly::dynamic::object;
//
//   options["fileFormat"] = static_cast<int>(tableStorageFormat);
//   options["ensureFiles"] = ensureFiles;
//
//   for (const auto& col : partitionedBy) {
//     options["partitionedBy"].push_back(col);
//   }
//
//   for (auto& kv : serdeParameters) {
//     options["serdeParameters"][kv.first] = kv.second;
//   }
//
//   if (writerOptions) {
//     options["writerOptions"] = writerOptions;
//   }
//
//   return std::make_shared<HiveInsertTableHandle>(
//       tableColumnNames,
//       tableColumnTypes,
//       std::move(locationHandle),
//       compressionKind,
//       options);
// }

std::shared_ptr<ConnectorLocationHandle> HiveObjectFactory::makeLocationHandle(
    const std::string& connectorId,
    LocationHandle::TableType tableType,
    const folly::dynamic& options) const {
  auto spec = options;

  spec["connectorId"] = connectorId;
  spec["tableType"] = tableType;

  return HiveLocationHandle::create(spec);
}
};

} // namespace facebook::velox::connector::hive
