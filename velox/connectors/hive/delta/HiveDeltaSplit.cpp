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

#include "velox/connectors/hive/delta/HiveDeltaSplit.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/serialization/Serializable.h"

namespace facebook::velox::connector::hive::delta {

std::string_view toString(DeltaColumnMappingMode mode) {
  switch (mode) {
    case DeltaColumnMappingMode::kNone:
      return "none";
    case DeltaColumnMappingMode::kName:
      return "name";
    case DeltaColumnMappingMode::kId:
      return "id";
  }
  VELOX_UNREACHABLE();
}

DeltaColumnMappingMode deltaColumnMappingModeFromString(std::string_view name) {
  if (name == "none") {
    return DeltaColumnMappingMode::kNone;
  }
  if (name == "name") {
    return DeltaColumnMappingMode::kName;
  }
  if (name == "id") {
    return DeltaColumnMappingMode::kId;
  }
  VELOX_USER_FAIL("Unknown Delta column mapping mode: {}", name);
}

HiveDeltaSplit::HiveDeltaSplit(
    const std::string& connectorId,
    const std::string& filePath,
    dwio::common::FileFormat fileFormat,
    uint64_t start,
    uint64_t length,
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKeys,
    std::optional<int32_t> tableBucketNumber,
    const std::unordered_map<std::string, std::string>& customSplitInfo,
    const std::shared_ptr<std::string>& extraFileInfo,
    bool cacheable,
    const std::unordered_map<std::string, std::string>& infoColumns,
    std::optional<FileProperties> fileProperties,
    bool hasDeletionVector,
    DeltaColumnMappingMode columnMappingMode)
    : HiveConnectorSplit(
          connectorId,
          filePath,
          fileFormat,
          start,
          length,
          partitionKeys,
          tableBucketNumber,
          customSplitInfo,
          extraFileInfo,
          /*serdeParameters=*/{},
          /*splitWeight=*/0,
          cacheable,
          infoColumns,
          std::move(fileProperties),
          std::nullopt,
          std::nullopt),
      hasDeletionVector(hasDeletionVector),
      columnMappingMode(columnMappingMode) {}

folly::dynamic HiveDeltaSplit::serialize() const {
  folly::dynamic obj = HiveConnectorSplit::serialize();
  obj["name"] = "HiveDeltaSplit";
  obj["hasDeletionVector"] = hasDeletionVector;
  // Key is namespaced to avoid clashing with HiveConnectorSplit's own
  // 'columnMappingMode' field, which uses dwio::common::ColumnMappingMode
  // names (kPosition, kParquetFieldId, ...) not the Delta names.
  obj["deltaColumnMappingMode"] =
      std::string(delta::toString(columnMappingMode));
  return obj;
}

// static
std::shared_ptr<HiveDeltaSplit> HiveDeltaSplit::create(
    const folly::dynamic& obj) {
  // Deserialize the base fields via HiveConnectorSplit::create, then rebuild
  // a HiveDeltaSplit around them. HiveConnectorSplit has several const fields
  // (splitWeight, columnMappingMode) plus a few (serdeParameters,
  // rowIdProperties, bucketConversion) that HiveDeltaSplit's constructor does
  // not currently accept, so they cannot be preserved via the current API;
  // that is intentional -- Delta splits produced by the Presto coordinator
  // only exercise the parameters listed below plus hasDeletionVector and the
  // Delta columnMappingMode.
  auto base = HiveConnectorSplit::create(obj);
  DeltaColumnMappingMode mode = DeltaColumnMappingMode::kNone;
  if (auto it = obj.find("deltaColumnMappingMode"); it != obj.items().end()) {
    mode = deltaColumnMappingModeFromString(it->second.asString());
  }
  return std::make_shared<HiveDeltaSplit>(
      base->connectorId,
      base->filePath,
      base->fileFormat,
      base->start,
      base->length,
      base->partitionKeys,
      base->tableBucketNumber,
      base->customSplitInfo,
      base->extraFileInfo,
      base->cacheable,
      base->infoColumns,
      base->properties,
      obj.getDefault("hasDeletionVector", false).asBool(),
      mode);
}

// static
void HiveDeltaSplit::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("HiveDeltaSplit", HiveDeltaSplit::create);
}

} // namespace facebook::velox::connector::hive::delta
