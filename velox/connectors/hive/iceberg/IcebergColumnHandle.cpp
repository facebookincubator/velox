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

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/PartitionValue.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/dwio/common/ParquetFieldId.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/functions/prestosql/types/UuidType.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

// Produces a compact string of the set attributes in an IcebergFieldMetadata
// node. Only set fields are emitted; returns an empty string when empty().
std::string fieldMetadataToString(const IcebergFieldMetadata& meta) {
  std::ostringstream out;
  bool first = true;
  auto append = [&](const std::string& key, const std::string& value) {
    if (!first) {
      out << ", ";
    }
    out << key << ": " << value;
    first = false;
  };
  if (meta.required.has_value()) {
    append("required", *meta.required ? "true" : "false");
  }
  if (meta.longType.has_value()) {
    append("longType", *meta.longType);
  }
  if (meta.timestampUnit.has_value()) {
    append("timestampUnit", *meta.timestampUnit);
  }
  if (meta.binaryType.has_value()) {
    append("binaryType", *meta.binaryType);
  }
  if (meta.structType.has_value()) {
    append("structType", *meta.structType);
  }
  if (meta.length.has_value()) {
    append("length", std::to_string(*meta.length));
  }
  return out.str();
}

// Produces a compact string representation of a ParquetFieldId tree.
// Format: <id> for leaf nodes, <id>[<child0>, <child1>, ...] for nested nodes.
std::string fieldIdToString(const parquet::ParquetFieldId& fieldId) {
  std::string result = std::to_string(fieldId.fieldId);
  if (!fieldId.children.empty()) {
    result += "[";
    for (size_t i = 0; i < fieldId.children.size(); ++i) {
      if (i > 0) {
        result += ", ";
      }
      result += fieldIdToString(fieldId.children[i]);
    }
    result += "]";
  }
  return result;
}

// Serializes a ParquetFieldId tree to a folly::dynamic object.
folly::dynamic serializeFieldId(const parquet::ParquetFieldId& fieldId) {
  folly::dynamic obj = folly::dynamic::object;
  obj["fieldId"] = fieldId.fieldId;
  folly::dynamic children = folly::dynamic::array;
  for (const auto& child : fieldId.children) {
    children.push_back(serializeFieldId(child));
  }
  obj["children"] = children;
  return obj;
}

// Deserializes a ParquetFieldId tree from a folly::dynamic object.
parquet::ParquetFieldId deserializeFieldId(const folly::dynamic& obj) {
  parquet::ParquetFieldId fieldId;
  fieldId.fieldId = static_cast<int32_t>(obj["fieldId"].asInt());
  for (const auto& child : obj["children"]) {
    fieldId.children.push_back(deserializeFieldId(child));
  }
  return fieldId;
}

// Serializes an IcebergFieldMetadata node (and its children) to a
// folly::dynamic object. Only set optional fields are emitted; an all-empty
// node is serialized as an empty object so that the children array remains
// parallel to ParquetFieldId::children.
folly::dynamic serializeFieldMetadata(const IcebergFieldMetadata& meta) {
  folly::dynamic obj = folly::dynamic::object;
  if (meta.required.has_value()) {
    obj["required"] = *meta.required;
  }
  if (meta.longType.has_value()) {
    obj["longType"] = *meta.longType;
  }
  if (meta.timestampUnit.has_value()) {
    obj["timestampUnit"] = *meta.timestampUnit;
  }
  if (meta.binaryType.has_value()) {
    obj["binaryType"] = *meta.binaryType;
  }
  if (meta.structType.has_value()) {
    obj["structType"] = *meta.structType;
  }
  if (meta.length.has_value()) {
    obj["length"] = *meta.length;
  }
  folly::dynamic children = folly::dynamic::array;
  for (const auto& child : meta.children) {
    children.push_back(serializeFieldMetadata(child));
  }
  obj["children"] = children;
  return obj;
}

// Deserializes an IcebergFieldMetadata node (and its children) from a
// folly::dynamic object.
IcebergFieldMetadata deserializeFieldMetadata(const folly::dynamic& obj) {
  IcebergFieldMetadata meta;
  if (auto it = obj.find("required"); it != obj.items().end()) {
    meta.required = it->second.asBool();
  }
  if (auto it = obj.find("longType"); it != obj.items().end()) {
    meta.longType = it->second.asString();
  }
  if (auto it = obj.find("timestampUnit"); it != obj.items().end()) {
    meta.timestampUnit = it->second.asString();
  }
  if (auto it = obj.find("binaryType"); it != obj.items().end()) {
    meta.binaryType = it->second.asString();
  }
  if (auto it = obj.find("structType"); it != obj.items().end()) {
    meta.structType = it->second.asString();
  }
  if (auto it = obj.find("length"); it != obj.items().end()) {
    meta.length = static_cast<int32_t>(it->second.asInt());
  }
  if (auto it = obj.find("children"); it != obj.items().end()) {
    for (const auto& child : it->second) {
      meta.children.push_back(deserializeFieldMetadata(child));
    }
  }
  return meta;
}

} // namespace

IcebergColumnHandle::IcebergColumnHandle(
    const std::string& name,
    ColumnType columnType,
    TypePtr dataType,
    parquet::ParquetFieldId icebergField,
    std::vector<common::Subfield> requiredSubfields,
    std::optional<std::string> initialDefaultValue,
    IcebergFieldMetadata icebergMetadata,
    std::function<void(VectorPtr&)> postProcessor,
    std::optional<std::string> writeDefaultValue)
    : HiveColumnHandle(
          name,
          columnType,
          dataType,
          dataType,
          std::move(requiredSubfields),
          ColumnParseParameters{
              ColumnParseParameters::PartitionDateValueFormat::kDaysSinceEpoch},
          std::move(postProcessor)),
      field_(std::move(icebergField)),
      initialDefaultValue_(std::move(initialDefaultValue)),
      icebergMetadata_(std::move(icebergMetadata)),
      writeDefaultValue_(std::move(writeDefaultValue)) {}

const parquet::ParquetFieldId& IcebergColumnHandle::field() const {
  return field_;
}

std::string IcebergColumnHandle::toString() const {
  std::string fields = HiveColumnHandle::toStringFields();
  fields += ", field: " + fieldIdToString(field_);
  if (initialDefaultValue_.has_value()) {
    fields += ", initialDefaultValue: " + *initialDefaultValue_;
  }
  if (writeDefaultValue_.has_value()) {
    fields += ", writeDefaultValue: " + *writeDefaultValue_;
  }
  if (!icebergMetadata_.empty()) {
    fields +=
        ", icebergMetadata: {" + fieldMetadataToString(icebergMetadata_) + "}";
  }
  return fmt::format("IcebergColumnHandle [name: {}, {}]", name(), fields);
}

folly::dynamic IcebergColumnHandle::serialize() const {
  folly::dynamic obj = ColumnHandle::serializeBase("IcebergColumnHandle");
  obj["hiveColumnHandleName"] = name();
  obj["columnType"] = columnTypeName(columnType());
  obj["dataType"] = dataType()->serialize();

  folly::dynamic requiredSubfieldsArr = folly::dynamic::array;
  for (const auto& subfield : requiredSubfields()) {
    requiredSubfieldsArr.push_back(subfield.toString());
  }
  obj["requiredSubfields"] = requiredSubfieldsArr;

  obj["field"] = serializeFieldId(field_);

  if (initialDefaultValue_.has_value()) {
    obj["initialDefaultValue"] = *initialDefaultValue_;
  }

  if (writeDefaultValue_.has_value()) {
    obj["writeDefaultValue"] = *writeDefaultValue_;
  }

  // Only serialize icebergMetadata when at least one node carries a set
  // attribute, to keep the serialized form compact for callers that never
  // populate V3 metadata.
  if (!icebergMetadata_.empty() || !icebergMetadata_.children.empty()) {
    obj["icebergMetadata"] = serializeFieldMetadata(icebergMetadata_);
  }

  return obj;
}

// static
ColumnHandlePtr IcebergColumnHandle::create(const folly::dynamic& obj) {
  auto name = obj["hiveColumnHandleName"].asString();
  auto columnType = columnTypeFromName(obj["columnType"].asString());
  auto dataType = ISerializable::deserialize<Type>(obj["dataType"]);

  std::vector<common::Subfield> requiredSubfields;
  for (const auto& s : obj["requiredSubfields"]) {
    requiredSubfields.emplace_back(s.asString());
  }

  auto field = deserializeFieldId(obj["field"]);

  std::optional<std::string> initialDefaultValue;
  if (auto it = obj.find("initialDefaultValue"); it != obj.items().end()) {
    initialDefaultValue = it->second.asString();
  }

  std::optional<std::string> writeDefaultValue;
  if (auto it = obj.find("writeDefaultValue"); it != obj.items().end()) {
    writeDefaultValue = it->second.asString();
  }

  IcebergFieldMetadata icebergMetadata;
  if (auto it = obj.find("icebergMetadata"); it != obj.items().end()) {
    icebergMetadata = deserializeFieldMetadata(it->second);
  }

  return std::make_shared<IcebergColumnHandle>(
      name,
      columnType,
      std::move(dataType),
      std::move(field),
      std::move(requiredSubfields),
      std::move(initialDefaultValue),
      std::move(icebergMetadata),
      /*postProcessor=*/std::function<void(VectorPtr&)>{},
      std::move(writeDefaultValue));
}

// static
void IcebergColumnHandle::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("IcebergColumnHandle", IcebergColumnHandle::create);
}

namespace {

Variant parseDefaultValue(const std::string& value, const Type& type) {
  return PartitionValue::fromString(
      value,
      type,
      PartitionValue::TimestampMode::kUtc,
      PartitionValue::DateMode::kIsoString);
}

Variant icebergDefaultValueToVariant(
    const std::string& value,
    const TypePtr& type) {
  if (type->isVarbinary()) {
    return Variant::binary(encoding::Base64::decode(value));
  }
  if (isUuidType(type)) {
    // Same conversion as CAST(varchar AS uuid) in UuidRegistration.cpp.
    const auto uuid = boost::lexical_cast<boost::uuids::uuid>(value);
    int128_t parsed;
    memcpy(&parsed, &uuid, sizeof(parsed));
    return Variant::create<TypeKind::HUGEINT>(DecimalUtil::bigEndian(parsed));
  }
  if (type->isTime()) {
    // Iceberg time is microseconds since midnight; Presto TIME is milliseconds.
    const auto micros =
        parseDefaultValue(value, *BIGINT()).value<TypeKind::BIGINT>();
    return Variant(type.get() == TIME().get() ? micros / 1'000 : micros);
  }
  if (isTimestampWithTimeZoneType(type)) {
    // Iceberg stores timestamptz values in UTC.
    static const auto kUtcZoneId = tz::getTimeZoneID("UTC");
    return Variant(pack(
        parseDefaultValue(value, *TIMESTAMP()).value<TypeKind::TIMESTAMP>(),
        kUtcZoneId));
  }
  return parseDefaultValue(value, *type);
}

} // namespace

VectorPtr newIcebergDefaultValueConstant(
    const TypePtr& type,
    const std::optional<std::string>& value,
    memory::MemoryPool* pool) {
  if (!value.has_value()) {
    return BaseVector::createNullConstant(type, 1, pool);
  }
  return BaseVector::createConstant(
      type, icebergDefaultValueToVariant(value.value(), type), 1, pool);
}

} // namespace facebook::velox::connector::hive::iceberg
