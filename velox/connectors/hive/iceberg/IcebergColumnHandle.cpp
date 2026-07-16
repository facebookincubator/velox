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

#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/dwio/common/ParquetFieldId.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

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

} // namespace

IcebergColumnHandle::IcebergColumnHandle(
    const std::string& name,
    ColumnType columnType,
    TypePtr dataType,
    parquet::ParquetFieldId icebergField,
    std::vector<common::Subfield> requiredSubfields,
    std::optional<std::string> initialDefaultValue,
    IcebergFieldMetadata icebergMetadata)
    : HiveColumnHandle(
          name,
          columnType,
          dataType,
          dataType,
          std::move(requiredSubfields),
          ColumnParseParameters{ColumnParseParameters::
                                    PartitionDateValueFormat::kDaysSinceEpoch}),
      field_(std::move(icebergField)),
      initialDefaultValue_(std::move(initialDefaultValue)),
      icebergMetadata_(std::move(icebergMetadata)) {}

const parquet::ParquetFieldId& IcebergColumnHandle::field() const {
  return field_;
}

std::string IcebergColumnHandle::toString() const {
  std::ostringstream out;
  out << HiveColumnHandle::toString();
  out << ", iceberg_field_id: " << field_.fieldId;
  if (initialDefaultValue_.has_value()) {
    out << ", initial_default: " << *initialDefaultValue_;
  }
  return out.str();
}

folly::dynamic IcebergColumnHandle::serialize() const {
  folly::dynamic obj = ColumnHandle::serializeBase("IcebergColumnHandle");
  obj["hiveColumnHandleName"] = name();
  obj["columnType"] = columnTypeName(columnType());
  obj["dataType"] = dataType()->serialize();
  obj["hiveType"] = schemaType()->serialize();

  folly::dynamic requiredSubfieldsArr = folly::dynamic::array;
  for (const auto& subfield : requiredSubfields()) {
    requiredSubfieldsArr.push_back(subfield.toString());
  }
  obj["requiredSubfields"] = requiredSubfieldsArr;

  obj["icebergField"] = serializeFieldId(field_);

  if (initialDefaultValue_.has_value()) {
    obj["initialDefaultValue"] = *initialDefaultValue_;
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

  auto icebergField = deserializeFieldId(obj["icebergField"]);

  std::optional<std::string> initialDefaultValue;
  if (auto it = obj.find("initialDefaultValue"); it != obj.items().end()) {
    initialDefaultValue = it->second.asString();
  }

  return std::make_shared<IcebergColumnHandle>(
      name,
      columnType,
      std::move(dataType),
      std::move(icebergField),
      std::move(requiredSubfields),
      std::move(initialDefaultValue));
}

// static
void IcebergColumnHandle::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("IcebergColumnHandle", IcebergColumnHandle::create);
}

} // namespace facebook::velox::connector::hive::iceberg
