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
#include "velox/connectors/hive/paimon/PaimonSchema.h"

#include <folly/String.h>
#include <algorithm>
#include <limits>
#include <tuple>
#include <unordered_set>

namespace facebook::velox::connector::hive::paimon {
namespace {

constexpr int32_t kSystemFieldIdStart = std::numeric_limits<int32_t>::max() / 2;
constexpr uint32_t kMaxTypeDepth = 128;

int32_t readInt32(const folly::dynamic& value) {
  VELOX_USER_CHECK(value.isInt(), "Expected a Paimon integer");
  const auto number = value.asInt();
  VELOX_USER_CHECK(
      number >= std::numeric_limits<int32_t>::min() &&
          number <= std::numeric_limits<int32_t>::max(),
      "Paimon integer out of range: {}",
      number);
  return static_cast<int32_t>(number);
}

// Container wire is expanded so a schema never depends on a thread-local
// RowType serialization cache.
folly::dynamic serializeType(const TypePtr& type) {
  if (!type->isRow() && !type->isArray() && !type->isMap()) {
    return type->serialize();
  }
  folly::dynamic obj =
      folly::dynamic::object("name", "Type")("type", type->kindName());
  if (type->isRow()) {
    obj["names"] = ISerializable::serialize(type->asRow().names());
  }
  obj["cTypes"] = folly::dynamic::array;
  for (column_index_t i = 0; i < type->size(); ++i) {
    obj["cTypes"].push_back(serializeType(type->childAt(i)));
  }
  return obj;
}

// Validate before Type deserialization, which recursively decodes children and
// narrows decimal parameters to uint8_t.
void validateTypeWire(const folly::dynamic& obj, uint32_t depth) {
  VELOX_USER_CHECK_LE(depth, kMaxTypeDepth, "Paimon type nesting is too deep");
  VELOX_USER_CHECK(obj.isObject(), "Expected Paimon Velox type object");
  VELOX_USER_CHECK(
      !obj.count("ref"), "Paimon schema type must be self-contained");
  VELOX_USER_CHECK(obj.at("type").isString(), "Expected Velox type name");
  const auto& name = obj.at("type").getString();
  VELOX_USER_CHECK(
      obj.at("name").isString() &&
          obj.at("name").getString() == (name == "DATE" ? "DateType" : "Type"),
      "Invalid Paimon type serde name");
  if (name == "DECIMAL") {
    const auto precision = readInt32(obj.at("precision"));
    const auto scale = readInt32(obj.at("scale"));
    VELOX_USER_CHECK(
        precision >= 1 && precision <= 38 && scale >= 0 && scale <= precision,
        "Invalid Paimon decimal precision/scale");
  }
  if (name == "ROW" || name == "ARRAY" || name == "MAP") {
    const auto& children = obj.at("cTypes");
    VELOX_USER_CHECK(children.isArray(), "Expected Velox type children array");
    if (name == "ROW") {
      VELOX_USER_CHECK(obj.at("names").isArray(), "Expected ROW names array");
      VELOX_USER_CHECK_EQ(
          obj.at("names").size(), children.size(), "Paimon ROW shape mismatch");
      for (const auto& fieldName : obj.at("names")) {
        VELOX_USER_CHECK(fieldName.isString(), "Expected Paimon field name");
      }
    } else {
      VELOX_USER_CHECK_EQ(
          children.size(),
          name == "ARRAY" ? 1 : 2,
          "Paimon type shape mismatch");
    }
    for (const auto& child : children) {
      validateTypeWire(child, depth + 1);
    }
  } else {
    VELOX_USER_CHECK(
        name == "BOOLEAN" || name == "TINYINT" || name == "SMALLINT" ||
            name == "INTEGER" || name == "BIGINT" || name == "REAL" ||
            name == "DOUBLE" || name == "DATE" || name == "VARCHAR" ||
            name == "VARBINARY" || name == "DECIMAL" || name == "TIMESTAMP",
        "Unsupported Paimon Velox type: {}",
        name);
    VELOX_USER_CHECK(
        !obj.count("cTypes") ||
            (obj.at("cTypes").isArray() && obj.at("cTypes").empty()),
        "Paimon scalar type must not have children");
  }
}

void validateType(
    const TypePtr& type,
    const PaimonTypeMetadata& metadata,
    bool isField,
    std::unordered_set<int32_t>& ids,
    uint32_t depth) {
  VELOX_USER_CHECK_NOT_NULL(type, "Missing Paimon field type");
  VELOX_USER_CHECK_LE(depth, kMaxTypeDepth, "Paimon type nesting is too deep");
  VELOX_USER_CHECK_EQ(
      metadata.fieldId.has_value(),
      isField,
      "Paimon DataField IDs are required only for ROW children");
  if (isField) {
    const auto id = *metadata.fieldId;
    VELOX_USER_CHECK(
        id >= 0 && id < kSystemFieldIdStart,
        "Invalid Paimon DataField ID: {}",
        id);
    VELOX_USER_CHECK(
        ids.insert(id).second, "Duplicate Paimon field ID: {}", id);
  }

  const auto& root = metadata.logicalType;
  TypePtr expected;
  size_t parameterCount = 0;
  if (root == "BOOLEAN") {
    expected = BOOLEAN();
  } else if (root == "TINYINT") {
    expected = TINYINT();
  } else if (root == "SMALLINT") {
    expected = SMALLINT();
  } else if (root == "INTEGER") {
    expected = INTEGER();
  } else if (root == "BIGINT") {
    expected = BIGINT();
  } else if (root == "FLOAT") {
    expected = REAL();
  } else if (root == "DOUBLE") {
    expected = DOUBLE();
  } else if (root == "DATE") {
    expected = DATE();
  } else if (root == "CHAR" || root == "VARCHAR") {
    expected = VARCHAR();
    parameterCount = 1;
  } else if (root == "BINARY" || root == "VARBINARY") {
    expected = VARBINARY();
    parameterCount = 1;
  } else if (
      root == "TIMESTAMP_WITHOUT_TIME_ZONE" ||
      root == "TIMESTAMP_WITH_LOCAL_TIME_ZONE") {
    expected = TIMESTAMP();
    parameterCount = 1;
  } else if (root == "DECIMAL") {
    VELOX_USER_CHECK(
        type->isDecimal(), "Paimon DECIMAL requires a decimal type");
    parameterCount = 2;
  } else if (root == "ROW") {
    VELOX_USER_CHECK(type->isRow(), "Paimon ROW requires a ROW type");
    std::unordered_set<std::string> names;
    for (const auto& name : type->asRow().names()) {
      VELOX_USER_CHECK(
          !folly::trimWhitespace(name).empty(),
          "Paimon field name must not be empty or blank");
      VELOX_USER_CHECK(
          names.insert(name).second, "Duplicate Paimon field name: {}", name);
    }
  } else if (root == "ARRAY") {
    VELOX_USER_CHECK(type->isArray(), "Paimon ARRAY requires an ARRAY type");
  } else if (root == "MAP") {
    VELOX_USER_CHECK(type->isMap(), "Paimon MAP requires a MAP type");
  } else {
    VELOX_UNSUPPORTED("Unsupported Paimon logical type: {}", root);
  }
  if (expected) {
    VELOX_USER_CHECK(
        *type == *expected,
        "Paimon logical type {} does not match Velox type {}",
        root,
        type->toString());
  }
  VELOX_USER_CHECK_EQ(
      metadata.parameters.size(),
      parameterCount,
      "Invalid parameters for Paimon type {}",
      root);
  if (root == "DECIMAL") {
    const auto [precision, scale] = getDecimalPrecisionScale(*type);
    VELOX_USER_CHECK(
        metadata.parameters[0] == precision && metadata.parameters[1] == scale,
        "Paimon decimal precision/scale do not match {}",
        type->toString());
  } else if (parameterCount == 1) {
    const auto parameter = metadata.parameters[0];
    if (type->isTimestamp()) {
      VELOX_USER_CHECK(
          parameter >= 0 && parameter <= 9,
          "Invalid Paimon timestamp precision: {}",
          parameter);
    } else {
      VELOX_USER_CHECK_GT(parameter, 0, "Paimon type length must be positive");
    }
  }
  VELOX_USER_CHECK_EQ(
      metadata.children.size(),
      type->size(),
      "Paimon type metadata shape mismatch");
  for (column_index_t i = 0; i < type->size(); ++i) {
    validateType(
        type->childAt(i), metadata.children[i], type->isRow(), ids, depth + 1);
  }
}

PaimonTypeMetadata readMetadata(const folly::dynamic& obj, uint32_t depth) {
  VELOX_USER_CHECK_LE(depth, kMaxTypeDepth, "Paimon type nesting is too deep");
  VELOX_USER_CHECK(obj.isObject(), "Expected Paimon type metadata object");
  VELOX_USER_CHECK(
      obj.at("logicalType").isString(), "Expected Paimon logical type name");
  VELOX_USER_CHECK(
      obj.at("nullable").isBool(), "Expected Paimon nullability boolean");
  PaimonTypeMetadata result;
  result.logicalType = obj.at("logicalType").asString();
  result.nullable = obj.at("nullable").asBool();
  if (!obj.at("fieldId").isNull()) {
    result.fieldId = readInt32(obj.at("fieldId"));
  }
  VELOX_USER_CHECK(
      obj.at("parameters").isArray(), "Expected type parameters array");
  for (const auto& parameter : obj.at("parameters")) {
    result.parameters.push_back(readInt32(parameter));
  }
  VELOX_USER_CHECK(
      obj.at("children").isArray(), "Expected type metadata children array");
  for (const auto& child : obj.at("children")) {
    result.children.push_back(readMetadata(child, depth + 1));
  }
  return result;
}

void indexFields(
    const PaimonTypeMetadata& metadata,
    std::vector<column_index_t>& path,
    std::unordered_map<int32_t, std::vector<column_index_t>>& paths) {
  if (metadata.fieldId) {
    paths.emplace(*metadata.fieldId, path);
  }
  for (column_index_t i = 0; i < metadata.children.size(); ++i) {
    path.push_back(i);
    indexFields(metadata.children[i], path, paths);
    path.pop_back();
  }
}

std::vector<int32_t> readIds(const folly::dynamic& array) {
  VELOX_USER_CHECK(array.isArray(), "Expected Paimon field ID array");
  std::vector<int32_t> ids;
  for (const auto& value : array) {
    ids.push_back(readInt32(value));
  }
  return ids;
}

} // namespace

bool PaimonTypeMetadata::operator==(const PaimonTypeMetadata& other) const {
  return std::tie(logicalType, nullable, fieldId, parameters, children) ==
      std::tie(
             other.logicalType,
             other.nullable,
             other.fieldId,
             other.parameters,
             other.children);
}

void PaimonTypeMetadata::validate(const TypePtr& type, bool isField) const {
  std::unordered_set<int32_t> ids;
  validateType(type, *this, isField, ids, 0);
}

folly::dynamic PaimonTypeMetadata::serialize() const {
  folly::dynamic obj =
      folly::dynamic::object("logicalType", logicalType)("nullable", nullable);
  obj["fieldId"] = fieldId ? folly::dynamic(*fieldId) : folly::dynamic(nullptr);
  obj["parameters"] = folly::dynamic::array;
  for (auto parameter : parameters) {
    obj["parameters"].push_back(parameter);
  }
  obj["children"] = folly::dynamic::array;
  for (const auto& child : children) {
    obj["children"].push_back(child.serialize());
  }
  return obj;
}

PaimonTypeMetadata PaimonTypeMetadata::create(const folly::dynamic& obj) {
  return readMetadata(obj, 0);
}

PaimonSchema::PaimonSchema(
    int64_t schemaId,
    int32_t schemaVersion,
    RowTypePtr rowType,
    PaimonTypeMetadata metadata,
    std::vector<int32_t> partitionKeys,
    std::vector<int32_t> primaryKeys,
    std::unordered_map<std::string, std::string> options)
    : schemaId_(schemaId),
      schemaVersion_(schemaVersion),
      rowType_(std::move(rowType)),
      metadata_(std::move(metadata)),
      partitionKeys_(std::move(partitionKeys)),
      primaryKeys_(std::move(primaryKeys)),
      options_(std::move(options)) {
  VELOX_USER_CHECK_GE(schemaId_, 0, "Paimon schema ID must be non-negative");
  VELOX_USER_CHECK(
      schemaVersion_ >= 1 && schemaVersion_ <= 3,
      "Unsupported Paimon schema version: {}",
      schemaVersion_);
  VELOX_USER_CHECK_NOT_NULL(rowType_, "Missing Paimon schema type");
  metadata_.validate(rowType_, false);
  std::vector<column_index_t> path;
  indexFields(metadata_, path, fieldIndexPaths_);
  const auto validateKeys = [&](const auto& keys, std::string_view label) {
    std::unordered_set<int32_t> unique;
    for (auto id : keys) {
      VELOX_USER_CHECK(
          unique.insert(id).second,
          "Duplicate Paimon {} field ID: {}",
          label,
          id);
      VELOX_USER_CHECK_EQ(
          fieldIndexPath(id).size(),
          1,
          "Paimon {} must reference top-level fields",
          label);
      const auto& type = fieldType(id);
      VELOX_USER_CHECK(
          !type->isRow() && !type->isArray() && !type->isMap(),
          "Paimon {} must reference scalar fields",
          label);
    }
  };
  validateKeys(partitionKeys_, "partition key");
  validateKeys(primaryKeys_, "primary key");
  if (!primaryKeys_.empty()) {
    const std::unordered_set<int32_t> partitions(
        partitionKeys_.begin(), partitionKeys_.end());
    VELOX_USER_CHECK(
        std::any_of(
            primaryKeys_.begin(),
            primaryKeys_.end(),
            [&](int32_t id) { return partitions.count(id) == 0; }),
        "Paimon primary keys must include a non-partition field");
  }
}

const std::vector<column_index_t>& PaimonSchema::fieldIndexPath(
    int32_t fieldId) const {
  const auto it = fieldIndexPaths_.find(fieldId);
  VELOX_USER_CHECK(
      it != fieldIndexPaths_.end(),
      "Unknown Paimon field ID {} in schema {}",
      fieldId,
      schemaId_);
  return it->second;
}

TypePtr PaimonSchema::fieldType(int32_t fieldId) const {
  TypePtr type = rowType_;
  for (auto index : fieldIndexPath(fieldId)) {
    type = type->childAt(index);
  }
  return type;
}

const PaimonTypeMetadata& PaimonSchema::fieldMetadata(int32_t fieldId) const {
  const auto* metadata = &metadata_;
  for (auto index : fieldIndexPath(fieldId)) {
    metadata = &metadata->children[index];
  }
  return *metadata;
}

std::vector<int32_t> PaimonSchema::fieldIdPath(
    const common::Subfield& subfield) const {
  const auto& path = subfield.path();
  VELOX_USER_CHECK(!path.empty(), "Empty Paimon field path");
  const auto* root = path.front()->as<common::Subfield::NestedField>();
  VELOX_USER_CHECK_NOT_NULL(
      root, "Paimon field path must start with a column name");
  const auto index = rowType_->getChildIdxIfExists(root->name());
  VELOX_USER_CHECK(index.has_value(), "Unknown Paimon field: {}", root->name());
  auto type = rowType_->childAt(*index);
  const auto* metadata = &metadata_.children[*index];
  std::vector<int32_t> ids{*metadata->fieldId};
  for (size_t i = 1; i < path.size(); ++i) {
    const auto& element = path[i];
    column_index_t child = 0;
    if (const auto* field = element->as<common::Subfield::NestedField>()) {
      VELOX_USER_CHECK(
          type->isRow(), "ROW required in Paimon path {}", subfield.toString());
      const auto childIndex = type->asRow().getChildIdxIfExists(field->name());
      VELOX_USER_CHECK(
          childIndex.has_value(),
          "Unknown Paimon subfield: {}",
          subfield.toString());
      child = *childIndex;
      ids.push_back(*metadata->children[child].fieldId);
    } else if (type->isArray()) {
      const auto* subscript = element->as<common::Subfield::LongSubscript>();
      VELOX_USER_CHECK(
          element->is(common::SubfieldKind::kAllSubscripts) ||
              (subscript && subscript->index() > 0),
          "Invalid ARRAY subscript in Paimon path {}",
          subfield.toString());
    } else if (type->isMap()) {
      const auto& keyType = type->childAt(0);
      VELOX_USER_CHECK(
          element->is(common::SubfieldKind::kAllSubscripts) ||
              (element->is(common::SubfieldKind::kStringSubscript) &&
               keyType->isVarchar()) ||
              (element->is(common::SubfieldKind::kLongSubscript) &&
               (keyType->isTinyint() || keyType->isSmallint() ||
                keyType->isInteger() || keyType->isBigint()) &&
               !keyType->isDate() && !keyType->isDecimal()),
          "Invalid MAP subscript in Paimon path {}",
          subfield.toString());
      child = 1;
    } else {
      VELOX_USER_FAIL("Invalid Paimon subfield path: {}", subfield.toString());
    }
    type = type->childAt(child);
    metadata = &metadata->children[child];
  }
  return ids;
}

folly::dynamic PaimonSchema::serialize() const {
  folly::dynamic obj = folly::dynamic::object("version", kSerdeVersion)(
      "schemaId", schemaId_)("schemaVersion", schemaVersion_);
  obj["rowType"] = serializeType(rowType_);
  obj["metadata"] = metadata_.serialize();
  obj["partitionKeys"] = folly::dynamic::array;
  obj["primaryKeys"] = folly::dynamic::array;
  for (auto id : partitionKeys_) {
    obj["partitionKeys"].push_back(id);
  }
  for (auto id : primaryKeys_) {
    obj["primaryKeys"].push_back(id);
  }
  obj["options"] = folly::dynamic::object;
  for (const auto& [key, value] : options_) {
    obj["options"][key] = value;
  }
  return obj;
}

PaimonSchemaPtr PaimonSchema::create(const folly::dynamic& obj) {
  VELOX_USER_CHECK(obj.isObject(), "Expected Paimon schema object");
  VELOX_USER_CHECK(
      obj.at("version").isInt() && obj.at("version").asInt() == kSerdeVersion,
      "Unsupported Paimon schema wire version");
  VELOX_USER_CHECK(obj.at("schemaId").isInt(), "Expected Paimon schema ID");
  validateTypeWire(obj.at("rowType"), 0);
  auto type = ISerializable::deserialize<Type>(obj.at("rowType"));
  VELOX_USER_CHECK(type && type->isRow(), "Paimon schema must have a ROW type");
  VELOX_USER_CHECK(
      obj.at("options").isObject(), "Expected Paimon schema options object");
  std::unordered_map<std::string, std::string> options;
  for (const auto& [key, value] : obj.at("options").items()) {
    VELOX_USER_CHECK(
        key.isString() && value.isString(),
        "Paimon schema options must be strings");
    options.emplace(key.asString(), value.asString());
  }
  return std::make_shared<PaimonSchema>(
      obj.at("schemaId").asInt(),
      readInt32(obj.at("schemaVersion")),
      asRowType(type),
      PaimonTypeMetadata::create(obj.at("metadata")),
      readIds(obj.at("partitionKeys")),
      readIds(obj.at("primaryKeys")),
      std::move(options));
}

} // namespace facebook::velox::connector::hive::paimon
