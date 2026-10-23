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

#include <folly/Unicode.h>
#include <algorithm>
#include <limits>
#include <tuple>
#include <unordered_set>

#include "velox/functions/lib/string/StringImpl.h"

namespace facebook::velox::connector::hive::paimon {
namespace {

constexpr int32_t kSystemFieldIdStart = std::numeric_limits<int32_t>::max() / 2;
// Bound recursive schema decoding independently of Paimon's logical types.
constexpr uint32_t kMaxTypeDepth = 128;

bool isBlankFieldName(const std::string& name) {
  const auto* position = reinterpret_cast<const unsigned char*>(name.data());
  const auto* end = position + name.size();
  while (position < end) {
    const auto codePoint = folly::utf8ToCodePoint(position, end, false);
    // Matches Paimon's Java Character.isWhitespace semantics.
    if (!functions::stringImpl::isUnicodeWhiteSpace(codePoint)) {
      return false;
    }
  }
  return true;
}

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

// Bound recursive type decoding and prevent decimal parameter truncation.
void validateTypeWire(const folly::dynamic& obj, uint32_t depth) {
  VELOX_USER_CHECK_LE(depth, kMaxTypeDepth, "Paimon type nesting is too deep");
  VELOX_USER_CHECK(obj.isObject(), "Expected Paimon Velox type object");
  VELOX_USER_CHECK(
      !obj.count("ref"), "Paimon schema type must be self-contained");
  if (isDecimalName(obj.at("type").asString())) {
    ISerializable::deserialize<uint8_t>(obj.at("precision"));
    ISerializable::deserialize<uint8_t>(obj.at("scale"));
  }
  if (const auto* children = obj.get_ptr("cTypes")) {
    VELOX_USER_CHECK(children->isArray(), "Expected Velox type children array");
    for (const auto& child : *children) {
      validateTypeWire(child, depth + 1);
    }
  }
}

void validateType(
    const TypePtr& type,
    const PaimonTypeMetadata& metadata,
    bool isField,
    std::vector<column_index_t>& path,
    std::unordered_map<int32_t, std::vector<column_index_t>>& paths) {
  VELOX_USER_CHECK_NOT_NULL(type, "Missing Paimon field type");
  VELOX_USER_CHECK_LE(
      path.size(), kMaxTypeDepth, "Paimon type nesting is too deep");
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
        paths.emplace(id, path).second, "Duplicate Paimon field ID: {}", id);
  }

  const auto& root = metadata.logicalType;
  size_t parameterCount = 0;
  if (root == "DECIMAL") {
    VELOX_USER_CHECK(
        type->isDecimal(), "Paimon DECIMAL requires a decimal type");
    parameterCount = 2;
  } else if (root == "ROW") {
    VELOX_USER_CHECK(type->isRow(), "Paimon ROW requires a ROW type");
    std::unordered_set<std::string> names;
    for (const auto& name : type->asRow().names()) {
      VELOX_USER_CHECK(
          !isBlankFieldName(name),
          "Paimon field name must not be empty or blank");
      VELOX_USER_CHECK(
          names.insert(name).second, "Duplicate Paimon field name: {}", name);
    }
  } else if (root == "ARRAY") {
    VELOX_USER_CHECK(type->isArray(), "Paimon ARRAY requires an ARRAY type");
  } else if (root == "MAP") {
    VELOX_USER_CHECK(type->isMap(), "Paimon MAP requires a MAP type");
  } else {
    // Logical aliases share a Velox type; their parameters remain in metadata.
    static const std::unordered_map<std::string, TypePtr> kScalarTypes{
        {"BOOLEAN", BOOLEAN()},
        {"TINYINT", TINYINT()},
        {"SMALLINT", SMALLINT()},
        {"INTEGER", INTEGER()},
        {"BIGINT", BIGINT()},
        {"FLOAT", REAL()},
        {"DOUBLE", DOUBLE()},
        {"DATE", DATE()},
        {"CHAR", VARCHAR()},
        {"VARCHAR", VARCHAR()},
        {"BINARY", VARBINARY()},
        {"VARBINARY", VARBINARY()},
        {"TIMESTAMP_WITHOUT_TIME_ZONE", TIMESTAMP()},
        {"TIMESTAMP_WITH_LOCAL_TIME_ZONE", TIMESTAMP()}};
    const auto it = kScalarTypes.find(root);
    if (it == kScalarTypes.end()) {
      VELOX_UNSUPPORTED("Unsupported Paimon logical type: {}", root);
    }
    const auto& expected = it->second;
    VELOX_USER_CHECK(
        *type == *expected,
        "Paimon logical type {} does not match Velox type {}",
        root,
        type->toString());
    if (expected->isVarchar() || expected->isVarbinary() ||
        expected->isTimestamp()) {
      parameterCount = 1;
    }
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
    path.push_back(i);
    validateType(
        type->childAt(i), metadata.children[i], type->isRow(), path, paths);
    path.pop_back();
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
  std::vector<column_index_t> path;
  validateType(rowType_, metadata_, false, path, fieldIndexPaths_);
  const auto validateKeys = [&](const auto& keys, std::string_view label) {
    std::unordered_set<int32_t> unique;
    for (auto id : keys) {
      VELOX_USER_CHECK(
          unique.insert(id).second,
          "Duplicate Paimon {} field ID: {}",
          label,
          id);
      const auto& fieldPath = fieldIndexPath(id);
      VELOX_USER_CHECK_EQ(
          fieldPath.size(),
          1,
          "Paimon {} must reference top-level fields",
          label);
      const auto& type = rowType_->childAt(fieldPath.front());
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

bool PaimonSchema::operator==(const PaimonSchema& other) const {
  return schemaId_ == other.schemaId_ &&
      schemaVersion_ == other.schemaVersion_ && *rowType_ == *other.rowType_ &&
      metadata_ == other.metadata_ && partitionKeys_ == other.partitionKeys_ &&
      primaryKeys_ == other.primaryKeys_ && options_ == other.options_;
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
    } else if (type->isArray() || type->isMap()) {
      child = type->isMap() ? 1 : 0;
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
