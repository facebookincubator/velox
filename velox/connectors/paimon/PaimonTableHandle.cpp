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
#include "velox/connectors/paimon/PaimonTableHandle.h"

#include <fmt/ranges.h>
#include <folly/String.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_set>

namespace facebook::velox::connector::paimon {
namespace {

std::string toLower(std::string_view value) {
  std::string lower(value);
  std::transform(lower.begin(), lower.end(), lower.begin(), [](auto c) {
    return std::tolower(static_cast<unsigned char>(c));
  });
  return lower;
}

} // namespace

std::string paimonMergeEngineString(PaimonMergeEngine mergeEngine) {
  switch (mergeEngine) {
    case PaimonMergeEngine::kDeduplicate:
      return "deduplicate";
    case PaimonMergeEngine::kPartialUpdate:
      return "partial-update";
    case PaimonMergeEngine::kAggregation:
      return "aggregation";
    case PaimonMergeEngine::kFirstRow:
      return "first-row";
    default:
      VELOX_FAIL(
          "Unknown PaimonMergeEngine: {}", static_cast<int>(mergeEngine));
  }
}

PaimonMergeEngine paimonMergeEngineFromString(const std::string& str) {
  const auto lower = toLower(str);
  for (auto mergeEngine :
       {PaimonMergeEngine::kDeduplicate,
        PaimonMergeEngine::kPartialUpdate,
        PaimonMergeEngine::kAggregation,
        PaimonMergeEngine::kFirstRow}) {
    if (lower == paimonMergeEngineString(mergeEngine)) {
      return mergeEngine;
    }
  }
  VELOX_USER_FAIL("Unknown Paimon merge engine: {}", str);
}

namespace {

// Paimon writes partition columns into data files at their table positions, so
// every table column is stored. Readers that bind file columns by position
// need the partition columns in place; partition values still come from the
// split.
RowTypePtr storedDataColumns(
    const RowTypePtr& rowType,
    const std::vector<std::string>& partitionKeys) {
  VELOX_USER_CHECK_NOT_NULL(rowType, "Missing Paimon table row type");
  std::unordered_set<column_index_t> partitions;
  for (const auto& name : partitionKeys) {
    const auto index = rowType->getChildIdxIfExists(name);
    VELOX_USER_CHECK(
        index.has_value(), "Unknown Paimon partition column '{}'", name);
    VELOX_USER_CHECK(
        partitions.insert(*index).second,
        "Duplicate Paimon partition column '{}'",
        name);
  }
  return rowType;
}

// Checks the primary keys and returns those that are not partition columns, as
// Paimon Java's TableSchema.trimmedPrimaryKeys does.
std::vector<std::string> trimPrimaryKeys(
    PaimonTableType tableType,
    const RowTypePtr& rowType,
    const std::vector<std::string>& primaryKeys,
    const std::vector<std::string>& partitionKeys) {
  if (primaryKeys.empty()) {
    return {};
  }
  VELOX_USER_CHECK_EQ(
      tableType,
      PaimonTableType::kPrimaryKey,
      "Paimon append-only table cannot have primary keys");
  std::unordered_set<std::string> keys;
  std::vector<std::string> trimmed;
  for (const auto& key : primaryKeys) {
    VELOX_USER_CHECK(
        rowType->containsChild(key), "Unknown Paimon primary key: {}", key);
    VELOX_USER_CHECK(
        keys.insert(key).second, "Duplicate Paimon primary key: {}", key);
    if (std::find(partitionKeys.begin(), partitionKeys.end(), key) ==
        partitionKeys.end()) {
      trimmed.push_back(key);
    }
  }
  VELOX_USER_CHECK(
      !trimmed.empty(),
      "Paimon primary keys must not all be partition columns: [{}]",
      fmt::join(primaryKeys, ", "));
  return trimmed;
}

bool parseBooleanOption(std::string_view key, std::string_view value) {
  const auto lower = toLower(value);
  if (lower == "true") {
    return true;
  }
  VELOX_USER_CHECK_EQ(
      lower, "false", "Paimon table option must be true or false: {}", key);
  return false;
}

// Parses the merge-on-read options of a primary-key table as Paimon Java's
// CoreOptions does, and checks them as its SchemaValidation does.
PaimonTableHandle::MergeOptions parseMergeOptions(
    PaimonTableType tableType,
    const RowTypePtr& rowType,
    const std::unordered_map<std::string, std::string>& options) {
  PaimonTableHandle::MergeOptions mergeOptions;
  if (tableType != PaimonTableType::kPrimaryKey) {
    return mergeOptions;
  }
  if (auto it = options.find("merge-engine"); it != options.end()) {
    mergeOptions.mergeEngine = paimonMergeEngineFromString(it->second);
  }
  for (const auto* key :
       {"ignore-delete",
        "first-row.ignore-delete",
        "deduplicate.ignore-delete",
        "partial-update.ignore-delete"}) {
    if (auto it = options.find(key); it != options.end()) {
      mergeOptions.ignoreDelete = parseBooleanOption(key, it->second);
      break;
    }
  }
  if (auto it = options.find("sequence.field"); it != options.end()) {
    // Java's String.split drops trailing empty fields before each field is
    // trimmed.
    std::vector<std::string_view> fields;
    folly::split(',', it->second, fields);
    while (!fields.empty() && fields.back().empty()) {
      fields.pop_back();
    }
    std::unordered_set<std::string> names;
    for (auto field : fields) {
      std::string name(folly::trimWhitespace(field));
      VELOX_USER_CHECK(
          rowType->containsChild(name),
          "Unknown Paimon sequence field: {}",
          name);
      VELOX_USER_CHECK(
          names.insert(name).second,
          "Duplicate Paimon sequence field: {}",
          name);
      mergeOptions.sequenceFields.push_back(std::move(name));
    }
  }
  if (auto it = options.find("sequence.field.sort-order");
      it != options.end()) {
    const auto order = toLower(it->second);
    VELOX_USER_CHECK(
        order == "ascending" || order == "descending",
        "Unknown Paimon sequence.field.sort-order: {}",
        it->second);
    mergeOptions.sequenceFieldsAscending = order == "ascending";
  }
  VELOX_USER_CHECK(
      mergeOptions.sequenceFields.empty() ||
          mergeOptions.mergeEngine != PaimonMergeEngine::kFirstRow,
      "Paimon first-row merge engine does not support sequence.field");
  return mergeOptions;
}

} // namespace

PaimonTableHandle::PaimonTableHandle(
    std::string connectorId,
    std::string tableName,
    int64_t schemaId,
    RowTypePtr rowType,
    std::vector<int32_t> fieldIds,
    PaimonTableType tableType,
    std::vector<std::string> partitionKeys,
    std::unordered_map<std::string, std::string> options,
    common::SubfieldFilters subfieldFilters,
    core::TypedExprPtr remainingFilter,
    std::vector<connector::FileColumnHandlePtr> filterColumnHandles,
    double sampleRate,
    std::string dbName,
    std::vector<std::string> primaryKeys)
    : connector::FileTableHandle(
          std::move(connectorId),
          std::move(tableName),
          std::move(subfieldFilters),
          std::move(remainingFilter),
          storedDataColumns(rowType, partitionKeys),
          /*tableParameters=*/{},
          sampleRate,
          std::move(dbName)),
      schemaId_(schemaId),
      rowType_(std::move(rowType)),
      fieldIds_(std::move(fieldIds)),
      tableType_(tableType),
      partitionKeys_(std::move(partitionKeys)),
      options_(std::move(options)),
      filterColumnHandles_(std::move(filterColumnHandles)),
      primaryKeys_(std::move(primaryKeys)),
      trimmedPrimaryKeys_(
          trimPrimaryKeys(tableType_, rowType_, primaryKeys_, partitionKeys_)),
      mergeOptions_(parseMergeOptions(tableType_, rowType_, options_)) {
  VELOX_USER_CHECK_GE(schemaId_, 0, "Paimon schema ID must be nonnegative");
  VELOX_USER_CHECK_EQ(
      fieldIds_.size(),
      rowType_->size(),
      "Paimon field IDs must be aligned to table columns");
  std::unordered_set<int32_t> ids;
  for (auto id : fieldIds_) {
    VELOX_USER_CHECK_GE(id, 0, "Paimon field ID must be nonnegative");
    VELOX_USER_CHECK(ids.insert(id).second, "Duplicate Paimon field ID {}", id);
  }
  VELOX_USER_CHECK(
      std::isfinite(sampleRate) && sampleRate > 0 && sampleRate <= 1,
      "Paimon sample rate must be finite and in (0, 1]");
}

std::string PaimonTableHandle::toString() const {
  return fmt::format(
      "PaimonTableHandle[table {}, schemaId {}, tableType {}, dataColumns {}, "
      "sampleRate {}]",
      tableName(),
      schemaId_,
      tableType_,
      dataColumns()->toString(),
      sampleRate());
}

folly::dynamic PaimonTableHandle::serialize() const {
  for (const auto& [subfield, filter] : subfieldFilters()) {
    VELOX_USER_CHECK_NOT_NULL(
        filter,
        "Null filter for subfield '{}' in table '{}'",
        subfield.toString(),
        tableName());
  }
  for (size_t i = 0; i < filterColumnHandles_.size(); ++i) {
    VELOX_USER_CHECK_NOT_NULL(
        filterColumnHandles_[i],
        "Null filter column handle at index {} in table '{}'",
        i,
        tableName());
  }

  folly::dynamic obj = serializeFileFields("PaimonTableHandle");
  obj["version"] = kSerdeVersion;
  obj["schemaId"] = schemaId_;
  obj["rowType"] = rowType_->serialize();
  obj["fieldIds"] = ISerializable::serialize(fieldIds_);
  obj["tableType"] = paimonTableTypeString(tableType_);
  obj["partitionKeys"] = ISerializable::serialize(partitionKeys_);
  obj["options"] = folly::dynamic::object;
  for (const auto& [key, value] : options_) {
    obj["options"][key] = value;
  }
  if (!primaryKeys_.empty()) {
    obj["primaryKeys"] = ISerializable::serialize(primaryKeys_);
  }
  return obj;
}

ConnectorTableHandlePtr PaimonTableHandle::create(
    const folly::dynamic& obj,
    void* context) {
  VELOX_USER_CHECK(
      obj.at("version").isInt() && obj.at("version").asInt() == kSerdeVersion,
      "Unsupported Paimon table handle wire version");

  std::string connectorId;
  std::string tableName;
  std::string dbName;
  double sampleRate{1.0};
  core::TypedExprPtr remainingFilter;
  common::SubfieldFilters filters;
  std::vector<connector::FileColumnHandlePtr> filterColumns;
  VELOX_USER_CHECK(
      obj["subfieldFilters"].isArray(), "Expected Paimon filter array");
  if (auto it = obj.find("filterColumnHandles"); it != obj.items().end()) {
    VELOX_USER_CHECK(
        it->second.isArray(), "Expected Paimon filter column handle array");
  }
  deserializeFileFields(
      obj,
      context,
      connectorId,
      tableName,
      filters,
      remainingFilter,
      filterColumns,
      sampleRate,
      dbName);
  VELOX_USER_CHECK_EQ(
      filters.size(),
      obj["subfieldFilters"].size(),
      "Duplicate Paimon subfield filter");
  if (auto it = obj.find("remainingFilter");
      it != obj.items().end() && !it->second.isNull()) {
    VELOX_USER_CHECK_NOT_NULL(
        remainingFilter, "Invalid Paimon remaining filter");
  }

  std::unordered_map<std::string, std::string> options;
  for (const auto& [key, value] : obj.at("options").items()) {
    options.emplace(key.asString(), value.asString());
  }
  std::vector<std::string> primaryKeys;
  if (auto it = obj.find("primaryKeys"); it != obj.items().end()) {
    primaryKeys =
        ISerializable::deserialize<std::vector<std::string>>(it->second);
  }

  return std::make_shared<PaimonTableHandle>(
      std::move(connectorId),
      std::move(tableName),
      ISerializable::deserialize<int64_t>(obj.at("schemaId")),
      ISerializable::deserialize<RowType>(obj.at("rowType"), context),
      ISerializable::deserialize<std::vector<int32_t>>(obj.at("fieldIds")),
      paimonTableTypeFromString(obj.at("tableType").asString()),
      ISerializable::deserialize<std::vector<std::string>>(
          obj.at("partitionKeys")),
      std::move(options),
      std::move(filters),
      std::move(remainingFilter),
      std::move(filterColumns),
      sampleRate,
      std::move(dbName),
      std::move(primaryKeys));
}

void PaimonTableHandle::registerSerDe() {
  DeserializationWithContextRegistryForSharedPtr().Register(
      "PaimonTableHandle", create);
}

} // namespace facebook::velox::connector::paimon
