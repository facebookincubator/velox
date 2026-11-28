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
#include <cmath>
#include <unordered_set>

namespace facebook::velox::connector::paimon {
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
std::vector<std::string> checkPrimaryKeys(
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
  std::vector<std::string> nonPartitionKeys;
  for (const auto& key : primaryKeys) {
    VELOX_USER_CHECK(
        rowType->containsChild(key), "Unknown Paimon primary key: {}", key);
    VELOX_USER_CHECK(
        keys.insert(key).second, "Duplicate Paimon primary key: {}", key);
    if (std::find(partitionKeys.begin(), partitionKeys.end(), key) ==
        partitionKeys.end()) {
      nonPartitionKeys.push_back(key);
    }
  }
  VELOX_USER_CHECK(
      !nonPartitionKeys.empty(),
      "Paimon primary keys must not all be partition columns: [{}]",
      fmt::join(primaryKeys, ", "));
  return nonPartitionKeys;
}

// Parses 'value' of the boolean table option 'key', in any capitalization.
bool parseBooleanOption(std::string_view key, std::string_view value) {
  std::string lower{value};
  folly::toLowerAscii(lower);
  if (lower == "true") {
    return true;
  }
  VELOX_USER_CHECK_EQ(
      lower, "false", "Paimon table option must be true or false: {}", key);
  return false;
}

// Returns 'value' of the 'changelog-producer' table option in lower case, after
// checking that it is a PaimonChangelogProducer value in any capitalization.
std::string parseChangelogProducer(std::string_view value) {
  std::string lower{value};
  folly::toLowerAscii(lower);
  VELOX_USER_CHECK(
      lower == PaimonChangelogProducer::kNone ||
          lower == PaimonChangelogProducer::kInput ||
          lower == PaimonChangelogProducer::kFullCompaction ||
          lower == PaimonChangelogProducer::kLookup,
      "Unknown Paimon changelog-producer: {}",
      value);
  return lower;
}

bool contains(const std::vector<std::string>& names, const std::string& name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

// Resolves the aggregate function of every column of an aggregation table as
// Paimon Java's AggregateMergeFunction.getAggFuncName does.
std::unordered_map<
    std::string,
    PaimonTableHandle::MergeOptions::FieldAggregation>
parseFieldAggregations(
    const RowTypePtr& rowType,
    const std::vector<std::string>& primaryKeys,
    const std::vector<std::string>& sequenceFields,
    const std::unordered_map<std::string, std::string>& options) {
  const auto defaultIt =
      options.find(PaimonTableOptions::kFieldsDefaultAggregateFunction);
  std::unordered_map<
      std::string,
      PaimonTableHandle::MergeOptions::FieldAggregation>
      fieldAggregations;
  for (const auto& name : rowType->names()) {
    auto& aggregation = fieldAggregations[name];
    if (contains(sequenceFields, name)) {
      aggregation.function = PaimonAggregateFunction::kLastValue;
    } else if (contains(primaryKeys, name)) {
      aggregation.function = PaimonAggregateFunction::kPrimaryKey;
    } else if (auto it = options.find(
                   PaimonTableOptions::fieldAggregateFunction(name));
               it != options.end()) {
      aggregation.function = it->second;
    } else if (defaultIt != options.end()) {
      aggregation.function = defaultIt->second;
    } else {
      aggregation.function = PaimonAggregateFunction::kLastNonNullValue;
    }
    const auto ignoreRetractKey = PaimonTableOptions::fieldIgnoreRetract(name);
    if (auto it = options.find(ignoreRetractKey); it != options.end()) {
      aggregation.ignoreRetract =
          parseBooleanOption(ignoreRetractKey, it->second);
    }
  }
  return fieldAggregations;
}

// Parses the merge-on-read options of a primary-key table as Paimon Java's
// CoreOptions does, and checks them as its SchemaValidation does.
PaimonTableHandle::MergeOptions parseMergeOptions(
    PaimonTableType tableType,
    const RowTypePtr& rowType,
    const std::vector<std::string>& primaryKeys,
    const std::unordered_map<std::string, std::string>& options) {
  PaimonTableHandle::MergeOptions mergeOptions;
  if (tableType != PaimonTableType::kPrimaryKey) {
    return mergeOptions;
  }
  if (auto it = options.find(PaimonTableOptions::kMergeEngine);
      it != options.end()) {
    mergeOptions.mergeEngine = paimonMergeEngineFromString(it->second);
  }
  for (const auto* key :
       {PaimonTableOptions::kIgnoreDelete,
        PaimonTableOptions::kFirstRowIgnoreDelete,
        PaimonTableOptions::kDeduplicateIgnoreDelete,
        PaimonTableOptions::kPartialUpdateIgnoreDelete}) {
    if (auto it = options.find(key); it != options.end()) {
      mergeOptions.ignoreDelete = parseBooleanOption(key, it->second);
      break;
    }
  }
  if (auto it = options.find(PaimonTableOptions::kSequenceField);
      it != options.end()) {
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
  if (auto it = options.find(PaimonTableOptions::kSequenceFieldSortOrder);
      it != options.end()) {
    std::string order{it->second};
    folly::toLowerAscii(order);
    VELOX_USER_CHECK(
        order == "ascending" || order == "descending",
        "Unknown Paimon sequence.field.sort-order: {}",
        it->second);
    mergeOptions.sequenceFieldsAscending = order == "ascending";
  }
  if (auto it = options.find(PaimonTableOptions::kSequenceSnapshotOrdering);
      it != options.end()) {
    mergeOptions.sequenceSnapshotOrdering =
        parseBooleanOption(it->first, it->second);
  }
  VELOX_USER_CHECK(
      mergeOptions.sequenceFields.empty() ||
          mergeOptions.mergeEngine != PaimonMergeEngine::kFirstRow,
      "Paimon first-row merge engine does not support sequence.field");

  // Paimon Java's CoreOptions.lookupStrategy().
  mergeOptions.needLookup =
      mergeOptions.mergeEngine == PaimonMergeEngine::kFirstRow;
  if (auto it = options.find(PaimonTableOptions::kChangelogProducer);
      it != options.end()) {
    mergeOptions.needLookup |=
        parseChangelogProducer(it->second) == PaimonChangelogProducer::kLookup;
  }
  for (const auto* key :
       {PaimonTableOptions::kDeletionVectorsEnabled,
        PaimonTableOptions::kForceLookup}) {
    if (auto it = options.find(key); it != options.end()) {
      mergeOptions.needLookup |= parseBooleanOption(key, it->second);
    }
  }

  if (mergeOptions.mergeEngine == PaimonMergeEngine::kAggregation) {
    mergeOptions.fieldAggregations = parseFieldAggregations(
        rowType, primaryKeys, mergeOptions.sequenceFields, options);
    if (auto it =
            options.find(PaimonTableOptions::kAggregationRemoveRecordOnDelete);
        it != options.end()) {
      mergeOptions.aggregationRemoveRecordOnDelete =
          parseBooleanOption(it->first, it->second);
    }
  }
  return mergeOptions;
}

} // namespace

// static
std::string PaimonTableOptions::fieldAggregateFunction(
    std::string_view column) {
  return fmt::format("fields.{}.aggregate-function", column);
}

// static
std::string PaimonTableOptions::fieldIgnoreRetract(std::string_view column) {
  return fmt::format("fields.{}.ignore-retract", column);
}

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
      nonPartitionPrimaryKeys_(
          checkPrimaryKeys(tableType_, rowType_, primaryKeys_, partitionKeys_)),
      mergeOptions_(
          parseMergeOptions(tableType_, rowType_, primaryKeys_, options_)) {
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
