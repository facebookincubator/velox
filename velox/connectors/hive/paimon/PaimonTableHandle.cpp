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
#include "velox/connectors/hive/paimon/PaimonTableHandle.h"

#include <cmath>
#include <unordered_set>

#include "velox/connectors/hive/PartitionValue.h"
#include "velox/core/Expressions.h"

namespace facebook::velox::connector::hive::paimon {
namespace {

constexpr std::string_view kSameSchemaAppendBatch = "SAME_SCHEMA_APPEND_BATCH";

TypePtr columnType(const PaimonSchema& schema, const std::string& name) {
  const auto index = schema.rowType()->getChildIdxIfExists(name);
  VELOX_USER_CHECK(
      index.has_value(), "Unknown Paimon column '{}' in target schema", name);
  return schema.rowType()->childAt(*index);
}

bool isPartitionColumn(const PaimonSchema& schema, const std::string& name) {
  for (auto id : schema.partitionKeys()) {
    if (schema.rowType()->nameOf(schema.fieldIndexPath(id).front()) == name) {
      return true;
    }
  }
  return false;
}

TypePtr subfieldType(const PaimonSchema& schema, const common::Subfield& path) {
  schema.fieldIdPath(path);
  auto type = columnType(
      schema, path.path().front()->as<common::Subfield::NestedField>()->name());
  for (size_t i = 1; i < path.path().size(); ++i) {
    if (const auto* field =
            path.path()[i]->as<common::Subfield::NestedField>()) {
      type = type->asRow().findChild(field->name());
    } else {
      type = type->childAt(type->isMap() ? 1 : 0);
    }
  }
  return type;
}

void validateFilterType(const TypePtr& type, const common::Filter& filter) {
  using common::FilterKind;
  bool supported = false;
  switch (filter.kind()) {
    case FilterKind::kAlwaysFalse:
    case FilterKind::kAlwaysTrue:
    case FilterKind::kIsNull:
    case FilterKind::kIsNotNull:
      return;
    case FilterKind::kBoolValue:
      supported = type->isBoolean();
      break;
    case FilterKind::kBigintRange:
    case FilterKind::kBigintValuesUsingHashTable:
    case FilterKind::kBigintValuesUsingBitmask:
    case FilterKind::kNegatedBigintRange:
    case FilterKind::kNegatedBigintValuesUsingHashTable:
    case FilterKind::kNegatedBigintValuesUsingBitmask:
    case FilterKind::kBigintMultiRange:
    case FilterKind::kBigintValuesUsingBloomFilter:
      supported = (type->isInteger() || type->isBigint()) && !type->isDate();
      break;
    case FilterKind::kDoubleRange:
      supported = type->isDouble();
      break;
    case FilterKind::kBytesRange:
    case FilterKind::kNegatedBytesRange:
    case FilterKind::kBytesValues:
    case FilterKind::kNegatedBytesValues:
      supported = type->isVarchar();
      break;
    case FilterKind::kMultiRange: {
      const auto* ranges = dynamic_cast<const common::MultiRange*>(&filter);
      VELOX_USER_CHECK_NOT_NULL(ranges, "Invalid Paimon MultiRange filter");
      VELOX_USER_CHECK_GE(
          ranges->filters().size(),
          2,
          "Paimon MultiRange requires at least two filters");
      for (const auto& range : ranges->filters()) {
        VELOX_USER_CHECK_NOT_NULL(range, "Missing Paimon range filter");
        validateFilterType(type, *range);
      }
      supported = type->isDouble() || type->isVarchar();
      break;
    }
    default:
      break;
  }
  VELOX_USER_CHECK(
      supported,
      "Paimon filter {} is not supported for target type {}",
      filter.kind(),
      type->toString());
}

std::unordered_set<std::string> validateExpression(
    const PaimonSchema& schema,
    const core::TypedExprPtr& expression) {
  std::unordered_set<std::string> columns;
  if (!expression) {
    return columns;
  }
  VELOX_USER_CHECK(
      expression->type() && expression->type()->isBoolean(),
      "Paimon remaining filter must have BOOLEAN type");
  std::vector<const core::ITypedExpr*> pending{expression.get()};
  std::unordered_set<const core::ITypedExpr*> visited;
  while (!pending.empty()) {
    const auto* current = pending.back();
    pending.pop_back();
    VELOX_USER_CHECK_NOT_NULL(current, "Missing Paimon expression input");
    if (!visited.insert(current).second) {
      continue;
    }
    VELOX_USER_CHECK_NOT_NULL(
        current->type(), "Missing Paimon expression type");
    VELOX_USER_CHECK(
        !current->isLambdaKind(),
        "Paimon same-schema append profile does not support Lambda filters");
    VELOX_USER_CHECK(
        !current->isInputKind(),
        "Paimon remaining filter must access input columns by name");
    if (const auto* field =
            current->asUnchecked<core::FieldAccessTypedExpr>()) {
      TypePtr expected;
      if (field->isInputColumn()) {
        expected = columnType(schema, field->name());
        columns.insert(field->name());
        if (!field->inputs().empty()) {
          const auto& input = field->inputs().front();
          VELOX_USER_CHECK(
              input && input->isInputKind() && input->type() &&
                  *input->type() == *schema.rowType(),
              "Paimon expression input type does not match target schema");
        }
      } else {
        VELOX_USER_CHECK(
            field->inputs().size() == 1 && field->inputs().front(),
            "Invalid Paimon expression field input");
        const auto& parent = field->inputs().front()->type();
        VELOX_USER_CHECK(
            parent && parent->isRow(),
            "Paimon field access requires ROW input");
        const auto index = parent->asRow().getChildIdxIfExists(field->name());
        VELOX_USER_CHECK(
            index.has_value(), "Unknown Paimon expression subfield");
        expected = parent->childAt(*index);
      }
      VELOX_USER_CHECK(
          *field->type() == *expected,
          "Paimon expression field '{}' type does not match target schema",
          field->name());
      if (field->isInputColumn()) {
        continue;
      }
    } else if (
        const auto* field =
            current->asUnchecked<core::DereferenceTypedExpr>()) {
      VELOX_USER_CHECK(
          field->inputs().size() == 1 && field->inputs().front(),
          "Invalid Paimon expression dereference input");
      const auto& parent = field->inputs().front()->type();
      VELOX_USER_CHECK(
          parent && parent->isRow() && field->index() < parent->size(),
          "Invalid Paimon expression dereference");
      VELOX_USER_CHECK(
          *field->type() == *parent->childAt(field->index()),
          "Paimon expression subfield type does not match target schema");
    }
    for (const auto& input : current->inputs()) {
      pending.push_back(input.get());
    }
  }
  return columns;
}

RowTypePtr deriveDataColumns(const PaimonSchemaPtr& schema) {
  VELOX_USER_CHECK_NOT_NULL(schema, "Missing Paimon table schema");
  if (schema->partitionKeys().empty()) {
    return schema->rowType();
  }
  std::unordered_set<column_index_t> partitions;
  for (auto id : schema->partitionKeys()) {
    partitions.insert(schema->fieldIndexPath(id).front());
  }
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (column_index_t i = 0; i < schema->rowType()->size(); ++i) {
    if (!partitions.count(i)) {
      names.push_back(schema->rowType()->nameOf(i));
      types.push_back(schema->rowType()->childAt(i));
    }
  }
  return ROW(std::move(names), std::move(types));
}

// A stringified Subfield cannot distinguish literal dots or brackets in field
// names from path separators. Preserve the path elements in the handle wire.
folly::dynamic serializeSubfield(const common::Subfield& subfield) {
  folly::dynamic path = folly::dynamic::array;
  for (const auto& element : subfield.path()) {
    folly::dynamic obj = folly::dynamic::object;
    if (const auto* field = element->as<common::Subfield::NestedField>()) {
      obj["kind"] = "field";
      obj["value"] = field->name();
    } else if (
        const auto* subscript =
            element->as<common::Subfield::LongSubscript>()) {
      obj["kind"] = "long";
      obj["value"] = subscript->index();
    } else if (
        const auto* subscript =
            element->as<common::Subfield::StringSubscript>()) {
      obj["kind"] = "string";
      obj["value"] = subscript->index();
    } else {
      VELOX_USER_CHECK(element->is(common::SubfieldKind::kAllSubscripts));
      obj["kind"] = "all";
    }
    path.push_back(std::move(obj));
  }
  return path;
}

common::Subfield deserializeSubfield(const folly::dynamic& obj) {
  VELOX_USER_CHECK(
      obj.isArray() && !obj.empty(), "Expected Paimon filter path");
  std::vector<std::unique_ptr<common::Subfield::PathElement>> path;
  for (const auto& element : obj) {
    VELOX_USER_CHECK(element.isObject(), "Expected Paimon path element");
    const auto& kind = element.at("kind");
    VELOX_USER_CHECK(kind.isString(), "Expected Paimon path element kind");
    if (kind.getString() == "field") {
      VELOX_USER_CHECK(
          element.at("value").isString(), "Expected Paimon path field name");
      path.push_back(
          std::make_unique<common::Subfield::NestedField>(
              element.at("value").asString()));
    } else if (kind.getString() == "long") {
      VELOX_USER_CHECK(
          element.at("value").isInt(), "Expected Paimon integer subscript");
      path.push_back(
          std::make_unique<common::Subfield::LongSubscript>(
              element.at("value").asInt()));
    } else if (kind.getString() == "string") {
      VELOX_USER_CHECK(
          element.at("value").isString(), "Expected Paimon string subscript");
      path.push_back(
          std::make_unique<common::Subfield::StringSubscript>(
              element.at("value").asString()));
    } else {
      VELOX_USER_CHECK_EQ(kind.getString(), "all", "Unknown Paimon path kind");
      path.push_back(std::make_unique<common::Subfield::AllSubscripts>());
    }
  }
  VELOX_USER_CHECK(
      path.front()->is(common::SubfieldKind::kNestedField),
      "Paimon filter path must start with a field name");
  return common::Subfield(std::move(path));
}

} // namespace

PaimonTableHandle::PaimonTableHandle(
    std::string connectorId,
    std::string tableName,
    PaimonSchemaPtr schema,
    ReadProfile readProfile,
    common::SubfieldFilters subfieldFilters,
    core::TypedExprPtr remainingFilter,
    std::vector<FileColumnHandlePtr> filterColumnHandles,
    std::unordered_map<std::string, std::string> queryOptions,
    double sampleRate,
    std::string dbName)
    : FileTableHandle(std::move(connectorId)),
      tableName_(std::move(tableName)),
      dbName_(std::move(dbName)),
      schema_(std::move(schema)),
      readProfile_(readProfile),
      subfieldFilters_(std::move(subfieldFilters)),
      remainingFilter_(std::move(remainingFilter)),
      filterColumnHandles_(std::move(filterColumnHandles)),
      queryOptions_(std::move(queryOptions)),
      sampleRate_(sampleRate),
      dataColumns_(deriveDataColumns(schema_)) {
  VELOX_USER_CHECK(!this->connectorId().empty(), "Empty Paimon connector ID");
  VELOX_USER_CHECK(!tableName_.empty(), "Empty Paimon table name");
  VELOX_USER_CHECK(
      readProfile_ == ReadProfile::kSameSchemaAppendBatch,
      "Unsupported Paimon read profile");
  VELOX_USER_CHECK(
      std::isfinite(sampleRate_) && sampleRate_ > 0 && sampleRate_ <= 1,
      "Paimon sample rate must be finite and in (0, 1]");
  validateExpression(*schema_, remainingFilter_);
  for (const auto& [subfield, filter] : subfieldFilters_) {
    VELOX_USER_CHECK(subfield.valid(), "Invalid Paimon filter path");
    VELOX_USER_CHECK_NOT_NULL(filter, "Missing Paimon subfield filter");
    validateFilterType(subfieldType(*schema_, subfield), *filter);
  }
  std::unordered_set<std::string> filterColumnNames;
  for (const auto& handle : filterColumnHandles_) {
    VELOX_USER_CHECK_NOT_NULL(handle, "Missing Paimon filter column handle");
    validateColumn(*handle);
    VELOX_USER_CHECK(
        filterColumnNames.insert(handle->name()).second,
        "Duplicate Paimon filter column handle '{}'",
        handle->name());
  }
}

PaimonTableType PaimonTableHandle::tableType() const {
  return schema_->primaryKeys().empty() ? PaimonTableType::kAppendOnly
                                        : PaimonTableType::kPrimaryKey;
}

bool PaimonTableHandle::hasSameTableDefinition(
    const PaimonTableHandle& other) const {
  return connectorId() == other.connectorId() &&
      tableName_ == other.tableName_ && dbName_ == other.dbName_ &&
      (schema_ == other.schema_ || *schema_ == *other.schema_);
}

bool PaimonTableHandle::hasSameScanConditions(
    const PaimonTableHandle& other) const {
  if (readProfile_ != other.readProfile_ ||
      queryOptions_ != other.queryOptions_ ||
      sampleRate_ != other.sampleRate_ ||
      static_cast<bool>(remainingFilter_) !=
          static_cast<bool>(other.remainingFilter_) ||
      (remainingFilter_ && remainingFilter_ != other.remainingFilter_ &&
       !(*remainingFilter_ == *other.remainingFilter_)) ||
      subfieldFilters_.size() != other.subfieldFilters_.size()) {
    return false;
  }
  for (const auto& [path, filter] : subfieldFilters_) {
    const auto it = other.subfieldFilters_.find(path);
    if (it == other.subfieldFilters_.end() ||
        (filter != it->second && !filter->testingEquals(*it->second))) {
      return false;
    }
  }
  return true;
}

void PaimonTableHandle::validateReadProfile() const {
  VELOX_USER_CHECK(
      tableType() == PaimonTableType::kAppendOnly,
      "Paimon same-schema append profile does not support primary-key tables");
  const auto& options = schema_->options();
  const auto requireOption = [&](const std::string& key,
                                 std::string_view value) {
    const auto it = options.find(key);
    VELOX_USER_CHECK(
        it != options.end(),
        "Missing explicit Paimon profile option '{}'",
        key);
    VELOX_USER_CHECK(
        it->second == value,
        "Paimon same-schema append profile requires '{}'='{}'",
        key,
        value);
  };
  requireOption("type", "table");
  requireOption("batch-scan-mode", "none");
  requireOption("deletion-vectors.enabled", "false");
  requireOption("data-evolution.enabled", "false");
  requireOption("data-file.thin-mode", "false");
  const auto tracking = options.find("row-tracking.enabled");
  VELOX_USER_CHECK(
      tracking != options.end() &&
          (tracking->second == "true" || tracking->second == "false"),
      "Paimon profile requires explicit boolean 'row-tracking.enabled'");
  for (const auto& [key, value] : options) {
    if (key == "file.format") {
      VELOX_USER_CHECK(
          value == "nimble" || value == "dwrf" || value == "orc" ||
              value == "parquet",
          "Unsupported Paimon schema file.format '{}'",
          value);
    } else if (key == "bucket") {
      const auto parsed = folly::tryTo<int32_t>(value);
      VELOX_USER_CHECK(
          parsed.hasValue() && (parsed.value() == -1 || parsed.value() > 0),
          "Unsupported Paimon append bucket option '{}'",
          value);
    } else {
      VELOX_USER_CHECK(
          key == "type" || key == "batch-scan-mode" ||
              key == "deletion-vectors.enabled" ||
              key == "data-evolution.enabled" || key == "data-file.thin-mode" ||
              key == "row-tracking.enabled",
          "Paimon same-schema append profile does not support option '{}'",
          key);
    }
  }
  VELOX_USER_CHECK(
      queryOptions_.empty(),
      "Paimon same-schema append profile does not support query options");
  for (const auto& field : schema_->metadata().children) {
    const auto& type = field.logicalType;
    VELOX_USER_CHECK(
        type == "BOOLEAN" || type == "INTEGER" || type == "BIGINT" ||
            type == "DOUBLE" || type == "VARCHAR",
        "Unsupported Paimon same-schema append logical type '{}'",
        type);
  }
  for (const auto& name : schema_->rowType()->names()) {
    VELOX_USER_CHECK(
        name != "_ROW_ID" && name != "_SEQUENCE_NUMBER" &&
            name != "_VALUE_KIND" && name != "_LEVEL" && name != "rowkind" &&
            !name.starts_with("_KEY_"),
        "Reserved Paimon field name '{}' in target schema",
        name);
  }
}

void PaimonTableHandle::validateColumn(const FileColumnHandle& column) const {
  const auto type = columnType(*schema_, column.name());
  VELOX_USER_CHECK(
      column.schemaType() && *column.schemaType() == *type,
      "Paimon column schema type does not match target schema: '{}'",
      column.name());
  VELOX_USER_CHECK(
      column.dataType() && *column.dataType() == *type,
      "Paimon column output type does not match target schema: '{}'",
      column.name());
  const auto role = isPartitionColumn(*schema_, column.name())
      ? FileColumnHandle::ColumnType::kPartitionKey
      : FileColumnHandle::ColumnType::kRegular;
  VELOX_USER_CHECK_EQ(
      column.columnType(),
      role,
      "Paimon column role does not match target schema: '{}'",
      column.name());
  VELOX_USER_CHECK(
      column.extractions().empty() && !column.postProcessor(),
      "Paimon same-schema append profile does not support column extractions or callbacks");
  for (const auto& path : column.requiredSubfields()) {
    subfieldType(*schema_, path);
    VELOX_USER_CHECK_EQ(
        path.path().front()->as<common::Subfield::NestedField>()->name(),
        column.name(),
        "Paimon required subfield does not match column name");
    VELOX_USER_CHECK(
        common::Subfield(path.toString()) == path,
        "Paimon FileColumnHandle required subfields must round-trip by name");
  }
}

void PaimonTableHandle::validateQuery(
    const RowTypePtr& outputType,
    const ColumnHandleMap& assignments) const {
  validateReadProfile();
  VELOX_USER_CHECK_NOT_NULL(outputType, "Missing Paimon output type");
  std::unordered_set<std::string> partitionColumns;
  std::unordered_set<std::string> assignedColumns;
  const auto validateHandle = [&](const ColumnHandlePtr& handle) {
    VELOX_USER_CHECK_NOT_NULL(handle, "Missing Paimon column handle");
    const auto* column = dynamic_cast<const FileColumnHandle*>(handle.get());
    VELOX_USER_CHECK_NOT_NULL(
        column, "Paimon requires FileColumnHandle inputs");
    validateColumn(*column);
    if (column->isPartitionKey()) {
      partitionColumns.insert(column->name());
    }
    return column;
  };
  for (const auto& [alias, handle] : assignments) {
    const auto* column = validateHandle(handle);
    VELOX_USER_CHECK(
        assignedColumns.insert(column->name()).second ||
            column->isPartitionKey(),
        "Paimon regular column '{}' cannot map to multiple assignments; use a projection",
        column->name());
  }
  for (const auto& handle : filterColumnHandles_) {
    validateHandle(handle);
  }
  std::unordered_set<std::string> outputNames;
  for (column_index_t i = 0; i < outputType->size(); ++i) {
    const auto& alias = outputType->nameOf(i);
    VELOX_USER_CHECK(
        outputNames.insert(alias).second,
        "Duplicate Paimon output channel name '{}'",
        alias);
    const auto it = assignments.find(alias);
    VELOX_USER_CHECK(
        it != assignments.end(),
        "Missing Paimon output assignment '{}'",
        alias);
    const auto* column = static_cast<const FileColumnHandle*>(it->second.get());
    VELOX_USER_CHECK(
        *outputType->childAt(i) == *column->dataType(),
        "Paimon output channel type does not match column handle: '{}'",
        alias);
  }
  auto conditionColumns = validateExpression(*schema_, remainingFilter_);
  for (const auto& [path, filter] : subfieldFilters_) {
    conditionColumns.insert(
        path.path().front()->as<common::Subfield::NestedField>()->name());
  }
  for (const auto& name : conditionColumns) {
    VELOX_USER_CHECK(
        !isPartitionColumn(*schema_, name) || partitionColumns.count(name),
        "Paimon partition condition requires a partition column handle: '{}'",
        name);
  }
}

void PaimonTableHandle::validateFileSchema(const RowTypePtr& fileType) const {
  validateReadProfile();
  VELOX_USER_CHECK_NOT_NULL(fileType, "Missing Paimon file schema");
  auto expectedType = dataColumns_;
  for (auto id : schema_->partitionKeys()) {
    if (fileType->containsChild(
            schema_->rowType()->nameOf(schema_->fieldIndexPath(id).front()))) {
      // Ordinary Paimon files may retain the physical partition columns.
      // The split remains authoritative for their values during adaptation.
      expectedType = schema_->rowType();
      break;
    }
  }
  VELOX_USER_CHECK_GE(
      fileType->size(),
      expectedType->size(),
      "Paimon file schema contradicts same-schema append profile");
  for (column_index_t i = 0; i < expectedType->size(); ++i) {
    VELOX_USER_CHECK(
        fileType->nameOf(i) == expectedType->nameOf(i) &&
            *fileType->childAt(i) == *expectedType->childAt(i),
        "Paimon file schema contradicts target column '{}'",
        expectedType->nameOf(i));
  }
  const auto extraColumns = fileType->size() - expectedType->size();
  VELOX_USER_CHECK(
      extraColumns == 0 ||
          (schema_->options().at("row-tracking.enabled") == "true" &&
           extraColumns == 2 &&
           fileType->nameOf(expectedType->size()) == "_ROW_ID" &&
           fileType->nameOf(expectedType->size() + 1) == "_SEQUENCE_NUMBER" &&
           *fileType->childAt(expectedType->size()) == *BIGINT() &&
           *fileType->childAt(expectedType->size() + 1) == *BIGINT()),
      "Paimon file layout contradicts same-schema append profile");
}

void PaimonTableHandle::validateDynamicFilter(
    const TypePtr& type,
    const common::Filter& filter) const {
  validateFilterType(type, filter);
}

void PaimonTableHandle::validateSplit(const PaimonConnectorSplit& split) const {
  VELOX_USER_CHECK_EQ(
      split.connectorId,
      connectorId(),
      "Paimon split connector does not match table handle");
  VELOX_USER_CHECK_EQ(
      split.tableType(),
      tableType(),
      "Paimon split table type does not match table schema");
  validateReadProfile();
  VELOX_USER_CHECK(
      split.rawConvertible(),
      "Paimon same-schema append profile requires rawConvertible=true");
  const auto format = split.fileFormat();
  VELOX_USER_CHECK(
      format == dwio::common::FileFormat::DWRF ||
          format == dwio::common::FileFormat::ORC ||
          format == dwio::common::FileFormat::PARQUET ||
          format == dwio::common::FileFormat::NIMBLE,
      "Unsupported Paimon same-schema append file format {}",
      dwio::common::FileFormatName::toName(format));
  for (const auto& file : split.dataFiles()) {
    VELOX_USER_CHECK(
        !file.deletionFile,
        "Paimon same-schema append profile does not support deletion vectors");
    VELOX_USER_CHECK_EQ(
        file.type,
        PaimonDataFile::Type::kData,
        "Paimon same-schema append profile requires data files");
    VELOX_USER_CHECK_EQ(
        file.level,
        0,
        "Paimon same-schema append profile requires level 0 files");
    VELOX_USER_CHECK_EQ(
        file.deleteRowCount,
        0,
        "Paimon same-schema append profile requires no delete records");
  }
  const auto& partitions = split.partitionKeys();
  VELOX_USER_CHECK_EQ(
      partitions.size(),
      schema_->partitionKeys().size(),
      "Paimon split partition names do not match table schema");
  for (auto id : schema_->partitionKeys()) {
    const auto& name =
        schema_->rowType()->nameOf(schema_->fieldIndexPath(id).front());
    VELOX_USER_CHECK(
        partitions.count(name),
        "Missing Paimon split partition '{}' declared by table schema",
        name);
    const auto& value = partitions.at(name);
    if (value) {
      PartitionValue::fromString(
          *value,
          *schema_->fieldType(id),
          PartitionValue::TimestampMode::kUtc,
          PartitionValue::DateMode::kIsoString);
    } else {
      VELOX_USER_CHECK(
          schema_->fieldMetadata(id).nullable,
          "NULL Paimon partition '{}' is not nullable in target schema",
          name);
    }
  }
}

std::string PaimonTableHandle::toString() const {
  return fmt::format(
      "PaimonTableHandle[table {}, schemaId {}, tableType {}, readProfile {}, dataColumns {}, "
      "sampleRate {}]",
      tableName_,
      schema_->schemaId(),
      tableType(),
      kSameSchemaAppendBatch,
      dataColumns_->toString(),
      sampleRate_);
}

folly::dynamic PaimonTableHandle::serialize() const {
  auto obj = ConnectorTableHandle::serializeBase("PaimonTableHandle");
  obj["version"] = kSerdeVersion;
  obj["tableName"] = tableName_;
  obj["dbName"] = dbName_;
  obj["schema"] = schema_->serialize();
  obj["readProfile"] = std::string(kSameSchemaAppendBatch);
  obj["subfieldFilters"] = folly::dynamic::array;
  for (const auto& [subfield, filter] : subfieldFilters_) {
    obj["subfieldFilters"].push_back(
        folly::dynamic::object("subfield", serializeSubfield(subfield))(
            "filter", filter->serialize()));
  }
  obj["remainingFilter"] = remainingFilter_ ? remainingFilter_->serialize()
                                            : folly::dynamic(nullptr);
  obj["filterColumnHandles"] = folly::dynamic::array;
  for (const auto& handle : filterColumnHandles_) {
    VELOX_USER_CHECK(
        !handle->postProcessor(),
        "Paimon filter column callbacks cannot be serialized");
    obj["filterColumnHandles"].push_back(handle->serialize());
  }
  obj["queryOptions"] = folly::dynamic::object;
  for (const auto& [key, value] : queryOptions_) {
    obj["queryOptions"][key] = value;
  }
  obj["sampleRate"] = sampleRate_;
  return obj;
}

ConnectorTableHandlePtr PaimonTableHandle::create(
    const folly::dynamic& obj,
    void* context) {
  VELOX_USER_CHECK(obj.isObject(), "Expected Paimon table handle object");
  VELOX_USER_CHECK(
      obj.at("version").isInt() && obj.at("version").asInt() == kSerdeVersion,
      "Unsupported Paimon table handle wire version");
  for (const auto* key : {"name", "connectorId", "tableName", "dbName"}) {
    VELOX_USER_CHECK(obj.at(key).isString(), "Expected Paimon string: {}", key);
  }
  VELOX_USER_CHECK_EQ(
      obj.at("name").getString(), "PaimonTableHandle", "Invalid handle name");
  auto schema = PaimonSchema::create(obj.at("schema"));
  VELOX_USER_CHECK(
      obj.at("readProfile").isString() &&
          obj.at("readProfile").getString() == kSameSchemaAppendBatch,
      "Unsupported Paimon read profile");
  common::SubfieldFilters filters;
  VELOX_USER_CHECK(
      obj.at("subfieldFilters").isArray(), "Expected Paimon filter array");
  for (const auto& entry : obj.at("subfieldFilters")) {
    VELOX_USER_CHECK(entry.isObject(), "Expected Paimon filter entry");
    auto path = deserializeSubfield(entry.at("subfield"));
    auto filter =
        ISerializable::deserialize<common::Filter>(entry.at("filter"));
    VELOX_USER_CHECK_NOT_NULL(filter, "Missing Paimon subfield filter");
    VELOX_USER_CHECK(
        filters.emplace(std::move(path), filter->clone()).second,
        "Duplicate Paimon subfield filter");
  }
  core::TypedExprPtr remainingFilter;
  if (!obj.at("remainingFilter").isNull()) {
    remainingFilter = ISerializable::deserialize<core::ITypedExpr>(
        obj.at("remainingFilter"), context);
    VELOX_USER_CHECK_NOT_NULL(
        remainingFilter, "Invalid Paimon remaining filter");
  }
  std::vector<FileColumnHandlePtr> filterColumnHandles;
  VELOX_USER_CHECK(
      obj.at("filterColumnHandles").isArray(),
      "Expected Paimon filter column handle array");
  for (const auto& handle : obj.at("filterColumnHandles")) {
    filterColumnHandles.push_back(
        ISerializable::deserialize<FileColumnHandle>(handle, context));
  }
  VELOX_USER_CHECK(
      obj.at("queryOptions").isObject(),
      "Expected Paimon query options object");
  std::unordered_map<std::string, std::string> queryOptions;
  for (const auto& [key, value] : obj.at("queryOptions").items()) {
    VELOX_USER_CHECK(
        key.isString() && value.isString(),
        "Paimon query options must be strings");
    queryOptions.emplace(key.asString(), value.asString());
  }
  const auto& sampleRate = obj.at("sampleRate");
  VELOX_USER_CHECK(
      sampleRate.isDouble() || sampleRate.isInt(),
      "Expected Paimon numeric sample rate");
  return std::make_shared<PaimonTableHandle>(
      obj.at("connectorId").asString(),
      obj.at("tableName").asString(),
      std::move(schema),
      ReadProfile::kSameSchemaAppendBatch,
      std::move(filters),
      std::move(remainingFilter),
      std::move(filterColumnHandles),
      std::move(queryOptions),
      sampleRate.asDouble(),
      obj.at("dbName").asString());
}

void PaimonTableHandle::registerSerDe() {
  DeserializationWithContextRegistryForSharedPtr().Register(
      "PaimonTableHandle", create);
}

} // namespace facebook::velox::connector::hive::paimon
