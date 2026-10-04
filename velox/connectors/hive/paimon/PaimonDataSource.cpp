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
#include "velox/connectors/hive/paimon/PaimonDataSource.h"

#include "velox/connectors/hive/PartitionValue.h"
#include "velox/connectors/hive/paimon/PaimonSplitReader.h"
#include "velox/core/Expressions.h"

namespace facebook::velox::connector::hive::paimon {
namespace {

bool supportedType(const Type& type) {
  return type == *BOOLEAN() || type == *INTEGER() || type == *BIGINT() ||
      type == *DOUBLE() || type == *VARCHAR();
}

void validateField(
    const PaimonSchema& schema,
    const std::string& name,
    const TypePtr& type = nullptr) {
  const auto index = schema.rowType->getChildIdxIfExists(name);
  VELOX_USER_CHECK(
      index.has_value(),
      "Unsupported Paimon field '{}' (including hidden fields)",
      name);
  if (type) {
    VELOX_USER_CHECK(
        schema.rowType->childAt(*index)->equivalent(*type),
        "Paimon field '{}' type differs from target schema",
        name);
  }
}

void validateExpression(
    const PaimonSchema& schema,
    const core::TypedExprPtr& expr) {
  if (!expr) {
    return;
  }
  if (const auto* field =
          dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get())) {
    validateField(schema, field->name(), field->type());
  }
  for (const auto& input : expr->inputs()) {
    validateExpression(schema, input);
  }
}

ConnectorTableHandlePtr validateRequest(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& handle,
    const ColumnHandleMap& assignments,
    const PaimonConfig& config,
    const ConnectorQueryCtx& ctx) {
  const auto table = std::dynamic_pointer_cast<const PaimonTableHandle>(handle);
  VELOX_USER_CHECK_NOT_NULL(
      table,
      "PaimonTableHandle with explicit table semantics and schema bundle is required");
  VELOX_USER_CHECK(
      !config.ignoreMissingFiles(ctx.sessionProperties()),
      "Paimon cannot enable ignore_missing_files: snapshot files are required");
  const auto& schema = table->targetSchema();
  for (auto i = 0; i < schema.fieldIds.size(); ++i) {
    const auto& name = schema.rowType->nameOf(i);
    VELOX_USER_CHECK(
        name != "_ROW_ID" && name != "_SEQUENCE_NUMBER" &&
            name != "_VALUE_KIND" && name != "_LEVEL" && name != "rowkind" &&
            name.find("_KEY_") != 0,
        "Paimon hidden/system fields are not supported in this profile");
    VELOX_USER_CHECK_LT(
        schema.fieldIds[i],
        1073741823,
        "Paimon hidden/system fields are not supported in this profile");
    VELOX_USER_CHECK(
        supportedType(*schema.rowType->childAt(i)),
        "Unsupported Paimon M1.1 type for '{}': {}",
        schema.rowType->nameOf(i),
        schema.rowType->childAt(i)->toString());
  }
  for (const auto& [key, value] : table->options()) {
    if (key == "row-tracking.enabled") {
      VELOX_USER_CHECK(
          value == "true" || value == "false",
          "Invalid Paimon boolean option {}",
          key);
      VELOX_USER_CHECK(
          value == "false" ||
              table->tableType() == PaimonTableType::kAppendOnly,
          "Unsupported Paimon row tracking on a primary-key table");
      // Ordinary user columns do not depend on reconstructed row IDs. Hidden
      // field requests are rejected separately, even when only used in filters.
    } else if (key == "merge-engine") {
      VELOX_USER_CHECK_EQ(
          value, "deduplicate", "Unsupported Paimon merge-engine");
    } else if (
        key == "data-evolution.enabled" || key == "deletion-vectors.enabled" ||
        key == "force-lookup" || key == "ignore-delete") {
      VELOX_USER_CHECK_EQ(
          value, "false", "Unsupported Paimon option {}={}", key, value);
    } else {
      VELOX_USER_FAIL("Unsupported Paimon execution option: {}={}", key, value);
    }
  }
  if (table->tableType() == PaimonTableType::kPrimaryKey) {
    VELOX_USER_CHECK(
        table->options().count("merge-engine"),
        "Paimon primary-key input requires an explicit merge-engine");
  }
  for (const auto& [alias, column] : assignments) {
    const auto* paimon = dynamic_cast<const PaimonColumnHandle*>(column.get());
    VELOX_USER_CHECK_NOT_NULL(
        paimon, "PaimonColumnHandle with field ID is required");
    validateField(schema, paimon->name(), paimon->dataType());
    const auto index = schema.rowType->getChildIdx(paimon->name());
    VELOX_USER_CHECK_EQ(
        paimon->fieldId(),
        schema.fieldIds[index],
        "Paimon assignment field ID differs from target schema");
    const auto& partitionIds = table->partitionFieldIds();
    const bool isPartition =
        std::find(
            partitionIds.begin(), partitionIds.end(), paimon->fieldId()) !=
        partitionIds.end();
    VELOX_USER_CHECK_EQ(
        paimon->columnType(),
        isPartition ? FileColumnHandle::ColumnType::kPartitionKey
                    : FileColumnHandle::ColumnType::kRegular,
        "Unsupported Paimon column role (including Hive row IDs)");
    VELOX_USER_CHECK(
        paimon->requiredSubfields().empty(),
        "Paimon M1.1 does not support requiredSubfields");
    if (auto output = outputType->getChildIdxIfExists(alias)) {
      VELOX_USER_CHECK(
          outputType->childAt(*output)->equivalent(*paimon->dataType()),
          "Paimon output type differs from assignment");
    }
  }
  for (const auto& [field, filter] : table->subfieldFilters()) {
    VELOX_USER_CHECK_EQ(
        field.path().size(), 1, "Paimon M1.1 supports only top-level filters");
    validateField(schema, getColumnName(field));
  }
  validateExpression(schema, table->remainingFilter());
  return table;
}

std::unordered_map<std::string, std::optional<std::string>> normalizePartition(
    const PaimonTableHandle& table,
    const PaimonConnectorSplit& split) {
  const auto& schema = table.targetSchema();
  std::unordered_map<std::string, std::optional<std::string>> result;
  const auto count =
      split.partitionKeys().size() + split.partitionValues().size();
  VELOX_USER_CHECK_EQ(
      count,
      table.partitionFieldIds().size(),
      "Paimon split partition metadata is incomplete");
  for (const auto id : table.partitionFieldIds()) {
    const auto index =
        std::find(schema.fieldIds.begin(), schema.fieldIds.end(), id) -
        schema.fieldIds.begin();
    const auto& type = schema.rowType->childAt(index);
    const auto& name = schema.rowType->nameOf(index);
    VELOX_USER_CHECK(
        type->isVarchar() || type->isInteger() || type->isBigint() ||
            type->isBoolean(),
        "Unsupported Paimon partition type: {}",
        type->toString());
    if (!split.partitionKeys().empty()) {
      const auto it = split.partitionKeys().find(name);
      VELOX_USER_CHECK(
          it != split.partitionKeys().end(),
          "Missing Paimon partition '{}'",
          name);
      if (it->second) {
        PartitionValue::fromString(
            *it->second,
            *type,
            PartitionValue::TimestampMode::kUtc,
            PartitionValue::DateMode::kIsoString);
      }
      result.emplace(name, it->second);
      continue;
    }
    const auto it = split.partitionValues().find(id);
    VELOX_USER_CHECK(
        it != split.partitionValues().end(),
        "Missing Paimon partition fieldId {}",
        id);
    const auto& value = it->second;
    VELOX_USER_CHECK_EQ(
        value.kind(), type->kind(), "Paimon typed partition type mismatch");
    if (value.isNull()) {
      result.emplace(name, std::nullopt);
    } else if (type->isVarchar()) {
      result.emplace(name, value.value<TypeKind::VARCHAR>());
    } else {
      result.emplace(name, value.toJson(type));
    }
  }
  return result;
}

} // namespace

PaimonDataSource::PaimonDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& assignments,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<PaimonConfig>& paimonConfig)
    : FileDataSource(
          outputType,
          validateRequest(
              outputType,
              tableHandle,
              assignments,
              *paimonConfig,
              *connectorQueryCtx),
          assignments,
          fileHandleFactory,
          ioExecutor,
          connectorQueryCtx,
          paimonConfig,
          FileScanOptions{
              .allowSampling = false,
              .applyLowercaseColumnNames = false}),
      paimonTable_(
          std::static_pointer_cast<const PaimonTableHandle>(tableHandle)) {}

std::unique_ptr<FileScanReader> PaimonDataSource::createScanReader() {
  const auto split =
      std::dynamic_pointer_cast<PaimonConnectorSplit>(activeSplit_);
  VELOX_USER_CHECK_NOT_NULL(split, "PaimonConnectorSplit is required");
  VELOX_USER_CHECK_EQ(
      split->connectorId,
      paimonTable_->connectorId(),
      "Paimon connectorId mismatch");
  VELOX_USER_CHECK_EQ(
      split->tableType(),
      paimonTable_->tableType(),
      "Paimon tableType mismatch");
  if (snapshotId_) {
    VELOX_USER_CHECK_EQ(
        *snapshotId_,
        split->snapshotId(),
        "Paimon snapshot changed within a task scan");
  } else {
    snapshotId_ = split->snapshotId();
  }
  VELOX_USER_CHECK(
      split->rawConvertible(), "Paimon merge-on-read is not yet implemented");
  for (const auto& file : split->dataFiles()) {
    VELOX_USER_CHECK(
        file.schemaId.has_value(),
        "Missing Paimon schemaId for file '{}'",
        file.path);
    const auto& schema = paimonTable_->schema(*file.schemaId);
    VELOX_USER_CHECK_EQ(
        schema.id,
        paimonTable_->targetSchema().id,
        "Paimon cross-schema reads require M1.2");
    const auto format = file.fileFormat.value_or(split->fileFormat());
    VELOX_USER_CHECK(
        format == dwio::common::FileFormat::PARQUET ||
            format == dwio::common::FileFormat::DWRF,
        "Unsupported Paimon file format: {}",
        dwio::common::FileFormatName::toName(format));
    VELOX_USER_CHECK(
        !file.deletionFile,
        "Paimon deletion vector reading is not yet implemented");
    VELOX_USER_CHECK_EQ(
        file.type,
        PaimonDataFile::Type::kData,
        "Paimon changelog file reading is not yet supported");
    if (split->tableType() == PaimonTableType::kPrimaryKey) {
      VELOX_USER_CHECK(
          file.deleteRowCount.has_value() && *file.deleteRowCount == 0,
          "Paimon primary-key raw reads require known zero deleteRowCount; KV fallback is not yet implemented");
    } else {
      VELOX_USER_CHECK(
          !file.deleteRowCount || *file.deleteRowCount == 0,
          "Paimon append file contains RowKind deletes");
    }
  }
  return std::make_unique<PaimonSplitReader>(
      split,
      paimonTable_,
      normalizePartition(*paimonTable_, *split),
      &partitionKeys_,
      connectorQueryCtx_,
      fileConfig_,
      readerOutputType_,
      dataIoStats_,
      metadataIoStats_,
      ioStats_,
      fileHandleFactory_,
      ioExecutor_,
      runtimeStats_,
      [this]() { return newFileScanState(); },
      remainingFilterColumns());
}

void PaimonDataSource::setFromDataSource(
    std::unique_ptr<DataSource> /*source*/) {
  VELOX_UNSUPPORTED("Paimon split preload/state takeover is not supported");
}

} // namespace facebook::velox::connector::hive::paimon
