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

#include "velox/connectors/hive/TableHandle.h"

namespace facebook::velox::connector::hive {

HiveColumnHandle::HiveColumnHandle(
    const std::string& name,
    ColumnType columnType,
    TypePtr dataType,
    TypePtr hiveType,
    std::vector<common::Subfield> requiredSubfields,
    std::vector<NamedExtraction> extractions,
    ColumnParseParameters columnParseParameters,
    std::function<void(VectorPtr&)> postProcessor)
    : FileColumnHandle(
          name,
          columnType,
          std::move(dataType),
          std::move(hiveType),
          std::move(requiredSubfields),
          std::move(extractions),
          columnParseParameters.partitionDateValueFormat ==
              ColumnParseParameters::kDaysSinceEpoch,
          std::move(postProcessor)) {
  if (this->extractions().empty()) {
    // No extractions: dataType and hiveType must match (existing behavior).
    VELOX_USER_CHECK(
        this->dataType()->equivalent(*this->hiveType()),
        "data type {} and hive type {} do not match",
        this->dataType()->toString(),
        this->hiveType()->toString());
  } else {
    // Validate each extraction chain against hiveType and verify output types.
    for (const auto& extraction : this->extractions()) {
      auto derivedType =
          deriveExtractionOutputType(this->hiveType(), extraction.chain);
      VELOX_USER_CHECK(
          derivedType->equivalent(*extraction.dataType),
          "Extraction '{}' declared output type {} does not match "
          "derived type: {}",
          extraction.outputName,
          extraction.dataType->toString(),
          derivedType->toString());
    }
  }
}

folly::dynamic HiveColumnHandle::serialize() const {
  folly::dynamic obj = ColumnHandle::serializeBase("HiveColumnHandle");
  obj["hiveColumnHandleName"] = name();
  obj["columnType"] = columnTypeName(columnType());
  obj["dataType"] = dataType()->serialize();
  obj["hiveType"] = hiveType()->serialize();
  folly::dynamic requiredSubfields = folly::dynamic::array;
  for (const auto& subfield : this->requiredSubfields()) {
    requiredSubfields.push_back(serializeSubfield(subfield));
  }
  obj["requiredSubfields"] = requiredSubfields;
  if (!extractions().empty()) {
    folly::dynamic extractions = folly::dynamic::array;
    for (const auto& extraction : this->extractions()) {
      extractions.push_back(serializeNamedExtraction(extraction));
    }
    obj["extractions"] = extractions;
  }
  return obj;
}

std::string HiveColumnHandle::toStringFields() const {
  std::ostringstream out;
  out << fmt::format(
      "columnType: {}, dataType: {},",
      columnTypeName(columnType()),
      dataType()->toString());
  out << " requiredSubfields: [";
  for (const auto& subfield : requiredSubfields()) {
    out << " " << subfield.toString();
  }
  out << " ]";
  if (!extractions().empty()) {
    out << ", extractions: [";
    for (size_t i = 0; i < extractions().size(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      const auto& extraction = extractions()[i];
      out << "{outputName: " << extraction.outputName << ", chain: [";
      for (size_t j = 0; j < extraction.chain.size(); ++j) {
        if (j > 0) {
          out << ", ";
        }
        const auto& elem = *extraction.chain[j];
        out << extractionStepName(elem.step());
        if (elem.step() == ExtractionStep::kStructField) {
          out << "("
              << static_cast<const StructFieldExtractionPathElement&>(elem)
                     .fieldName()
              << ")";
        }
        if (elem.step() == ExtractionStep::kMapKeyFilter) {
          out << "(";
          if (auto* strFilter =
                  dynamic_cast<const StringMapKeyFilterExtractionPathElement*>(
                      &elem)) {
            for (size_t k = 0; k < strFilter->filterKeys().size(); ++k) {
              if (k > 0) {
                out << ", ";
              }
              out << "\"" << strFilter->filterKeys()[k] << "\"";
            }
          } else if (
              auto* intFilter =
                  dynamic_cast<const IntMapKeyFilterExtractionPathElement*>(
                      &elem)) {
            for (size_t k = 0; k < intFilter->filterKeys().size(); ++k) {
              if (k > 0) {
                out << ", ";
              }
              out << intFilter->filterKeys()[k];
            }
          }
          out << ")";
        }
      }
      out << "], dataType: " << extraction.dataType->toString() << "}";
    }
    out << "]";
  }
  return out.str();
}

std::string HiveColumnHandle::toString() const {
  return fmt::format(
      "HiveColumnHandle [name: {}, {}]", name(), toStringFields());
}

ColumnHandlePtr HiveColumnHandle::create(const folly::dynamic& obj) {
  auto name = obj["hiveColumnHandleName"].asString();
  auto columnType = columnTypeFromName(obj["columnType"].asString());
  auto dataType = ISerializable::deserialize<Type>(obj["dataType"]);
  auto hiveType = ISerializable::deserialize<Type>(obj["hiveType"]);

  const auto& arr = obj["requiredSubfields"];
  std::vector<common::Subfield> requiredSubfields;
  requiredSubfields.reserve(arr.size());
  for (auto& s : arr) {
    requiredSubfields.push_back(deserializeSubfield(s));
  }

  std::vector<NamedExtraction> extractions;
  if (auto it = obj.find("extractions"); it != obj.items().end()) {
    for (const auto& extraction : it->second) {
      extractions.push_back(deserializeNamedExtraction(extraction));
    }
  }

  return std::make_shared<HiveColumnHandle>(
      name,
      columnType,
      dataType,
      hiveType,
      std::move(requiredSubfields),
      std::move(extractions));
}

void HiveColumnHandle::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("HiveColumnHandle", HiveColumnHandle::create);
}

HiveTableHandle::HiveTableHandle(
    std::string connectorId,
    const std::string& tableName,
    common::SubfieldFilters subfieldFilters,
    const core::TypedExprPtr& remainingFilter,
    const RowTypePtr& dataColumns,
    std::vector<std::string> indexColumns,
    const std::unordered_map<std::string, std::string>& tableParameters,
    std::vector<HiveColumnHandlePtr> filterColumnHandles,
    double sampleRate,
    std::string dbName,
    std::vector<int32_t> dataColumnFieldIds)
    : FileTableHandle(
          std::move(connectorId),
          tableName,
          std::move(subfieldFilters),
          remainingFilter,
          dataColumns,
          tableParameters,
          sampleRate,
          std::move(dbName)),
      dataColumnFieldIds_(std::move(dataColumnFieldIds)),
      indexColumns_(std::move(indexColumns)),
      filterColumnHandles_(std::move(filterColumnHandles)) {
  VELOX_CHECK_GT(sampleRate, 0.0, "Sample rate must be positive");
  VELOX_CHECK_LE(sampleRate, 1.0, "Sample rate must not exceed 1.0");
  VELOX_CHECK(
      dataColumnFieldIds_.empty() ||
          (dataColumns != nullptr &&
           dataColumnFieldIds_.size() == dataColumns->size()),
      "Data column field IDs must be empty or aligned to data columns");
}

HiveTableHandle::HiveTableHandle(
    std::string connectorId,
    const std::string& tableName,
    common::SubfieldFilters subfieldFilters,
    const core::TypedExprPtr& remainingFilter,
    const RowTypePtr& dataColumns,
    const std::unordered_map<std::string, std::string>& tableParameters,
    std::vector<HiveColumnHandlePtr> filterColumnHandles,
    double sampleRate)
    : HiveTableHandle(
          std::move(connectorId),
          tableName,
          std::move(subfieldFilters),
          remainingFilter,
          dataColumns,
          /*indexColumns=*/{},
          tableParameters,
          std::move(filterColumnHandles),
          sampleRate) {}

std::string HiveTableHandle::toString() const {
  std::stringstream out;
  out << "table: " << tableName();
  if (!subfieldFilters().empty()) {
    // Sort filters by subfield for deterministic output.
    std::map<std::string, common::Filter*> orderedFilters;
    for (const auto& [field, filter] : subfieldFilters()) {
      orderedFilters[field.toString()] = filter.get();
    }
    out << ", range filters: [";
    bool notFirstFilter = false;
    for (const auto& [field, filter] : orderedFilters) {
      if (notFirstFilter) {
        out << ", ";
      }
      out << "(" << field << ", " << filter->toString() << ")";
      notFirstFilter = true;
    }
    out << "]";
  }
  if (sampleRate() < 1.0) {
    out << ", sample rate: " << sampleRate();
  }
  if (remainingFilter()) {
    out << ", remaining filter: (" << remainingFilter()->toString() << ")";
  }
  if (dataColumns()) {
    out << ", data columns: " << dataColumns()->toString();
  }
  if (!tableParameters().empty()) {
    std::map<std::string, std::string> orderedTableParameters{
        tableParameters().begin(), tableParameters().end()};
    out << ", table parameters: [";
    bool firstParam = true;
    for (const auto& param : orderedTableParameters) {
      if (!firstParam) {
        out << ", ";
      }
      out << param.first << ":" << param.second;
      firstParam = false;
    }
    out << "]";
  }
  if (!filterColumnHandles_.empty()) {
    out << ", filter column handles: [";
    bool first = true;
    for (auto& handle : filterColumnHandles_) {
      if (first) {
        first = false;
      } else {
        out << ", ";
      }
      out << handle->toString();
    }
    out << "]";
  }
  return out.str();
}

folly::dynamic HiveTableHandle::serializeHiveFields(
    std::string_view typeName) const {
  folly::dynamic obj = serializeFileFields(typeName);

  if (dataColumns()) {
    obj["dataColumns"] = dataColumns()->serialize();
  }

  if (!dataColumnFieldIds_.empty()) {
    folly::dynamic dataColumnFieldIds = folly::dynamic::array;
    for (const auto fieldId : dataColumnFieldIds_) {
      dataColumnFieldIds.push_back(fieldId);
    }
    obj["dataColumnFieldIds"] = std::move(dataColumnFieldIds);
  }

  folly::dynamic tableParameters = folly::dynamic::object;
  for (const auto& param : this->tableParameters()) {
    tableParameters[param.first] = param.second;
  }
  obj["tableParameters"] = tableParameters;

  if (!indexColumns_.empty()) {
    folly::dynamic indexColumns = folly::dynamic::array;
    for (const auto& column : indexColumns_) {
      indexColumns.push_back(column);
    }
    obj["indexColumns"] = indexColumns;
  }

  return obj;
}

folly::dynamic HiveTableHandle::serialize() const {
  return serializeHiveFields("HiveTableHandle");
}

// static
void HiveTableHandle::deserializeHiveFields(
    const folly::dynamic& obj,
    void* context,
    std::string& connectorId,
    std::string& tableName,
    common::SubfieldFilters& subfieldFilters,
    core::TypedExprPtr& remainingFilter,
    double& sampleRate,
    RowTypePtr& dataColumns,
    std::unordered_map<std::string, std::string>& tableParameters,
    std::vector<HiveColumnHandlePtr>& filterColumnHandles,
    std::vector<std::string>& indexColumns,
    std::string& dbName,
    std::vector<int32_t>& dataColumnFieldIds) {
  // Hive represents an absent remaining filter by omitting the field.
  if (auto it = obj.find("remainingFilter"); it != obj.items().end()) {
    VELOX_USER_CHECK(
        !it->second.isNull(),
        "Hive remaining filter must not be null when present");
  }

  std::vector<FileColumnHandlePtr> fileFilterColumns;
  deserializeFileFields(
      obj,
      context,
      connectorId,
      tableName,
      subfieldFilters,
      remainingFilter,
      fileFilterColumns,
      sampleRate,
      dbName);
  for (const auto& handle : fileFilterColumns) {
    filterColumnHandles.push_back(
        std::dynamic_pointer_cast<const HiveColumnHandle>(handle));
  }

  if (auto it = obj.find("dataColumns"); it != obj.items().end()) {
    dataColumns = ISerializable::deserialize<RowType>(it->second, context);
  }

  if (auto it = obj.find("dataColumnFieldIds"); it != obj.items().end()) {
    dataColumnFieldIds.reserve(it->second.size());
    for (const auto& fieldId : it->second) {
      dataColumnFieldIds.push_back(fieldId.asInt());
    }
  }

  const auto& tableParametersObj = obj["tableParameters"];
  for (const auto& key : tableParametersObj.keys()) {
    tableParameters.emplace(key.asString(), tableParametersObj[key].asString());
  }

  if (auto it = obj.find("indexColumns"); it != obj.items().end()) {
    for (const auto& col : it->second) {
      indexColumns.push_back(col.asString());
    }
  }
}

ConnectorTableHandlePtr HiveTableHandle::create(
    const folly::dynamic& obj,
    void* context) {
  std::string connectorId;
  std::string tableName;
  std::string dbName;
  common::SubfieldFilters subfieldFilters;
  core::TypedExprPtr remainingFilter;
  double sampleRate{1.0};
  RowTypePtr dataColumns;
  std::unordered_map<std::string, std::string> tableParameters;
  std::vector<HiveColumnHandlePtr> filterColumnHandles;
  std::vector<std::string> indexColumns;
  std::vector<int32_t> dataColumnFieldIds;

  deserializeHiveFields(
      obj,
      context,
      connectorId,
      tableName,
      subfieldFilters,
      remainingFilter,
      sampleRate,
      dataColumns,
      tableParameters,
      filterColumnHandles,
      indexColumns,
      dbName,
      dataColumnFieldIds);

  return std::make_shared<const HiveTableHandle>(
      std::move(connectorId),
      tableName,
      std::move(subfieldFilters),
      remainingFilter,
      dataColumns,
      std::move(indexColumns),
      tableParameters,
      std::move(filterColumnHandles),
      sampleRate,
      std::move(dbName),
      std::move(dataColumnFieldIds));
}

void HiveTableHandle::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("HiveTableHandle", create);
}

} // namespace facebook::velox::connector::hive
