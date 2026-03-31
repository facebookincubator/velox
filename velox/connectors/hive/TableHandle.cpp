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

namespace {
std::unordered_map<HiveColumnHandle::ColumnType, std::string>
columnTypeNames() {
  return {
      {HiveColumnHandle::ColumnType::kPartitionKey, "PartitionKey"},
      {HiveColumnHandle::ColumnType::kRegular, "Regular"},
      {HiveColumnHandle::ColumnType::kSynthesized, "Synthesized"},
      {HiveColumnHandle::ColumnType::kRowIndex, "RowIndex"},
      {HiveColumnHandle::ColumnType::kRowId, "RowId"},
  };
}

template <typename K, typename V>
std::unordered_map<V, K> invertMap(const std::unordered_map<K, V>& mapping) {
  std::unordered_map<V, K> inverted;
  for (const auto& [key, value] : mapping) {
    inverted.emplace(value, key);
  }
  return inverted;
}

std::unordered_map<ExtractionStep, std::string> extractionStepNames() {
  return {
      {ExtractionStep::kStructField, "STRUCT_FIELD"},
      {ExtractionStep::kMapKeys, "MAP_KEYS"},
      {ExtractionStep::kMapValues, "MAP_VALUES"},
      {ExtractionStep::kMapKeyFilter, "MAP_KEY_FILTER"},
      {ExtractionStep::kArrayElements, "ARRAY_ELEMENTS"},
      {ExtractionStep::kSize, "SIZE"},
  };
}

/// Validate the extraction chain against the input type for step
/// compatibility.  Each step's output feeds as input to the next step.
void validateExtractionChain(
    const TypePtr& inputType,
    const std::vector<ExtractionPathElementPtr>& chain) {
  auto currentType = inputType;
  for (size_t i = 0; i < chain.size(); ++i) {
    const auto& element = chain[i];
    switch (element->step()) {
      case ExtractionStep::kStructField: {
        VELOX_USER_CHECK(
            currentType->isRow(),
            "Extraction step StructField requires ROW input, got: {}",
            currentType->toString());
        const auto& rowType = currentType->asRow();
        auto& fieldName =
            static_cast<const StructFieldExtractionPathElement&>(*element)
                .fieldName();
        VELOX_USER_CHECK(
            rowType.containsChild(fieldName),
            "Extraction step StructField references non-existent field: {}",
            fieldName);
        currentType = rowType.findChild(fieldName);
        break;
      }
      case ExtractionStep::kMapKeys: {
        VELOX_USER_CHECK(
            currentType->isMap(),
            "Extraction step MapKeys requires MAP input, got: {}",
            currentType->toString());
        currentType = ARRAY(currentType->asMap().keyType());
        break;
      }
      case ExtractionStep::kMapValues: {
        VELOX_USER_CHECK(
            currentType->isMap(),
            "Extraction step MapValues requires MAP input, got: {}",
            currentType->toString());
        currentType = ARRAY(currentType->asMap().valueType());
        break;
      }
      case ExtractionStep::kMapKeyFilter: {
        VELOX_USER_CHECK(
            currentType->isMap(),
            "Extraction step MapKeyFilter requires MAP input, got: {}",
            currentType->toString());
        // Type-preserving: MAP(K, V) -> MAP(K, V).
        break;
      }
      case ExtractionStep::kArrayElements: {
        VELOX_USER_CHECK(
            currentType->isArray(),
            "Extraction step ArrayElements requires ARRAY input, got: {}",
            currentType->toString());
        currentType = currentType->asArray().elementType();
        break;
      }
      case ExtractionStep::kSize: {
        VELOX_USER_CHECK(
            currentType->isMap() || currentType->isArray(),
            "Extraction step Size requires MAP or ARRAY input, got: {}",
            currentType->toString());
        VELOX_USER_CHECK_EQ(
            i,
            chain.size() - 1,
            "Extraction step Size must be the last step in the chain.");
        currentType = BIGINT();
        break;
      }
    }
  }
}

/// Recursively derive the output type from the input type and extraction
/// chain.  This follows the derivation rules in the design document.
TypePtr deriveOutputTypeImpl(
    const TypePtr& inputType,
    const std::vector<ExtractionPathElementPtr>& chain,
    size_t index) {
  if (index >= chain.size()) {
    return inputType;
  }

  const auto& element = chain[index];
  switch (element->step()) {
    case ExtractionStep::kStructField: {
      VELOX_CHECK(inputType->isRow());
      auto& fieldName =
          static_cast<const StructFieldExtractionPathElement&>(*element)
              .fieldName();
      auto childType = inputType->asRow().findChild(fieldName);
      return deriveOutputTypeImpl(childType, chain, index + 1);
    }
    case ExtractionStep::kMapKeys: {
      VELOX_CHECK(inputType->isMap());
      auto keyType = inputType->asMap().keyType();
      if (index + 1 >= chain.size()) {
        return ARRAY(keyType);
      }
      VELOX_CHECK_EQ(
          static_cast<int>(chain[index + 1]->step()),
          static_cast<int>(ExtractionStep::kArrayElements));
      return ARRAY(deriveOutputTypeImpl(keyType, chain, index + 2));
    }
    case ExtractionStep::kMapValues: {
      VELOX_CHECK(inputType->isMap());
      auto valueType = inputType->asMap().valueType();
      if (index + 1 >= chain.size()) {
        return ARRAY(valueType);
      }
      VELOX_CHECK_EQ(
          static_cast<int>(chain[index + 1]->step()),
          static_cast<int>(ExtractionStep::kArrayElements));
      return ARRAY(deriveOutputTypeImpl(valueType, chain, index + 2));
    }
    case ExtractionStep::kMapKeyFilter: {
      VELOX_CHECK(inputType->isMap());
      return deriveOutputTypeImpl(inputType, chain, index + 1);
    }
    case ExtractionStep::kArrayElements: {
      VELOX_CHECK(inputType->isArray());
      auto elementType = inputType->asArray().elementType();
      return ARRAY(deriveOutputTypeImpl(elementType, chain, index + 1));
    }
    case ExtractionStep::kSize: {
      VELOX_CHECK(inputType->isMap() || inputType->isArray());
      return BIGINT();
    }
  }
  VELOX_UNREACHABLE();
}

folly::dynamic serializeExtractionPathElement(
    const ExtractionPathElement& element) {
  folly::dynamic obj = folly::dynamic::object;
  obj["step"] = extractionStepName(element.step());
  if (element.step() == ExtractionStep::kStructField) {
    obj["fieldName"] =
        static_cast<const StructFieldExtractionPathElement&>(element)
            .fieldName();
  }
  if (element.step() == ExtractionStep::kMapKeyFilter) {
    if (auto* stringFilter =
            dynamic_cast<const StringMapKeyFilterExtractionPathElement*>(
                &element)) {
      folly::dynamic keys = folly::dynamic::array;
      for (const auto& key : stringFilter->filterKeys()) {
        keys.push_back(key);
      }
      obj["stringFilterKeys"] = keys;
    } else if (
        auto* intFilter =
            dynamic_cast<const IntMapKeyFilterExtractionPathElement*>(
                &element)) {
      folly::dynamic keys = folly::dynamic::array;
      for (const auto& key : intFilter->filterKeys()) {
        keys.push_back(key);
      }
      obj["intFilterKeys"] = keys;
    }
  }
  return obj;
}

ExtractionPathElementPtr deserializeExtractionPathElement(
    const folly::dynamic& obj) {
  auto step = extractionStepFromName(obj["step"].asString());
  switch (step) {
    case ExtractionStep::kStructField:
      return ExtractionPathElement::structField(obj["fieldName"].asString());
    case ExtractionStep::kMapKeyFilter: {
      std::vector<std::string> stringKeys;
      std::vector<int64_t> intKeys;
      if (auto it = obj.find("stringFilterKeys"); it != obj.items().end()) {
        for (const auto& key : it->second) {
          stringKeys.push_back(key.asString());
        }
      }
      if (auto it = obj.find("intFilterKeys"); it != obj.items().end()) {
        for (const auto& key : it->second) {
          intKeys.push_back(key.asInt());
        }
      }
      if (!stringKeys.empty()) {
        return ExtractionPathElement::mapKeyFilter(std::move(stringKeys));
      }
      return ExtractionPathElement::mapKeyFilter(std::move(intKeys));
    }
    case ExtractionStep::kMapKeys:
    case ExtractionStep::kMapValues:
    case ExtractionStep::kArrayElements:
    case ExtractionStep::kSize:
      return ExtractionPathElement::simple(step);
  }
  VELOX_UNREACHABLE();
}

folly::dynamic serializeNamedExtraction(const NamedExtraction& extraction) {
  folly::dynamic obj = folly::dynamic::object;
  obj["outputName"] = extraction.outputName;
  obj["dataType"] = extraction.dataType->serialize();
  folly::dynamic chain = folly::dynamic::array;
  for (const auto& element : extraction.chain) {
    chain.push_back(serializeExtractionPathElement(*element));
  }
  obj["chain"] = chain;
  return obj;
}

NamedExtraction deserializeNamedExtraction(const folly::dynamic& obj) {
  NamedExtraction extraction;
  extraction.outputName = obj["outputName"].asString();
  extraction.dataType = ISerializable::deserialize<Type>(obj["dataType"]);
  const auto& chainArr = obj["chain"];
  extraction.chain.reserve(chainArr.size());
  for (const auto& element : chainArr) {
    extraction.chain.push_back(deserializeExtractionPathElement(element));
  }
  return extraction;
}

} // namespace

bool extractionPathElementEquals(
    const ExtractionPathElement& lhs,
    const ExtractionPathElement& rhs) {
  if (lhs.step() != rhs.step()) {
    return false;
  }
  switch (lhs.step()) {
    case ExtractionStep::kStructField:
      return static_cast<const StructFieldExtractionPathElement&>(lhs)
                 .fieldName() ==
          static_cast<const StructFieldExtractionPathElement&>(rhs).fieldName();
    case ExtractionStep::kMapKeyFilter: {
      if (auto* lStr =
              dynamic_cast<const StringMapKeyFilterExtractionPathElement*>(
                  &lhs)) {
        auto* rStr =
            dynamic_cast<const StringMapKeyFilterExtractionPathElement*>(&rhs);
        return rStr && lStr->filterKeys() == rStr->filterKeys();
      }
      if (auto* lInt =
              dynamic_cast<const IntMapKeyFilterExtractionPathElement*>(&lhs)) {
        auto* rInt =
            dynamic_cast<const IntMapKeyFilterExtractionPathElement*>(&rhs);
        return rInt && lInt->filterKeys() == rInt->filterKeys();
      }
      return false;
    }
    case ExtractionStep::kMapKeys:
    case ExtractionStep::kMapValues:
    case ExtractionStep::kArrayElements:
    case ExtractionStep::kSize:
      return true;
  }
  VELOX_UNREACHABLE();
}

/*static*/ std::shared_ptr<const ExtractionPathElement>
ExtractionPathElement::simple(ExtractionStep step) {
  return std::make_shared<SimpleExtractionPathElement>(step);
}

/*static*/ std::shared_ptr<const ExtractionPathElement>
ExtractionPathElement::structField(const std::string& name) {
  return std::make_shared<StructFieldExtractionPathElement>(name);
}

/*static*/ std::shared_ptr<const ExtractionPathElement>
ExtractionPathElement::mapKeyFilter(std::vector<std::string> keys) {
  return std::make_shared<StringMapKeyFilterExtractionPathElement>(
      std::move(keys));
}

/*static*/ std::shared_ptr<const ExtractionPathElement>
ExtractionPathElement::mapKeyFilter(std::vector<int64_t> keys) {
  return std::make_shared<IntMapKeyFilterExtractionPathElement>(
      std::move(keys));
}

bool NamedExtraction::operator==(const NamedExtraction& other) const {
  if (outputName != other.outputName || chain.size() != other.chain.size() ||
      !dataType->equivalent(*other.dataType)) {
    return false;
  }
  for (size_t i = 0; i < chain.size(); ++i) {
    if (!extractionPathElementEquals(*chain[i], *other.chain[i])) {
      return false;
    }
  }
  return true;
}

std::string extractionStepName(ExtractionStep step) {
  static const auto names = extractionStepNames();
  return names.at(step);
}

ExtractionStep extractionStepFromName(const std::string& name) {
  static const auto nameToStep = invertMap(extractionStepNames());
  return nameToStep.at(name);
}

TypePtr deriveExtractionOutputType(
    const TypePtr& inputType,
    const std::vector<ExtractionPathElementPtr>& chain) {
  validateExtractionChain(inputType, chain);
  return deriveOutputTypeImpl(inputType, chain, 0);
}

std::string HiveColumnHandle::columnTypeName(
    HiveColumnHandle::ColumnType type) {
  static const auto ctNames = columnTypeNames();
  return ctNames.at(type);
}

HiveColumnHandle::ColumnType HiveColumnHandle::columnTypeFromName(
    const std::string& name) {
  static const auto nameColumnTypes = invertMap(columnTypeNames());
  return nameColumnTypes.at(name);
}

HiveColumnHandle::HiveColumnHandle(
    const std::string& name,
    ColumnType columnType,
    TypePtr dataType,
    TypePtr hiveType,
    std::vector<common::Subfield> requiredSubfields,
    std::vector<NamedExtraction> extractions,
    ColumnParseParameters columnParseParameters,
    std::function<void(VectorPtr&)> postProcessor)
    : name_(name),
      columnType_(columnType),
      dataType_(std::move(dataType)),
      hiveType_(std::move(hiveType)),
      requiredSubfields_(std::move(requiredSubfields)),
      extractions_(std::move(extractions)),
      columnParseParameters_(columnParseParameters),
      postProcessor_(std::move(postProcessor)) {
  VELOX_USER_CHECK(
      extractions_.empty() || requiredSubfields_.empty(),
      "Extractions and requiredSubfields are mutually exclusive on column: {}",
      name_);

  if (extractions_.empty()) {
    // No extractions: dataType and hiveType must match (existing behavior).
    VELOX_USER_CHECK(
        dataType_->equivalent(*hiveType_),
        "data type {} and hive type {} do not match",
        dataType_->toString(),
        hiveType_->toString());
  } else {
    // Validate each extraction chain against hiveType and verify output types.
    for (const auto& extraction : extractions_) {
      auto derivedType =
          deriveExtractionOutputType(hiveType_, extraction.chain);
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
  obj["hiveColumnHandleName"] = name_;
  obj["columnType"] = columnTypeName(columnType_);
  obj["dataType"] = dataType_->serialize();
  obj["hiveType"] = hiveType_->serialize();
  folly::dynamic requiredSubfields = folly::dynamic::array;
  for (const auto& subfield : requiredSubfields_) {
    requiredSubfields.push_back(subfield.toString());
  }
  obj["requiredSubfields"] = requiredSubfields;
  if (!extractions_.empty()) {
    folly::dynamic extractions = folly::dynamic::array;
    for (const auto& extraction : extractions_) {
      extractions.push_back(serializeNamedExtraction(extraction));
    }
    obj["extractions"] = extractions;
  }
  return obj;
}

std::string HiveColumnHandle::toString() const {
  std::ostringstream out;
  out << fmt::format(
      "HiveColumnHandle [name: {}, columnType: {}, dataType: {},",
      name_,
      columnTypeName(columnType_),
      dataType_->toString());
  out << " requiredSubfields: [";
  for (const auto& subfield : requiredSubfields_) {
    out << " " << subfield.toString();
  }
  out << " ]";
  if (!extractions_.empty()) {
    out << ", extractions: [";
    for (size_t i = 0; i < extractions_.size(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      const auto& extraction = extractions_[i];
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
  out << "]";
  return out.str();
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
    requiredSubfields.emplace_back(s.asString());
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
    std::string dbName)
    : ConnectorTableHandle(std::move(connectorId)),
      tableName_(tableName),
      subfieldFilters_(std::move(subfieldFilters)),
      remainingFilter_(remainingFilter),
      sampleRate_(sampleRate),
      dataColumns_(dataColumns),
      indexColumns_(std::move(indexColumns)),
      tableParameters_(tableParameters),
      filterColumnHandles_(std::move(filterColumnHandles)),
      dbName_(std::move(dbName)) {
  VELOX_CHECK_GT(sampleRate_, 0.0, "Sample rate must be positive");
  VELOX_CHECK_LE(sampleRate_, 1.0, "Sample rate must not exceed 1.0");
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
  out << "table: " << tableName_;
  if (!subfieldFilters_.empty()) {
    // Sort filters by subfield for deterministic output.
    std::map<std::string, common::Filter*> orderedFilters;
    for (const auto& [field, filter] : subfieldFilters_) {
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
  if (sampleRate_ < 1.0) {
    out << ", sample rate: " << sampleRate_;
  }
  if (remainingFilter_) {
    out << ", remaining filter: (" << remainingFilter_->toString() << ")";
  }
  if (dataColumns_) {
    out << ", data columns: " << dataColumns_->toString();
  }
  if (!tableParameters_.empty()) {
    std::map<std::string, std::string> orderedTableParameters{
        tableParameters_.begin(), tableParameters_.end()};
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

folly::dynamic HiveTableHandle::serialize() const {
  folly::dynamic obj = ConnectorTableHandle::serializeBase("HiveTableHandle");
  obj["tableName"] = tableName_;

  folly::dynamic subfieldFilters = folly::dynamic::array;
  for (const auto& [subfield, filter] : subfieldFilters_) {
    folly::dynamic pair = folly::dynamic::object;
    pair["subfield"] = subfield.toString();
    pair["filter"] = filter->serialize();
    subfieldFilters.push_back(pair);
  }

  obj["subfieldFilters"] = subfieldFilters;
  if (remainingFilter_) {
    obj["remainingFilter"] = remainingFilter_->serialize();
  }

  if (sampleRate_ < 1.0) {
    obj["sampleRate"] = sampleRate_;
  }

  if (dataColumns_) {
    obj["dataColumns"] = dataColumns_->serialize();
  }
  folly::dynamic tableParameters = folly::dynamic::object;
  for (const auto& param : tableParameters_) {
    tableParameters[param.first] = param.second;
  }
  obj["tableParameters"] = tableParameters;
  if (!filterColumnHandles_.empty()) {
    folly::dynamic filterColumnHandles = folly::dynamic::array;
    for (const auto& handle : filterColumnHandles_) {
      filterColumnHandles.push_back(handle->serialize());
    }
    obj["filterColumnHandles"] = filterColumnHandles;
  }
  if (!indexColumns_.empty()) {
    folly::dynamic indexColumns = folly::dynamic::array;
    for (const auto& column : indexColumns_) {
      indexColumns.push_back(column);
    }
    obj["indexColumns"] = indexColumns;
  }

  if (!dbName_.empty()) {
    obj["dbName"] = dbName_;
  }

  return obj;
}

ConnectorTableHandlePtr HiveTableHandle::create(
    const folly::dynamic& obj,
    void* context) {
  auto connectorId = obj["connectorId"].asString();
  auto tableName = obj["tableName"].asString();

  core::TypedExprPtr remainingFilter;
  if (auto it = obj.find("remainingFilter"); it != obj.items().end()) {
    remainingFilter =
        ISerializable::deserialize<core::ITypedExpr>(it->second, context);
  }

  common::SubfieldFilters subfieldFilters;
  folly::dynamic subfieldFiltersObj = obj["subfieldFilters"];
  for (const auto& subfieldFilter : subfieldFiltersObj) {
    common::Subfield subfield(subfieldFilter["subfield"].asString());
    auto filter =
        ISerializable::deserialize<common::Filter>(subfieldFilter["filter"]);
    subfieldFilters[common::Subfield(std::move(subfield.path()))] =
        filter->clone();
  }

  double sampleRate = 1.0;
  if (obj.count("sampleRate")) {
    sampleRate = obj["sampleRate"].asDouble();
  }

  RowTypePtr dataColumns;
  if (auto it = obj.find("dataColumns"); it != obj.items().end()) {
    dataColumns = ISerializable::deserialize<RowType>(it->second, context);
  }

  std::unordered_map<std::string, std::string> tableParameters{};
  const auto& tableParametersObj = obj["tableParameters"];
  for (const auto& key : tableParametersObj.keys()) {
    const auto& value = tableParametersObj[key];
    tableParameters.emplace(key.asString(), value.asString());
  }

  std::vector<HiveColumnHandlePtr> filterColumnHandles;
  if (auto it = obj.find("filterColumnHandles"); it != obj.items().end()) {
    for (const auto& handle : it->second) {
      filterColumnHandles.push_back(
          ISerializable::deserialize<HiveColumnHandle>(handle, context));
    }
  }

  std::vector<std::string> indexColumns;
  if (auto it = obj.find("indexColumns"); it != obj.items().end()) {
    for (const auto& column : it->second) {
      indexColumns.push_back(column.asString());
    }
  }

  std::string dbName;
  if (auto it = obj.find("dbName"); it != obj.items().end()) {
    dbName = it->second.asString();
  }

  return std::make_shared<const HiveTableHandle>(
      connectorId,
      tableName,
      std::move(subfieldFilters),
      remainingFilter,
      dataColumns,
      std::move(indexColumns),
      tableParameters,
      std::move(filterColumnHandles),
      sampleRate,
      std::move(dbName));
}

void HiveTableHandle::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("HiveTableHandle", create);
}

} // namespace facebook::velox::connector::hive
