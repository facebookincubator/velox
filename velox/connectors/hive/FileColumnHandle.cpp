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

#include "velox/connectors/hive/FileColumnHandle.h"

namespace facebook::velox::connector::hive {
namespace {

const std::unordered_map<FileColumnHandle::ColumnType, std::string>&
columnTypeNames() {
  static const std::unordered_map<FileColumnHandle::ColumnType, std::string>
      kColumnTypeNames = {
          {FileColumnHandle::ColumnType::kPartitionKey, "PartitionKey"},
          {FileColumnHandle::ColumnType::kRegular, "Regular"},
          {FileColumnHandle::ColumnType::kSynthesized, "Synthesized"},
          {FileColumnHandle::ColumnType::kRowIndex, "RowIndex"},
          {FileColumnHandle::ColumnType::kRowId, "RowId"},
      };
  return kColumnTypeNames;
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

// Validate the extraction chain against the input type for step
// compatibility.  Each step's output feeds as input to the next step.
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

// Recursively derive the output type from the input type and extraction
// chain.  This follows the derivation rules in the design document.
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

std::string FileColumnHandle::columnTypeName(
    FileColumnHandle::ColumnType columnType) {
  const auto& names = columnTypeNames();
  auto it = names.find(columnType);
  VELOX_CHECK(it != names.end(), "Unknown column type");
  return it->second;
}

FileColumnHandle::ColumnType FileColumnHandle::columnTypeFromName(
    const std::string& name) {
  const auto& names = columnTypeNames();
  for (const auto& [type, typeName] : names) {
    if (typeName == name) {
      return type;
    }
  }
  VELOX_FAIL("Unknown column type name: {}", name);
}

} // namespace facebook::velox::connector::hive
