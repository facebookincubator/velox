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
#include "velox/connectors/hive/paimon/PaimonColumnHandle.h"

namespace facebook::velox::connector::hive::paimon {
namespace {

TypePtr outputType(
    const TypePtr& schemaType,
    const std::vector<NamedExtraction>& extractions) {
  VELOX_USER_CHECK_NOT_NULL(schemaType, "Missing Paimon column schema type");
  if (extractions.empty()) {
    return schemaType;
  }
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (const auto& extraction : extractions) {
    const auto derived =
        deriveExtractionOutputType(schemaType, extraction.chain);
    VELOX_USER_CHECK(
        extraction.dataType && derived->equivalent(*extraction.dataType),
        "Paimon extraction '{}' output type must be {}",
        extraction.outputName,
        derived->toString());
    names.push_back(extraction.outputName);
    types.push_back(extraction.dataType);
  }
  return types.size() == 1 ? types.front()
                           : ROW(std::move(names), std::move(types));
}

} // namespace

PaimonColumnHandle::PaimonColumnHandle(
    std::string name,
    int32_t fieldId,
    ColumnType columnType,
    TypePtr schemaType,
    std::vector<common::Subfield> requiredSubfields,
    std::vector<NamedExtraction> extractions,
    bool partitionDateValueDaysSinceEpoch,
    std::function<void(VectorPtr&)> postProcessor)
    : FileColumnHandle{
          std::move(name),
          columnType,
          // Braces keep type derivation before moving its inputs.
          outputType(schemaType, extractions),
          std::move(schemaType),
          std::move(requiredSubfields),
          std::move(extractions),
          partitionDateValueDaysSinceEpoch,
          std::move(postProcessor)},
      fieldId_(fieldId) {
  VELOX_USER_CHECK(
      !this->name().empty(), "Paimon column name must not be empty");
  VELOX_USER_CHECK_GE(fieldId_, 0, "Paimon field ID must be nonnegative");
}

std::string PaimonColumnHandle::toString() const {
  return fmt::format(
      "PaimonColumnHandle[name {}, fieldId {}, columnType {}, schemaType {}, "
      "dataType {}]",
      name(),
      fieldId_,
      columnType(),
      schemaType()->toString(),
      dataType()->toString());
}

folly::dynamic PaimonColumnHandle::serialize() const {
  VELOX_USER_CHECK(
      !postProcessor(), "Paimon column callbacks cannot be serialized");
  auto obj = ColumnHandle::serializeBase("PaimonColumnHandle");
  obj["version"] = kSerdeVersion;
  obj["columnName"] = name();
  obj["fieldId"] = fieldId_;
  obj["columnType"] = columnTypeName(columnType());
  obj["schemaType"] = schemaType()->serialize();
  obj["requiredSubfields"] = folly::dynamic::array;
  for (const auto& path : requiredSubfields()) {
    obj["requiredSubfields"].push_back(serializeSubfield(path));
  }
  if (!extractions().empty()) {
    obj["extractions"] = folly::dynamic::array;
    for (const auto& extraction : extractions()) {
      obj["extractions"].push_back(serializeNamedExtraction(extraction));
    }
  }
  if (isPartitionDateValueDaysSinceEpoch()) {
    obj["partitionDateValueDaysSinceEpoch"] = true;
  }
  return obj;
}

ColumnHandlePtr PaimonColumnHandle::create(const folly::dynamic& obj) {
  VELOX_USER_CHECK_EQ(
      obj.at("version").asInt(),
      kSerdeVersion,
      "Unsupported Paimon column handle wire version");
  std::vector<common::Subfield> subfields;
  for (const auto& path : obj.at("requiredSubfields")) {
    subfields.push_back(deserializeSubfield(path));
  }
  std::vector<NamedExtraction> extractions;
  if (auto it = obj.find("extractions"); it != obj.items().end()) {
    for (const auto& extraction : it->second) {
      extractions.push_back(deserializeNamedExtraction(extraction));
    }
  }
  return std::make_shared<PaimonColumnHandle>(
      obj.at("columnName").asString(),
      ISerializable::deserialize<int32_t>(obj.at("fieldId")),
      columnTypeFromName(obj.at("columnType").asString()),
      ISerializable::deserialize<Type>(obj.at("schemaType")),
      std::move(subfields),
      std::move(extractions),
      obj.getDefault("partitionDateValueDaysSinceEpoch", false).asBool());
}

void PaimonColumnHandle::registerSerDe() {
  DeserializationRegistryForSharedPtr().Register("PaimonColumnHandle", create);
}

} // namespace facebook::velox::connector::hive::paimon
