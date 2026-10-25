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

#include "velox/connectors/hive/paimon/PaimonSubfield.h"

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
    : name_(std::move(name)),
      fieldId_(fieldId),
      columnType_(columnType),
      schemaType_(std::move(schemaType)),
      requiredSubfields_(std::move(requiredSubfields)),
      extractions_(std::move(extractions)),
      partitionDateValueDaysSinceEpoch_(partitionDateValueDaysSinceEpoch),
      postProcessor_(std::move(postProcessor)),
      dataType_(outputType(schemaType_, extractions_)) {
  VELOX_USER_CHECK(!name_.empty(), "Paimon column name must not be empty");
  VELOX_USER_CHECK_GE(fieldId_, 0, "Paimon field ID must be nonnegative");
  VELOX_USER_CHECK(
      extractions_.empty() || requiredSubfields_.empty(),
      "Extractions and requiredSubfields are mutually exclusive on column: {}",
      name_);
  for (const auto& path : requiredSubfields_) {
    VELOX_USER_CHECK(
        path.valid() && path.baseName() == name_,
        "Required subfield does not match column name '{}'",
        name_);
    resolveSubfieldType(ROW({name_}, {schemaType_}), path);
  }
}

std::string PaimonColumnHandle::toString() const {
  return fmt::format(
      "PaimonColumnHandle[name {}, fieldId {}, columnType {}, schemaType {}, "
      "dataType {}]",
      name_,
      fieldId_,
      columnType_,
      schemaType_->toString(),
      dataType_->toString());
}

folly::dynamic PaimonColumnHandle::serialize() const {
  VELOX_USER_CHECK(
      !postProcessor_, "Paimon column callbacks cannot be serialized");
  auto obj = ColumnHandle::serializeBase("PaimonColumnHandle");
  obj["version"] = kSerdeVersion;
  obj["columnName"] = name_;
  obj["fieldId"] = fieldId_;
  obj["columnType"] = columnTypeName(columnType_);
  obj["schemaType"] = schemaType_->serialize();
  obj["requiredSubfields"] = folly::dynamic::array;
  for (const auto& path : requiredSubfields_) {
    obj["requiredSubfields"].push_back(serializeSubfield(path));
  }
  obj["extractions"] = folly::dynamic::array;
  for (const auto& extraction : extractions_) {
    obj["extractions"].push_back(serializeNamedExtraction(extraction));
  }
  obj["partitionDateValueDaysSinceEpoch"] = partitionDateValueDaysSinceEpoch_;
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
  for (const auto& extraction : obj.at("extractions")) {
    extractions.push_back(deserializeNamedExtraction(extraction));
  }
  return std::make_shared<PaimonColumnHandle>(
      obj.at("columnName").asString(),
      ISerializable::deserialize<int32_t>(obj.at("fieldId")),
      columnTypeFromName(obj.at("columnType").asString()),
      ISerializable::deserialize<Type>(obj.at("schemaType")),
      std::move(subfields),
      std::move(extractions),
      obj.at("partitionDateValueDaysSinceEpoch").asBool());
}

void PaimonColumnHandle::registerSerDe() {
  DeserializationRegistryForSharedPtr().Register("PaimonColumnHandle", create);
}

} // namespace facebook::velox::connector::hive::paimon
