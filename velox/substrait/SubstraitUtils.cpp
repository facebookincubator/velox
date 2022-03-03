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

#include "SubstraitUtils.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::substrait {

std::shared_ptr<SubstraitParser::SubstraitType> SubstraitParser::parseType(
    const ::substrait::Type& sType) {
  std::string typeName;
  ::substrait::Type_Nullability nullability;
  switch (sType.kind_case()) {
    case ::substrait::Type::KindCase::kBool: {
      typeName = "BOOL";
      nullability = sType.bool_().nullability();
      break;
    }
    case ::substrait::Type::KindCase::kFp64: {
      typeName = "FP64";
      nullability = sType.fp64().nullability();
      break;
    }
    case ::substrait::Type::KindCase::kStruct: {
      // TODO: Support for Struct is not fully added.
      typeName = "Struct";
      auto sStruct = sType.struct_();
      auto sTypes = sStruct.types();
      for (const auto& type : sTypes) {
        parseType(type);
      }
      break;
    }
    case ::substrait::Type::KindCase::kString: {
      typeName = "STRING";
      nullability = sType.string().nullability();
      break;
    }
    default:
      VELOX_NYI("Substrait parsing for type {} not supported.", typeName);
  }
  bool nullable;
  switch (nullability) {
    case ::substrait::Type_Nullability::
        Type_Nullability_NULLABILITY_UNSPECIFIED:
      nullable = true;
      break;
    case ::substrait::Type_Nullability::Type_Nullability_NULLABILITY_NULLABLE:
      nullable = true;
      break;
    case ::substrait::Type_Nullability::Type_Nullability_NULLABILITY_REQUIRED:
      nullable = false;
      break;
    default:
      VELOX_NYI(
          "Substrait parsing for nullability {} not supported.", nullability);
  }
  std::shared_ptr<SubstraitType> substraitType =
      std::make_shared<SubstraitType>(typeName, nullable);
  return substraitType;
}

std::vector<std::shared_ptr<SubstraitParser::SubstraitType>>
SubstraitParser::parseNamedStruct(const ::substrait::NamedStruct& namedStruct) {
  const auto& sNames = namedStruct.names();
  std::vector<std::string> nameList;
  nameList.reserve(sNames.size());
  for (const auto& sName : sNames) {
    nameList.emplace_back(sName);
  }
  // Parse Struct.
  const auto& sStruct = namedStruct.struct_();
  const auto& sTypes = sStruct.types();
  std::vector<std::shared_ptr<SubstraitParser::SubstraitType>>
      substraitTypeList;
  substraitTypeList.reserve(sTypes.size());
  for (const auto& type : sTypes) {
    substraitTypeList.emplace_back(parseType(type));
  }
  return substraitTypeList;
}

int32_t SubstraitParser::parseReferenceSegment(
    const ::substrait::Expression::ReferenceSegment& sRef) {
  auto typeCase = sRef.reference_type_case();
  switch (typeCase) {
    case ::substrait::Expression::ReferenceSegment::ReferenceTypeCase::
        kStructField: {
      auto sField = sRef.struct_field();
      auto fieldId = sField.field();
      return fieldId;
    }
    default:
      VELOX_NYI(
          "Substrait conversion not supported for ReferenceSegment '{}'",
          typeCase);
  }
}

std::vector<std::string> SubstraitParser::makeNames(
    const std::string& prefix,
    int size) {
  std::vector<std::string> names;
  names.reserve(size);
  for (int i = 0; i < size; i++) {
    names.emplace_back(fmt::format("{}_{}", prefix, i));
  }
  return names;
}

std::string SubstraitParser::makeNodeName(int node_id, int col_idx) {
  return fmt::format("n{}_{}", node_id, col_idx);
}

std::string SubstraitParser::findSubstraitFunction(
    const std::unordered_map<uint64_t, std::string>& functionMap,
    const uint64_t& id) const {
  if (functionMap.find(id) == functionMap.end()) {
    VELOX_FAIL("Could not find function id {} in function map.", id);
  }
  std::unordered_map<uint64_t, std::string>& map =
      const_cast<std::unordered_map<uint64_t, std::string>&>(functionMap);
  return map[id];
}

std::string SubstraitParser::findVeloxFunction(
    const std::unordered_map<uint64_t, std::string>& functionMap,
    const uint64_t& id) const {
  std::string subFunc = findSubstraitFunction(functionMap, id);
  std::string veloxFunc = mapToVeloxFunction(subFunc);
  return veloxFunc;
}

std::string SubstraitParser::mapToVeloxFunction(
    const std::string& subFunc) const {
  if (substraitVeloxFunctionMap.find(subFunc) ==
      substraitVeloxFunctionMap.end()) {
    VELOX_FAIL(
        "Could not find Substrait function {} in function map.", subFunc);
  }
  std::unordered_map<std::string, std::string>& map =
      const_cast<std::unordered_map<std::string, std::string>&>(
          substraitVeloxFunctionMap);
  return map[subFunc];
}

} // namespace facebook::velox::substrait
