/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "SubstraitType.h"

namespace facebook::velox::substrait {

SubstraitTypePtr SubstraitTypeUtil::fromVelox(const TypePtr& type) {
  return VELOX_DYNAMIC_TYPE_DISPATCH(
      SubstraitTypeUtil::substraitTypeMaker, type->kind());
}

SubstraitTypePtr SubstraitTypeUtil::fromString(const std::string& type) {
  std::string lowerCaseRawType = type;
  std::transform(
      lowerCaseRawType.begin(),
      lowerCaseRawType.end(),
      lowerCaseRawType.begin(),
      [](unsigned char c) { return std::tolower(c); });

  const auto& scalarTypes = SubstraitTypeUtil::scalarTypes();
  for (const auto& [matchingType, substraitType] : scalarTypes) {
    if (lowerCaseRawType.rfind(matchingType, 0) == 0) {
      return substraitType;
    }
  }
  if (lowerCaseRawType.rfind("any", 0) == 0) {
    return std::make_shared<const SubstraitAnyType>(type);

  } else if (lowerCaseRawType.rfind("unknown", 0) == 0) {
    return std::make_shared<const SubstraitUnknownType>();
  } else {
    VELOX_NYI("Returning Substrait Type not supported for raw type {}.", type);
  }
}

std::string SubstraitTypeUtil::signature(
    const std::string& functionName,
    const std::vector<SubstraitTypePtr>& substraitTypes) {
  std::stringstream signature;
  signature << functionName;
  if (!substraitTypes.empty()) {
    signature << ":";
    for (auto it = substraitTypes.begin(); it != substraitTypes.end(); ++it) {
      const auto& typeSign = (*it)->signature();
      if (it == substraitTypes.end() - 1) {
        signature << typeSign;
      } else {
        signature << typeSign << "_";
      }
    }
  }

  return signature.str();
}
std::unordered_map<std::string, SubstraitTypePtr>&
SubstraitTypeUtil::scalarTypes() {
  static std::unordered_map<std::string, SubstraitTypePtr> map{
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kBool),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kI8),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kI16),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kI32),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kI64),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kFp32),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kFp64),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kString),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kBinary),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kTimestamp),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kTimestampTz),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kDate),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kTime),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kIntervalDay),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kIntervalYear),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kUuid),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kFixedChar),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kVarchar),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kFixedBinary),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kDecimal),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kStruct),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kList),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kMap),
      SUBSTRAIT_SCALAR_TYPE_MAPPING(SubstraitTypeKind::kUserDefined),
  };
  return map;
}

} // namespace facebook::velox::substrait
