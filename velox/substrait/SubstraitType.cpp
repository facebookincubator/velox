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


const std::unordered_map<::substrait::Type::KindCase, SubstraitTypePtr>
    SubstraitTypeUtil::TYPES = {
        {SubstraitTypeKind::kBool, std::make_shared<BooleanType>()},
        {SubstraitTypeKind::kI8, std::make_shared<TinyintType>()},
        {SubstraitTypeKind::kI16, std::make_shared<SmallintType>()},
        {SubstraitTypeKind::kI32, std::make_shared<IntegerType>()},
        {SubstraitTypeKind::kI64, std::make_shared<BigintType>()},
        {SubstraitTypeKind::kFp32, std::make_shared<RealType>()},
        {SubstraitTypeKind::kFp64, std::make_shared<DoubleType>()},
        {SubstraitTypeKind::kString, std::make_shared<StringType>()},
        {SubstraitTypeKind::kBinary, std::make_shared<BinaryType>()},
        {SubstraitTypeKind::kTimestamp, std::make_shared<TimestampType>()},
        {SubstraitTypeKind::kDate, std::make_shared<DateType>()},
        {SubstraitTypeKind::kTime, std::make_shared<TimeType>()},
        {SubstraitTypeKind::kIntervalYear,
         std::make_shared<IntervalYearType>()},
        {SubstraitTypeKind::kIntervalDay, std::make_shared<IntervalDayType>()},
        {SubstraitTypeKind::kTimestampTz, std::make_shared<TimestampTzType>()},
        {SubstraitTypeKind::kUuid, std::make_shared<UuidType>()},
        {SubstraitTypeKind::kFixedChar, std::make_shared<FixedCharType>()},
        {SubstraitTypeKind::kVarchar, std::make_shared<VarcharType>()},
        {SubstraitTypeKind::kFixedBinary, std::make_shared<FixedBinaryType>()},
        {SubstraitTypeKind::kDecimal, std::make_shared<DecimalType>()},
        {SubstraitTypeKind::kStruct, std::make_shared<StructType>()},
        {SubstraitTypeKind::kList, std::make_shared<ListType>()},
        {SubstraitTypeKind::kMap, std::make_shared<MapType>()},
        {SubstraitTypeKind::kUserDefined, std::make_shared<UserDefinedType>()},
};

const SubstraitTypePtr SubstraitTypeUtil::ANY_TYPE =
    std::make_unique<facebook::velox::substrait::SubstraitAnyType>();

const SubstraitTypePtr SubstraitTypeUtil::UNKNOWN_TYPE =
    std::make_unique<facebook::velox::substrait::SubstraitUnknownType>();

const std::string SubstraitTypeUtil::typeToSignature(
    const ::substrait::Type& type) {
  if (type.kind_case() == substrait::SubstraitTypeKind::kUserDefined) {
    return ANY_TYPE->signature();
  } else {
    if (TYPES.find(type.kind_case()) != TYPES.end()) {
      return TYPES.at(type.kind_case())->signature();
    } else {
      VELOX_NYI(
          "Returning Substrait signature of Substrait Type not supported for Substrait type {}.",
          type.kind_case());
    }
  }
}

const SubstraitTypePtr SubstraitTypeUtil::parseType(
    const std::string& rawType) {
  std::string lowerCaseRawType = rawType;
  std::transform(
      lowerCaseRawType.begin(), lowerCaseRawType.end(), lowerCaseRawType.begin(), [](unsigned char c) {
        return std::tolower(c);
      });

  if (lowerCaseRawType.rfind(ANY_TYPE->rawType(), 0) == 0) {
    return ANY_TYPE;
  } else if (lowerCaseRawType.rfind(UNKNOWN_TYPE->rawType(), 0) == 0) {
    return UNKNOWN_TYPE;
  } else {
    for (auto& [typeKind, type] : TYPES) {
      if (lowerCaseRawType.rfind(type->rawType(), 0) == 0) {
        return type;
      }
    }
    VELOX_NYI(
        "Returning Substrait Type not supported for raw type {}.", rawType);
  }
}

std::string SubstraitTypeUtil::signature(
    const std::string& functionName,
    const std::vector<::substrait::Type>& substraitTypes) {
  std::stringstream signature;
  signature << functionName << ":";
  for (auto it = substraitTypes.begin(); it != substraitTypes.end(); ++it) {
    const auto& typeSign = typeToSignature(*it);
    if (it == substraitTypes.end() - 1) {
      signature << typeSign;
    } else {
      signature << typeSign << "_";
    }
  }
  return signature.str();
}

} // namespace facebook::velox::substrait
