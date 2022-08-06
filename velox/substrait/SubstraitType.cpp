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

const std::string SubstraitType::BOOL = "bool";
const std::string SubstraitType::I8 = "i8";
const std::string SubstraitType::I16 = "i16";
const std::string SubstraitType::I32 = "i32";
const std::string SubstraitType::I64 = "i64";
const std::string SubstraitType::FP32 = "fp32";
const std::string SubstraitType::FP64 = "fp64";
const std::string SubstraitType::STRING = "str";
const std::string SubstraitType::BINARY = "vbin";
const std::string SubstraitType::TIMESTAMP = "ts";
const std::string SubstraitType::TIMESTAMP_TZ = "tstz";
const std::string SubstraitType::DATE = "date";
const std::string SubstraitType::TIME = "time";
const std::string SubstraitType::INTERVAL_YEAR = "iyear";
const std::string SubstraitType::INTERVAL_DAY = "iday";
const std::string SubstraitType::UUID = "uuid";
const std::string SubstraitType::FIXED_CHAR = "fchar";
const std::string SubstraitType::VARCHAR = "vchar";
const std::string SubstraitType::FIXED_BINARY = "fbin";
const std::string SubstraitType::DECIMAL = "dec";
const std::string SubstraitType::STRUCT = "struct";
const std::string SubstraitType::LIST = "list";
const std::string SubstraitType::MAP = "map";
const std::string SubstraitType::ANY = "any";
const std::string SubstraitType::USER_DEFINED = "u!name";
const std::unordered_map<std::string, std::string>
    SubstraitType::ARGUMENT_TO_SIGNATURE = {
        {"boolean", BOOL},
        {"i8", I8},
        {"i16", I16},
        {"i32", I32},
        {"i64", I64},
        {"fp32", FP32},
        {"fp64", FP64},
        {"string", STRING},
        {"binary", BINARY},
        {"timestamp", TIMESTAMP},
        {"timestamp_tz", TIMESTAMP_TZ},
        {"date", DATE},
        {"time", TIME},
        {"interval_year", INTERVAL_YEAR},
        {"interval_day", INTERVAL_DAY},
        {"uuid", UUID},
        {"fixedchar", FIXED_CHAR},
        {"varchar", VARCHAR},
        {"fixedbinary", FIXED_BINARY},
        {"decimal", DECIMAL},
        {"struct", STRUCT},
        {"list", LIST},
        {"map", MAP},
        {"any", ANY},
        {"user defined type", USER_DEFINED}};

const std::unordered_map<::substrait::Type::KindCase, std::string>
    SubstraitType::TYPE_KIND_TO_SIGNATURE = {
        {::substrait::Type::KindCase::kBool, BOOL},
        {::substrait::Type::KindCase::kI8, I8},
        {::substrait::Type::KindCase::kI16, I16},
        {::substrait::Type::KindCase::kI32, I32},
        {::substrait::Type::KindCase::kI64, I64},
        {::substrait::Type::KindCase::kFp32, FP32},
        {::substrait::Type::KindCase::kFp64, FP64},
        {::substrait::Type::KindCase::kString, STRING},
        {::substrait::Type::KindCase::kBinary, BINARY},
        {::substrait::Type::KindCase::kTimestamp, TIMESTAMP},
        {::substrait::Type::KindCase::kDate, DATE},
        {::substrait::Type::KindCase::kTime, TIME},
        {::substrait::Type::KindCase::kIntervalYear, INTERVAL_YEAR},
        {::substrait::Type::KindCase::kIntervalDay, INTERVAL_DAY},
        {::substrait::Type::KindCase::kTimestampTz, TIMESTAMP_TZ},
        {::substrait::Type::KindCase::kUuid, UUID},
        {::substrait::Type::KindCase::kFixedChar, FIXED_CHAR},
        {::substrait::Type::KindCase::kVarchar, VARCHAR},
        {::substrait::Type::KindCase::kFixedBinary, FIXED_BINARY},
        {::substrait::Type::KindCase::kDecimal, DECIMAL},
        {::substrait::Type::KindCase::kStruct, STRUCT},
        {::substrait::Type::KindCase::kList, LIST},
        {::substrait::Type::KindCase::kMap, MAP},
        {::substrait::Type::KindCase::kUserDefinedTypeReference, USER_DEFINED},
};

const std::string& SubstraitType::typeToSignature(
    const ::substrait::Type& type) {
  if (TYPE_KIND_TO_SIGNATURE.find(type.kind_case()) !=
      TYPE_KIND_TO_SIGNATURE.end()) {
    return TYPE_KIND_TO_SIGNATURE.at(type.kind_case());
  } else {
    VELOX_NYI(
        "Returning Substrait signature of Substrait Type not supported for Substrait type {}.",
        type.kind_case());
  }
}

const std::string& SubstraitType::argumentToSignature(
    const std::string& argumentType) {
  if (isWildcard(argumentType)) {
    return ANY;
  }
  if (ARGUMENT_TO_SIGNATURE.find(argumentType) != ARGUMENT_TO_SIGNATURE.end()) {
    return ARGUMENT_TO_SIGNATURE.at(argumentType);
  } else {
    for (auto& [key, value] : ARGUMENT_TO_SIGNATURE) {
      if (argumentType.rfind(key, 0) == 0) {
        return value;
      }
    }
  }
  VELOX_NYI(
      "Returning Substrait signature of argument type not supported for argumentType  {}.",
      argumentType);
}

bool SubstraitType::isWildcard(const std::string& argumentType) {
  return argumentType.rfind("any", 0) == 0 || argumentType.rfind("ANY", 0) == 0;
}

std::string SubstraitType::signature(
    const std::string& name,
    const std::vector<::substrait::Type>& types) {
  std::stringstream signature;
  signature << name << ":";
  for (auto& type : types) {
    signature << "_" << typeToSignature(type);
  }
  const auto& signatureStr = signature.str();
}

} // namespace facebook::velox::substrait
