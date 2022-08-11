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

#pragma once

#include "iostream"
#include "velox/substrait/TypeUtils.h"
#include "velox/substrait/proto/substrait/algebra.pb.h"

namespace facebook::velox::substrait {

using SubstraitTypeKind = ::substrait::Type::KindCase;

template <SubstraitTypeKind KIND>
struct SubstraitTypeTraits {};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kBool> {
  static constexpr const char* signature = "bool";
  static constexpr const char* rawType = "boolean";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kI8> {
  static constexpr const char* signature = "i8";
  static constexpr const char* rawType = "i8";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kI16> {
  static constexpr const char* signature = "i16";
  static constexpr const char* rawType = "i16";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kI32> {
  static constexpr const char* signature = "i32";
  static constexpr const char* rawType = "i32";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kI64> {
  static constexpr const char* signature = "i64";
  static constexpr const char* rawType = "i64";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kFp32> {
  static constexpr const char* signature = "fp32";
  static constexpr const char* rawType = "fp32";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kFp64> {
  static constexpr const char* signature = "fp64";
  static constexpr const char* rawType = "fp64";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kString> {
  static constexpr const char* signature = "str";
  static constexpr const char* rawType = "string";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kBinary> {
  static constexpr const char* signature = "vbin";
  static constexpr const char* rawType = "binary";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kTimestamp> {
  static constexpr const char* signature = "ts";
  static constexpr const char* rawType = "timestamp";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kTimestampTz> {
  static constexpr const char* signature = "tstz";
  static constexpr const char* rawType = "timestamp_tz";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kDate> {
  static constexpr const char* signature = "date";
  static constexpr const char* rawType = "date";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kTime> {
  static constexpr const char* signature = "time";
  static constexpr const char* rawType = "time";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kIntervalYear> {
  static constexpr const char* signature = "iyear";
  static constexpr const char* rawType = "interval_year";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kIntervalDay> {
  static constexpr const char* signature = "iday";
  static constexpr const char* rawType = "interval_day";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kUuid> {
  static constexpr const char* signature = "uuid";
  static constexpr const char* rawType = "uuid";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kFixedChar> {
  static constexpr const char* signature = "fchar";
  static constexpr const char* rawType = "fixedchar";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kVarchar> {
  static constexpr const char* signature = "vchar";
  static constexpr const char* rawType = "varchar";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kFixedBinary> {
  static constexpr const char* signature = "fbin";
  static constexpr const char* rawType = "fixedbinary";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kDecimal> {
  static constexpr const char* signature = "dec";
  static constexpr const char* rawType = "decimal";
};
template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kStruct> {
  static constexpr const char* signature = "struct";
  static constexpr const char* rawType = "struct";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kList> {
  static constexpr const char* signature = "list";
  static constexpr const char* rawType = "list";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kMap> {
  static constexpr const char* signature = "map";
  static constexpr const char* rawType = "map";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kUserDefined> {
  static constexpr const char* signature = "u!name";
  static constexpr const char* rawType = "user defined type";
};

class SubstraitType {
 public:
  virtual const char* signature() const = 0;
  virtual const bool isWildcard() const = 0;
  virtual const char* rawType() const = 0;
};

/// Types used in function argument declarations.
template <SubstraitTypeKind Kind>
class SubstraitTypeBase : public SubstraitType {
 public:
 private:
  const char* signature() const override {
    return SubstraitTypeTraits<Kind>::signature;
  }
  const bool isWildcard() const override {
    return false;
  }
  const char* rawType() const override {
    return SubstraitTypeTraits<Kind>::rawType;
  }
};

class SubstraitAnyType : public SubstraitType {
 public:
  const char* signature() const override {
    return "any";
  }
  const bool isWildcard() const override {
    return true;
  }

  const char* rawType() const override {
    return "any";
  }
};

struct SubstraitTypeAnchor {
  std::string uri;
  std::string name;

  bool operator==(const SubstraitTypeAnchor& other) const {
    return (uri == other.uri && name == other.name);
  }
};

using SubstraitTypeAnchorPtr = std::shared_ptr<SubstraitTypeAnchor>;

class SubstraitUnknownType : public SubstraitType {
 public:
  const char* signature() const override {
    return "unknown";
  }
  const bool isWildcard() const override {
    return false;
  }
  const char* rawType() const override {
    return "unknown";
  }
};

using SubstraitTypePtr = std::shared_ptr<const SubstraitType>;

class SubstraitTypeUtil {
 private:
  /// Return the Substrait signature according to the substrait type.
  static const std::string typeToSignature(const ::substrait::Type& type);

  /// A map store the mapping of substrait type kind to corresponding type.
  static const std::unordered_map<::substrait::Type::KindCase, SubstraitTypePtr>
      TYPES;
  static const SubstraitTypePtr ANY_TYPE;
  static const SubstraitTypePtr UNKNOWN_TYPE;

 public:
  static const SubstraitTypePtr parseType(const std::string& rawType);

  // Return function signature according to the given function name and
  // substrait types.
  static std::string signature(
      const std::string& functionName,
      const std::vector<::substrait::Type>& types);
};

using BooleanType = SubstraitTypeBase<SubstraitTypeKind::kBool>;
using TinyintType = SubstraitTypeBase<SubstraitTypeKind::kI8>;
using SmallintType = SubstraitTypeBase<SubstraitTypeKind::kI16>;
using IntegerType = SubstraitTypeBase<SubstraitTypeKind::kI32>;
using BigintType = SubstraitTypeBase<SubstraitTypeKind::kI64>;
using RealType = SubstraitTypeBase<SubstraitTypeKind::kFp32>;
using DoubleType = SubstraitTypeBase<SubstraitTypeKind::kFp64>;
using StringType = SubstraitTypeBase<SubstraitTypeKind::kString>;
using BinaryType = SubstraitTypeBase<SubstraitTypeKind::kBinary>;
using TimestampType = SubstraitTypeBase<SubstraitTypeKind::kTimestamp>;
using DateType = SubstraitTypeBase<SubstraitTypeKind::kDate>;
using TimeType = SubstraitTypeBase<SubstraitTypeKind::kTime>;
using IntervalYearType = SubstraitTypeBase<SubstraitTypeKind::kIntervalYear>;
using IntervalDayType = SubstraitTypeBase<SubstraitTypeKind::kIntervalDay>;
using TimestampTzType = SubstraitTypeBase<SubstraitTypeKind::kTimestampTz>;
using UuidType = SubstraitTypeBase<SubstraitTypeKind::kUuid>;
using FixedCharType = SubstraitTypeBase<SubstraitTypeKind::kFixedChar>;
using VarcharType = SubstraitTypeBase<SubstraitTypeKind::kVarchar>;
using FixedBinaryType = SubstraitTypeBase<SubstraitTypeKind::kFixedBinary>;
using DecimalType = SubstraitTypeBase<SubstraitTypeKind::kDecimal>;
using StructType = SubstraitTypeBase<SubstraitTypeKind::kStruct>;
using ListType = SubstraitTypeBase<SubstraitTypeKind::kList>;
using MapType = SubstraitTypeBase<SubstraitTypeKind::kMap>;
using UserDefinedType = SubstraitTypeBase<SubstraitTypeKind::kUserDefined>;

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

} // namespace facebook::velox::substrait

namespace std {
/// hash function of facebook::velox::substrait::SubstraitTypeAnchor
template <>
struct hash<facebook::velox::substrait::SubstraitTypeAnchor> {
  size_t operator()(
      const facebook::velox::substrait::SubstraitTypeAnchor& k) const {
    return hash<std::string>()(k.name) ^ hash<std::string>()(k.uri);
  }
};

}; // namespace std