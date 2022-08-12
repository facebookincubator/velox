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
  static constexpr const char* matchingType = "boolean";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kI8> {
  static constexpr const char* signature = "i8";
  static constexpr const char* matchingType = "i8";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kI16> {
  static constexpr const char* signature = "i16";
  static constexpr const char* matchingType = "i16";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kI32> {
  static constexpr const char* signature = "i32";
  static constexpr const char* matchingType = "i32";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kI64> {
  static constexpr const char* signature = "i64";
  static constexpr const char* matchingType = "i64";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kFp32> {
  static constexpr const char* signature = "fp32";
  static constexpr const char* matchingType = "fp32";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kFp64> {
  static constexpr const char* signature = "fp64";
  static constexpr const char* matchingType = "fp64";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kString> {
  static constexpr const char* signature = "str";
  static constexpr const char* matchingType = "string";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kBinary> {
  static constexpr const char* signature = "vbin";
  static constexpr const char* matchingType = "binary";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kTimestamp> {
  static constexpr const char* signature = "ts";
  static constexpr const char* matchingType = "timestamp";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kTimestampTz> {
  static constexpr const char* signature = "tstz";
  static constexpr const char* matchingType = "timestamp_tz";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kDate> {
  static constexpr const char* signature = "date";
  static constexpr const char* matchingType = "date";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kTime> {
  static constexpr const char* signature = "time";
  static constexpr const char* matchingType = "time";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kIntervalYear> {
  static constexpr const char* signature = "iyear";
  static constexpr const char* matchingType = "interval_year";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kIntervalDay> {
  static constexpr const char* signature = "iday";
  static constexpr const char* matchingType = "interval_day";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kUuid> {
  static constexpr const char* signature = "uuid";
  static constexpr const char* matchingType = "uuid";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kFixedChar> {
  static constexpr const char* signature = "fchar";
  static constexpr const char* matchingType = "fixedchar";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kVarchar> {
  static constexpr const char* signature = "vchar";
  static constexpr const char* matchingType = "varchar";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kFixedBinary> {
  static constexpr const char* signature = "fbin";
  static constexpr const char* matchingType = "fixedbinary";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kDecimal> {
  static constexpr const char* signature = "dec";
  static constexpr const char* matchingType = "decimal";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kStruct> {
  static constexpr const char* signature = "struct";
  static constexpr const char* matchingType = "struct";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kList> {
  static constexpr const char* signature = "list";
  static constexpr const char* matchingType = "list";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kMap> {
  static constexpr const char* signature = "map";
  static constexpr const char* matchingType = "map";
};

template <>
struct SubstraitTypeTraits<SubstraitTypeKind::kUserDefined> {
  static constexpr const char* signature = "u!name";
  static constexpr const char* matchingType = "user defined type";
};

class SubstraitType {
 public:
  virtual const char* signature() const = 0;
  virtual const bool isWildcard() const {
    return false;
  }
  virtual const char* matchingType() const = 0;
  virtual const bool isUnknown() const {
    return false;
  }
  virtual const bool isKind() const {
    return !isWildcard() && !isUnknown();
  }
};

using SubstraitTypePtr = std::shared_ptr<const SubstraitType>;

using SubstraitTypePtr = std::shared_ptr<const SubstraitType>;

/// Types used in function argument declarations.
template <SubstraitTypeKind Kind>
class SubstraitTypeBase : public SubstraitType {
 public:
  const char* signature() const override {
    return SubstraitTypeTraits<Kind>::signature;
  }
  const char* matchingType() const override {
    return SubstraitTypeTraits<Kind>::matchingType;
  }
  const SubstraitTypeKind kind() const {
    return Kind;
  }
};

class SubstraitAnyType : public SubstraitType {
 public:
  SubstraitAnyType(const std::string& value) : value_(value) {}
  const char* signature() const override {
    return "any";
  }
  const bool isWildcard() const override {
    return true;
  }

  const char* matchingType() const override {
    return "any";
  }
  const std::string& value() const {
    return value_;
  }

 private:
  const std::string value_;
};

class SubstraitUnknownType : public SubstraitType {
 public:
  const char* signature() const override {
    return "unknown";
  }
  const bool isUnknown() const override {
    return true;
  }

  const char* matchingType() const override {
    return "unknown";
  }
};

struct SubstraitTypeAnchor {
  std::string uri;
  std::string name;

  bool operator==(const SubstraitTypeAnchor& other) const {
    return (uri == other.uri && name == other.name);
  }
};

template <TypeKind T>
class SubstraitTypeCreator {};

#define SUBSTRAITY_TYPE_OF(typeKind) \
  std::make_shared<SubstraitTypeBase<typeKind>>(SubstraitTypeBase<typeKind>())

template <>
class SubstraitTypeCreator<TypeKind::BOOLEAN> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kBool);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::TINYINT> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kI8);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::SMALLINT> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kI16);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::INTEGER> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kI32);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::BIGINT> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kI64);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::REAL> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kFp32);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::DOUBLE> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kFp64);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::VARCHAR> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kVarchar);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::VARBINARY> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kBinary);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::TIMESTAMP> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kTimestamp);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::DATE> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kDate);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::INTERVAL_DAY_TIME> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kIntervalDay);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::SHORT_DECIMAL> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kDecimal);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::LONG_DECIMAL> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kDecimal);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::ARRAY> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kList);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::MAP> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kMap);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::ROW> {
 public:
  static SubstraitTypePtr create() {
    return SUBSTRAITY_TYPE_OF(SubstraitTypeKind::kStruct);
  }
};

template <>
class SubstraitTypeCreator<TypeKind::UNKNOWN> {
 public:
  static SubstraitTypePtr create() {
    return std::make_shared<SubstraitUnknownType>();
  }
};

template <>
class SubstraitTypeCreator<TypeKind::FUNCTION> {
 public:
  static SubstraitTypePtr create() {
    throw std::runtime_error("FUNCTION type not supported");
  }
};

template <>
class SubstraitTypeCreator<TypeKind::OPAQUE> {
 public:
  static SubstraitTypePtr create() {
    throw std::runtime_error("OPAQUE type not supported");
  }
};

template <>
class SubstraitTypeCreator<TypeKind::INVALID> {
 public:
  static SubstraitTypePtr create() {
    throw std::runtime_error("Invalid type not supported");
  }
};

#define SUBSTRAIT_SCALAR_TYPE_MAPPING(typeKind)        \
  {                                                    \
    SubstraitTypeTraits<typeKind>::matchingType,       \
        std::make_shared<SubstraitTypeBase<typeKind>>( \
            SubstraitTypeBase<typeKind>())             \
  }

using SubstraitTypeAnchorPtr = std::shared_ptr<SubstraitTypeAnchor>;

class SubstraitTypeUtil {
 public:
  /// parsing substrait extenstion raw type string into Substrait extension
  /// type.
  /// @param type - substrait extension raw type, e.g.('string'/'i8').
  static SubstraitTypePtr fromString(const std::string& type);

  template <TypeKind T>
  static SubstraitTypePtr substraitTypeMaker() {
    return SubstraitTypeCreator<T>::create();
  }

  /// Return the Substrait extension type  according to the velox type.
  static SubstraitTypePtr fromVelox(const TypePtr& type);

  // Return function signature according to the given function name and
  // substrait types.
  static std::string signature(
      const std::string& functionName,
      const std::vector<SubstraitTypePtr>& types);

 private:
  static std::unordered_map<std::string, SubstraitTypePtr>& scalarTypes();
};

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