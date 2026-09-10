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
#pragma once

/// The compile-time half of Velox's type system: TypeKind, the enumeration of
/// the logical types; TypeTraits, the properties of each one; and
/// SimpleTypeTrait, which maps a C++ type back to the kind it denotes. Together
/// they are the type set that Type.h documents as "TypeKind"; the runtime Type
/// hierarchy that maps onto them lives there. Type.h includes this header.

#include <compare>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include "velox/common/EnumDeclare.h"

namespace facebook::velox {

/// Declarations suffice: TypeTraits only names these to form type aliases.
struct StringView;
struct Timestamp;

using int128_t = __int128_t;

/// Simple enum with type category.
enum class TypeKind : int8_t {
  BOOLEAN = 0,
  TINYINT = 1,
  SMALLINT = 2,
  INTEGER = 3,
  BIGINT = 4,
  REAL = 5,
  DOUBLE = 6,
  VARCHAR = 7,
  VARBINARY = 8,
  TIMESTAMP = 9,
  HUGEINT = 10,
  // Enum values for ComplexTypes start after 30 to leave
  // some values space to accommodate adding new scalar/native
  // types above.
  ARRAY = 30,
  MAP = 31,
  ROW = 32,
  UNKNOWN = 33,
  FUNCTION = 34,
  OPAQUE = 35,
  INVALID = 36
};

VELOX_DECLARE_ENUM_NAME(TypeKind);

template <TypeKind KIND>
class ScalarType;
class ShortDecimalType;
class LongDecimalType;
class ArrayType;
class MapType;
class RowType;
class FunctionType;
class OpaqueType;
class UnknownType;

struct UnknownValue {
  bool operator==(const UnknownValue& /* b */) const {
    return true;
  }

  auto operator<=>(const UnknownValue& /* b */) const {
    return std::strong_ordering::equal;
  }
};

template <typename T>
void toAppend(
    const ::facebook::velox::UnknownValue& /* value */,
    T* /* result */) {
  // TODO Implement
}

template <TypeKind KIND>
struct TypeTraits {};

template <>
struct TypeTraits<TypeKind::BOOLEAN> {
  using ImplType = ScalarType<TypeKind::BOOLEAN>;
  using NativeType = bool;
  using DeepCopiedType = NativeType;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::BOOLEAN;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = true;
  static constexpr const char* name = "BOOLEAN";
};

template <>
struct TypeTraits<TypeKind::TINYINT> {
  using ImplType = ScalarType<TypeKind::TINYINT>;
  using NativeType = int8_t;
  using DeepCopiedType = NativeType;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::TINYINT;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = true;
  static constexpr const char* name = "TINYINT";
};

template <>
struct TypeTraits<TypeKind::SMALLINT> {
  using ImplType = ScalarType<TypeKind::SMALLINT>;
  using NativeType = int16_t;
  using DeepCopiedType = NativeType;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::SMALLINT;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = true;
  static constexpr const char* name = "SMALLINT";
};

template <>
struct TypeTraits<TypeKind::INTEGER> {
  using ImplType = ScalarType<TypeKind::INTEGER>;
  using NativeType = int32_t;
  using DeepCopiedType = NativeType;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::INTEGER;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = true;
  static constexpr const char* name = "INTEGER";
};

template <>
struct TypeTraits<TypeKind::BIGINT> {
  using ImplType = ScalarType<TypeKind::BIGINT>;
  using NativeType = int64_t;
  using DeepCopiedType = NativeType;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::BIGINT;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = true;
  static constexpr const char* name = "BIGINT";
};

template <>
struct TypeTraits<TypeKind::REAL> {
  using ImplType = ScalarType<TypeKind::REAL>;
  using NativeType = float;
  using DeepCopiedType = NativeType;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::REAL;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = true;
  static constexpr const char* name = "REAL";
};

template <>
struct TypeTraits<TypeKind::DOUBLE> {
  using ImplType = ScalarType<TypeKind::DOUBLE>;
  using NativeType = double;
  using DeepCopiedType = NativeType;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::DOUBLE;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = true;
  static constexpr const char* name = "DOUBLE";
};

template <>
struct TypeTraits<TypeKind::VARCHAR> {
  using ImplType = ScalarType<TypeKind::VARCHAR>;
  using NativeType = velox::StringView;
  using DeepCopiedType = std::string;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::VARCHAR;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = false;
  static constexpr const char* name = "VARCHAR";
};

template <>
struct TypeTraits<TypeKind::TIMESTAMP> {
  using ImplType = ScalarType<TypeKind::TIMESTAMP>;
  using NativeType = Timestamp;
  using DeepCopiedType = Timestamp;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::TIMESTAMP;
  // isPrimitiveType in the type traits indicate whether it is a leaf type.
  // So only types which have other sub types, should be set to false.
  // Timestamp does not contain other types, so it is set to true.
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = true;
  static constexpr const char* name = "TIMESTAMP";
};

template <>
struct TypeTraits<TypeKind::HUGEINT> {
  using ImplType = ScalarType<TypeKind::HUGEINT>;
  using NativeType = int128_t;
  using DeepCopiedType = NativeType;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::HUGEINT;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = true;
  static constexpr const char* name = "HUGEINT";
};

template <>
struct TypeTraits<TypeKind::VARBINARY> {
  using ImplType = ScalarType<TypeKind::VARBINARY>;
  using NativeType = velox::StringView;
  using DeepCopiedType = std::string;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::VARBINARY;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = false;
  static constexpr const char* name = "VARBINARY";
};

template <>
struct TypeTraits<TypeKind::ARRAY> {
  using ImplType = ArrayType;
  using NativeType = void;
  using DeepCopiedType = void;
  static constexpr uint32_t minSubTypes = 1;
  static constexpr uint32_t maxSubTypes = 1;
  static constexpr TypeKind typeKind = TypeKind::ARRAY;
  static constexpr bool isPrimitiveType = false;
  static constexpr bool isFixedWidth = false;
  static constexpr const char* name = "ARRAY";
};

template <>
struct TypeTraits<TypeKind::MAP> {
  using ImplType = MapType;
  using NativeType = void;
  using DeepCopiedType = void;
  static constexpr uint32_t minSubTypes = 2;
  static constexpr uint32_t maxSubTypes = 2;
  static constexpr TypeKind typeKind = TypeKind::MAP;
  static constexpr bool isPrimitiveType = false;
  static constexpr bool isFixedWidth = false;
  static constexpr const char* name = "MAP";
};

template <>
struct TypeTraits<TypeKind::ROW> {
  using ImplType = RowType;
  using NativeType = void;
  using DeepCopiedType = void;
  static constexpr uint32_t minSubTypes = 1;
  static constexpr uint32_t maxSubTypes = std::numeric_limits<char16_t>::max();
  static constexpr TypeKind typeKind = TypeKind::ROW;
  static constexpr bool isPrimitiveType = false;
  static constexpr bool isFixedWidth = false;
  static constexpr const char* name = "ROW";
};

template <>
struct TypeTraits<TypeKind::UNKNOWN> {
  using ImplType = UnknownType;
  using NativeType = UnknownValue;
  using DeepCopiedType = UnknownValue;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::UNKNOWN;
  static constexpr bool isPrimitiveType = true;
  static constexpr bool isFixedWidth = true;
  static constexpr const char* name = "UNKNOWN";
};

template <>
struct TypeTraits<TypeKind::INVALID> {
  using ImplType = void;
  using NativeType = void;
  using DeepCopiedType = void;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::INVALID;
  static constexpr bool isPrimitiveType = false;
  static constexpr bool isFixedWidth = false;
  static constexpr const char* name = "INVALID";
};

template <>
struct TypeTraits<TypeKind::FUNCTION> {
  using ImplType = FunctionType;
  using NativeType = void;
  using DeepCopiedType = void;
  static constexpr uint32_t minSubTypes = 1;
  static constexpr uint32_t maxSubTypes = std::numeric_limits<char16_t>::max();
  static constexpr TypeKind typeKind = TypeKind::FUNCTION;
  static constexpr bool isPrimitiveType = false;
  static constexpr bool isFixedWidth = false;
  static constexpr const char* name = "FUNCTION";
};

template <>
struct TypeTraits<TypeKind::OPAQUE> {
  using ImplType = OpaqueType;
  using NativeType = std::shared_ptr<void>;
  using DeepCopiedType = std::shared_ptr<void>;
  static constexpr uint32_t minSubTypes = 0;
  static constexpr uint32_t maxSubTypes = 0;
  static constexpr TypeKind typeKind = TypeKind::OPAQUE;
  static constexpr bool isPrimitiveType = false;
  static constexpr bool isFixedWidth = false;
  static constexpr const char* name = "OPAQUE";
};

// Convenience constexpr function to check for string-like and nested type
// kinds.
constexpr bool is_string_kind(TypeKind kind) {
  return kind == TypeKind::VARCHAR || kind == TypeKind::VARBINARY;
}

constexpr bool is_nested_kind(TypeKind kind) {
  return kind == TypeKind::ARRAY || kind == TypeKind::MAP ||
      kind == TypeKind::ROW;
}

/// Evaluates to true only for std::shared_ptr<T>, the physical value type
/// stored in OPAQUE vectors.
template <typename>
struct is_shared_ptr : public std::false_type {};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : public std::true_type {};

/// Maps a C++ type used in a simple function signature to the TypeTraits of the
/// logical type it denotes, which is how registration derives a signature from
/// template arguments. Specializations for the signature tag types (Varchar,
/// Map, Array, Row, ...) are in SimpleFunctionApi.h, which is where the tags
/// themselves are declared.
template <typename T>
struct SimpleTypeTrait {};

template <>
struct SimpleTypeTrait<int128_t> : public TypeTraits<TypeKind::HUGEINT> {};

template <>
struct SimpleTypeTrait<int64_t> : public TypeTraits<TypeKind::BIGINT> {};

template <>
struct SimpleTypeTrait<int32_t> : public TypeTraits<TypeKind::INTEGER> {};

template <>
struct SimpleTypeTrait<int16_t> : public TypeTraits<TypeKind::SMALLINT> {};

template <>
struct SimpleTypeTrait<int8_t> : public TypeTraits<TypeKind::TINYINT> {};

template <>
struct SimpleTypeTrait<float> : public TypeTraits<TypeKind::REAL> {};

template <>
struct SimpleTypeTrait<double> : public TypeTraits<TypeKind::DOUBLE> {};

template <>
struct SimpleTypeTrait<bool> : public TypeTraits<TypeKind::BOOLEAN> {};

template <>
struct SimpleTypeTrait<Timestamp> : public TypeTraits<TypeKind::TIMESTAMP> {};

template <>
struct SimpleTypeTrait<UnknownValue> : public TypeTraits<TypeKind::UNKNOWN> {};

template <TypeKind KIND>
struct TypeFactory;

} // namespace facebook::velox
