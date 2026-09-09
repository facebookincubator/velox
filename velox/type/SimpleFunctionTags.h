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

/// Tag types naming the argument and return types of simple functions.
/// SimpleFunctionApi.h maps these onto the runtime type system.

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>

#include <fmt/format.h>

#include "velox/type/TypeKind.h"

namespace facebook::velox {

// Declared, not defined: an initialize() signature names these by reference,
// and a reference needs only a declaration. Keeps Type.h and QueryConfig.h out
// of headers that merely define simple functions.
class Type;
using TypePtr = std::shared_ptr<const Type>;

namespace core {
class QueryConfig;
} // namespace core

template <typename UNDERLYING_TYPE>
struct Variadic {
  using underlying_type = UNDERLYING_TYPE;

  Variadic() = delete;
};

// A type that can be used in simple function to represent any type.
// Two Generics with the same type variables should bound to the same type.
template <size_t id>
struct TypeVariable {
  static size_t getId() {
    return id;
  }
};

using T1 = TypeVariable<1>;
using T2 = TypeVariable<2>;
using T3 = TypeVariable<3>;
using T4 = TypeVariable<4>;
using T5 = TypeVariable<5>;
using T6 = TypeVariable<6>;
using T7 = TypeVariable<7>;
using T8 = TypeVariable<8>;

template <size_t id>
struct IntegerVariable {
  static size_t getId() {
    return id;
  }

  static std::string name() {
    return fmt::format("i{}", id);
  }
};

using P1 = IntegerVariable<1>;
using P2 = IntegerVariable<2>;
using P3 = IntegerVariable<3>;
using P4 = IntegerVariable<4>;
using S1 = IntegerVariable<5>;
using S2 = IntegerVariable<6>;
using S3 = IntegerVariable<7>;
using S4 = IntegerVariable<8>;

template <size_t id>
struct EnumVariable {
  static size_t getId() {
    return id;
  }

  static std::string name() {
    return fmt::format("E{}", id);
  }
};

using E1 = EnumVariable<1>;
using E2 = EnumVariable<2>;

template <typename P, typename S>
struct ShortDecimal {
 private:
  ShortDecimal() {}
};

template <typename P, typename S>
struct LongDecimal {
 private:
  LongDecimal() {}
};

struct AnyType {};

template <typename T = AnyType, bool comparable = false, bool orderable = false>
struct Generic {
  Generic() = delete;
  static_assert(!(orderable && !comparable), "Orderable implies comparable.");
};

using Any = Generic<>;

template <typename T>
using Comparable = Generic<T, true, false>;

// Orderable implies comparable.
template <typename T>
using Orderable = Generic<T, true, true>;

template <typename>
struct isVariadicType : public std::false_type {};

template <typename T>
struct isVariadicType<Variadic<T>> : public std::true_type {};

template <typename>
struct isGenericType : public std::false_type {};

template <typename T, bool comparable, bool orderable>
struct isGenericType<Generic<T, comparable, orderable>>
    : public std::true_type {};

// std::shared_ptr<T> is the C++ representation of the OPAQUE type.
template <typename T>
struct isOpaqueType : public is_shared_ptr<T> {};

template <typename KEY, typename VALUE>
struct Map {
  using key_type = KEY;
  using value_type = VALUE;

  static_assert(
      !isVariadicType<key_type>::value,
      "Map keys cannot be Variadic");
  static_assert(
      !isVariadicType<value_type>::value,
      "Map values cannot be Variadic");

 private:
  Map() {}
};

template <typename ELEMENT>
struct Array {
  using element_type = ELEMENT;

  static_assert(
      !isVariadicType<element_type>::value,
      "Array elements cannot be Variadic");

 private:
  Array() {}
};

template <typename ELEMENT>
using ArrayWriterT = Array<ELEMENT>;

template <typename... T>
struct Row {
  template <size_t idx>
  using type_at = typename std::tuple_element<idx, std::tuple<T...>>::type;

  static const size_t size_ = sizeof...(T);

  static_assert(
      std::conjunction<std::bool_constant<!isVariadicType<T>::value>...>::value,
      "Struct fields cannot be Variadic");

 private:
  Row() {}
};

struct DynamicRow {
 private:
  DynamicRow() {}
};

// T must be a struct with T::type being a built-in type and T::typeName
// type name to use in FunctionSignature.
// providesCustomComparison must be set to true to ensure values are wrapped in
// a view that exposes the custom comparison operations in Simple Functions.
template <typename T, bool providesCustomComparison_ = false>
struct CustomType {
  static constexpr bool providesCustomComparison = providesCustomComparison_;

 private:
  CustomType() {}
};

template <typename T>
struct UnwrapCustomType {
  using type = T;
};

template <typename T, bool providesCustomComparison>
struct UnwrapCustomType<CustomType<T, providesCustomComparison>> {
  using type = typename T::type;
};

template <typename T>
struct providesCustomComparison {
  static constexpr bool value = false;
};

template <typename T>
struct providesCustomComparison<CustomType<T, true>> {
  static constexpr bool value = true;
};

struct IntervalDayTime {
 private:
  IntervalDayTime() {}
};

struct IntervalYearMonth {
 private:
  IntervalYearMonth() {}
};

struct Date {
 private:
  Date() {}
};

struct Varbinary {
 private:
  Varbinary() {}
};

struct Varchar {
 private:
  Varchar() {}
};

struct TimestampUtcT {
  using type = Timestamp;

  static constexpr const char* typeName = "timestamp utc";
};

using TimestampUtc = CustomType<TimestampUtcT>;

// Type to use for inputs and outputs of simple functions with BigintEnum types.
// E.g. arg_type<BigintEnum<E1>> and out_type<BigintEnum<E1>>.
template <typename E>
struct BigintEnumT {
  using type = int64_t;

  static inline const std::string typeName = "bigint_enum(" + E::name() + ")";
};

template <typename E>
using BigintEnum = CustomType<BigintEnumT<E>>;

// Type to use for inputs and outputs of simple functions with VarcharEnum
// types. E.g. arg_type<VarcharEnum<E1>> and out_type<VarcharEnum<E1>>.
template <typename E>
struct VarcharEnumT {
  using type = Varchar;

  static inline const std::string typeName = "varchar_enum(" + E::name() + ")";
};

template <typename E>
using VarcharEnum = CustomType<VarcharEnumT<E>>;

template <typename T>
struct Constant {};

template <typename T>
struct UnwrapConstantType {
  using type = T;
};

template <typename T>
struct UnwrapConstantType<Constant<T>> {
  using type = T;
};

template <typename T>
struct isConstantType {
  static constexpr bool value = false;
};

template <typename T>
struct isConstantType<Constant<T>> {
  static constexpr bool value = true;
};

template <typename... TArgs>
struct ConstantChecker {
  static constexpr bool isConstant[sizeof...(TArgs)] = {
      isConstantType<TArgs>::value...};
};

} // namespace facebook::velox
