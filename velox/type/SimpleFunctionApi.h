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

/// The simple function tag types plus their mappings onto the runtime type
/// system. Include SimpleFunctionTags.h instead if you only need the tags.

#include "velox/type/CppToType.h"
#include "velox/type/SimpleFunctionTags.h"

namespace facebook::velox {

/// CppToType templates for the tag types in SimpleFunctionTags.h.

template <>
struct CppToType<Varchar> : public CppToTypeBase<TypeKind::VARCHAR> {};

template <>
struct CppToType<Varbinary> : public CppToTypeBase<TypeKind::VARBINARY> {};

template <>
struct CppToType<Date> : public CppToTypeBase<TypeKind::INTEGER> {};

template <>
struct CppToType<TimestampUtc> : public CppToTypeBase<TypeKind::TIMESTAMP> {
  static auto create() {
    return TimestampUtcType::get();
  }
};

template <typename T>
struct CppToType<Generic<T>> : public CppToTypeBase<TypeKind::UNKNOWN> {};

template <typename KEY, typename VAL>
struct CppToType<Map<KEY, VAL>> : public TypeTraits<TypeKind::MAP> {
  static auto create() {
    return MAP(CppToType<KEY>::create(), CppToType<VAL>::create());
  }
};

template <typename ELEMENT>
struct CppToType<Array<ELEMENT>> : public TypeTraits<TypeKind::ARRAY> {
  static auto create() {
    return ARRAY(CppToType<ELEMENT>::create());
  }
};

template <typename... T>
struct CppToType<Row<T...>> : public TypeTraits<TypeKind::ROW> {
  static auto create() {
    return ROW({CppToType<T>::create()...});
  }
};

template <>
struct CppToType<DynamicRow> : public TypeTraits<TypeKind::ROW> {
  static std::shared_ptr<const Type> create() {
    throw std::logic_error{"can't determine exact type for DynamicRow"};
  }
};

template <typename T, bool providesCustomComparison>
struct CppToType<CustomType<T, providesCustomComparison>>
    : public CppToType<typename T::type> {
  static auto create() {
    return CppToType<typename T::type>::create();
  }
};

/// SimpleTypeTrait template.

template <typename P, typename S>
struct SimpleTypeTrait<ShortDecimal<P, S>>
    : public TypeTraits<TypeKind::BIGINT> {};

template <typename P, typename S>
struct SimpleTypeTrait<LongDecimal<P, S>>
    : public TypeTraits<TypeKind::HUGEINT> {};

template <>
struct SimpleTypeTrait<Varchar> : public TypeTraits<TypeKind::VARCHAR> {};

template <>
struct SimpleTypeTrait<Varbinary> : public TypeTraits<TypeKind::VARBINARY> {};

template <>
struct SimpleTypeTrait<Date> : public TypeTraits<TypeKind::INTEGER> {
  static constexpr const char* name = "DATE";
};

template <>
struct SimpleTypeTrait<IntervalDayTime> : public TypeTraits<TypeKind::BIGINT> {
  static constexpr const char* name = "INTERVAL DAY TO SECOND";
};

template <>
struct SimpleTypeTrait<IntervalYearMonth>
    : public TypeTraits<TypeKind::INTEGER> {
  static constexpr const char* name = "INTERVAL YEAR TO MONTH";
};

template <>
struct SimpleTypeTrait<Time> : public TypeTraits<TypeKind::BIGINT> {
  static constexpr const char* name = "TIME";
};

template <>
struct SimpleTypeTrait<TimeMicroUtc> : public TypeTraits<TypeKind::BIGINT> {
  static constexpr const char* name = "TIME MICRO UTC";
};

template <typename T, bool comparable, bool orderable>
struct SimpleTypeTrait<Generic<T, comparable, orderable>> {
  static constexpr TypeKind typeKind = TypeKind::INVALID;
  static constexpr bool isPrimitiveType = false;
  static constexpr bool isFixedWidth = false;
};

template <typename T>
struct SimpleTypeTrait<std::shared_ptr<T>>
    : public TypeTraits<TypeKind::OPAQUE> {};

template <typename KEY, typename VAL>
struct SimpleTypeTrait<Map<KEY, VAL>> : public TypeTraits<TypeKind::MAP> {};

template <typename ELEMENT>
struct SimpleTypeTrait<Array<ELEMENT>> : public TypeTraits<TypeKind::ARRAY> {};

template <typename... T>
struct SimpleTypeTrait<Row<T...>> : public TypeTraits<TypeKind::ROW> {};

template <>
struct SimpleTypeTrait<DynamicRow> : public TypeTraits<TypeKind::ROW> {};

// T is also a simple type that represent the physical type of the custom type.
template <typename T, bool providesCustomComparison>
struct SimpleTypeTrait<CustomType<T, providesCustomComparison>>
    : public SimpleTypeTrait<typename T::type> {
  using physical_t = SimpleTypeTrait<typename T::type>;
  static constexpr TypeKind typeKind = physical_t::typeKind;
  static constexpr bool isPrimitiveType = physical_t::isPrimitiveType;
  static constexpr bool isFixedWidth = physical_t::isFixedWidth;

  // This is different than the physical type name.
  static constexpr const char* name = T::typeName;
};

/// MaterializeType template.

template <typename T>
struct MaterializeType {
  using null_free_t = T;
  using nullable_t = T;
  static constexpr bool requiresMaterialization = false;
};

template <typename V>
struct MaterializeType<Array<V>> {
  using null_free_t = std::vector<typename MaterializeType<V>::null_free_t>;
  using nullable_t =
      std::vector<std::optional<typename MaterializeType<V>::nullable_t>>;
  static constexpr bool requiresMaterialization = true;
};

template <typename K, typename V>
struct MaterializeType<Map<K, V>> {
  using key_t = typename MaterializeType<K>::null_free_t;

  using nullable_t = folly::
      F14FastMap<key_t, std::optional<typename MaterializeType<V>::nullable_t>>;

  using null_free_t =
      folly::F14FastMap<key_t, typename MaterializeType<V>::null_free_t>;
  static constexpr bool requiresMaterialization = true;
};

template <typename... T>
struct MaterializeType<Row<T...>> {
  using nullable_t =
      std::tuple<std::optional<typename MaterializeType<T>::nullable_t>...>;

  using null_free_t = std::tuple<typename MaterializeType<T>::null_free_t...>;
  static constexpr bool requiresMaterialization = true;
};

template <typename T>
struct MaterializeType<std::shared_ptr<T>> {
  using nullable_t = T;
  using null_free_t = T;
  static constexpr bool requiresMaterialization = false;
};

template <typename T, bool providesCustomComparison>
struct MaterializeType<CustomType<T, providesCustomComparison>> {
  using inner_materialize_t = MaterializeType<typename T::type>;
  using nullable_t = typename inner_materialize_t::nullable_t;
  using null_free_t = typename inner_materialize_t::null_free_t;
  static constexpr bool requiresMaterialization =
      inner_materialize_t::requiresMaterialization;
};

template <>
struct MaterializeType<Varchar> {
  using nullable_t = std::string;
  using null_free_t = std::string;
  static constexpr bool requiresMaterialization = true;
};

template <>
struct MaterializeType<Varbinary> {
  using nullable_t = std::string;
  using null_free_t = std::string;
  static constexpr bool requiresMaterialization = true;
};

} // namespace facebook::velox
