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

// GPU shadow for velox/type/SimpleFunctionApi.h. Keeps the type tags, which
// SimpleFunctionTags.h defines and which parse under nvcc, and drops the
// host-only function-signature machinery the real header stacks on them.
//
// The SimpleTypeTrait specialisations below are copied from the real header,
// where they sit above the host-only half. Registration reads
// SimpleTypeTrait<T>::name to build the signature strings the host matches
// against, so they must stay identical.
#pragma once

#include "velox/type/SimpleFunctionTags.h"

namespace facebook::velox {

// Defined in Type.h, which the device side only declares; named here as a
// template argument alone.
struct Time;

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

// SimpleTypeTrait<TimeMicroUtc> is the one specialisation not carried over:
// unlike the tags above, TimeMicroUtc is declared in Type.h rather than
// SimpleFunctionTags.h, so the type does not exist in a device translation unit
// to specialise on. A function taking one cannot be registered here anyway.

} // namespace facebook::velox
