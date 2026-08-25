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

// GPU shadow for velox/type/SimpleFunctionApi.h.
//
// The real header is two things stacked: the type tags a call() body names, and
// the function-signature machinery (FunctionSignature, SignatureBuilder, the
// simple-function reflection utilities) that is host-only. Only the first half
// is wanted here.
//
// The tags themselves are no longer restated. SimpleFunctionTags.h holds them
// and parses under nvcc on its own, so it is included directly -- one
// definition of Date and Generic<> rather than a copy that can drift from it.
// What remains is the SimpleTypeTrait specialisations, which live in the real
// SimpleFunctionApi.h above the host-only half and so cannot be reached
// without it. They are reproduced verbatim; registration reads
// SimpleTypeTrait<T>::name to build the signature strings the host bridge
// matches against, so a divergence here would show up as a function that
// silently never resolves.
#pragma once

#include "velox/type/SimpleFunctionTags.h"

namespace facebook::velox {

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

// SimpleTypeTrait<TimeMicroUtc> is the one specialisation not carried over:
// unlike the tags above, TimeMicroUtc is declared in Type.h rather than
// SimpleFunctionTags.h, so the type does not exist in a device translation unit
// to specialise on. A function taking one cannot be registered here anyway.

} // namespace facebook::velox
