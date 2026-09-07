/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <span>
#include <type_traits>

#include "velox/dwio/nimble/common/Types.h"

namespace facebook::nimble {

// This class is used to manage semantic type and its corresponding encoding
// type. For most data types, they are the same. They will be different for
// floating point types. The main reason it is different for floating point
// numbers are because of +/-0.0 ( +0.0 == -0.0 but their sign bits are
// different), and NaNs can have different payload bits but even if NaNs
// with the same bits, == will return false, that causes trouble for encoding.
// So they will be treated as integers during encoding (their type will
// still be encoded as double/float)
template <typename T>
struct EncodingPhysicalType {
  using type = typename TypeTraits<T>::physicalType;

  static std::span<type> asEncodingPhysicalTypeSpan(std::span<T> values) {
    return std::span<type>(
        reinterpret_cast<type*>(values.data()), values.size());
  }

  static std::span<const type> asEncodingPhysicalTypeSpan(
      std::span<const T> values) {
    return std::span<const type>(
        reinterpret_cast<const type*>(values.data()), values.size());
  }

  static type asEncodingPhysicalType(T v) {
    return *reinterpret_cast<type*>(&v);
  }

  static T asEncodingLogicalType(type v) {
    return *reinterpret_cast<T*>(&v);
  }
};

#define NIMBLE_AS(D, v) *(reinterpret_cast<D*>(&(v)))

#define NIMBLE_AS_CONST(D, v) *(reinterpret_cast<const D*>(&(v)))
} // namespace facebook::nimble
