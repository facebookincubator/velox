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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <stdexcept>
#include <string>
#include <string_view>

#include "velox/dwio/nimble/common/Types.h"

// ---------------------------------------------------------------------------
// Element type selection for the ML ID drivers.
// ---------------------------------------------------------------------------
// Every driver runs one element type per invocation, chosen by --mlidc_dtype.
//
// The supported set is exactly the one SubIntSplitEncoding documents at
// SubIntSplitEncoding.h:50-52 and its own typed-test suite uses at
// encodings/tests/SubIntSplitEncodingTest.cpp:280: int32, uint32, int64,
// uint64, float, double. It is a ceiling rather than a preference. SubIntSplit
// static_asserts sizeof(physicalType) == 4 || == 8 (SubIntSplitEncoding.h:77),
// so the 8- and 16-bit types cannot be instantiated at all while SubIntSplit is
// in the encoder suite, and bool/string are not numeric.
//
// Floats reach the encodings through their physical type
// (TypeTraits<float>::physicalType == uint32_t, common/Types.h:315), so the
// encodings compress the bit pattern. Nothing here needs to know that; it
// matters in the drivers that hand samples to the SubIntSplit sampler.
//
// DataType is reused as the type tag rather than declaring a parallel enum, so
// the names reported in the CSV come from nimble's own toString(DataType).

namespace facebook::nimble::mlidc {

// The --mlidc_dtype values this suite accepts, in the order they are dispatched.
inline constexpr std::string_view kElemTypeNames =
    "int32, uint32, int64, uint64, float, double";

// Maps a --mlidc_dtype value onto the DataType tag used to dispatch.
//
// Throws rather than falling back to a default: a typo that silently ran int64
// would produce a plausible-looking CSV attributed to the wrong type.
inline DataType parseElemDataType(std::string_view name) {
  if (name == "int32") {
    return DataType::Int32;
  }
  if (name == "uint32") {
    return DataType::Uint32;
  }
  if (name == "int64") {
    return DataType::Int64;
  }
  if (name == "uint64") {
    return DataType::Uint64;
  }
  if (name == "float") {
    return DataType::Float;
  }
  if (name == "double") {
    return DataType::Double;
  }
  throw std::runtime_error(
      "Unknown --mlidc_dtype value: '" + std::string(name) +
      "'. Supported: " + std::string(kElemTypeNames) +
      ". The 8- and 16-bit types are excluded because SubIntSplitEncoding "
      "only supports 32- and 64-bit types.");
}

// Name reported in the dtype CSV column, e.g. "Int64", "Double".
template <typename T>
inline std::string elemTypeName() {
  return toString(TypeTraits<T>::dataType);
}

// Calls fn.operator()<T>() with T fixed to the type `dataType` names.
//
// fn must be a generic lambda taking an explicit template argument:
//
//   dispatchElemType(dt, [&]<typename T>() { return run<T>(); });
//
// This is a restricted twin of NIMBLE_RETURN_BY_NUMERIC_DATA_TYPE_OR
// (common/DataTypeDispatch.h:194), which cannot be used directly because it
// also expands the 8- and 16-bit cases that SubIntSplit rejects at compile
// time.
template <typename Fn>
decltype(auto) dispatchElemType(DataType dataType, Fn&& fn) {
  switch (dataType) {
    case DataType::Int32:
      return fn.template operator()<int32_t>();
    case DataType::Uint32:
      return fn.template operator()<uint32_t>();
    case DataType::Int64:
      return fn.template operator()<int64_t>();
    case DataType::Uint64:
      return fn.template operator()<uint64_t>();
    case DataType::Float:
      return fn.template operator()<float>();
    case DataType::Double:
      return fn.template operator()<double>();
    default:
      throw std::runtime_error(
          "Unsupported element type: " + toString(dataType) +
          ". Supported: " + std::string(kElemTypeNames));
  }
}

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
