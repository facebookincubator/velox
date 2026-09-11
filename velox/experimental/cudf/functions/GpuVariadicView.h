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

#include "velox/experimental/cudf/functions/GpuFunctionRegistry.h"
#include "velox/experimental/cudf/types/GpuProxyTypes.cuh"

#include <cudf/utilities/bit.hpp>

#include <cstdint>

/// Row access for a GpuArgView, and the view a variadic argument arrives as.
///
/// These live together because they are the same idea at two arities: reading
/// row r out of one argument, and reading row r out of a run of arguments whose
/// length is not known until the expression is compiled.
namespace facebook::velox::cudf_velox::gpu_sfi {

namespace detail {

/// Row-to-element mapping for one argument. A constant resolves to element 0
/// for every row, which is the same indirection DecodedVector performs on the
/// CPU side and the reason a literal argument needs no special case here.
GPU_HOST_DEVICE inline cudf::size_type argIndex(
    const GpuArgView& argument,
    cudf::size_type row) {
  return argument.isConstant ? 0 : row + argument.offset;
}

GPU_HOST_DEVICE inline bool argIsNull(
    const GpuArgView& argument,
    cudf::size_type row) {
  return argument.nullMask != nullptr &&
      !cudf::bit_is_set(argument.nullMask, argIndex(argument, row));
}

template <typename T>
GPU_HOST_DEVICE inline const T& argValue(
    const GpuArgView& argument,
    cudf::size_type row) {
  return static_cast<const T*>(argument.data)[argIndex(argument, row)];
}

} // namespace detail

/// One element of a variadic pack: a value, or nothing.
///
/// Named after std::optional's interface rather than Velox's OptionalAccessor
/// because that is what a call() body reads -- has_value() and value() -- and
/// std::optional itself is not usable from device code. Deliberately not
/// convertible to bool: `if (arg)` on a three-valued input is the mistake this
/// type exists to make hard to write.
template <typename T>
class GpuOptionalValue {
 public:
  GPU_HOST_DEVICE explicit GpuOptionalValue(const T* value) : value_(value) {}

  GPU_HOST_DEVICE bool has_value() const {
    return value_ != nullptr;
  }

  GPU_HOST_DEVICE const T& value() const {
    return *value_;
  }

 private:
  const T* value_;
};

/// A variadic argument as the kernel sees it.
///
/// Mirrors Velox's VariadicView: a lazy window onto the arguments in the pack,
/// materialising nothing. Velox's version wraps a range of DecodedVectors and
/// hands out OptionalAccessors; this wraps the tail of the GpuArgView array and
/// hands out GpuOptionalValues, which is the same shape over the same
/// indirection.
///
/// Elements are always read through at(), never as a contiguous block, because
/// the arguments in a pack are separate columns -- adjacent in the descriptor
/// array, unrelated in memory.
template <typename T>
class GpuVariadicView {
 public:
  GPU_HOST_DEVICE GpuVariadicView(
      const GpuArgView* arguments,
      int32_t size,
      cudf::size_type row)
      : arguments_(arguments), size_(size), row_(row) {}

  GPU_HOST_DEVICE int32_t size() const {
    return size_;
  }

  /// The i-th argument of the pack at this row, empty when that argument is
  /// null here. No bounds check: callers iterate to size().
  GPU_HOST_DEVICE GpuOptionalValue<T> at(int32_t i) const {
    if (detail::argIsNull(arguments_[i], row_)) {
      return GpuOptionalValue<T>{nullptr};
    }
    return GpuOptionalValue<T>{&detail::argValue<T>(arguments_[i], row_)};
  }

 private:
  const GpuArgView* arguments_;
  int32_t size_;
  cudf::size_type row_;
};

template <typename>
struct isGpuVariadicView : std::false_type {};

template <typename T>
struct isGpuVariadicView<GpuVariadicView<T>> : std::true_type {};

} // namespace facebook::velox::cudf_velox::gpu_sfi
