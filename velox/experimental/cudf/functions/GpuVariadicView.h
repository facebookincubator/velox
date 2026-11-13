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

#include "velox/common/base/Macros.h"

#include <cudf/utilities/bit.hpp>

#include <cstdint>

/// Row access for a GpuArgView, and the view a variadic argument arrives as.
namespace facebook::velox::cudf_velox::gpu_sfi {

namespace detail {

/// Row-to-element mapping for one argument. A constant reads element 0 for
/// every row.
VELOX_GPU_COMPATIBLE inline cudf::size_type argIndex(
    const GpuArgView& argument,
    cudf::size_type row) {
  return argument.isConstant ? 0 : row + argument.offset;
}

VELOX_GPU_COMPATIBLE inline bool argIsNull(
    const GpuArgView& argument,
    cudf::size_type row) {
  return argument.nullMask != nullptr &&
      !cudf::bit_is_set(argument.nullMask, argIndex(argument, row));
}

template <typename T>
VELOX_GPU_COMPATIBLE inline const T& argValue(
    const GpuArgView& argument,
    cudf::size_type row) {
  return static_cast<const T*>(argument.data)[argIndex(argument, row)];
}

} // namespace detail

/// One element of a variadic pack: a value, or nothing. Offers the
/// has_value()/value() interface a call() body reads, since std::optional is
/// not usable on the device. Not convertible to bool, so that `if (arg)` on a
/// three-valued input does not compile.
template <typename T>
class GpuOptionalValue {
 public:
  VELOX_GPU_COMPATIBLE explicit GpuOptionalValue(const T* value)
      : value_(value) {}

  VELOX_GPU_COMPATIBLE bool has_value() const {
    return value_ != nullptr;
  }

  VELOX_GPU_COMPATIBLE const T& value() const {
    return *value_;
  }

 private:
  const T* value_;
};

/// A variadic argument as the kernel sees it: like Velox's VariadicView, a lazy
/// window onto the tail of the GpuArgView array. Each element is a separate
/// column, so elements are read only through at().
template <typename T>
class GpuVariadicView {
 public:
  VELOX_GPU_COMPATIBLE GpuVariadicView(
      const GpuArgView* arguments,
      int32_t size,
      cudf::size_type row)
      : arguments_(arguments), size_(size), row_(row) {}

  VELOX_GPU_COMPATIBLE int32_t size() const {
    return size_;
  }

  /// The i-th argument of the pack at this row, empty when that argument is
  /// null here. No bounds check: callers iterate to size().
  VELOX_GPU_COMPATIBLE GpuOptionalValue<T> at(int32_t i) const {
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
