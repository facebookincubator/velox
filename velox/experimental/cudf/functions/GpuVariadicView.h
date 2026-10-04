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

#include "velox/common/base/CheckedArithmetic.h"
#include "velox/type/Timestamp.h"

#include <cudf/utilities/bit.hpp>

#include <cstdint>

/// Row access for a GpuArgView, and the view a variadic argument arrives as.
namespace facebook::velox::cudf_velox::gpu_sfi {

namespace detail {

/// Row-to-element mapping for one argument. A constant reads element 0 for
/// every row.
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

/// A timestamp argument at this row, split from cuDF's one integer per row
/// into the seconds and nanoseconds in [0, 1e9) Velox's Timestamp holds; one
/// tick before the epoch is second -1.
GPU_HOST_DEVICE inline Timestamp argTimestamp(
    const GpuArgView& argument,
    cudf::size_type row) {
  const int64_t ticks = argValue<int64_t>(argument, row);
  const int64_t ticksPerSecond = argument.ticksPerSecond;
  int64_t seconds = ticks / ticksPerSecond;
  int64_t remainder = ticks % ticksPerSecond;
  if (remainder < 0) {
    seconds -= 1;
    remainder += ticksPerSecond;
  }
  return Timestamp(
      seconds,
      static_cast<uint64_t>(remainder * (1'000'000'000 / ticksPerSecond)));
}

/// A timestamp result as cuDF's one integer per row in the output column's
/// unit: the reverse of argTimestamp(). Nanoseconds finer than the unit are
/// dropped, as Timestamp::toMicros() drops them. False when the unit cannot
/// hold the instant, as a nanosecond column cannot past the year 2262.
GPU_HOST_DEVICE inline bool timestampTicks(
    const Timestamp& timestamp,
    int64_t ticksPerSecond,
    int64_t& ticks) {
  int64_t wholeSeconds{0};
  if (::facebook::velox::detail::mulOverflow(
          timestamp.getSeconds(), ticksPerSecond, &wholeSeconds)) {
    return false;
  }
  const int64_t subSecond = static_cast<int64_t>(timestamp.getNanos()) /
      (1'000'000'000 / ticksPerSecond);
  return !::facebook::velox::detail::addOverflow(
      wholeSeconds, subSecond, &ticks);
}

} // namespace detail

/// One element of a variadic pack: a value, or nothing. Offers the
/// has_value()/value() interface a call() body reads, since std::optional is
/// not usable on the device. Not convertible to bool, so that `if (arg)` on a
/// three-valued input does not compile.
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

/// A variadic argument as the kernel sees it: like Velox's VariadicView, a lazy
/// window onto the tail of the GpuArgView array. Each element is a separate
/// column, so elements are read only through at().
template <typename T>
class GpuVariadicView {
  // at() hands out a pointer into the column, and a timestamp has to be
  // converted on the way out of it.
  static_assert(
      !std::is_same_v<T, Timestamp>,
      "Variadic timestamp arguments are not supported yet");

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
