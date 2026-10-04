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

#include "velox/experimental/cudf/types/GpuTimestamp.cuh"

#include <cstdint>

namespace facebook::velox {
template <typename P, typename S>
struct ShortDecimal;
template <typename P, typename S>
struct LongDecimal;
struct Date;
struct IntervalDayTime;
struct IntervalYearMonth;
struct Time;
class Timestamp;
template <typename T>
struct Variadic;
template <typename T, bool providesCustomComparison>
struct CustomType;
} // namespace facebook::velox

namespace facebook::velox::cudf_velox::gpu_sfi {
/// Declared, not included: the resolver only names it, and its definition
/// pulls in the cudf headers.
template <typename T>
class GpuVariadicView;
} // namespace facebook::velox::cudf_velox::gpu_sfi

namespace facebook::velox::gpu {

namespace detail {

template <typename T>
struct resolver {
  using in_type = T;
  using out_type = T;
  using null_free_in_type = T;
};

template <typename P, typename S>
struct resolver<ShortDecimal<P, S>> {
  using in_type = int64_t;
  using out_type = int64_t;
  using null_free_in_type = int64_t;
};

template <typename P, typename S>
struct resolver<LongDecimal<P, S>> {
  using in_type = __int128;
  using out_type = __int128;
  using null_free_in_type = __int128;
};

template <>
struct resolver<Date> {
  using in_type = int32_t;
  using out_type = int32_t;
  using null_free_in_type = int32_t;
};

template <>
struct resolver<IntervalDayTime> {
  using in_type = int64_t;
  using out_type = int64_t;
  using null_free_in_type = int64_t;
};

template <>
struct resolver<IntervalYearMonth> {
  using in_type = int32_t;
  using out_type = int32_t;
  using null_free_in_type = int32_t;
};

template <>
struct resolver<Time> {
  using in_type = int64_t;
  using out_type = int64_t;
  using null_free_in_type = int64_t;
};

template <>
struct resolver<Timestamp> {
  using in_type = GpuTimestamp;
  using out_type = GpuTimestamp;
  using null_free_in_type = GpuTimestamp;
};

/// A custom type resolves to its physical type, so TimestampWithTimezone
/// arrives as the packed int64 the column holds. Velox wraps a comparable one
/// in a view that reaches its Type for compare(); a kernel has no Type, so a
/// body reads the value directly.
template <typename T, bool providesCustomComparison>
struct resolver<CustomType<T, providesCustomComparison>> {
  using in_type = typename resolver<typename T::type>::in_type;
  using out_type = typename resolver<typename T::type>::out_type;
  using null_free_in_type =
      typename resolver<typename T::type>::null_free_in_type;
};

/// A variadic pack resolves to a view over its element type. Only in_type is
/// meaningful: a pack is never a return type, and the view reports nullity per
/// element.
template <typename T>
struct resolver<Variadic<T>> {
  using in_type =
      cudf_velox::gpu_sfi::GpuVariadicView<typename resolver<T>::in_type>;
  using out_type = void;
  using null_free_in_type = in_type;
};

} // namespace detail

struct GpuExec {
  template <typename T>
  using resolver = detail::resolver<T>;
};

} // namespace facebook::velox::gpu
