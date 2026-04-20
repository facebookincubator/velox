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

#include "velox/experimental/cudf/types/GpuStringView.cuh"
#include "velox/experimental/cudf/types/GpuTimestamp.cuh"
#include <cstdint>

namespace facebook::velox {
struct Varchar;
struct Varbinary;
template <typename P, typename S>
struct ShortDecimal;
template <typename P, typename S>
struct LongDecimal;
struct Date;
struct IntervalDayTime;
struct IntervalYearMonth;
struct Time;
class Timestamp;
} // namespace facebook::velox

namespace facebook::velox::gpu {

namespace detail {

template <typename T>
struct resolver {
  using in_type = T;
  using out_type = T;
  using null_free_in_type = T;
};

template <>
struct resolver<Varchar> {
  using in_type = GpuStringView;
  using out_type = GpuStringView;
  using null_free_in_type = GpuStringView;
};

template <>
struct resolver<Varbinary> {
  using in_type = GpuStringView;
  using out_type = GpuStringView;
  using null_free_in_type = GpuStringView;
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

} // namespace detail

struct GpuExec {
  template <typename T>
  using resolver = detail::resolver<T>;
};

} // namespace facebook::velox::gpu
