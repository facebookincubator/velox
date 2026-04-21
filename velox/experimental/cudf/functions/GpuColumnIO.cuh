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

#include "velox/experimental/cudf/types/GpuProxyTypes.cuh"
#include "velox/experimental/cudf/types/GpuStringView.cuh"
#include "velox/experimental/cudf/types/GpuTimestamp.cuh"

#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/span.hpp>

namespace facebook::velox::gpu {

// ------------------------------------------------------------------
// GpuColumnReader: per-row device-side reader for cuDF columns
// ------------------------------------------------------------------

template <typename T>
struct GpuColumnReader {
  cudf::device_span<const T> data;

  GPU_DEVICE T read(cudf::size_type row) const {
    return data[row];
  }
};

template <>
struct GpuColumnReader<GpuStringView> {
  const int32_t* offsets;
  const char* chars;

  GPU_DEVICE GpuStringView read(cudf::size_type row) const {
    int32_t start = offsets[row];
    int32_t len = offsets[row + 1] - start;
    return GpuStringView(chars + start, len);
  }
};

template <>
struct GpuColumnReader<GpuTimestamp> {
  cudf::device_span<const int64_t> epochNanos;

  GPU_DEVICE GpuTimestamp read(cudf::size_type row) const {
    int64_t ns = epochNanos[row];
    return GpuTimestamp(
        ns / 1000000000LL,
        static_cast<uint64_t>(ns % 1000000000LL));
  }
};

// ------------------------------------------------------------------
// GpuColumnWriter: per-row device-side writer for output columns
// ------------------------------------------------------------------

template <typename T>
struct GpuColumnWriter {
  T* data;
  cudf::bitmask_type* nullBitmask;

  GPU_DEVICE void write(cudf::size_type row, const T& value) {
    data[row] = value;
  }

  GPU_DEVICE void setNull(cudf::size_type row) {
    cudf::clear_bit(nullBitmask, row);
  }

  GPU_DEVICE void setValid(cudf::size_type row) {
    cudf::set_bit(nullBitmask, row);
  }
};

template <>
struct GpuColumnWriter<GpuTimestamp> {
  int64_t* epochNanos;
  cudf::bitmask_type* nullBitmask;

  GPU_DEVICE void write(cudf::size_type row, const GpuTimestamp& value) {
    epochNanos[row] = value.seconds * 1000000000LL +
        static_cast<int64_t>(value.nanos);
  }

  GPU_DEVICE void setNull(cudf::size_type row) {
    cudf::clear_bit(nullBitmask, row);
  }

  GPU_DEVICE void setValid(cudf::size_type row) {
    cudf::set_bit(nullBitmask, row);
  }
};

} // namespace facebook::velox::gpu
